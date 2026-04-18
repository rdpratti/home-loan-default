"""
graph_analytics.py
------------------
Graph analytics module for Home Credit Default prediction.
Builds a customer relationship graph from shared attributes
and computes per-customer graph features for use in the
Bayesian Network.

Designed to plug into the existing home-credit-bayesian.py pipeline.

Usage in get_merged_data():
    from graph_analytics import build_graph, compute_graph_features, log_graph_diagnostics
    
    G           = build_graph(app_train, logger)
    graph_df    = compute_graph_features(G, app_train, logger)
    log_graph_diagnostics(graph_df, logger)
    
    # Then merge graph_df into merged_df like any other summary
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


# ─── 1. BUILD THE GRAPH ──────────────────────────────────────────────────────

def build_graph(app_train: pd.DataFrame, logger) -> nx.Graph:
    """
    Build a bipartite graph connecting customers to shared attributes.

    Nodes:
        - Customer nodes:     "cust_{SK_ID_CURR}"
        - Organization nodes: "org_{ORGANIZATION_TYPE}"
        - Region nodes:       "reg_{REGION_POPULATION_RELATIVE}"  (binned)
        - Occupation nodes:   "occ_{OCCUPATION_TYPE}"

    Edges:
        - Customer ↔ Organization  (works at same org type)
        - Customer ↔ Region        (lives in same population region)
        - Customer ↔ Occupation    (same occupation type)

    Each customer node carries the TARGET label so neighbor
    default rates can be computed.
    """
    logger.debug("build_graph: constructing customer relationship graph")
    G = nx.Graph()

    # ── Add customer nodes ────────────────────────────────────────────────
    for _, row in app_train.iterrows():
        G.add_node(
            f"cust_{row['SK_ID_CURR']}",
            node_type  = 'customer',
            defaulted  = int(row['TARGET']) if pd.notna(row['TARGET']) else -1,
            sk_id_curr = row['SK_ID_CURR']
        )

    # ── Add organization edges ────────────────────────────────────────────
    # Customers sharing an organization type are connected via an org node
    org_col = 'ORGANIZATION_TYPE'
    if org_col in app_train.columns:
        for org, group in app_train.groupby(org_col):
            org_node = f"org_{org}"
            G.add_node(org_node, node_type='organization')
            for sk_id in group['SK_ID_CURR']:
                G.add_edge(f"cust_{sk_id}", org_node, edge_type='organization')
        logger.debug(f"build_graph: added organization edges for "
                     f"{app_train[org_col].nunique()} org types")

    # ── Add region edges ──────────────────────────────────────────────────
    # Bin continuous region population into 5 buckets first
    reg_col = 'REGION_POPULATION_RELATIVE'
    if reg_col in app_train.columns:
        app_train = app_train.copy()
        app_train['region_bin'] = pd.qcut(
            app_train[reg_col], q=5, labels=False, duplicates='drop'
        )
        for reg, group in app_train.groupby('region_bin'):
            reg_node = f"reg_{reg}"
            G.add_node(reg_node, node_type='region')
            for sk_id in group['SK_ID_CURR']:
                G.add_edge(f"cust_{sk_id}", reg_node, edge_type='region')
        logger.debug(f"build_graph: added region edges for 5 region bins")

    # ── Add occupation edges ──────────────────────────────────────────────
    occ_col = 'OCCUPATION_TYPE'
    if occ_col in app_train.columns:
        occ_data = app_train.dropna(subset=[occ_col])
        for occ, group in occ_data.groupby(occ_col):
            occ_node = f"occ_{occ}"
            G.add_node(occ_node, node_type='occupation')
            for sk_id in group['SK_ID_CURR']:
                G.add_edge(f"cust_{sk_id}", occ_node, edge_type='occupation')
        logger.debug(f"build_graph: added occupation edges for "
                     f"{occ_data[occ_col].nunique()} occupation types")

    n_customers = sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'customer')
    logger.debug(f"build_graph: graph complete — "
                 f"{G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges, "
                 f"{n_customers} customer nodes")
    return G


# ─── 2. COMPUTE GRAPH FEATURES ───────────────────────────────────────────────

def compute_graph_features(G: nx.Graph, logger) -> pd.DataFrame:
    """
    Compute per-customer graph features:

        neighbor_default_rate   : proportion of customer's customer-neighbors
                                  who defaulted — guilt by association signal
        customer_degree         : number of shared-attribute connections
                                  (highly connected = unusual pattern)
        org_default_rate        : default rate of all customers in same org
        region_default_rate     : default rate of all customers in same region

    All required customer data (SK_ID_CURR, defaulted flag) is read directly
    from node attributes on *G* — no separate DataFrame is needed.

    Returns a DataFrame with SK_ID_CURR + graph feature columns.
    """
    logger.debug("compute_graph_features: computing per-customer graph features")

    # Pre-compute org and region default rates for efficiency
    org_default_rates    = _compute_hub_default_rates(G, 'organization')
    region_default_rates = _compute_hub_default_rates(G, 'region')

    records = []
    customer_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get('node_type') == 'customer'
    ]

    for node, data in customer_nodes:
        sk_id = data['sk_id_curr']

        # Neighbor customer default rate
        neighbor_defaults = []
        for neighbor in G.neighbors(node):
            nd = G.nodes[neighbor]
            if nd.get('node_type') == 'customer' and nd.get('defaulted') != -1:
                neighbor_defaults.append(nd['defaulted'])
        neighbor_default_rate = (
            np.mean(neighbor_defaults) if neighbor_defaults else np.nan
        )

        # Degree — total connections (customer + hub nodes)
        degree = G.degree(node)

        # Org default rate via hub node
        org_rate = np.nan
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor].get('node_type') == 'organization':
                org_rate = org_default_rates.get(neighbor, np.nan)
                break

        # Region default rate via hub node
        region_rate = np.nan
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor].get('node_type') == 'region':
                region_rate = region_default_rates.get(neighbor, np.nan)
                break

        records.append({
            'SK_ID_CURR'          : sk_id,
            'neighbor_default_rate': neighbor_default_rate,
            'customer_degree'      : degree,
            'org_default_rate'     : org_rate,
            'region_default_rate'  : region_rate,
        })

    graph_df = pd.DataFrame(records)
    logger.debug(f"compute_graph_features: completed — {len(graph_df)} customer records")
    return graph_df


def _compute_hub_default_rates(G: nx.Graph, hub_type: str) -> dict:
    """
    For each hub node (org/region), compute the default rate
    of all customer nodes connected to it.
    """
    rates = {}
    for node, data in G.nodes(data=True):
        if data.get('node_type') == hub_type:
            defaults = [
                G.nodes[n]['defaulted']
                for n in G.neighbors(node)
                if G.nodes[n].get('node_type') == 'customer'
                and G.nodes[n].get('defaulted') != -1
            ]
            rates[node] = np.mean(defaults) if defaults else np.nan
    return rates


# ─── 3. BIN FOR BAYESIAN NETWORK ─────────────────────────────────────────────

def bin_graph_features(graph_df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Convert continuous graph features into categorical bins
    suitable for the Discrete Bayesian Network.

    Adds columns:
        NeighborRisk    : Low / Medium / High / VeryHigh
        OrgRisk         : Low / Medium / High
        RegionRisk      : Low / Medium / High
        CustomerHub     : Low / Medium / High  (degree-based)
    """
    df = graph_df.copy()

    df['NeighborRisk'] = pd.cut(
        df['neighbor_default_rate'],
        bins  = [-0.01, 0.10, 0.20, 0.30, 1.01],
        labels= ['Low', 'Medium', 'High', 'VeryHigh']
    )

    df['OrgRisk'] = pd.cut(
        df['org_default_rate'],
        bins  = [-0.01, 0.15, 0.25, 1.01],
        labels= ['Low', 'Medium', 'High']
    )

    df['RegionRisk'] = pd.cut(
        df['region_default_rate'],
        bins  = [-0.01, 0.15, 0.25, 1.01],
        labels= ['Low', 'Medium', 'High']
    )

    df['CustomerHub'] = pd.cut(
        df['customer_degree'],
        bins  = [-1, 2, 4, 999],
        labels= ['Low', 'Medium', 'High']
    )

    # Fill nulls — customers with no neighbor data get 'Unknown'
    for col in ['NeighborRisk', 'OrgRisk', 'RegionRisk', 'CustomerHub']:
        df[col] = df[col].cat.add_categories('Unknown').fillna('Unknown')

    logger.debug(f"bin_graph_features: binned graph features\n"
                 f"{df[['NeighborRisk','OrgRisk','RegionRisk','CustomerHub']].value_counts().head(10).to_string()}")
    return df


# ─── 4. DIAGNOSTICS ──────────────────────────────────────────────────────────

def log_graph_diagnostics(graph_df: pd.DataFrame, app_train: pd.DataFrame, logger):
    """
    Log raw graph feature distributions split by TARGET
    to assess discriminating power before binning.
    """
    labeled = app_train[['SK_ID_CURR', 'TARGET']].dropna(subset=['TARGET'])
    labeled['LoanOutcome'] = np.where(labeled['TARGET'] == 0, 'Repaid', 'Defaulted')

    diag = graph_df.merge(labeled[['SK_ID_CURR', 'LoanOutcome']], on='SK_ID_CURR')

    numeric_cols = graph_df.select_dtypes(include='number')\
                           .columns\
                           .difference(['SK_ID_CURR'])\
                           .tolist()

    for col in numeric_cols:
        logger.debug(f"graph feature '{col}' by LoanOutcome:\n"
                     f"{diag.groupby('LoanOutcome')[col].describe().to_string()}")


def plot_graph_feature_distributions(graph_df: pd.DataFrame,
                                      app_train: pd.DataFrame,
                                      logger):
    """
    Plot histograms of graph features split by LoanOutcome.
    Saved to logs/ directory.
    """
    labeled = app_train[['SK_ID_CURR', 'TARGET']].dropna(subset=['TARGET'])
    labeled['LoanOutcome'] = np.where(labeled['TARGET'] == 0, 'Repaid', 'Defaulted')
    diag = graph_df.merge(labeled[['SK_ID_CURR', 'LoanOutcome']], on='SK_ID_CURR')

    features = ['neighbor_default_rate', 'org_default_rate',
                'region_default_rate',   'customer_degree']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, features):
        for outcome, color in [('Repaid', 'steelblue'), ('Defaulted', 'tomato')]:
            subset = diag[diag['LoanOutcome'] == outcome][col].dropna()
            ax.hist(subset, bins=40, alpha=0.6, color=color,
                    label=outcome, edgecolor='black', linewidth=0.3)
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()

    fig.suptitle('Graph Feature Distributions by LoanOutcome', fontsize=13)
    fig.tight_layout()

    Path('logs').mkdir(exist_ok=True)
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'logs/graph_features_{ts}.png'
    fig.savefig(plot_path)
    plt.close(fig)
    logger.debug(f"graph feature plot saved: {plot_path}")


def plot_subgraph(G: nx.Graph, sk_id_curr: int, logger, hops: int = 2):
    """
    Visualize the local neighborhood of a single customer
    up to `hops` away. Useful for inspecting high-risk customers.

    Color coding:
        Red (tomato)    = Defaulted customer
        Blue (steelblue) = Repaid customer
        Grey (lightgrey) = Hub node (org/region/occupation)
    """
    root = f"cust_{sk_id_curr}"
    if root not in G:
        logger.debug(f"plot_subgraph: customer {sk_id_curr} not found in graph")
        return

    # Get subgraph within N hops
    nodes  = nx.single_source_shortest_path_length(G, root, cutoff=hops)
    sub_G  = G.subgraph(nodes.keys()).copy()

    color_map = []
    for node in sub_G.nodes():
        nd = sub_G.nodes[node]
        if nd.get('node_type') == 'customer':
            color_map.append('tomato' if nd.get('defaulted') == 1 else 'steelblue')
        else:
            color_map.append('lightgrey')

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(sub_G, seed=42)
    nx.draw_networkx(
        sub_G, pos, ax=ax,
        node_color=color_map,
        node_size=300,
        font_size=7,
        edge_color='grey',
        alpha=0.85
    )
    ax.set_title(f"Local Graph — Customer {sk_id_curr} ({hops}-hop neighborhood)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tomato',    label='Defaulted'),
        Patch(facecolor='steelblue', label='Repaid'),
        Patch(facecolor='lightgrey', label='Hub node'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    Path('logs').mkdir(exist_ok=True)
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'logs/subgraph_{sk_id_curr}_{ts}.png'
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    logger.debug(f"subgraph plot saved: {plot_path}")


# ─── 5. INTEGRATION HELPER ───────────────────────────────────────────────────

def get_graph_summary(app_train: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Top-level function to call from get_merged_data().
    Builds graph, computes features, bins them, runs diagnostics.

    Returns a DataFrame with SK_ID_CURR + binned graph feature columns
    ready to merge into merged_df.

    Example in get_merged_data():
        from graph_analytics import get_graph_summary
        graph_summary = get_graph_summary(app_data, logger)
        merged_df = merged_df.merge(
            graph_summary[['SK_ID_CURR', 'NeighborRisk', 'OrgRisk']],
            on='SK_ID_CURR', how='left'
        )
    """
    logger.debug("get_graph_summary: starting graph analytics pipeline")

    G          = build_graph(app_train, logger)
    graph_df   = compute_graph_features(G, logger)

    log_graph_diagnostics(graph_df, app_train, logger)
    plot_graph_feature_distributions(graph_df, app_train, logger)

    binned_df  = bin_graph_features(graph_df, logger)

    logger.debug("get_graph_summary: complete")
    return binned_df


# ─── EXAMPLE EXPERT LIST ADDITIONS ───────────────────────────────────────────
#
# After merging graph_summary into merged_df, add to expert_list:
#
# ('NeighborRisk', 'LoanOutcome')   # guilt by association
# ('OrgRisk',      'LoanOutcome')   # employer sector risk
# ('RegionRisk',   'LoanOutcome')   # geographic risk
#
# Or as indirect influences:
# ('NeighborRisk', 'PriorLoanApproved')
# ('OrgRisk',      'IncomeType')
