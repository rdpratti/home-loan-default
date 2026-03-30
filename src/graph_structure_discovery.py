"""
graph_structure_discovery.py
----------------------------
Builds a feature-dependency graph and applies graph-theoretic algorithms
to suggest edges for the expert Bayesian Network (expert_list).

Unlike graph_analytics.py (which builds a customer-relationship graph for
per-customer features), this module builds a VARIABLE-DEPENDENCY graph
where nodes are feature columns and edges represent statistical associations.

Entry point:
    suggested, report = suggest_expert_list(df, target, logger)
    → suggested : list of (parent, child) tuples for expert_list
    → report    : dict with intermediate results for inspection

Techniques:
    1. NMI with target        — rank direct parent candidates
    2. Pairwise NMI matrix     — weighted feature-dependency graph
    3. Chow-Liu spanning tree  — strongest non-redundant dependency backbone
    4. Conditional NMI         — direct vs mediated association test
    5. Community detection     — cluster features; suggest intermediate edges
    6. Betweenness centrality  — identify natural mediator / hub nodes

Usage in main():
    from graph_structure_discovery import suggest_expert_list, plot_all

    suggested, report = suggest_expert_list(train_df, 'LoanOutcome', logger)
    plot_all(report, 'LoanOutcome', logger)
    # inspect suggested, then hand-tune into expert_list
"""

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import normalized_mutual_info_score
from itertools import combinations
from pathlib import Path
from datetime import datetime


# ─── 1. FEATURE-TARGET NMI ───────────────────────────────────────────────────

def compute_feature_target_mi(df: pd.DataFrame, target: str, logger) -> pd.Series:
    """
    NMI(feature, target) for every feature column.
    Higher = stronger direct association with the outcome.
    Returns a Series sorted descending.
    """
    features = [c for c in df.columns if c != target]
    t = df[target].astype(str)

    scores = {
        col: normalized_mutual_info_score(
            df[col].astype(str), t, average_method='arithmetic'
        )
        for col in features
    }
    result = pd.Series(scores).sort_values(ascending=False)

    logger.debug(f"feature→target NMI ({target}):\n"
                 + "\n".join(f"  {k:<28} {v:.4f}" for k, v in result.items()))
    return result


# ─── 2. PAIRWISE NMI MATRIX ──────────────────────────────────────────────────

def compute_pairwise_nmi(df: pd.DataFrame, target: str, logger) -> pd.DataFrame:
    """
    All-pairs NMI between feature columns (target excluded).
    Returns a symmetric n×n DataFrame.

    NOTE: runs C(n,2) NMI calls — fast for n≤30 on 300k rows.
    """
    features = [c for c in df.columns if c != target]
    pairs    = list(combinations(features, 2))
    logger.debug(f"compute_pairwise_nmi: {len(features)} features → "
                 f"{len(pairs)} pairs")

    matrix = pd.DataFrame(0.0, index=features, columns=features)
    for f1, f2 in pairs:
        nmi = normalized_mutual_info_score(
            df[f1].astype(str), df[f2].astype(str),
            average_method='arithmetic'
        )
        matrix.loc[f1, f2] = nmi
        matrix.loc[f2, f1] = nmi
    for f in features:
        matrix.at[f, f] = 1.0

    # Log top 10 pairs
    pair_scores = sorted(
        [(f1, f2, matrix.loc[f1, f2]) for f1, f2 in pairs],
        key=lambda x: x[2], reverse=True
    )
    logger.debug("top 10 pairwise NMI:\n"
                 + "\n".join(f"  {f1:<25} ↔ {f2:<25} {s:.4f}"
                              for f1, f2, s in pair_scores[:10]))
    return matrix


# ─── 3. BUILD MI GRAPH ───────────────────────────────────────────────────────

def build_mi_graph(nmi_matrix: pd.DataFrame, threshold: float,
                   logger) -> nx.Graph:
    """
    Weighted undirected graph from the NMI matrix.
    Edges with NMI < threshold are omitted (noise suppression).
    """
    G   = nx.Graph()
    G.add_nodes_from(nmi_matrix.columns)
    features = nmi_matrix.columns.tolist()

    for f1, f2 in combinations(features, 2):
        w = nmi_matrix.loc[f1, f2]
        if w >= threshold:
            G.add_edge(f1, f2, weight=w)

    logger.debug(f"build_mi_graph: threshold={threshold:.3f} → "
                 f"{G.number_of_edges()} edges kept")
    return G


# ─── 4. CHOW-LIU MAXIMUM SPANNING TREE ───────────────────────────────────────

def chow_liu_tree(nmi_matrix: pd.DataFrame, logger) -> nx.Graph:
    """
    Maximum spanning tree of the NMI graph (Chow-Liu algorithm).

    The MST is the best tree-structured approximation to the joint
    distribution — each edge in it is a strong, non-redundant
    direct dependency.  Cycle-free, so safe to use as a BN skeleton.
    """
    G_full = nx.Graph()
    features = nmi_matrix.columns.tolist()
    for f1, f2 in combinations(features, 2):
        w = nmi_matrix.loc[f1, f2]
        if w > 0:
            G_full.add_edge(f1, f2, weight=w)

    mst = nx.maximum_spanning_tree(G_full, weight='weight')

    logger.debug(f"chow_liu_tree: {mst.number_of_edges()} edges")
    for u, v, d in sorted(mst.edges(data=True),
                           key=lambda x: x[2]['weight'], reverse=True):
        logger.debug(f"  {u:<28} — {v:<28} {d['weight']:.4f}")
    return mst


# ─── 5. CONDITIONAL NMI ──────────────────────────────────────────────────────

def conditional_nmi(df: pd.DataFrame, x_col: str, y_col: str,
                    z_col: str) -> float:
    """
    MI(X ; Y | Z) — association between X and Y after controlling for Z.

    Computed as the Z-stratum-weighted average of NMI(X,Y) within each
    value of Z.  Requires at least 10 rows per stratum.

    Interpretation:
        conditional_nmi >> 0   → X-Y are directly linked (Z doesn't explain it)
        conditional_nmi ≈ 0    → Z mediates the X-Y association
    """
    n     = len(df)
    total = 0.0
    for zv, sub in df.groupby(z_col):
        if len(sub) < 10:
            continue
        w     = len(sub) / n
        score = normalized_mutual_info_score(
            sub[x_col].astype(str), sub[y_col].astype(str),
            average_method='arithmetic'
        )
        total += w * score
    return total


# ─── 6. COMMUNITY DETECTION ──────────────────────────────────────────────────

def detect_communities(G: nx.Graph, logger) -> dict:
    """
    Greedy modularity community detection on the MI graph.
    Features in the same community share information structure —
    they are natural candidates for intermediate BN edges.

    Returns dict: feature → community_id (int).
    """
    if G.number_of_edges() == 0:
        logger.debug("detect_communities: no edges, single community")
        return {n: 0 for n in G.nodes()}

    comms = list(nx.community.greedy_modularity_communities(G, weight='weight'))
    mapping = {node: cid for cid, comm in enumerate(comms) for node in comm}

    logger.debug(f"detect_communities: {len(comms)} communities")
    for cid, comm in enumerate(comms):
        logger.debug(f"  [{cid}] {sorted(comm)}")
    return mapping


# ─── 7. CENTRALITY ───────────────────────────────────────────────────────────

def compute_centrality(G: nx.Graph, logger) -> pd.DataFrame:
    """
    Betweenness and degree centrality on the MI graph.

    High betweenness → this feature sits on many shortest paths
                       between others → natural mediator in the BN.
    High degree       → correlated with many other features.
    """
    degree  = nx.degree_centrality(G)
    between = nx.betweenness_centrality(G, weight='weight', normalized=True)

    df_c = pd.DataFrame({
        'degree_centrality'     : degree,
        'betweenness_centrality': between,
    }).sort_values('betweenness_centrality', ascending=False)

    logger.debug(f"centrality (sorted by betweenness):\n{df_c.to_string()}")
    return df_c


# ─── 8. MEDIATION ANALYSIS ───────────────────────────────────────────────────

def find_mediators(df: pd.DataFrame, target: str,
                   target_mi: pd.Series, top_k: int,
                   drop_fraction: float, logger) -> dict:
    """
    For every pair among the top-k direct parent candidates, test whether
    one mediates the other's association with target.

    Rule: if  cond_NMI(f1, target | f2)  <  drop_fraction × NMI(f1, target)
          then f2 likely mediates f1→target.
          Suggest: f1 → f2 → target  rather than  f1 → target directly.

    Returns dict: {mediated_feature: mediating_feature}
    """
    candidates = target_mi.head(top_k).index.tolist()
    mediated   = {}

    logger.debug(f"find_mediators: testing top {top_k} candidates")
    for f1, f2 in combinations(candidates, 2):
        mi_f1 = target_mi[f1]
        if mi_f1 < 1e-4:
            continue
        cond = conditional_nmi(df, f1, target, f2)
        ratio = cond / mi_f1
        logger.debug(f"  NMI({f1}→{target}|{f2}): {cond:.4f}  "
                     f"(marginal={mi_f1:.4f}, ratio={ratio:.2f})")
        if ratio < drop_fraction:
            if f1 not in mediated:
                mediated[f1] = f2
                logger.debug(f"  → {f2} mediates {f1}→{target}")

    return mediated


# ─── 9. ORIENT EDGES ─────────────────────────────────────────────────────────

def _orient(u: str, v: str, target: str,
            target_mi: pd.Series) -> tuple:
    """
    Orient edge (u, v) as (parent, child):
      • If one endpoint is target → other node is parent
      • Otherwise → higher MI-with-target node is parent
        (more informative about outcome = closer to root)
    """
    if v == target:
        return (u, target)
    if u == target:
        return (v, target)
    return (u, v) if target_mi.get(u, 0) >= target_mi.get(v, 0) else (v, u)


# ─── 10. MAIN ENTRY POINT ────────────────────────────────────────────────────

def suggest_expert_list(df: pd.DataFrame, target: str, logger,
                        mi_threshold: float   = 0.002,
                        drop_fraction: float  = 0.50,
                        mediation_top_k: int  = 10) -> tuple:
    """
    Full pipeline. Returns:
        suggested   : list of (parent, child) tuples, best edges first
        report      : dict with all intermediate artefacts for inspection

    Parameters
    ----------
    df              : merged modeling DataFrame (all object dtype, no SK_ID_CURR)
    target          : outcome column ('LoanOutcome')
    mi_threshold    : minimum NMI to include an edge (noise floor)
    drop_fraction   : fraction drop in conditional NMI that signals mediation
    mediation_top_k : how many top features to test for mediation
    """
    logger.debug("=" * 60)
    logger.debug("suggest_expert_list: starting structure discovery")
    logger.debug("=" * 60)

    # ── 1. NMI with target ───────────────────────────────────────────────────
    target_mi = compute_feature_target_mi(df, target, logger)

    # ── 2. Pairwise NMI matrix ───────────────────────────────────────────────
    nmi_matrix = compute_pairwise_nmi(df, target, logger)

    # ── 3. MI graph ──────────────────────────────────────────────────────────
    G = build_mi_graph(nmi_matrix, threshold=mi_threshold, logger=logger)

    # ── 4. Chow-Liu tree ─────────────────────────────────────────────────────
    mst = chow_liu_tree(nmi_matrix, logger)

    # ── 5. Community detection ───────────────────────────────────────────────
    communities = detect_communities(G, logger)

    # ── 6. Centrality ────────────────────────────────────────────────────────
    centrality = compute_centrality(G, logger)

    # ── 7. Mediation analysis ─────────────────────────────────────────────────
    mediated = find_mediators(df, target, target_mi,
                              top_k=mediation_top_k,
                              drop_fraction=drop_fraction,
                              logger=logger)

    # ── 8. Collect candidate edges ────────────────────────────────────────────
    seen     = set()
    scored   = []   # (parent, child, nmi_score, source_tag)

    def _add(parent, child, score, tag):
        key = (parent, child)
        if key not in seen and parent != child:
            seen.add(key)
            scored.append((parent, child, score, tag))

    # 8a. Direct → target: all features above threshold, strongest first
    for feat, mi_score in target_mi.items():
        if mi_score >= mi_threshold and feat not in mediated:
            p, c = _orient(feat, target, target, target_mi)
            _add(p, c, mi_score, 'direct')

    # 8b. Mediated features: f → mediator (mediator already goes to target above)
    for feat, mediator in mediated.items():
        p, c = _orient(feat, mediator, target, target_mi)
        mi   = nmi_matrix.loc[feat, mediator]
        _add(p, c, mi, 'mediated→indirect')

    # 8c. Chow-Liu tree edges between features (not involving target)
    for u, v, d in mst.edges(data=True):
        if u == target or v == target:
            continue
        p, c = _orient(u, v, target, target_mi)
        _add(p, c, d['weight'], 'chow-liu')

    # 8d. Intra-community edges not already in MST:
    #     strongest pairwise NMI within each community (adds depth)
    comm_edges = {}
    features   = [c for c in df.columns if c != target]
    for f1, f2 in combinations(features, 2):
        if communities.get(f1) == communities.get(f2):
            w = nmi_matrix.loc[f1, f2]
            comm_edges[(f1, f2)] = w
    for (f1, f2), w in sorted(comm_edges.items(),
                               key=lambda x: x[1], reverse=True)[:20]:
        p, c = _orient(f1, f2, target, target_mi)
        _add(p, c, w, 'community')

    # ── 9. Sort: direct edges first, then by NMI ─────────────────────────────
    order = {'direct': 0, 'mediated→indirect': 1, 'chow-liu': 2, 'community': 3}
    scored.sort(key=lambda x: (order.get(x[3], 9), -x[2]))

    # ── 10. Log suggested expert_list ─────────────────────────────────────────
    logger.debug(f"\n{'='*60}")
    logger.debug(f"SUGGESTED expert_list  ({len(scored)} edges)")
    logger.debug(f"{'='*60}")
    logger.debug(f"  {'Parent':<28} {'Child':<28} {'NMI':>6}  Source")
    logger.debug(f"  {'-'*72}")
    for parent, child, score, tag in scored:
        logger.debug(f"  {parent:<28} {child:<28} {score:>6.4f}  [{tag}]")

    logger.debug(f"\nexpert_list = [")
    for parent, child, *_ in scored:
        logger.debug(f"    ('{parent}', '{child}'),")
    logger.debug(f"]")

    report = {
        'target_mi'    : target_mi,
        'nmi_matrix'   : nmi_matrix,
        'G'            : G,
        'mst'          : mst,
        'communities'  : communities,
        'centrality'   : centrality,
        'mediated'     : mediated,
        'expert_scored': scored,
    }
    suggested = [(p, c) for p, c, _, _ in scored]

    # ── 11. Auto-generate plots ───────────────────────────────────────────────
    plot_dependency_graph(G, target_mi, communities, target, logger)
    plot_chow_liu_tree(mst, target_mi, target, logger)

    return suggested, report


# ─── 11. DIRECT-PARENT SELECTION ─────────────────────────────────────────────

def select_direct_parents(df: pd.DataFrame, target: str,
                          candidates: list, max_parents: int,
                          logger) -> pd.DataFrame:
    """Greedily select the best direct parents for a target node.

    Uses **forward selection driven by conditional NMI**: starting from an
    empty parent set, at each step the candidate that still contributes the
    most information about ``target`` — *given what has already been selected*
    — is added.  Candidates that are redundant with already-selected parents
    have near-zero conditional NMI and are naturally de-prioritised.

    This answers the question "which of these candidates should be *direct*
    parents vs. connected only through an intermediate node?" without needing
    to run a full structure search.

    Algorithm
    ---------
    Round 0 (no parents selected yet):
        Score each candidate by its marginal NMI with ``target``.
        Select the highest scorer.

    Round k (k parents already selected):
        For each remaining candidate C, compute
            MI(C ; target | selected_1, …, selected_k)
        approximated by conditioning on the *joint* parent set one variable
        at a time and averaging (tractable without assuming independence).
        Select the candidate with the highest conditional score.

    Stop when ``max_parents`` is reached or the best remaining candidate's
    conditional NMI drops below 10 % of its marginal NMI (diminishing
    returns).

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling DataFrame (object dtype columns, including ``target``).
    target : str
        The node whose parents we are selecting (e.g. ``'LoanOutcome'``).
    candidates : list[str]
        Pool of features to evaluate as potential direct parents.
    max_parents : int
        Hard upper limit on the number of parents selected.
    logger :
        Standard Python logger.

    Returns
    -------
    pandas.DataFrame
        One row per selected parent, columns:

        ``parent``
            Feature name.
        ``marginal_nmi``
            NMI(feature, target) — unconditional baseline.
        ``conditional_nmi``
            NMI(feature, target | already-selected parents) at the round it
            was chosen.  Equal to ``marginal_nmi`` for the first selection.
        ``retention_pct``
            ``100 × conditional_nmi / marginal_nmi`` — how much of the
            original signal survives after conditioning.  Low values (<30 %)
            suggest the candidate's effect is mostly mediated by the already-
            selected parents; it is likely better placed as an *indirect*
            parent (grandparent) rather than a direct one.
        ``round``
            Selection order (1 = chosen first).

    Notes
    -----
    The remaining (not selected) candidates are also logged with their final
    conditional NMI, showing how much *residual* signal they would add beyond
    the chosen set.  Features with near-zero residual are good candidates for
    indirect (grandparent) roles in the DAG.
    """
    logger.debug("=" * 60)
    logger.debug(f"select_direct_parents: target={target}, "
                 f"candidates={candidates}, max_parents={max_parents}")
    logger.debug("=" * 60)

    # Marginal NMI for every candidate — the Round-0 baseline
    marginal = {}
    for c in candidates:
        marginal[c] = normalized_mutual_info_score(
            df[c].astype(str), df[target].astype(str),
            average_method='arithmetic'
        )
    logger.debug("Marginal NMI with target:")
    for c, v in sorted(marginal.items(), key=lambda x: -x[1]):
        logger.debug(f"  {c:<28} {v:.4f}")

    selected  = []   # chosen parents in order
    remaining = list(candidates)
    results   = []

    for rnd in range(1, max_parents + 1):
        if not remaining:
            break

        best_feat  = None
        best_score = -1.0

        for c in remaining:
            if not selected:
                # Round 1: no conditioning yet — use marginal NMI
                score = marginal[c]
            else:
                # Condition on each already-selected parent one at a time
                # and average — approximates full joint conditioning.
                scores_per_parent = [
                    conditional_nmi(df, c, target, z_col=p)
                    for p in selected
                ]
                score = sum(scores_per_parent) / len(scores_per_parent)

            if score > best_score:
                best_score = score
                best_feat  = c

        if best_feat is None:
            break

        # Stop if conditional NMI has dropped to < 10 % of marginal
        retention = best_score / marginal[best_feat] if marginal[best_feat] > 0 else 0
        logger.debug(f"Round {rnd}: select '{best_feat}'  "
                     f"conditional_nmi={best_score:.4f}  "
                     f"marginal={marginal[best_feat]:.4f}  "
                     f"retention={retention*100:.1f}%")

        results.append({
            'round':           rnd,
            'parent':          best_feat,
            'marginal_nmi':    marginal[best_feat],
            'conditional_nmi': best_score,
            'retention_pct':   round(retention * 100, 1),
        })

        selected.append(best_feat)
        remaining.remove(best_feat)

        if retention < 0.10:
            logger.debug("  → retention < 10 %, stopping early")
            break

    # Log residual signal in candidates that were NOT selected
    logger.debug("\nResidual signal in non-selected candidates "
                 "(conditional on full selected set):")
    for c in remaining:
        if selected:
            scores_per_parent = [
                conditional_nmi(df, c, target, z_col=p) for p in selected
            ]
            residual = sum(scores_per_parent) / len(scores_per_parent)
        else:
            residual = marginal[c]
        logger.debug(f"  {c:<28} residual_nmi={residual:.4f}  "
                     f"marginal={marginal[c]:.4f}  "
                     f"retained={100*residual/marginal[c]:.1f}%"
                     if marginal[c] > 0 else
                     f"  {c:<28} marginal=0")

    result_df = pd.DataFrame(results)[
        ['round', 'parent', 'marginal_nmi', 'conditional_nmi', 'retention_pct']
    ]
    logger.debug(f"\nSelected direct parents for '{target}':\n"
                 f"{result_df.to_string(index=False)}")
    return result_df


# ─── 12. VISUALIZATIONS ──────────────────────────────────────────────────────

def plot_dependency_graph(G: nx.Graph, target_mi: pd.Series,
                          communities: dict, target: str, logger):
    """
    Feature-dependency graph.
      Node size  ∝ NMI with target
      Node color = community
      Edge width ∝ NMI weight
      Target node in gold
    """
    if G.number_of_nodes() == 0:
        return

    cmap   = plt.cm.get_cmap('tab10')
    n_comm = max(communities.values()) + 1 if communities else 1

    node_colors = [
        'gold' if n == target
        else cmap((communities.get(n, 0) % 10) / 10)
        for n in G.nodes()
    ]
    node_sizes  = [
        1400 if n == target
        else 300 + 3000 * target_mi.get(n, 0)
        for n in G.nodes()
    ]
    weights     = [G[u][v]['weight'] for u, v in G.edges()]
    max_w       = max(weights) if weights else 1
    edge_widths = [3.0 * w / max_w for w in weights]

    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(G, weight='weight', seed=42, k=2.0)

    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edge_color='grey',      width=edge_widths,
        font_size=8,            font_weight='bold',
        alpha=0.9
    )

    handles = [mpatches.Patch(color='gold', label=f'{target} (target)')]
    for cid in range(n_comm):
        handles.append(mpatches.Patch(
            color=cmap((cid % 10) / 10), label=f'Community {cid}'
        ))
    ax.legend(handles=handles, loc='upper left', fontsize=7)
    ax.set_title(
        'Feature Dependency Graph  '
        '(node size ∝ MI with target | color = community | edge ∝ NMI)',
        fontsize=11
    )

    Path('logs').mkdir(exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/dependency_graph_{ts}.png'
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.debug(f"dependency graph saved: {path}")


def plot_chow_liu_tree(mst: nx.Graph, target_mi: pd.Series,
                       target: str, logger):
    """
    Chow-Liu maximum spanning tree with NMI edge labels.
    Tree structure = best non-redundant dependency backbone.
    """
    if mst.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(mst, weight='weight', seed=42, k=2.5)

    node_colors = ['gold' if n == target else 'steelblue' for n in mst.nodes()]
    node_sizes  = [
        1400 if n == target
        else 300 + 3000 * target_mi.get(n, 0)
        for n in mst.nodes()
    ]
    edge_labels = {(u, v): f"{d['weight']:.3f}"
                   for u, v, d in mst.edges(data=True)}

    nx.draw_networkx(
        mst, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edge_color='steelblue',  width=2,
        font_size=8,             font_weight='bold',
        alpha=0.9
    )
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels,
                                 font_size=7, ax=ax)
    ax.set_title(
        'Chow-Liu Maximum Spanning Tree  '
        '(optimal tree-structured feature dependency backbone)',
        fontsize=11
    )

    Path('logs').mkdir(exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/chow_liu_tree_{ts}.png'
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.debug(f"Chow-Liu tree saved: {path}")


def plot_nmi_heatmap(nmi_matrix: pd.DataFrame, target_mi: pd.Series,
                     target: str, logger):
    """
    Heatmap of the full pairwise NMI matrix, with features sorted
    by their NMI with the target (strongest predictors at top/left).
    """
    import seaborn as sns

    # Sort by MI with target
    order      = target_mi.index.tolist()
    mat_sorted = nmi_matrix.loc[order, order]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        mat_sorted,
        annot=True, fmt='.2f', cmap='YlOrRd',
        linewidths=0.5, ax=ax,
        annot_kws={'size': 7},
        vmin=0, vmax=0.5
    )
    ax.set_title(
        f'Pairwise NMI Heatmap  (features sorted by NMI with {target})',
        fontsize=11
    )
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)

    Path('logs').mkdir(exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/nmi_heatmap_{ts}.png'
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.debug(f"NMI heatmap saved: {path}")


def plot_all(report: dict, target: str, logger):
    """
    Convenience: generate all three plots from the report dict.
    """
    plot_nmi_heatmap(
        report['nmi_matrix'], report['target_mi'], target, logger
    )
    plot_dependency_graph(
        report['G'], report['target_mi'],
        report['communities'], target, logger
    )
    plot_chow_liu_tree(
        report['mst'], report['target_mi'], target, logger
    )
