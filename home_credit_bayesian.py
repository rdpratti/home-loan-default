"""
batch_runner.py
---------------
Simple sequential batch controller for Python scripts.
 
Usage:
    python batch_runner.py                  # uses jobs defined in JOBS list below
    python batch_runner.py --stop-on-fail   # halt immediately if any job fails
    python batch_runner.py --log-dir logs   # write per-job logs to a folder
"""
 
import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import statistics
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (HillClimbSearch, BayesianEstimator, MaximumLikelihoodEstimator, PC, ExpertKnowledge)
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, ConfusionMatrixDisplay, f1_score, recall_score, 
                             balanced_accuracy_score, precision_score)
from sklearn.model_selection import train_test_split
import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)
import networkx as nx
sys.path.insert(0, 'src')
from graph_analytics import get_graph_summary

# module-level logger
logger = logging.getLogger(__name__)

#setup logging
def setup_logging(level):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = f'logs/bayesian_model_{timestamp}.log'
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.setLevel(level)
    logger.debug(f"Logger initialized — level: {logging.getLevelName(level)}, file: {log_file}")
    
    return 
 
# Payment on_time_rate and Payment History. Were recent installments paid on time? 
# Analogous to PriorFills in Pre-Auth
# (analog: did member fill prior prescriptions as expected?)
def generate_install_summary(data_dir):
    
    logger.debug(f"Running generate_install_summary dir: {data_dir}")
    
    
    install      = pd.read_csv(f"{data_dir}\\installments_payments.csv")
    
    fill_summary = install.groupby('SK_ID_CURR').apply(
        lambda x: pd.Series({
            'on_time_rate': (x['DAYS_ENTRY_PAYMENT'] <= x['DAYS_INSTALMENT']).mean(),
            'total_fills':  len(x)
        })
        ).reset_index()

    # Bucket on_time_rate into categorical PaymentHistory
    fill_summary['PaymentHistory'] = pd.cut(
            fill_summary['on_time_rate'],
            bins=[0, 0.5, 0.8, 1.01],
            labels=['Low', 'Medium', 'High']
        )
    
    return fill_summary

# Add Credit Utilization
def generate_util_summary(data_dir):
    
    logger.debug(f"Running generate_util_summary dir: {data_dir}")
    
    credit_card  = pd.read_csv(f"{data_dir}\\credit_card_balance.csv")

    credit_card['CREDIT_UTILIZATION'] = np.where(
    credit_card['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
    credit_card['AMT_BALANCE'] / credit_card['AMT_CREDIT_LIMIT_ACTUAL'], 0 )

    util_summary = credit_card.groupby('SK_ID_CURR').agg(
                        avg_utilization = ('CREDIT_UTILIZATION', 'mean'),
                        max_utilization = ('CREDIT_UTILIZATION', 'max')
                    ).reset_index()

    util_summary['CreditUtilization'] = pd.cut(
                        util_summary['avg_utilization'],
                        bins=[-0.01, 0.30, 0.60, 0.90, 999],
                        labels=['Low', 'Medium', 'High', 'MaxedOut']
                    )
    return util_summary

# Add Positive Cash
def generate_pos_summary(data_dir):
    
    logger.debug(f"Running generate_pos_summary dir: {data_dir}")
    
    pos_cash     = pd.read_csv(f"{data_dir}\\POS_CASH_balance.csv")

    pos_summary = pos_cash.groupby('SK_ID_CURR').agg(
                      max_dpd          = ('SK_DPD',               'max'),
                      max_dpd_def      = ('SK_DPD_DEF',           'max'),
                      active_contracts = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum())
                  ).reset_index()

    pos_summary['DPD'] = pd.cut(
                            pos_summary['max_dpd'],
                            bins=[-1, 0, 30, 60, 999999],
                            labels=['None', 'Low', 'Medium', 'High']
                         )

    pos_summary['ContractStatus'] = np.where(pos_summary['active_contracts'] > 0, 'Active', 'Inactive')
    
    return pos_summary

# Add bureau data
# bureau → bureau_summary
def generate_bureau_summary(data_dir):
    
    logger.debug(f"Running generate_bureau_summary dir: {data_dir}")
    
    bureau       = pd.read_csv(f"{data_dir}\\bureau.csv")
    
    bureau_summary = bureau.groupby('SK_ID_CURR').agg(
                        max_days_overdue = ('CREDIT_DAY_OVERDUE',    'max'),
                        max_overdue_amt  = ('AMT_CREDIT_MAX_OVERDUE', 'max'),
                        total_debt       = ('AMT_CREDIT_SUM_DEBT',    'sum'),
                        total_prolonged  = ('CNT_CREDIT_PROLONG',     'sum'),
                        active_credits   = ('CREDIT_ACTIVE', lambda x: (x == 'Active').sum())
                    ).reset_index()

    bureau_summary['DaysOverdue'] = pd.cut(
                                        bureau_summary['max_days_overdue'],
                                        bins=[-1, 0, 30, 90, 999999],
                                        labels=['None', 'Low', 'Medium', 'High']
                                    )
            
    bureau_summary['MaxOverdue'] = pd.cut(
                                        bureau_summary['max_overdue_amt'],
                                        bins=[-1, 0, 1000, 10000, 999999999],
                                        labels=['None', 'Low', 'Medium', 'High']
                                    )
    
    bureau_summary['DebtLoad'] = pd.cut(
                                        bureau_summary['total_debt'],
                                        bins=[-1, 0, 50000, 200000, 999999999],
                                        labels=['None', 'Low', 'Medium', 'High']
                                )
    
    bureau_summary['CreditProlonged'] = pd.cut(
                                            bureau_summary['total_prolonged'],
                                            bins=[-1, 0, 1, 3, 999],
                                            labels=['Never', 'Once', 'Several', 'Many']
                                        )
    
    bureau_summary['ActiveCredits'] = pd.cut(
                                        bureau_summary['active_credits'],
                                        bins=[-1, 0, 1, 3, 999],
                                        labels=['None', 'One', 'Few', 'Many']
                                      )
    
    return bureau_summary

#prev_app → prev_summary
def generate_prev_summary(data_dir):
    
    logger.debug(f"Running generate_prev_summary dir: {data_dir}")
    
    prev_app     = pd.read_csv(f"{data_dir}\\previous_application.csv")

    prev_summary = prev_app.groupby('SK_ID_CURR').agg(
                        prior_approved = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
                        prior_refused  = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
                        prior_total    = ('NAME_CONTRACT_STATUS', 'count')
                    ).reset_index()
    
    prev_summary['PriorLoanApproved'] = np.where(prev_summary['prior_approved'] > 0, 'Yes', 'No')

    prev_summary['PrevRejected'] = pd.cut(
                prev_summary['prior_refused'],
                bins=[-1, 0, 1, 2, 999999],   # ← safer upper bound
                labels=['None', 'Once', 'Twice', 'Multiple']
    )
    
    return prev_summary

def get_app_data2(data_dir):
    
    logger.debug(f"Running get_app_data dir: {data_dir}")
    
    app_train  = pd.read_csv(f"{data_dir}\\application_train.csv")
    app_test   = pd.read_csv(f"{data_dir}\\application_test.csv")
    app_data   = pd.concat([app_train, app_test], ignore_index=True)
    
    # IncomeType: is the applicant's income from a stable source?
    # (analog: is the diagnosis code consistent with the drug requested?)
    new_cols = pd.DataFrame({
                    'IncomeType': np.where(
                        app_data['NAME_INCOME_TYPE'].isin(['Working', 'Commercial associate']),
                            'Stable', 'Unstable'
                ),
                    'OccupationType': np.where(
                        app_data['OCCUPATION_TYPE'].isna(), 'Unknown', np.where(
                                app_data['OCCUPATION_TYPE'].isin(['Managers', 'Core staff',
                                                                   'High skill tech staff',
                                                                   'Medicine staff', 'Accountants']),
                                                                    'Professional', 'Laborer'
                                                                 )
                ),
                        'IncomeBracket': pd.qcut(app_data['AMT_INCOME_TOTAL'], q=3, labels=['Low', 'Medium', 'High']
                ),
                        'LoanOutcome': np.where(app_data['TARGET'] == 0, 'Repaid', 'Defaulted')
                ,
                        'ExtSource1Risk': pd.cut(app_data['EXT_SOURCE_1'],
                                                  bins  = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01],
                                                  labels= ['VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow']),

                        'ExtSource2Risk': pd.cut(app_data['EXT_SOURCE_2'],
                                                  bins  = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01],
                                                  labels= ['VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow']),

                        'ExtSource3Risk': pd.cut(app_data['EXT_SOURCE_3'],
                                                  bins  = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01],
                                                  labels= ['VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow']),

                        'AmtCredit': pd.cut(app_data['AMT_CREDIT'],
                                                  bins  = [0, 252000, 270000, 675000, 1125000, 1350000, 4100000],
                                                  labels= ['MedHigh', 'Med', 'High', 'MedLow', 'VeryLow', 'Low']),

                        'AmtGoodsPrice': pd.cut(app_data['AMT_GOODS_PRICE'],
                                                  bins  = [0, 171000, 270000, 450000, 454500, 675000, 4100000],
                                                  labels= ['High', 'Medium', 'VeryHigh', 'Low', 'MedHigh', 'VeryLow']),

                        'AmtReqCreditBureauMon': pd.cut(app_data['AMT_REQ_CREDIT_BUREAU_MON'],
                                                  bins  = [0, 1, 27],
                                                  labels= ['Medium', 'VeryLow']),                                                                              
    }, index=app_data.index)
    
    temp_df = pd.concat([app_data, new_cols], axis=1).copy()
     
    
    # Summary Counts
    logger.debug(f"IncomeBracket distribution:\n{temp_df['IncomeBracket'].value_counts(dropna=False).to_string()}")
    logger.debug(f"LoanOutcome distribution:\n{temp_df['LoanOutcome'].value_counts(dropna=False).to_string()}")
    for col in ['ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk', 'AmtCredit', 'AmtGoodsPrice', 'AmtReqCreditBureauMon']:
        logger.debug(f"{col} distribution:\n"
                     f"{temp_df[col].value_counts(dropna=False).to_string()}")

    return temp_df

def get_merged_data(data_dir):
    
    logger.debug(f"Running get merged data dir: {data_dir}")

    app_data = get_app_data2(data_dir)
    bureau_summary = generate_bureau_summary(data_dir)
    install_summary = generate_install_summary(data_dir)
    util_summary = generate_util_summary(data_dir)
    pos_summary = generate_pos_summary(data_dir)
    prev_summary = generate_prev_summary(data_dir)

    # Graph analytics — uses TARGET (NaN for test rows → defaulted=-1, excluded
    # from default rate calculations).
    graph_summary = get_graph_summary(app_data, logger)

    # Merge all into merged_df

    merged_df = (
            app_data[['SK_ID_CURR', 'IncomeType', 'OccupationType',
                        'IncomeBracket', 'LoanOutcome',
                        'ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk',
                        'AmtCredit','AmtGoodsPrice', 'AmtReqCreditBureauMon']]
                    .merge(install_summary[['SK_ID_CURR', 'PaymentHistory']],
                       on='SK_ID_CURR', how='left')
                    .merge(prev_summary[['SK_ID_CURR', 'PriorLoanApproved', 'PrevRejected']],
                       on='SK_ID_CURR', how='left')
                    .merge(util_summary[['SK_ID_CURR', 'CreditUtilization']],
                       on='SK_ID_CURR', how='left')
                    .merge(pos_summary[['SK_ID_CURR', 'DPD', 'ContractStatus']],
                       on='SK_ID_CURR', how='left')
                    .merge(bureau_summary[['SK_ID_CURR', 'DaysOverdue', 'MaxOverdue',
                                        'DebtLoad', 'CreditProlonged', 'ActiveCredits']],
                       on='SK_ID_CURR', how='left')
                    .merge(graph_summary[['SK_ID_CURR', 'NeighborRisk', 'OrgRisk',
                                          'RegionRisk', 'CustomerHub']],
                       on='SK_ID_CURR', how='left')
                    .drop(columns=['SK_ID_CURR'])
     )

    # Fill nulls 
    fill_values = {
        'PaymentHistory':   'Low',
        'PriorLoanApproved':'No',
        'PrevRejected':     'None',
        'CreditUtilization':'Low',
        'DPD':              'None',
        'ContractStatus':   'Inactive',
        'DaysOverdue':      'None',
        'MaxOverdue':       'None',
        'DebtLoad':         'None',
        'CreditProlonged':  'Never',
        'ActiveCredits':    'None',
        'NeighborRisk':     'Unknown',
        'OrgRisk':          'Unknown',
        'RegionRisk':       'Unknown',
        'CustomerHub':      'Unknown',
        'ExtSource1Risk': 'Unknown',
        'ExtSource2Risk': 'Unknown',
        'ExtSource3Risk': 'Unknown',
        'AmtCredit': 'Unknown',
        'AmtGoodsPrice': 'Unknown',
        'AmtReqCreditBureauMon': 'Unknown',
    }

    for col, val in fill_values.items():
        if merged_df[col].dtype.name == 'category':
            # Only add category if it doesn't already exist
            if val not in merged_df[col].cat.categories:
                merged_df[col] = merged_df[col].cat.add_categories(val)
            merged_df[col] = merged_df[col].fillna(val)
        else:
            merged_df[col] = merged_df[col].fillna(val)
    
    #convert all colunns except SK_ID_CURR to string object
    model_cols = [col for col in merged_df.columns if col != 'SK_ID_CURR']
    merged_df[model_cols] = merged_df[model_cols].astype('object')

    # ── Confirm ───────────────────────────────────────────────────────────
    logger.debug(f"merged df dtypes:\n{merged_df.dtypes}")
    logger.debug(f"merged df shape:\n{merged_df.shape}")

    log_raw_diagnostics(app_data, install_summary, util_summary, 
                        pos_summary, bureau_summary, prev_summary)
    
    return merged_df

def log_raw_diagnostics(app_data, install_summary, util_summary, 
                         pos_summary, bureau_summary, prev_summary):
    
    labeled = app_data[['SK_ID_CURR', 'LoanOutcome']].dropna(subset=['LoanOutcome'])
    
    summaries = {
        'install_summary' : install_summary,
        'util_summary'    : util_summary,
        'pos_summary'     : pos_summary,
        'bureau_summary'  : bureau_summary,
        'prev_summary'    : prev_summary,
    }
    
    for name, summary in summaries.items():
        numeric_cols = summary.select_dtypes(include='number').columns.difference(['SK_ID_CURR']).tolist()
        diag = summary[['SK_ID_CURR'] + numeric_cols].merge(labeled, on='SK_ID_CURR')
        logger.debug(f"{name} numeric cols by LoanOutcome:\n"
                     f"{diag.groupby('LoanOutcome')[numeric_cols].describe().to_string()}")

    return

def create_expert_model(df, expert_list): 
    
    logger.debug(f"create_expert_model df rows, cols: {df.shape}")
    
#    print("Training data distribution:")
#    print(df['LoanOutcome'].value_counts())
#    print(df['LoanOutcome'].value_counts(normalize=True))
    model = DiscreteBayesianNetwork(expert_list)
    
    #5. FIT THE MODEL (learns CPTs from data) ──────────────────────────
    # BayesianEstimator with Dirichlet prior handles sparse cells gracefully

    model.fit(
            df,
            estimator=BayesianEstimator,
            prior_type='dirichlet',
            pseudo_counts=0.01   # ← let data dominate over smoothing
        )
    
    # Verify the model is valid
    logger.debug(f"Model valid:{model.check_model()}")
    logger.debug(f"Full Influence Model")
    show_influences(model, 'LoanOutcome')
    logger.debug(f"Model CPS")
    logger.debug(f"Model cpds:{model.get_cpds('LoanOutcome')}")
    
    return model


def create_auto_model(df, max_indegree=2, max_iter=int(1e4), scoring_method='bic-d'): 
    """
    Learns BN structure automatically from data using HillClimbSearch.
    
    Parameters:
        df            : modeling dataframe (all object dtype columns)
        max_indegree  : max number of parents per node (default 3)
        max_iter      : max search iterations (default 10000)
        scoring_method: scoring metric for structure search (default 'bic-d')
    
    Returns:
        fitted DiscreteBayesianNetwork model
    """
    
    logger.debug(f"create_auto_model df rows, cols: {df.shape}")
#    print("Training data distribution:")
#    print(df['LoanOutcome'].value_counts())
#    print(df['LoanOutcome'].value_counts(normalize=True))
    
    # Define forbidden edges — LoanOutcome cannot be a parent of anything
    forbidden = [('LoanOutcome', col) for col in df.columns if col != 'LoanOutcome']
    knowledge = ExpertKnowledge(forbidden_edges=forbidden)

    # Learn structure
    hc = HillClimbSearch(df)
    learned_dag = hc.estimate(
        scoring_method=scoring_method,
        max_indegree=max_indegree,
        max_iter=max_iter,
        expert_knowledge=knowledge
    )
    
    logger.debug(f"Structure search complete. Learned edges:\n{list(learned_dag.edges())}")
    
    # Fit probabilities
    auto_model = DiscreteBayesianNetwork(learned_dag.edges())
    auto_model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type='dirichlet',
        pseudo_counts=0.1   # ← let data dominate over smoothing
    )
    
    # Validate
    is_valid = auto_model.check_model()
    logger.debug(f"Auto Model valid: {is_valid}")
       
    if not is_valid:
        raise ValueError("Automated model failed validation check.")
    
    logger.debug(f"Full Influence Auto Model")
    show_influences(auto_model, 'LoanOutcome')
    
    logger.debug(f"Auto Model CPS")
    logger.debug(f"Full Influence Auto Model: {auto_model.get_cpds('LoanOutcome')}")
    
    return auto_model

def show_influences(model, target):
    direct   = set(model.get_parents(target))
    all_anc  = nx.ancestors(model, target)
    indirect = all_anc - direct

    logger.debug(f"\n{'='*40}")
    logger.debug(f"  Influence Map for: {target}")
    logger.debug(f"{'='*40}")
    #print(f"  Direct   ({len(direct)})  : {direct}")
    #print(f"  Indirect ({len(indirect)}): {indirect}")
    
    lines = ["Full chain:"]

    for src, dst in nx.edges(model):
        if dst == target:
            lines.append(f"  {src} ──→ {target}  [DIRECT]")

    for node in indirect:
        paths = list(nx.all_simple_paths(model, node, target))
        for path in paths:
            lines.append(f"  {' → '.join(path)}  [INDIRECT]")

    logger.debug("\n".join(lines))


def  display_confusion_matrix(actuals, predictions):

    # Core metrics
    logger.debug(f"Accuracy: {accuracy_score(actuals, predictions):.3f}")
    logger.debug(f"Classification Report:\n{classification_report(actuals, predictions)}")

    # Confusion matrix
    cm = confusion_matrix(actuals, predictions, labels=['Repaid', 'Defaulted'])
    logger.debug(f"Confusion Matrix (rows=actual, cols=predicted):\n"
                 f"                 Repaid  Defaulted\n"
                 f"  Repaid       {cm[0][0]:>6}  {cm[0][1]:>9}\n"
                 f"  Defaulted    {cm[1][0]:>6}  {cm[1][1]:>9}")

    return

def plot_probability_distribution(probs_list, threshold, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(probs_list, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
    
    # draw threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')
    
    # shade the "predicted Defaulted" region
    ax.axvspan(threshold, 1.0, alpha=0.1, color='red', label='Predicted Defaulted')
    
    ax.set_xlabel('P(Defaulted)')
    ax.set_ylabel('Count')
    ax.set_title(f'{model_name} — Predicted Default Probability Distribution')
    ax.legend()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'logs/{model_name}_prob_dist_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    logger.debug(f"Probability distribution plot saved: {plot_path}")
    return


def predict(test_df, model, target, default_threshold=0.2,mname=None):
    
    logger.debug(f"running predict test df rows, cols: {test_df.shape}")
#    print("Test data distribution:")
#    print(test_df['LoanOutcome'].value_counts())
#    print(test_df['LoanOutcome'].value_counts(normalize=True))
    
    infer = VariableElimination(model)
    # Only use columns that exist in the DAG
    dag_cols     = list(model.nodes())
    test_model   = test_df[dag_cols].copy().astype('object')
    feature_cols = [col for col in dag_cols if col != target]
    
    actuals    = []
    predictions = []
    probs_list = []

    #sample_df   = test_df[test_df['LoanOutcome'] == 'Defaulted']
    for _, row in test_df.iterrows():
        # Build evidence from all non-target columns
        evidence = {col: row[col] for col in feature_cols}
    
        # Query the network
        result = infer.query([target], evidence=evidence, show_progress=False)
        
        # Get probability of Defaulted specifically
        states = result.state_names[target]
        probs  = result.values
        default_prob = probs[states.index('Defaulted')]
        probs_list.append(default_prob)   # ← append each probability

        default_prob = probs[states.index('Defaulted')]
        # Use threshold instead of argmax
        pred = 'Defaulted' if default_prob >= default_threshold else 'Repaid'
        #pred = result.state_names[target][np.argmax(result.values)]
    
        actuals.append(row[target])
        predictions.append(pred)
    
    # log after the loop
    probs_array = np.array(probs_list)

    logger.debug(f"Predicted probability distribution:\n"
                 f"  min:    {probs_array.min():.3f}\n"
                 f"  max:    {probs_array.max():.3f}\n"
                 f"  mean:   {probs_array.mean():.3f}\n"
                 f"  median: {np.median(probs_array):.3f}\n"
                 f"  >0.20:  {(probs_array >= 0.20).sum()}\n"
                 f"  >0.30:  {(probs_array >= 0.30).sum()}\n"
                 f"  >0.40:  {(probs_array >= 0.40).sum()}")
        
    plot_probability_distribution(probs_list, default_threshold, mname)
    #create and print confusion matrix
    display_confusion_matrix(actuals, predictions)
    return actuals, predictions  

def balance_training_data(train_df):
    #guaranteee 75/25 paidoff/defaulted
    ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
    train_resampled, _ = ros.fit_resample(train_df, train_df['LoanOutcome'])
    logger.debug(f"Resampling LoanOutcome distribution:\n{train_resampled['LoanOutcome'].value_counts().to_string()}")
    return train_resampled

def run_sweep(test_df, model, auto_model):
    logger.debug("=== Expert Model Threshold Sweep ===")
    #for threshold in [0.10, 0.15, 0.20, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    for threshold in [0.40, 0.43, 0.45, 0.47, 0.48, 0.50, 0.52, 0.55]:
        acts, preds = predict(test_df, model, 'LoanOutcome', default_threshold=threshold, mname = 'Expert')
        precision = precision_score(acts, preds, pos_label='Defaulted', zero_division=0)
        recall    = recall_score(acts, preds,    pos_label='Defaulted', zero_division=0)
        f1        = f1_score(acts, preds,        pos_label='Defaulted', zero_division=0)
        bal_acc   = balanced_accuracy_score(acts, preds)
        logger.debug(f"Threshold {threshold:.2f} | Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  BalAcc: {bal_acc:.3f}")

    #logger.debug("\n=== Auto Model Threshold Sweep ===")
    #for threshold in [0.10, 0.15, 0.20, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    #    acts, preds = predict(test_df, auto_model, 'LoanOutcome',
    #                          default_threshold=threshold, mname = 'Auto')
    #    precision = precision_score(acts, preds, pos_label='Defaulted', zero_division=0)
    #    recall    = recall_score(acts, preds,    pos_label='Defaulted', zero_division=0)
    #    f1        = f1_score(acts, preds,        pos_label='Defaulted', zero_division=0)
    #    bal_acc   = balanced_accuracy_score(acts, preds)
    #    logger.debug(f"Threshold {threshold:.2f} | Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  BalAcc: {bal_acc:.3f}")
    return    
 
def main(target):

    setup_logging(logging.DEBUG)
    
    data_dir = "data/"
    df = get_merged_data(data_dir)    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    logger.debug(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    for col in [c for c in train_df.columns if c != 'LoanOutcome']:
        ct = pd.crosstab(train_df[col], train_df['LoanOutcome'], normalize='index')
        logger.debug(f"Feature vs LoanOutcome:\n{ct.to_string()}")
    
    train_bal = balance_training_data(train_df)
    
    # Phase 2 — Define and Train the Expert Network
    # DEFINE THE DAG STRUCTURE (domain knowledge)
    # Each tuple is (parent, child) — "parent influences child"
    

    expert_list = [('PrevRejected',          'PriorLoanApproved'),
                    ('PaymentHistory',        'PriorLoanApproved'),
                    ('ContractStatus',        'PriorLoanApproved'),
                    ('ActiveCredits',         'CreditProlonged'),
                    ('MaxOverdue',            'CreditProlonged'),
                    ('AmtCredit',             'AmtGoodsPrice'),
                    ('ExtSource1Risk',        'LoanOutcome'),
                    ('ExtSource2Risk',        'LoanOutcome'),
                    ('ExtSource3Risk',        'LoanOutcome'),
                    ('PriorLoanApproved',     'LoanOutcome'),
                    ('CreditProlonged',       'LoanOutcome'),
                    ('ContractStatus',       'LoanOutcome'),
                    ('AmtGoodsPrice',         'LoanOutcome'), 
                    ('AmtReqCreditBureauMon', 'LoanOutcome'),
                    
                ]

    model = create_expert_model(train_bal, expert_list)
    
    acts1, preds1 = predict(test_df, model, target, default_threshold=.50, mname="Expert")
        
    # Phase 2b — Define and Train the Auto Network
    # 
    auto_model = create_auto_model(df=train_bal,
                          max_indegree=3,        # stricter — each node can have at most 2 parents
                          max_iter=int(1e4),     # more search iterations
                          scoring_method='bic-d'    # alternative scoring method
                          )
    
    acts1, preds1 = predict(test_df, model, target, default_threshold=.50, mname="Auto")
    #run_sweep(test_df, model, auto_model)
    
    return test_df, model, target
 
if __name__ == "__main__":
    main('LoanOutcome')