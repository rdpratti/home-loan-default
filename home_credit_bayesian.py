"""
home_credit_bayesian.py
-----------------------
Bayesian Network pipeline for predicting home-loan default using the Home Credit dataset.

The pipeline:
  1. Loads and merges the application, installment, credit-card, POS-cash, bureau,
     and previous-application CSV files into a single discretised feature table.
  2. Trains two Discrete Bayesian Networks on a class-balanced training split:
       - Expert model  — DAG edges supplied by domain knowledge / graph analysis.
       - Auto model    — DAG learned automatically via Hill-Climb Search (BIC-D).
  3. Runs Variable-Elimination inference on the held-out test set, classifying each
     applicant as 'Defaulted' or 'Repaid' using an adjustable probability threshold.
  4. Reports a confusion matrix, classification metrics, and a predicted-probability
     histogram for each model.

Usage:
    python home_credit_bayesian.py
"""
 

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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay, f1_score, recall_score,
                             balanced_accuracy_score, precision_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging
import networkx as nx
sys.path.insert(0, 'src')
from data_prep import get_merged_data, MODEL_COLS
from graph_structure_discovery import select_direct_parents
#from graph_analytics import get_graph_summary

# module-level logger
logger = logging.getLogger(__name__)

#setup logging
def setup_logging(level):
    """Configure file and console logging for the pipeline run.

    Creates a timestamped log file under ``logs/`` and attaches both a
    ``FileHandler`` and a ``StreamHandler`` (stdout) to the module-level
    logger.  Call this once at the start of ``main()``.

    Parameters
    ----------
    level : int
        Logging level constant, e.g. ``logging.DEBUG`` or ``logging.INFO``.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = f'logs/bayesian_model_{timestamp}.log'

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(open(sys.stdout.fileno(), mode='w',
                                    encoding='utf-8', closefd=False))
    ch.setFormatter(formatter)

    # Attach handlers to the root logger so all module loggers
    # (data_prep, graph_structure_discovery, etc.) write to the same
    # file and console without needing their own handler setup.
    root = logging.getLogger()
    root.addHandler(fh)
    root.addHandler(ch)
    root.setLevel(level)

    # Silence noisy third-party loggers that would otherwise flood the output
    logging.getLogger('pgmpy').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    logger.debug(f"Logger initialized — level: {logging.getLevelName(level)}, file: {log_file}")
    
    return 
 

def create_expert_model(df, expert_list):
    """Fit a Discrete Bayesian Network using a hand-crafted DAG structure.

    The DAG edges in ``expert_list`` encode domain knowledge about how
    applicant and credit-history variables causally influence loan outcome.
    Conditional probability tables (CPTs) are estimated from the (balanced)
    training data using a Bayesian Estimator with a weak Dirichlet prior so
    that sparse parent-state combinations do not produce zero probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        Class-balanced training data (all columns as ``object`` dtype).
    expert_list : list of tuple[str, str]
        DAG edges as ``(parent, child)`` pairs.

    Returns
    -------
    pgmpy.models.DiscreteBayesianNetwork
        Fitted model with CPTs for every node in the DAG.
    """
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
    """Log all direct and indirect influence paths leading to a target node.

    Traverses the DAG to identify which nodes are direct parents of ``target``
    and which reach it through intermediate nodes, then logs each edge/path
    with a [DIRECT] or [INDIRECT] label.  Useful for auditing the model's
    causal structure after fitting.

    Parameters
    ----------
    model : pgmpy.models.DiscreteBayesianNetwork
        The fitted Bayesian Network.
    target : str
        Name of the variable whose influence map should be printed
        (typically ``'LoanOutcome'``).
    """
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


def display_confusion_matrix(actuals, predictions):
    """Log accuracy, a full classification report, and a formatted confusion matrix.

    The confusion matrix is arranged with the positive class (Defaulted) in the
    top-left so that the true-positive count (correctly identified defaults) is
    immediately visible.  Layout::

                     Defaulted   Repaid
        Defaulted      TP          FN
        Repaid         FP          TN

    Parameters
    ----------
    actuals : list[str]
        Ground-truth labels (``'Defaulted'`` or ``'Repaid'``).
    predictions : list[str]
        Model-predicted labels.
    """
    # Core metrics
    logger.debug(f"Accuracy: {accuracy_score(actuals, predictions):.3f}")
    logger.debug(f"Classification Report:\n{classification_report(actuals, predictions)}")

    # Confusion matrix
    cm = confusion_matrix(actuals, predictions, labels=['Defaulted','Repaid'])
    logger.debug(f"Confusion Matrix (rows=actual, cols=predicted):\n"
                 f"                 Defaulted  Repaid\n"
                 f"  Defaulted       {cm[0][0]:>6}  {cm[0][1]:>9}\n"
                 f"  Repaid          {cm[1][0]:>6}  {cm[1][1]:>9}")

    return

def plot_probability_distribution(probs_list, threshold, model_name):
    """Save a histogram of predicted default probabilities with the decision threshold marked.

    Saves a PNG to ``logs/<model_name>_prob_dist_<timestamp>.png``.  The red
    dashed line shows the classification threshold and the shaded region to its
    right highlights applicants classified as Defaulted.  Reviewing this plot
    helps diagnose probability calibration and threshold placement.

    Parameters
    ----------
    probs_list : list[float]
        Per-row P(Defaulted) values from the inference engine.
    threshold : float
        Decision threshold — rows above this value are labelled Defaulted.
    model_name : str
        Label used in the plot title and output filename (e.g. ``'Expert'``).
    """
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


def plot_reliability_diagram(probs, actuals, model_name, n_bins=10):
    """Save a reliability (calibration) diagram comparing predicted probabilities to observed rates.

    A perfectly calibrated model lies on the diagonal.  Points above the
    diagonal mean the model under-predicts default probability; points below
    mean it over-predicts.

    Parameters
    ----------
    probs : list[float]
        Raw (or calibrated) P(Defaulted) values from the inference engine.
    actuals : list[str]
        Ground-truth labels (``'Defaulted'`` or ``'Repaid'``).
    model_name : str
        Label used in the plot title and output filename.
    n_bins : int, optional
        Number of equal-width bins for the calibration curve.  Default 10.
    """
    true_labels = [1 if a == 'Defaulted' else 0 for a in actuals]
    fraction_pos, mean_pred = calibration_curve(true_labels, probs,
                                                n_bins=n_bins, strategy='uniform')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, fraction_pos, 's-', label=model_name)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'{model_name} — Reliability Diagram')
    ax.legend()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'logs/{model_name}_reliability_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.debug(f"Reliability diagram saved: {plot_path}")


def calibrate_model(cal_df, model, target, method='isotonic'):
    """Fit a post-hoc probability calibrator on a held-out calibration set.

    Runs Variable-Elimination inference on ``cal_df`` (which must NOT have been
    oversampled — it needs the true class proportions) to collect raw
    P(Defaulted) scores, then fits a monotonic calibrator that maps those
    scores to better-calibrated probabilities.

    Two methods are supported:

    - ``'isotonic'``  — non-parametric, more flexible, needs ≥ a few hundred
      samples to avoid overfitting.
    - ``'platt'``     — logistic regression (Platt scaling), smoother but
      assumes a sigmoid relationship between raw score and true probability.

    Parameters
    ----------
    cal_df : pandas.DataFrame
        Held-out calibration split (unbalanced, real class proportions).
    model : pgmpy.models.DiscreteBayesianNetwork
        Fitted Bayesian Network.
    target : str
        Name of the target variable (``'LoanOutcome'``).
    method : {'isotonic', 'platt'}, optional
        Calibration method.  Default ``'isotonic'``.

    Returns
    -------
    calibrator
        Fitted ``IsotonicRegression`` or ``LogisticRegression`` object.
        Pass this to ``predict()`` via the ``calibrator`` keyword argument.
    """
    infer        = VariableElimination(model)
    dag_cols     = list(model.nodes())
    feature_cols = [c for c in dag_cols if c != target]
    cal_model_df = cal_df[dag_cols].copy().astype('object')

    # Precompute valid states per feature so unknown values can be dropped rather
    # than crashing inference.  Unobserved variables are marginalised by pgmpy.
    model_states = {col: set(model.get_cpds(col).state_names[col]) for col in feature_cols}
    _warned = set()

    raw_probs, true_labels = [], []
    for _, row in tqdm(cal_model_df.iterrows(), total=len(cal_model_df), desc="Calibrating"):
        evidence = {}
        for col in feature_cols:
            val = row[col]
            if val in model_states[col]:
                evidence[col] = val
            elif col not in _warned:
                logger.debug(f"calibrate_model: dropping unknown state '{val}' for '{col}' "
                             f"(known: {sorted(model_states[col])})")
                _warned.add(col)
        result   = infer.query([target], evidence=evidence, show_progress=False)
        states   = result.state_names[target]
        raw_probs.append(float(result.values[states.index('Defaulted')]))
        true_labels.append(1 if row[target] == 'Defaulted' else 0)

    raw_probs   = np.array(raw_probs)
    true_labels = np.array(true_labels)

    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(raw_probs, true_labels)
    else:  # platt scaling
        calibrator = LogisticRegression()
        calibrator.fit(raw_probs.reshape(-1, 1), true_labels)

    n_defaults = true_labels.sum()
    logger.debug(f"Calibrator ({method}) fitted on {len(true_labels)} samples "
                 f"({n_defaults} defaults, {len(true_labels) - n_defaults} repaid)")
    return calibrator


def predict(test_df, model, target, default_threshold=0.2, mname=None, calibrator=None, plot_reliability=False):
    """Run Variable-Elimination inference on the test set and evaluate predictions.

    For each row in ``test_df``, queries the model with all non-target columns
    as evidence and retrieves P(Defaulted).  Any row whose P(Defaulted) meets
    or exceeds ``default_threshold`` is classified as Defaulted; otherwise
    Repaid.

    Also logs summary statistics of the predicted-probability distribution
    (min, max, mean, median, and counts above common thresholds) and saves a
    probability histogram plot.

    Parameters
    ----------
    test_df : pandas.DataFrame
        Held-out test set (includes the target column for evaluation).
    model : pgmpy.models.DiscreteBayesianNetwork
        Fitted Bayesian Network.
    target : str
        Name of the target variable (``'LoanOutcome'``).
    default_threshold : float, optional
        Probability threshold above which a case is labelled Defaulted.
        Default 0.2.  Lower values increase recall at the cost of precision.
    mname : str, optional
        Short model label used in logging and plot filenames.

    Returns
    -------
    actuals : list[str]
        True labels from ``test_df[target]``.
    predictions : list[str]
        Model predictions (``'Defaulted'`` or ``'Repaid'``).
    """
    logger.debug(f"running predict test df rows, cols: {test_df.shape}")
#    print("Test data distribution:")
#    print(test_df['LoanOutcome'].value_counts())
#    print(test_df['LoanOutcome'].value_counts(normalize=True))
    
    infer = VariableElimination(model)
    # Only use columns that exist in the DAG
    dag_cols     = list(model.nodes())
    test_model   = test_df[dag_cols].copy().astype('object')
    feature_cols = [col for col in dag_cols if col != target]

    # Precompute valid states per feature so unknown values can be dropped rather
    # than crashing inference.  Unobserved variables are marginalised by pgmpy.
    model_states = {col: set(model.get_cpds(col).state_names[col]) for col in feature_cols}
    _warned = set()

    actuals    = []
    predictions = []
    probs_list = []

    for _, row in tqdm(test_model.iterrows(), total=len(test_model), desc=f"Predicting ({mname})"):
        evidence = {}
        for col in feature_cols:
            val = row[col]
            if val in model_states[col]:
                evidence[col] = val
            elif col not in _warned:
                logger.debug(f"predict: dropping unknown state '{val}' for '{col}' "
                             f"(known: {sorted(model_states[col])})")
                _warned.add(col)

        # Query the network
        result = infer.query([target], evidence=evidence, show_progress=False)
        
        # Get probability of Defaulted specifically
        states = result.state_names[target]
        probs  = result.values
        default_prob = float(probs[states.index('Defaulted')])

        # Apply post-hoc calibration if a calibrator was supplied
        if calibrator is not None:
            if isinstance(calibrator, IsotonicRegression):
                default_prob = float(calibrator.predict([default_prob])[0])
            else:  # LogisticRegression (Platt)
                default_prob = float(calibrator.predict_proba([[default_prob]])[0, 1])

        probs_list.append(default_prob)

        # Use threshold instead of argmax
        pred = 'Defaulted' if default_prob >= default_threshold else 'Repaid'
        #pred = result.state_names[target][np.argmax(result.values)]
    
        actuals.append(row[target])
        predictions.append(pred)
    
    # log after the loop
    probs_array  = np.array(probs_list)
    true_binary  = np.array([1 if a == 'Defaulted' else 0 for a in actuals])

    # Brier score (lower = better; 0 = perfect)
    brier = brier_score_loss(true_binary, probs_array)

    # Expected Calibration Error — 10 equal-width bins over [0, 1]
    n_bins_ece = 10
    bin_edges  = np.linspace(0, 1, n_bins_ece + 1)
    bin_ids    = np.digitize(probs_array, bin_edges[1:-1])
    ece        = 0.0
    for b in range(n_bins_ece):
        mask = bin_ids == b
        if mask.sum() > 0:
            ece += (mask.sum() / len(true_binary)) * abs(true_binary[mask].mean() - probs_array[mask].mean())

    logger.debug(f"Predicted probability distribution:\n"
                 f"  min:    {probs_array.min():.3f}\n"
                 f"  max:    {probs_array.max():.3f}\n"
                 f"  mean:   {probs_array.mean():.3f}\n"
                 f"  median: {np.median(probs_array):.3f}\n"
                 f"  >0.20:  {(probs_array >= 0.20).sum()}\n"
                 f"  >0.30:  {(probs_array >= 0.30).sum()}\n"
                 f"  >0.40:  {(probs_array >= 0.40).sum()}\n"
                 f"  Brier:  {brier:.4f}\n"
                 f"  ECE:    {ece:.4f}")
        
    plot_probability_distribution(probs_list, default_threshold, mname)
    if plot_reliability:
        plot_reliability_diagram(probs_list, actuals, mname)
    #create and print confusion matrix
    display_confusion_matrix(actuals, predictions)
    return actuals, predictions

def balance_training_data(train_df):
    """Oversample the minority class to reduce class imbalance before training.

    Uses ``RandomOverSampler`` with ``sampling_strategy=0.35``, which means
    the minority class (Defaulted) is upsampled until it represents 35 % of
    the majority class count — yielding roughly a 74/26 Repaid/Defaulted split
    in the resampled data.  This is deliberately less aggressive than 50/50 so
    that predicted probabilities remain better calibrated to the true ~20 %
    base rate.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training split before resampling.

    Returns
    -------
    pandas.DataFrame
        Resampled training data with the same columns as ``train_df``.
    """
    #guaranteee 3/1 paidoff/defaulted
    ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
    train_resampled, _ = ros.fit_resample(train_df, train_df['LoanOutcome'])
    logger.debug(f"Resampling LoanOutcome distribution:\n{train_resampled['LoanOutcome'].value_counts().to_string()}")
    return train_resampled

def run_sweep(test_df, model, auto_model, expert_calibrator=None, auto_calibrator=None):
    """Evaluate both models across a range of decision thresholds.

    Iterates over a predefined list of probability thresholds for both the
    expert and auto models, calling ``predict()`` at each threshold and logging
    Precision, Recall, F1, and Balanced Accuracy.  Use this to identify the
    operating point that best balances catching defaults (recall) against false
    alarms (1 − precision).

    When calibrators are supplied the threshold range covers 0.04–0.20, which
    is where calibrated probabilities (mean ~0.08) carry useful signal.
    Without calibrators the range covers 0.10–0.60 to match the inflated
    raw probability distribution.

    Parameters
    ----------
    test_df : pandas.DataFrame
        Held-out test set.
    model : pgmpy.models.DiscreteBayesianNetwork
        Fitted expert Bayesian Network.
    auto_model : pgmpy.models.DiscreteBayesianNetwork
        Fitted auto (Hill-Climb) Bayesian Network.
    expert_calibrator : calibrator object or None
        Fitted calibrator for the expert model (from ``calibrate_model()``).
    auto_calibrator : calibrator object or None
        Fitted calibrator for the auto model (from ``calibrate_model()``).
    """
    if expert_calibrator is not None:
        thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    else:
        thresholds = [0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    logger.debug("=== Expert Model Threshold Sweep ===")
    for threshold in thresholds:
        acts, preds = predict(test_df, model, 'LoanOutcome',
                              default_threshold=threshold, mname='Expert',
                              calibrator=expert_calibrator)
        precision = precision_score(acts, preds, pos_label='Defaulted', zero_division=0)
        recall    = recall_score(acts, preds,    pos_label='Defaulted', zero_division=0)
        f1        = f1_score(acts, preds,        pos_label='Defaulted', zero_division=0)
        bal_acc   = balanced_accuracy_score(acts, preds)
        logger.debug(f"Threshold {threshold:.2f} | Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  BalAcc: {bal_acc:.3f}")

    if auto_calibrator is not None:
        thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    else:
        thresholds = [0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    logger.debug("\n=== Auto Model Threshold Sweep ===")
    for threshold in thresholds:
        acts, preds = predict(test_df, auto_model, 'LoanOutcome',
                              default_threshold=threshold, mname='Auto',
                              calibrator=auto_calibrator)
        precision = precision_score(acts, preds, pos_label='Defaulted', zero_division=0)
        recall    = recall_score(acts, preds,    pos_label='Defaulted', zero_division=0)
        f1        = f1_score(acts, preds,        pos_label='Defaulted', zero_division=0)
        bal_acc   = balanced_accuracy_score(acts, preds)
        logger.debug(f"Threshold {threshold:.2f} | Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  BalAcc: {bal_acc:.3f}")
    return
 
def main(target):
    """Execute the full Bayesian Network training and evaluation pipeline.

    Steps:
    1. Initialise logging.
    2. Load and merge all data sources into a single discretised DataFrame.
    3. Stratified 80/20 train/test split.
    4. Log feature-vs-outcome cross-tabulations for every predictor.
    5. Oversample the training minority class.
    6. Fit the expert model on the balanced training data.
    7. Evaluate the expert model on the test set at threshold 0.10.
    8. Fit the auto (Hill-Climb) model on the balanced training data.
    9. Evaluate the auto model on the test set at threshold 0.06.

    Parameters
    ----------
    target : str
        Name of the target column — must be ``'LoanOutcome'``.

    Returns
    -------
    test_df : pandas.DataFrame
        Held-out test set (useful for interactive exploration after the run).
    model : pgmpy.models.DiscreteBayesianNetwork
        Fitted expert model.
    target : str
        Echoes the ``target`` parameter for convenience.
    """
    setup_logging(logging.DEBUG)
    
    data_dir = "data/"
    merged_df = get_merged_data(data_dir)

    # Select only the discrete model columns for training/inference.
    # merged_df also contains raw numeric/string source columns for diagnostics;
    # those must be excluded before passing data to the Bayesian Network.
    df = merged_df[MODEL_COLS].dropna(subset=[target]).copy()

    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    train_df, cal_df = train_test_split(train_val_df, test_size=0.15, random_state=42, stratify=train_val_df[target])
    logger.debug(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows, Calibrate: {len(cal_df)} rows")
    
    #commented out 4/15/26 to return later
    #for col in [c for c in train_df.columns if c != 'LoanOutcome']:
    #    ct = pd.crosstab(train_df[col], train_df['LoanOutcome'], normalize='index')
    #    logger.debug(f"Feature vs LoanOutcome:\n{ct.to_string()}")
    
    train_bal = balance_training_data(train_df)

    # Evaluate which of the current direct parents of LoanOutcome are genuinely
    # adding information vs. redundant with stronger parents.  Results are
    # logged — use them to trim expert_list before the next run.
    current_direct_parents = [ 'AmtGoodsPrice', 'ExtSource1Risk', 
                               'ExtSource2Risk', 'ExtSource3Risk',
                             ]
    select_direct_parents(train_bal, target, current_direct_parents,
                          max_parents=4, logger=logger)

    # Phase 2 — Define and Train the Expert Network
    # DEFINE THE DAG STRUCTURE (domain knowledge)
    # Each tuple is (parent, child) — "parent influences child"

    expert_list = [('ContractStatus',    'PriorLoanApproved'),
                   ('PriorLoanApproved', 'PaymentHistory'),
                   ('PaymentHistory',    'DPD'),
                   ('AmtCredit',         'AmtGoodsPrice'),
                   ('ExtSource3Risk',    'ActiveCredits'),
                   ('ActiveCredits',     'MaxOverdue'),
                   ('MaxOverdue',        'CreditProlonged'),
                   ('OccupationType',    'IncomeType'),
                   ('ExtSource1Risk',    'OccupationType'),
                   ('AmtGoodsPrice',     'LoanOutcome'),
                   ('ExtSource3Risk',    'LoanOutcome'),
                   ('ExtSource1Risk',    'LoanOutcome'),
                   ('ExtSource2Risk',    'LoanOutcome'),
                   ('AgeEducation',      'LoanOutcome'),
                  ]



    logger.info("Phase 1: fitting expert model...")
    model = create_expert_model(train_bal, expert_list)
    logger.info("Phase 1: calibrating expert model...")
    expert_calibrator = calibrate_model(cal_df, model, target)
    logger.info("Phase 1: evaluating expert model on test set...")
    _, _ = predict(test_df, model, target, default_threshold=.12,
                   mname="Expert", calibrator=expert_calibrator, plot_reliability=True)
    logger.info("Phase 1: complete.")

    logger.info("Phase 2: fitting auto model (hill-climb search)...")
    auto_model = create_auto_model(df=train_bal,
                          max_indegree=3,
                          max_iter=int(1e4),
                          scoring_method='bic-d'
                          )
    logger.info("Phase 2: calibrating auto model...")
    auto_calibrator = calibrate_model(cal_df, auto_model, target)
    logger.info("Phase 2: evaluating auto model on test set...")
    _, _ = predict(test_df, auto_model, target, default_threshold=.12,
                   mname="Auto", calibrator=auto_calibrator, plot_reliability=True)
    logger.info("Phase 2: complete.")

    #
    logger.info("Phase 3: running threshold sweep...")
    run_sweep(test_df, model, auto_model,
              expert_calibrator=expert_calibrator,
              auto_calibrator=auto_calibrator)
    logger.info("Phase 3: complete. Pipeline finished.")
    
    return test_df, model, target
 
if __name__ == "__main__":
    main('LoanOutcome')