"""
home_credit_naive_bayes.py
──────────────────────────
Categorical Naive Bayes classifier for Home Credit Default dataset.
Designed as a direct baseline companion to home_credit_bayesian.py.

Uses the same preprocessing conventions, logging pattern, and output
structure so results are directly comparable to the Bayesian Network.

Pipeline
--------
1. Load and merge all six Home Credit CSV files via data_prep.get_merged_data.
2. Ordinal-encode the selected feature columns (CategoricalNB requires
   non-negative integers).
3. Optionally oversample the minority class on the training split.
4. Select the Laplace smoothing parameter (alpha) by stratified 5-fold CV.
5. Sweep the classification threshold and log precision/recall/F1/BalAcc.
6. Save probability-distribution, threshold-sweep, feature log-probability,
   and ROC-curve plots to logs/.

Usage (standalone — run from repo root):
    python src/home_credit_naive_bayes.py

Usage (as importable module):
    from home_credit_naive_bayes import run
    results = run(merged_df, logger)
"""

import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure root logger with a timestamped file handler.
    Safe to call multiple times in Jupyter — removes stale file handlers first.

    Parameters
    ----------
    level : int
        Logging level (e.g. logging.DEBUG).

    Returns
    -------
    logging.Logger
        Module-level logger.
    """
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = f'logs/naive_bayes_{timestamp}.log'

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root.addHandler(fh)

    # Suppress matplotlib's verbose font-cache DEBUG output
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug(f"Logger initialized - level: {logging.getLevelName(level)}, file: {log_file}")
    return logger


# ── Feature definitions ────────────────────────────────────────────────────────

TARGET = 'LoanOutcome'

# All MODEL_COLS features used as inputs — mirrors data_prep.MODEL_COLS minus LoanOutcome.
# NB has no CPT sparsity problem so we can use the full feature set.
FEATURES = [
    'IncomeType', 'OccupationType', 'IncomeBracket',
    'ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk',
    'AmtCredit', 'AmtGoodsPrice', 'AmtReqCreditBureauMon',
    'PaymentHistory',
    'PriorLoanApproved', 'PrevRejected',
    'CreditUtilization',
    'DPD', 'ContractStatus',
    'DaysOverdue', 'MaxOverdue', 'DebtLoad', 'CreditProlonged', 'ActiveCredits',
]

# Ordinal encoding orders — must match binning labels in data_prep.py.
# Sentinel / fill-value categories are listed first in each entry.
# CategoricalNB treats the encoding as nominal so order does not affect accuracy,
# but consistent ordering with the bin sequence makes the log-probability plot readable.
ORDINAL_CATEGORIES = {
    # Application-level features (no nulls in source; 'Unknown' handles missing occupation)
    'IncomeType':            ['Stable', 'Unstable'],
    'OccupationType':        ['Unknown', 'Laborer', 'Professional'],
    'IncomeBracket':         ['Low', 'Medium', 'High'],
    # External credit scores — ~1–63 % missing → filled 'Unknown'
    'ExtSource1Risk':        ['Unknown', 'VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow'],
    'ExtSource2Risk':        ['Unknown', 'VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow'],
    'ExtSource3Risk':        ['Unknown', 'VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow'],
    # Loan amount buckets (labels are frequency-based, not magnitude-ordered)
    'AmtCredit':             ['Unknown', 'MedHigh', 'Med', 'High', 'MedLow', 'VeryLow', 'Low'],
    'AmtGoodsPrice':         ['Unknown', 'High', 'Medium', 'VeryHigh', 'Low', 'MedHigh', 'VeryLow'],
    'AmtReqCreditBureauMon': ['Unknown', 'Medium', 'VeryLow'],
    # Instalment payment behaviour — 'Unknown' for applicants with no instalment history
    'PaymentHistory':        ['Unknown', 'Low', 'Medium', 'High'],
    # Previous application history
    'PriorLoanApproved':     ['No', 'Yes'],
    'PrevRejected':          ['None', 'Once', 'Twice', 'Multiple'],
    # Credit-card utilisation — 'Low' for applicants with no credit card
    'CreditUtilization':     ['Low', 'Medium', 'High', 'MaxedOut'],
    # POS-cash delinquency and contract status
    'DPD':                   ['None', 'Low', 'Medium', 'High'],
    'ContractStatus':        ['Inactive', 'Active'],
    # Bureau overdue / debt / prolongation / active-credit buckets
    'DaysOverdue':           ['None', 'Low', 'Medium', 'High'],
    'MaxOverdue':            ['None', 'Low', 'Medium', 'High'],
    'DebtLoad':              ['None', 'Low', 'Medium', 'High'],
    'CreditProlonged':       ['Never', 'Once', 'Several', 'Many'],
    'ActiveCredits':         ['None', 'One', 'Few', 'Many'],
}

TEST_SIZE   = 0.20
RANDOM_SEED = 42
CV_FOLDS    = 5

# alpha values to sweep — mirrors C sweep in LR L1 baseline
# CategoricalNB alpha is additive Laplace smoothing (1.0 = standard Laplace)
ALPHA_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

# Threshold sweep range — matches BN sweep
THRESHOLD_RANGE = np.arange(0.20, 0.65, 0.05)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    logger: logging.Logger
) -> tuple[pd.DataFrame, pd.Series, OrdinalEncoder]:
    """
    Encode features and binarize target.

    Parameters
    ----------
    df : pd.DataFrame
        merged_df from the main pipeline.
    logger : logging.Logger

    Returns
    -------
    X_encoded : pd.DataFrame
        Integer-encoded feature matrix (non-negative, required by CategoricalNB).
    y : pd.Series
        Binary target (1 = Defaulted, 0 = Repaid).
    encoder : OrdinalEncoder
        Fitted encoder for later inspection.
    """
    logger.debug("prepare_data: starting")

    df_model = df[FEATURES + [TARGET]].copy()

    # get_merged_data() fills nulls with CATEGORICAL_FILL_VALUES ('Unknown' for
    # Ext/Amt columns, 'Low' for PaymentHistory, etc.).  The fillna here is a
    # safety net for any residual pandas NaN that slips through; we map them to
    # 'Unknown' so they land in the sentinel category defined in ORDINAL_CATEGORIES.
    for col in FEATURES:
        df_model[col] = df_model[col].fillna('Unknown').astype(str)

    # Log class distribution before encoding
    logger.debug(f"LoanOutcome distribution:\n{df_model[TARGET].value_counts()}")

    # Binarise target
    y = (df_model[TARGET] == 'Defaulted').astype(int)
    X = df_model[FEATURES]

    # Log feature distributions
    for col in FEATURES:
        logger.debug(f"{col} distribution:\n{X[col].value_counts()}")

    # Ordinal encode — CategoricalNB requires non-negative integers
    categories = [ORDINAL_CATEGORIES[col] for col in FEATURES]
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1      # unseen states get -1; logged below
    )
    X_enc = encoder.fit_transform(X)

    # Check for unknown values introduced by unseen states
    unknown_mask = (X_enc == -1).any(axis=0)
    if unknown_mask.any():
        unknown_cols = [FEATURES[i] for i, v in enumerate(unknown_mask) if v]
        logger.warning(f"Unknown category values (-1) in: {unknown_cols}")

    # Shift to 0-based non-negative integers (CategoricalNB requires >= 0)
    X_enc = np.where(X_enc == -1, 0, X_enc).astype(int)

    X_encoded = pd.DataFrame(X_enc, columns=FEATURES, index=X.index)

    logger.debug(f"prepare_data: complete - shape {X_encoded.shape}, "
                 f"default rate {y.mean():.3f}")
    return X_encoded, y, encoder


def resample_balanced(
    X: pd.DataFrame,
    y: pd.Series,
    logger: logging.Logger,
    random_state: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Upsample minority class to match majority class count.
    Mirrors the resampling step in home_credit_bayesian.py.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    logger : logging.Logger
    random_state : int

    Returns
    -------
    X_resampled, y_resampled : pd.DataFrame, pd.Series
    """
    df = X.copy()
    df[TARGET] = y

    majority = df[df[TARGET] == 0]
    minority = df[df[TARGET] == 1]

    minority_upsampled = minority.sample(
        n=len(majority), replace=True, random_state=random_state
    )
    df_balanced = pd.concat([majority, minority_upsampled]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    logger.debug(f"resample_balanced: {len(majority)} Repaid / "
                 f"{len(minority)} Defaulted -> {len(df_balanced)} balanced rows")

    y_res = df_balanced[TARGET]
    X_res = df_balanced.drop(columns=[TARGET])
    return X_res, y_res


# ── Alpha selection via cross-validation ──────────────────────────────────────

def select_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    logger: logging.Logger
) -> float:
    """
    Cross-validate over ALPHA_GRID and return best alpha by balanced accuracy.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    logger : logging.Logger

    Returns
    -------
    float
        Best alpha value.
    """
    logger.debug(f"select_alpha: sweeping {ALPHA_GRID}")

    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = {}

    for alpha in ALPHA_GRID:
        model = CategoricalNB(alpha=alpha)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring='balanced_accuracy', n_jobs=-1
        )
        scores[alpha] = cv_scores.mean()
        logger.debug(f"  alpha={alpha:.3f}  BalAcc={cv_scores.mean():.4f} "
                     f"(+/- {cv_scores.std():.4f})")

    best_alpha = max(scores, key=scores.get)
    logger.debug(f"select_alpha: best alpha={best_alpha} "
                 f"(BalAcc={scores[best_alpha]:.4f})")
    return best_alpha


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def threshold_sweep(
    y_true: pd.Series,
    y_proba: np.ndarray,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Sweep classification threshold and log precision/recall/F1/BalAcc.

    Parameters
    ----------
    y_true : pd.Series
    y_proba : np.ndarray
        Predicted probabilities for Defaulted class.
    logger : logging.Logger

    Returns
    -------
    pd.DataFrame
        Sweep results table.
    """
    rows = []
    for t in THRESHOLD_RANGE:
        y_pred = (y_proba >= t).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        rows.append({
            'Threshold':  round(t, 2),
            'Precision':  round(report['1']['precision'], 3),
            'Recall':     round(report['1']['recall'],    3),
            'F1':         round(report['1']['f1-score'],  3),
            'BalAcc':     round(balanced_accuracy_score(y_true, y_pred), 3),
        })

    sweep_df = pd.DataFrame(rows)
    best_row = sweep_df.loc[sweep_df['F1'].idxmax()]
    logger.debug(f"threshold_sweep results:\n{sweep_df.to_string(index=False)}")
    logger.debug(f"Best threshold by F1: {best_row['Threshold']} "
                 f"(F1={best_row['F1']}, BalAcc={best_row['BalAcc']})")
    return sweep_df


def log_classification_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label: str,
    threshold: float,
    logger: logging.Logger
) -> None:
    """
    Log full classification report, confusion matrix, and ROC-AUC.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    y_proba : np.ndarray
    label : str
        Model label for log headers (e.g. 'NaiveBayes').
    threshold : float
    logger : logging.Logger
    """
    report = classification_report(
        y_true, y_pred,
        target_names=['Repaid', 'Defaulted'],
        zero_division=0
    )
    cm  = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    bal = balanced_accuracy_score(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=['Actual Repaid', 'Actual Defaulted'],
        columns=['Pred Repaid', 'Pred Defaulted']
    )

    logger.debug(f"{label} | threshold={threshold:.2f}")
    logger.debug(f"Accuracy: {(y_pred == y_true).mean():.3f}")
    logger.debug(f"Classification Report:\n{report}")
    logger.debug(f"Balanced Accuracy: {bal:.4f}")
    logger.debug(f"ROC-AUC: {auc:.4f}")
    logger.debug(f"Confusion Matrix (rows=actual, cols=predicted):\n{cm_df}")

    logger.debug(f"Predicted probability distribution:")
    logger.debug(f"  min:    {y_proba.min():.3f}")
    logger.debug(f"  max:    {y_proba.max():.3f}")
    logger.debug(f"  mean:   {y_proba.mean():.3f}")
    logger.debug(f"  median: {np.median(y_proba):.3f}")
    logger.debug(f"  >0.20:  {(y_proba > 0.20).sum()}")
    logger.debug(f"  >0.30:  {(y_proba > 0.30).sum()}")
    logger.debug(f"  >0.40:  {(y_proba > 0.40).sum()}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_probability_distribution(
    y_proba: np.ndarray,
    threshold: float,
    label: str,
    logger: logging.Logger
) -> str:
    """
    Plot predicted default probability distribution with threshold line.
    Matches visual style of Expert/Auto BN probability plots.

    Parameters
    ----------
    y_proba : np.ndarray
    threshold : float
    label : str
    logger : logging.Logger

    Returns
    -------
    str
        Saved file path.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/nb_prob_dist_{label.lower()}_{timestamp}.png'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_proba, bins=50, color='#6A9EC2', edgecolor='none', alpha=0.85)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold}')
    ax.axvspan(threshold, 1.0, alpha=0.12, color='red', label='Predicted Defaulted')
    ax.set_xlabel('P(Defaulted)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{label} - Predicted Default Probability Distribution', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.debug(f"Probability distribution plot saved: {path}")
    return path


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    label: str,
    logger: logging.Logger
) -> str:
    """
    Plot F1, BalAcc, Precision, Recall across threshold sweep.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of threshold_sweep().
    label : str
    logger : logging.Logger

    Returns
    -------
    str
        Saved file path.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/nb_threshold_sweep_{label.lower()}_{timestamp}.png'

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df['Threshold'], sweep_df['F1'],        marker='o', label='F1',
            color='#2171B5', linewidth=2.5)
    ax.plot(sweep_df['Threshold'], sweep_df['BalAcc'],    marker='o', label='Balanced Acc',
            color='#1A9850', linewidth=2.5)
    ax.plot(sweep_df['Threshold'], sweep_df['Recall'],    marker='s', label='Recall',
            color='#D73027', linewidth=1.5, linestyle='--')
    ax.plot(sweep_df['Threshold'], sweep_df['Precision'], marker='s', label='Precision',
            color='#7B2D8B', linewidth=1.5, linestyle='--')

    best_t = sweep_df.loc[sweep_df['F1'].idxmax(), 'Threshold']
    ax.axvline(best_t, color='gray', linestyle=':', linewidth=1.2,
               label=f'Best F1 threshold={best_t}')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score',     fontsize=12)
    ax.set_title(f'{label} - Threshold Sweep', fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.debug(f"Threshold sweep plot saved: {path}")
    return path


def plot_feature_log_probs(
    model: CategoricalNB,
    encoder: OrdinalEncoder,
    logger: logging.Logger,
    top_n: int = 10,
) -> str:
    """
    Plot per-feature log-probability differences (Defaulted - Repaid) per bin.
    Only the top_n features by signal spread (max diff − min diff) are shown,
    keeping the chart readable when the full feature set is large.

    Parameters
    ----------
    model : CategoricalNB
        Fitted model.
    encoder : OrdinalEncoder
        Fitted encoder (to recover bin labels).
    logger : logging.Logger
    top_n : int
        Number of features to plot, ranked by log-probability spread.

    Returns
    -------
    str
        Saved file path.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/nb_feature_log_probs_{timestamp}.png'

    lp = model.feature_log_prob_

    # ── Compute diff and spread for every feature ──────────────────────────
    feature_data = []
    for i, col in enumerate(FEATURES):
        cats      = ORDINAL_CATEGORIES[col]
        lp0       = lp[i][0]
        lp1       = lp[i][1]
        n_cats    = min(len(cats), len(lp0))
        diff      = lp1[:n_cats] - lp0[:n_cats]
        spread    = diff.max() - diff.min()
        feature_data.append((col, cats[:n_cats], diff, spread))

    # ── Rank by spread and keep top_n ──────────────────────────────────────
    feature_data.sort(key=lambda x: x[3], reverse=True)
    selected = feature_data[:top_n]

    skipped = [f[0] for f in feature_data[top_n:]]
    logger.debug(
        f"plot_feature_log_probs: showing top {top_n} of {len(FEATURES)} features "
        f"by log-prob spread. Omitted: {skipped}"
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(top_n, 1, figsize=(10, 3.5 * top_n))
    if top_n == 1:
        axes = [axes]

    for ax, (col, cats, diff, spread) in zip(axes, selected):
        colors = ['#D85A30' if d > 0 else '#378ADD' for d in diff]
        ax.bar(cats, diff, color=colors, edgecolor='none')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'{col}  (spread={spread:.2f})', fontsize=11, fontweight='500')
        ax.set_ylabel('log P(Default) -\nlog P(Repaid)', fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    handles = [
        Patch(color='#D85A30', label='Higher default probability'),
        Patch(color='#378ADD', label='Lower default probability'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Naive Bayes - Top {top_n} Features by Log-Probability Spread',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.debug(f"Feature log-probability plot saved: {path}")
    return path


def plot_roc_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    label: str,
    logger: logging.Logger
) -> str:
    """
    Plot ROC curve with AUC annotation.

    Parameters
    ----------
    y_true : pd.Series
    y_proba : np.ndarray
    label : str
    logger : logging.Logger

    Returns
    -------
    str
        Saved file path.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'logs/nb_roc_{label.lower()}_{timestamp}.png'

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc         = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#378ADD', linewidth=2.5,
            label=f'ROC AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Random baseline')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'{label} - ROC Curve', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.debug(f"ROC curve plot saved: {path}")
    return path


# ── Summary comparison table ───────────────────────────────────────────────────

def print_comparison_summary(
    sweep_df: pd.DataFrame,
    label: str,
    auc: float,
    logger: logging.Logger
) -> None:
    """
    Print a summary table formatted to match BN run output for direct comparison.

    Parameters
    ----------
    sweep_df : pd.DataFrame
    label : str
    auc : float
    logger : logging.Logger
    """
    best = sweep_df.loc[sweep_df['F1'].idxmax()]

    summary = f"""
{'='*55}
  {label} - Best Performance Summary
{'='*55}
  Threshold : {best['Threshold']:.2f}
  Precision : {best['Precision']:.3f}
  Recall    : {best['Recall']:.3f}
  F1        : {best['F1']:.3f}
  BalAcc    : {best['BalAcc']:.3f}
  ROC-AUC   : {auc:.4f}
{'='*55}
  Compare against BN Expert:
    F1=0.392  BalAcc=0.607  (threshold=0.50, 8 parents)
{'='*55}
"""
    print(summary)
    logger.debug(summary)


# ── Main entry point ───────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    logger: logging.Logger,
    threshold: float = None,
    resample: bool   = True
) -> dict:
    """
    Full Naive Bayes pipeline on merged_df.

    Parameters
    ----------
    df : pd.DataFrame
        merged_df from the main preprocessing pipeline.
    logger : logging.Logger
        Logger instance (from setup_logging or passed in from BN script).
    threshold : float, optional
        Classification threshold. If None, best threshold from sweep is used.
    resample : bool
        If True, upsample minority class to 50/50 before training.
        Set to False to compare on imbalanced data.

    Returns
    -------
    dict
        Keys: model, encoder, sweep_df, y_test, y_proba, best_threshold, auc
    """
    logger.debug("=" * 60)
    logger.debug("Naive Bayes pipeline: starting")
    logger.debug("=" * 60)
    logger.debug(f"Features ({len(FEATURES)}): {FEATURES}")

    # ── Prepare data
    X_encoded, y, encoder = prepare_data(df, logger)

    # ── Train/test split (stratified — mirrors BN split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )
    logger.debug(f"Train: {X_train.shape}  Test: {X_test.shape}")
    logger.debug(f"Default rate - train: {y_train.mean():.3f}  "
                 f"test: {y_test.mean():.3f}")

    # ── Optional resampling on train set only
    if resample:
        X_train, y_train = resample_balanced(X_train, y_train, logger)
        logger.debug(f"Post-resample train shape: {X_train.shape}")

    # ── Alpha selection
    best_alpha = select_alpha(X_train, y_train, logger)

    # ── Fit final model
    model = CategoricalNB(alpha=best_alpha)
    model.fit(X_train, y_train)
    logger.debug(f"Model fitted - alpha={best_alpha}, "
                 f"classes={model.classes_}, "
                 f"class log-priors={model.class_log_prior_}")

    # ── Predict
    y_proba = model.predict_proba(X_test)[:, 1]  # P(Defaulted)
    logger.debug(f"running predict test df rows, cols: {X_test.shape}")

    # ── Threshold sweep
    sweep_df = threshold_sweep(y_test, y_proba, logger)

    # ── Apply threshold
    if threshold is None:
        threshold = float(sweep_df.loc[sweep_df['F1'].idxmax(), 'Threshold'])
        logger.debug(f"Auto-selected best threshold: {threshold:.2f}")
    else:
        logger.debug(f"Using provided threshold: {threshold:.2f}")

    y_pred = (y_proba >= threshold).astype(int)

    # ── Log results
    log_classification_results(
        y_test, y_pred, y_proba, 'NaiveBayes', threshold, logger
    )

    # ── ROC-AUC
    auc = roc_auc_score(y_test, y_proba)

    # ── Summary comparison
    print_comparison_summary(sweep_df, 'NaiveBayes', auc, logger)

    # ── Plots
    plot_probability_distribution(y_proba, threshold, 'NaiveBayes', logger)
    plot_threshold_sweep(sweep_df, 'NaiveBayes', logger)
    plot_feature_log_probs(model, encoder, logger)
    plot_roc_curve(y_test, y_proba, 'NaiveBayes', logger)

    logger.debug("Naive Bayes pipeline: complete")

    return {
        'model':           model,
        'encoder':         encoder,
        'sweep_df':        sweep_df,
        'y_test':          y_test,
        'y_proba':         y_proba,
        'best_threshold':  threshold,
        'auc':             auc,
    }


# ── Standalone execution ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    import sys

    # Make data_prep importable: add src/ (this file's directory) to sys.path
    # so the script works whether invoked as:
    #   python src/home_credit_naive_bayes.py   (from repo root)
    #   python home_credit_naive_bayes.py       (from inside src/)
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from data_prep import get_merged_data, MODEL_COLS

    # Initialise logging — creates logs/naive_bayes_<timestamp>.log
    _logger = setup_logging()

    # Load and merge all six Home Credit CSVs.
    # data/ is relative to the repo root — run this script from there.
    _logger.debug("Loading merged data ...")
    _merged_df = get_merged_data('data/')

    # Drop Kaggle test rows (TARGET = NaN → LoanOutcome = None).
    # MODEL_COLS selects the 21 discrete model features used by both the
    # Bayesian Network and this Naive Bayes baseline.
    _df = _merged_df[MODEL_COLS].dropna(subset=[TARGET]).copy()
    _logger.debug(f"Labelled rows available for modelling: {len(_df)}")

    run(_df, _logger)
