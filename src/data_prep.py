"""
data_prep.py
------------
Data preparation pipeline for the Home Credit default-prediction project.

Entry point
-----------
``get_merged_data(data_dir, logger=None)`` — reads all six raw CSV files,
computes per-applicant summary features, and returns a single wide DataFrame
that contains **both** the raw numeric/string source columns and the derived
categorical model features used by the Bayesian Network.

Column naming convention
------------------------
- **lowercase_snake_case** — raw aggregated numeric/string values kept for
  reference and exploratory analysis (e.g. ``on_time_rate``, ``max_dpd``).
- **PascalCase** — derived categorical model features ready for use in the
  Bayesian Network (e.g. ``PaymentHistory``, ``DPD``).

Null handling
-------------
Raw numeric columns are left as ``NaN`` where data is absent (truthful).
Derived categorical columns are filled with conservative defaults defined in
``CATEGORICAL_FILL_VALUES`` so the Bayesian Network always receives a valid
state.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODEL_COLS
# ---------------------------------------------------------------------------
# Ordered list of the 21 discrete model features passed to the Bayesian
# Network.  This is the subset of the wide merged DataFrame that contains
# only PascalCase derived categorical columns — no raw numerics, no
# SK_ID_CURR key.  Use this list in main() to slice merged_df down to the
# inference-ready feature table before train/test split.
MODEL_COLS = [
    'LoanOutcome',
    'IncomeType', 'OccupationType', 'IncomeBracket',
    'ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk',
    'AmtCredit', 'AmtGoodsPrice', 'AmtReqCreditBureauMon',
    'PaymentHistory',
    'PriorLoanApproved', 'PrevRejected',
    'CreditUtilization',
    'DPD', 'ContractStatus',
    'DaysOverdue', 'MaxOverdue', 'DebtLoad', 'CreditProlonged', 'ActiveCredits',
]

# ---------------------------------------------------------------------------
# CATEGORICAL_FILL_VALUES
# ---------------------------------------------------------------------------
# Conservative defaults applied to derived categorical columns after the
# merge, so that applicants with no record in a subsidiary table
# (e.g. no bureau history, no prior POS-cash loans) receive a valid state
# rather than a NaN that would break Bayesian Network inference.
#
# The chosen defaults represent the least-informative / lowest-risk state
# for each feature — i.e. what we would assume about an applicant we know
# nothing about.  Raw numeric columns are intentionally left as NaN
# (truthful absence of data).
CATEGORICAL_FILL_VALUES = {
    'PaymentHistory':        'Unknown',   # no instalment history → no data, not worst tier
    'PriorLoanApproved':     'No',        # no prior application → never approved
    'PrevRejected':          'None',      # no prior application → never rejected
    'CreditUtilization':     'Low',       # no credit card → assume low utilisation
    'DPD':                   'None',      # no POS-cash loan → no days past due
    'ContractStatus':        'Inactive',  # no POS-cash loan → no active contract
    'DaysOverdue':           'None',      # no bureau record → no overdue days
    'MaxOverdue':            'None',      # no bureau record → no overdue amount
    'DebtLoad':              'None',      # no bureau record → no outstanding debt
    'CreditProlonged':       'Never',     # no bureau record → never prolonged
    'ActiveCredits':         'None',      # no bureau record → no active credits
    'ExtSource1Risk':        'Unknown',   # EXT_SOURCE_1 missing for ~63 % of rows
    'ExtSource2Risk':        'Unknown',   # EXT_SOURCE_2 missing for < 1 % of rows
    'ExtSource3Risk':        'Unknown',   # EXT_SOURCE_3 missing for ~23 % of rows
    'AmtCredit':             'Unknown',   # safety default (AMT_CREDIT rarely null)
    'AmtGoodsPrice':         'Unknown',   # AMT_GOODS_PRICE null for ~278 rows
    'AmtReqCreditBureauMon': 'Unknown',   # AMT_REQ_CREDIT_BUREAU_MON null for ~87 k rows
}


# ---------------------------------------------------------------------------
# Per-source summary builders
# ---------------------------------------------------------------------------

def get_app_data(data_dir: str) -> pd.DataFrame:
    """Load and enrich the application tables with model-ready categorical features.

    Concatenates ``application_train.csv`` and ``application_test.csv``, then
    derives one categorical model feature per raw signal:

    ========================  =============================================
    Raw column                Derived feature
    ========================  =============================================
    NAME_INCOME_TYPE          IncomeType  (Stable / Unstable)
    OCCUPATION_TYPE           OccupationType  (Professional / Laborer / Unknown)
    AMT_INCOME_TOTAL          IncomeBracket  (Low / Medium / High tercile)
    TARGET                    LoanOutcome  (Repaid / Defaulted; NaN for test)
    EXT_SOURCE_1              ExtSource1Risk  (VeryLow … VeryHigh risk bands)
    EXT_SOURCE_2              ExtSource2Risk
    EXT_SOURCE_3              ExtSource3Risk
    AMT_CREDIT                AmtCredit  (6 credit-amount buckets)
    AMT_GOODS_PRICE           AmtGoodsPrice  (6 goods-price buckets)
    AMT_REQ_CREDIT_BUREAU_MON AmtReqCreditBureauMon  (Medium / VeryLow)
    ========================  =============================================

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.

    Returns
    -------
    pandas.DataFrame
        Full application dataset (train + test rows) with both raw source
        columns and the derived categorical columns listed above.
        Test rows have ``LoanOutcome = NaN`` and ``TARGET = NaN``.
    """
    logger.debug(f"Running get_app_data dir: {data_dir}")

    app_train = pd.read_csv(f"{data_dir}application_train.csv")
    app_test  = pd.read_csv(f"{data_dir}application_test.csv")
    app_data  = pd.concat([app_train, app_test], ignore_index=True)

    derived = pd.DataFrame({
        'IncomeType': np.where(
            app_data['NAME_INCOME_TYPE'].isin(['Working', 'Commercial associate']),
            'Stable', 'Unstable'
        ),
        'OccupationType': np.where(
            app_data['OCCUPATION_TYPE'].isna(), 'Unknown',
            np.where(
                app_data['OCCUPATION_TYPE'].isin([
                    'Managers', 'Core staff', 'High skill tech staff',
                    'Medicine staff', 'Accountants'
                ]),
                'Professional', 'Laborer'
            )
        ),
        'IncomeBracket': pd.qcut(
            app_data['AMT_INCOME_TOTAL'], q=3, labels=['Low', 'Medium', 'High']
        ),
        'LoanOutcome': np.where(
            app_data['TARGET'] == 0, 'Repaid',
            np.where(app_data['TARGET'] == 1, 'Defaulted', None)
        ),
        'ExtSource1Risk': pd.cut(
            app_data['EXT_SOURCE_1'],
            bins=[0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01],
            labels=['VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow']
        ),
        'ExtSource2Risk': pd.cut(
            app_data['EXT_SOURCE_2'],
            bins=[0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01],
            labels=['VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow']
        ),
        'ExtSource3Risk': pd.cut(
            app_data['EXT_SOURCE_3'],
            bins=[0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01],
            labels=['VeryHigh', 'High', 'MedHigh', 'Medium', 'Low', 'VeryLow']
        ),
        'AmtCredit': pd.cut(
            app_data['AMT_CREDIT'],
            bins=[0, 252000, 270000, 675000, 1125000, 1350000, 4100000],
            labels=['MedHigh', 'Med', 'High', 'MedLow', 'VeryLow', 'Low']
        ),
        'AmtGoodsPrice': pd.cut(
            app_data['AMT_GOODS_PRICE'],
            bins=[0, 171000, 270000, 450000, 454500, 675000, 4100000],
            labels=['High', 'Medium', 'VeryHigh', 'Low', 'MedHigh', 'VeryLow']
        ),
        'AmtReqCreditBureauMon': pd.cut(
            app_data['AMT_REQ_CREDIT_BUREAU_MON'],
            bins=[0, 1, 27],
            labels=['Medium', 'VeryLow']
        ),
    }, index=app_data.index)

    result = pd.concat([app_data, derived], axis=1).copy()

    # Diagnostic distributions
    for col in ['IncomeBracket', 'LoanOutcome',
                'ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk',
                'AmtCredit', 'AmtGoodsPrice', 'AmtReqCreditBureauMon']:
        logger.debug(f"{col} distribution:\n"
                     f"{result[col].value_counts(dropna=False).to_string()}")

    return result


def generate_install_summary(data_dir: str) -> pd.DataFrame:
    """Summarise installment payment behaviour per applicant.

    Reads ``installments_payments.csv`` and computes the fraction of
    instalments paid on or before their scheduled due date.  That rate is
    bucketed into the ordinal ``PaymentHistory`` model feature.

    Analogous to a prior-fill compliance rate: did the applicant honour past
    repayment schedules as agreed?

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.

    Returns
    -------
    pandas.DataFrame
        One row per ``SK_ID_CURR`` with raw columns ``on_time_rate`` and
        ``total_fills``, plus derived column ``PaymentHistory``
        ∈ {Low, Medium, High}.
    """
    logger.debug(f"Running generate_install_summary dir: {data_dir}")

    install = pd.read_csv(f"{data_dir}installments_payments.csv")

    summary = (
        install.groupby('SK_ID_CURR')
        .apply(lambda x: pd.Series({
            'on_time_rate': (x['DAYS_ENTRY_PAYMENT'] <= x['DAYS_INSTALMENT']).mean(),
            'total_fills':  len(x),
        }))
        .reset_index()
    )

    summary['PaymentHistory'] = pd.cut(
        summary['on_time_rate'],
        bins=[0, 0.5, 0.8, 1.01],
        labels=['Low', 'Medium', 'High']
    )

    return summary


def generate_util_summary(data_dir: str) -> pd.DataFrame:
    """Summarise credit-card utilisation per applicant.

    Reads ``credit_card_balance.csv``, computes a per-snapshot utilisation
    ratio (balance ÷ limit), and averages across all snapshots.  The mean
    utilisation is bucketed into the ordinal ``CreditUtilization`` feature.

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.

    Returns
    -------
    pandas.DataFrame
        One row per ``SK_ID_CURR`` with raw columns ``avg_utilization`` and
        ``max_utilization``, plus derived column ``CreditUtilization``
        ∈ {Low, Medium, High, MaxedOut}.
    """
    logger.debug(f"Running generate_util_summary dir: {data_dir}")

    credit_card = pd.read_csv(f"{data_dir}credit_card_balance.csv")

    credit_card['CREDIT_UTILIZATION'] = np.where(
        credit_card['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
        credit_card['AMT_BALANCE'] / credit_card['AMT_CREDIT_LIMIT_ACTUAL'],
        0
    )

    summary = (
        credit_card.groupby('SK_ID_CURR')
        .agg(
            avg_utilization=('CREDIT_UTILIZATION', 'mean'),
            max_utilization=('CREDIT_UTILIZATION', 'max'),
        )
        .reset_index()
    )

    summary['CreditUtilization'] = pd.cut(
        summary['avg_utilization'],
        bins=[-0.01, 0.30, 0.60, 0.90, 999],
        labels=['Low', 'Medium', 'High', 'MaxedOut']
    )

    return summary


def generate_pos_summary(data_dir: str) -> pd.DataFrame:
    """Summarise POS-cash loan delinquency and contract status per applicant.

    Reads ``POS_CASH_balance.csv`` and extracts the worst days-past-due (DPD)
    seen across all POS/cash loan snapshots, plus whether the applicant
    currently holds any active contracts.

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.

    Returns
    -------
    pandas.DataFrame
        One row per ``SK_ID_CURR`` with raw columns ``max_dpd``,
        ``max_dpd_def``, and ``active_contracts``, plus derived columns
        ``DPD`` ∈ {None, Low, Medium, High} and
        ``ContractStatus`` ∈ {Active, Inactive}.
    """
    logger.debug(f"Running generate_pos_summary dir: {data_dir}")

    pos_cash = pd.read_csv(f"{data_dir}POS_CASH_balance.csv")

    summary = (
        pos_cash.groupby('SK_ID_CURR')
        .agg(
            max_dpd          =('SK_DPD',               'max'),
            max_dpd_def      =('SK_DPD_DEF',           'max'),
            active_contracts =('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
        )
        .reset_index()
    )

    summary['DPD'] = pd.cut(
        summary['max_dpd'],
        bins=[-1, 0, 30, 60, 999999],
        labels=['None', 'Low', 'Medium', 'High']
    )

    summary['ContractStatus'] = np.where(
        summary['active_contracts'] > 0, 'Active', 'Inactive'
    )

    return summary


def generate_bureau_summary(data_dir: str) -> pd.DataFrame:
    """Summarise external credit-bureau history per applicant.

    Reads ``bureau.csv`` (Credit Bureau records for external loans) and
    derives five categorical model features:

    - **DaysOverdue**     — worst days-overdue bucket across all bureau loans.
    - **MaxOverdue**      — highest overdue amount bucket.
    - **DebtLoad**        — total current outstanding debt bucket.
    - **CreditProlonged** — how many times loans were extended/prolonged.
    - **ActiveCredits**   — count of currently active bureau credits.

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.

    Returns
    -------
    pandas.DataFrame
        One row per ``SK_ID_CURR`` with raw columns ``max_days_overdue``,
        ``max_overdue_amt``, ``total_debt``, ``total_prolonged``, and
        ``active_credits``, plus the five derived categorical columns above.
    """
    logger.debug(f"Running generate_bureau_summary dir: {data_dir}")

    bureau = pd.read_csv(f"{data_dir}bureau.csv")

    summary = (
        bureau.groupby('SK_ID_CURR')
        .agg(
            max_days_overdue=('CREDIT_DAY_OVERDUE',    'max'),
            max_overdue_amt =('AMT_CREDIT_MAX_OVERDUE', 'max'),
            total_debt      =('AMT_CREDIT_SUM_DEBT',    'sum'),
            total_prolonged =('CNT_CREDIT_PROLONG',     'sum'),
            active_credits  =('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        )
        .reset_index()
    )

    summary['DaysOverdue'] = pd.cut(
        summary['max_days_overdue'],
        bins=[-1, 0, 30, 90, 999999],
        labels=['None', 'Low', 'Medium', 'High']
    )

    summary['MaxOverdue'] = pd.cut(
        summary['max_overdue_amt'],
        bins=[-1, 0, 1000, 10000, 999999999],
        labels=['None', 'Low', 'Medium', 'High']
    )

    summary['DebtLoad'] = pd.cut(
        summary['total_debt'],
        bins=[-1, 0, 50000, 200000, 999999999],
        labels=['None', 'Low', 'Medium', 'High']
    )

    summary['CreditProlonged'] = pd.cut(
        summary['total_prolonged'],
        bins=[-1, 0, 1, 3, 999],
        labels=['Never', 'Once', 'Several', 'Many']
    )

    summary['ActiveCredits'] = pd.cut(
        summary['active_credits'],
        bins=[-1, 0, 1, 3, 999],
        labels=['None', 'One', 'Few', 'Many']
    )

    return summary


def generate_prev_summary(data_dir: str) -> pd.DataFrame:
    """Summarise previous Home Credit application history per applicant.

    Reads ``previous_application.csv`` and derives whether the applicant had
    at least one prior approved loan and how many prior applications were
    refused.

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.

    Returns
    -------
    pandas.DataFrame
        One row per ``SK_ID_CURR`` with raw columns ``prior_approved``,
        ``prior_refused``, and ``prior_total``, plus derived columns
        ``PriorLoanApproved`` ∈ {Yes, No} and
        ``PrevRejected`` ∈ {None, Once, Twice, Multiple}.
    """
    logger.debug(f"Running generate_prev_summary dir: {data_dir}")

    prev_app = pd.read_csv(f"{data_dir}previous_application.csv")

    summary = (
        prev_app.groupby('SK_ID_CURR')
        .agg(
            prior_approved=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
            prior_refused =('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
            prior_total   =('NAME_CONTRACT_STATUS', 'count'),
        )
        .reset_index()
    )

    summary['PriorLoanApproved'] = np.where(
        summary['prior_approved'] > 0, 'Yes', 'No'
    )

    summary['PrevRejected'] = pd.cut(
        summary['prior_refused'],
        bins=[-1, 0, 1, 2, 999999],
        labels=['None', 'Once', 'Twice', 'Multiple']
    )

    return summary


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def log_raw_diagnostics(app_data, install_summary, util_summary,
                         pos_summary, bureau_summary, prev_summary):
    """Log descriptive statistics for raw numeric columns split by LoanOutcome.

    For each subsidiary summary table, joins it to the labelled application
    rows and calls ``describe()`` grouped by ``LoanOutcome``.  Use this to
    verify that the underlying continuous distributions differ between
    defaulters and repayers before and after discretisation.

    Parameters
    ----------
    app_data : pandas.DataFrame
        Full application DataFrame (must contain ``SK_ID_CURR`` and
        ``LoanOutcome``).
    install_summary : pandas.DataFrame
        Output of :func:`generate_install_summary`.
    util_summary : pandas.DataFrame
        Output of :func:`generate_util_summary`.
    pos_summary : pandas.DataFrame
        Output of :func:`generate_pos_summary`.
    bureau_summary : pandas.DataFrame
        Output of :func:`generate_bureau_summary`.
    prev_summary : pandas.DataFrame
        Output of :func:`generate_prev_summary`.
    """
    labeled = app_data[['SK_ID_CURR', 'LoanOutcome']].dropna(subset=['LoanOutcome'])

    summaries = {
        'install_summary': install_summary,
        'util_summary':    util_summary,
        'pos_summary':     pos_summary,
        'bureau_summary':  bureau_summary,
        'prev_summary':    prev_summary,
    }

    for name, summary in summaries.items():
        numeric_cols = (
            summary.select_dtypes(include='number')
            .columns.difference(['SK_ID_CURR'])
            .tolist()
        )
        diag = summary[['SK_ID_CURR'] + numeric_cols].merge(labeled, on='SK_ID_CURR')
        logger.debug(
            f"{name} numeric cols by LoanOutcome:\n"
            f"{diag.groupby('LoanOutcome')[numeric_cols].describe().to_string()}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _APP_RAW_COLS
# ---------------------------------------------------------------------------
# Raw columns from application_train / application_test carried forward into
# the wide merged DataFrame.  Keeping these alongside the derived features
# allows downstream exploratory analysis against the original continuous
# values without re-reading the CSVs.  SK_ID_CURR is included here as the
# join key; it is dropped from the model feature table (MODEL_COLS) before
# training.
_APP_RAW_COLS = [
    'SK_ID_CURR',
    'TARGET',               # 0 = Repaid, 1 = Defaulted, NaN = test row
    'NAME_INCOME_TYPE',     # raw income-source string → IncomeType
    'OCCUPATION_TYPE',      # raw occupation string → OccupationType
    'AMT_INCOME_TOTAL',     # raw annual income → IncomeBracket
    'EXT_SOURCE_1',         # external credit score 1 → ExtSource1Risk
    'EXT_SOURCE_2',         # external credit score 2 → ExtSource2Risk
    'EXT_SOURCE_3',         # external credit score 3 → ExtSource3Risk
    'AMT_CREDIT',           # loan amount → AmtCredit
    'AMT_GOODS_PRICE',      # financed goods value → AmtGoodsPrice
    'AMT_REQ_CREDIT_BUREAU_MON',  # bureau enquiries last month → AmtReqCreditBureauMon
]

# ---------------------------------------------------------------------------
# _APP_DERIVED_COLS
# ---------------------------------------------------------------------------
# PascalCase categorical features derived from the application table columns
# listed in _APP_RAW_COLS.  These are selected alongside the raw columns
# when building the merged DataFrame so that both forms are always available
# in one place.
_APP_DERIVED_COLS = [
    'IncomeType', 'OccupationType', 'IncomeBracket', 'LoanOutcome',
    'ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk',
    'AmtCredit', 'AmtGoodsPrice', 'AmtReqCreditBureauMon',
]


def get_merged_data(data_dir: str) -> pd.DataFrame:
    """Build the master DataFrame by merging all six Home Credit data sources.

    This is the single entry point for data preparation.  It:

    1. Loads application data and derives application-level model features.
    2. Builds per-applicant summaries from the five subsidiary tables
       (installments, credit-card, POS-cash, bureau, previous applications).
    3. Left-joins every summary onto the application data on ``SK_ID_CURR``.
    4. Fills missing derived categorical values with conservative defaults
       (see :data:`CATEGORICAL_FILL_VALUES`).
    5. Converts all derived categorical columns to ``object`` dtype so that
       pgmpy treats them as discrete variables.

    The resulting DataFrame contains **both** the raw source columns (kept for
    exploratory analysis and diagnostics) and the PascalCase derived model
    features (ready for use in the Bayesian Network).

    Column groups in the output
    ---------------------------
    Application raw
        ``SK_ID_CURR``, ``TARGET``, ``NAME_INCOME_TYPE``, ``OCCUPATION_TYPE``,
        ``AMT_INCOME_TOTAL``, ``EXT_SOURCE_1/2/3``, ``AMT_CREDIT``,
        ``AMT_GOODS_PRICE``, ``AMT_REQ_CREDIT_BUREAU_MON``

    Application derived
        ``IncomeType``, ``OccupationType``, ``IncomeBracket``, ``LoanOutcome``,
        ``ExtSource1/2/3Risk``, ``AmtCredit``, ``AmtGoodsPrice``,
        ``AmtReqCreditBureauMon``

    Installment raw / derived
        ``on_time_rate``, ``total_fills`` / ``PaymentHistory``

    Previous-application raw / derived
        ``prior_approved``, ``prior_refused``, ``prior_total`` /
        ``PriorLoanApproved``, ``PrevRejected``

    Credit-card raw / derived
        ``avg_utilization``, ``max_utilization`` / ``CreditUtilization``

    POS-cash raw / derived
        ``max_dpd``, ``max_dpd_def``, ``active_contracts`` /
        ``DPD``, ``ContractStatus``

    Bureau raw / derived
        ``max_days_overdue``, ``max_overdue_amt``, ``total_debt``,
        ``total_prolonged``, ``active_credits`` /
        ``DaysOverdue``, ``MaxOverdue``, ``DebtLoad``,
        ``CreditProlonged``, ``ActiveCredits``

    Parameters
    ----------
    data_dir : str
        Directory that contains the raw Home Credit CSV files.  Must end with
        a path separator (e.g. ``"data/"``).

    Returns
    -------
    pandas.DataFrame
        One row per applicant (train + test rows combined).
        Test rows have ``TARGET = NaN`` and ``LoanOutcome = NaN``.
        Raw numeric columns retain their original ``NaN`` values.
        Derived categorical columns have no nulls.
    """
    logger.debug(f"Running get_merged_data dir: {data_dir}")

    # Build each source summary
    app_data        = get_app_data(data_dir)
    bureau_summary  = generate_bureau_summary(data_dir)
    install_summary = generate_install_summary(data_dir)
    util_summary    = generate_util_summary(data_dir)
    pos_summary     = generate_pos_summary(data_dir)
    prev_summary    = generate_prev_summary(data_dir)

    # Select which raw app columns to carry forward (intersect with what exists)
    app_raw_cols = [c for c in _APP_RAW_COLS if c in app_data.columns]

    # Merge: start from app raw + derived, then join every summary
    merged_df = (
        app_data[app_raw_cols + _APP_DERIVED_COLS]
        .merge(
            install_summary[['SK_ID_CURR', 'on_time_rate', 'total_fills', 'PaymentHistory']],
            on='SK_ID_CURR', how='left'
        )
        .merge(
            prev_summary[['SK_ID_CURR', 'prior_approved', 'prior_refused', 'prior_total',
                          'PriorLoanApproved', 'PrevRejected']],
            on='SK_ID_CURR', how='left'
        )
        .merge(
            util_summary[['SK_ID_CURR', 'avg_utilization', 'max_utilization', 'CreditUtilization']],
            on='SK_ID_CURR', how='left'
        )
        .merge(
            pos_summary[['SK_ID_CURR', 'max_dpd', 'max_dpd_def', 'active_contracts',
                         'DPD', 'ContractStatus']],
            on='SK_ID_CURR', how='left'
        )
        .merge(
            bureau_summary[['SK_ID_CURR', 'max_days_overdue', 'max_overdue_amt',
                            'total_debt', 'total_prolonged', 'active_credits',
                            'DaysOverdue', 'MaxOverdue', 'DebtLoad',
                            'CreditProlonged', 'ActiveCredits']],
            on='SK_ID_CURR', how='left'
        )
    )

    # Fill nulls in derived categorical columns only
    for col, fill_val in CATEGORICAL_FILL_VALUES.items():
        if col not in merged_df.columns:
            continue
        if merged_df[col].dtype.name == 'category':
            if fill_val not in merged_df[col].cat.categories:
                merged_df[col] = merged_df[col].cat.add_categories(fill_val)
        merged_df[col] = merged_df[col].fillna(fill_val)

    # Cast derived categorical columns to plain object so pgmpy sees discrete states
    derived_cols = list(CATEGORICAL_FILL_VALUES.keys()) + [
        'IncomeType', 'OccupationType', 'IncomeBracket', 'LoanOutcome',
    ]
    for col in derived_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype('object')

    logger.debug(f"merged_df shape: {merged_df.shape}")
    logger.debug(f"merged_df dtypes:\n{merged_df.dtypes.to_string()}")

    log_raw_diagnostics(app_data, install_summary, util_summary,
                        pos_summary, bureau_summary, prev_summary)

    return merged_df
