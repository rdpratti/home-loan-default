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
from sklearn.impute import KNNImputer

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
    'CreditAge',
    'ClosedVsActive',
    'Age',
    'EmploymentYears',
    'AgeEducation',
    'EducationLevel', 'Gender', 'FamilyStatus',
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
    'CreditAge':             'Short',     # no bureau record → treat as short credit history
    'ClosedVsActive':        'Low',       # no bureau record → assume low closed-credit ratio
    'Age':                   'MidAge',    # age always present — default unused in practice
    'EmploymentYears':       'Short',     # unemployed / no employment data → short tenure
    'AgeEducation':          'OldLowEdu', # age always present — default unused in practice
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

def _knn_impute_app(app_data: pd.DataFrame) -> pd.DataFrame:
    """KNN-impute random-missing numeric columns before binning.

    Only columns whose nulls represent genuine data gaps (MAR/MCAR) are
    imputed.  Structural nulls — bureau join misses, credit-card join misses —
    are left as NaN and handled downstream by CATEGORICAL_FILL_VALUES.

    Data leakage prevention
    -----------------------
    The scaler statistics (mean, std) and the KNNImputer are fitted exclusively
    on training rows (TARGET is not null).  The fitted objects are then used to
    transform both training and test rows.  This mirrors the correct sklearn
    Pipeline pattern (fit on train → transform all) and ensures that test-set
    information cannot influence how training rows are imputed.

    Columns imputed
    ---------------
    EXT_SOURCE_1/2/3
        External credit-bureau risk scores [0, 1].  Missing for 54 %, 0.2 %,
        and 20 % of applicants respectively.  KNN uses five neighbours selected
        by Euclidean distance across a focused feature set; standardisation is
        applied using training-row mean/std so that scale differences do not
        distort distances.

    DAYS_EMPLOYED
        Days since employment start (stored as a negative integer).  The
        sentinel value 365243 flags pensioners/unemployed in the raw data and
        is deliberately excluded from binning downstream via a .where() guard.
        To preserve that semantics:
          - NaN rows where NAME_INCOME_TYPE is Pensioner or Unemployed are left
            as NaN — they will flow through the .where() guard and land in the
            EmploymentYears 'Short' default bucket.
          - NaN rows for all other income types are imputed with the median of
            actual employed training applicants (non-null, non-sentinel).
        Median is used here rather than KNN because the conditional exclusion
        logic would require a second separate KNN pass; the median is a
        defensible and cheap substitute for a column with only 18 % nulls.

    Parameters
    ----------
    app_data : pd.DataFrame
        Raw concatenated application data (train + test).  Train rows are
        identified by TARGET being non-null.

    Returns
    -------
    pd.DataFrame
        Copy of app_data with imputed values written into the source columns.
    """
    app_data = app_data.copy()

    # Identify training rows — imputer is fitted on these only.
    train_mask = app_data['TARGET'].notna()

    # ------------------------------------------------------------------
    # EXT_SOURCE_1 / 2 / 3  — KNN imputation
    # ------------------------------------------------------------------
    # Feature set chosen to capture applicant risk profile without
    # including the large-range sentinel-contaminated DAYS_EMPLOYED.
    knn_cols = ['DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    target_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    X = app_data[knn_cols].copy()

    # Standardise using per-column mean/std computed on TRAINING rows only.
    # pandas .mean()/.std() skip NaN by default, so this is safe even with
    # missing values present.  NaN values survive the arithmetic unchanged,
    # which is exactly what KNNImputer expects as input.
    col_means = X.loc[train_mask].mean()
    col_stds  = X.loc[train_mask].std().replace(0, 1)   # guard zero-variance
    X_scaled  = (X - col_means) / col_stds              # NaN rows remain NaN

    # Fit imputer on training rows only, then transform all rows.
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(X_scaled.loc[train_mask])

    X_imp_scaled = pd.DataFrame(
        imputer.transform(X_scaled),
        columns=knn_cols,
        index=X.index,
    )

    # Back-transform to original scale
    X_imputed = X_imp_scaled * col_stds + col_means

    # Write imputed values only into originally-null rows; clip to [0, 1]
    # since the EXT_SOURCE scores are bounded by definition.
    for col in target_cols:
        null_mask = app_data[col].isna()
        if null_mask.any():
            app_data.loc[null_mask, col] = X_imputed.loc[null_mask, col].clip(0, 1)
            logger.debug(
                f"KNN imputed {null_mask.sum():,} nulls in {col} "
                f"(min={app_data.loc[null_mask, col].min():.3f}, "
                f"max={app_data.loc[null_mask, col].max():.3f}, "
                f"mean={app_data.loc[null_mask, col].mean():.3f})"
            )

    # ------------------------------------------------------------------
    # DAYS_EMPLOYED  — conditional median imputation
    # ------------------------------------------------------------------
    # Pensioners and unemployed applicants have no employment duration by
    # definition; imputing a tenure for them would be semantically wrong.
    non_working   = app_data['NAME_INCOME_TYPE'].isin(['Pensioner', 'Unemployed'])
    emp_null_mask = app_data['DAYS_EMPLOYED'].isna()
    impute_emp    = emp_null_mask & ~non_working

    if impute_emp.any():
        # Median computed on TRAINING rows only — genuinely employed, non-sentinel.
        employed_vals = app_data.loc[
            train_mask & ~emp_null_mask & (app_data['DAYS_EMPLOYED'] != 365243),
            'DAYS_EMPLOYED'
        ]
        emp_median = employed_vals.median()
        app_data.loc[impute_emp, 'DAYS_EMPLOYED'] = emp_median
        logger.debug(
            f"Median imputed {impute_emp.sum():,} non-pensioner DAYS_EMPLOYED nulls "
            f"(median={emp_median:.0f} days, "
            f"~{abs(emp_median)/365.25:.1f} years)"
        )

    pensioner_null = emp_null_mask & non_working
    if pensioner_null.any():
        logger.debug(
            f"Left {pensioner_null.sum():,} Pensioner/Unemployed DAYS_EMPLOYED nulls "
            f"as NaN — will map to EmploymentYears 'Short' default."
        )

    return app_data


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

    # KNN-impute random-missing numeric columns before binning.
    # Structural nulls (bureau/credit-card join misses) are left untouched.
    app_data = _knn_impute_app(app_data)

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
        'Age': pd.qcut(
            app_data['DAYS_BIRTH'].abs() / 365.25,
            q=4,
            labels=['Young', 'MidAge', 'MidOld', 'Senior'],
        ),
        'EmploymentYears': pd.qcut(
            (app_data['DAYS_EMPLOYED'].abs() / 365.25).where(
                app_data['DAYS_EMPLOYED'] < 365243
            ),
            q=4,
            labels=['Short', 'MidShort', 'MidLong', 'Long'],
            duplicates='drop',
        ),
        'EducationLevel': app_data['NAME_EDUCATION_TYPE']
                          .fillna(app_data['NAME_EDUCATION_TYPE'].mode()[0]).astype(object),
        'Gender': app_data['CODE_GENDER']
                  .replace('XNA', app_data['CODE_GENDER'][app_data['CODE_GENDER'] != 'XNA'].mode()[0])
                  .fillna(app_data['CODE_GENDER'][app_data['CODE_GENDER'] != 'XNA'].mode()[0]).astype(object),
        'FamilyStatus': app_data['NAME_FAMILY_STATUS']
                        .replace('Unknown', app_data['NAME_FAMILY_STATUS']
                                 [app_data['NAME_FAMILY_STATUS'] != 'Unknown'].mode()[0])
                        .fillna(app_data['NAME_FAMILY_STATUS']
                                [app_data['NAME_FAMILY_STATUS'] != 'Unknown'].mode()[0]).astype(object),
    }, index=app_data.index)

    result = pd.concat([app_data, derived], axis=1).copy()

    # AgeEducation — interaction of age quartile and education level.
    # Captures the dominant synergy identified in pairwise IV screening:
    # young applicants with secondary education are the largest high-risk segment.
    age_q = pd.qcut(app_data['DAYS_BIRTH'].abs() / 365.25, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    young   = age_q.isin(['Q1', 'Q2'])
    high_edu = app_data['NAME_EDUCATION_TYPE'].fillna('').isin(
        ['Higher education', 'Academic degree']
    )
    result['AgeEducation'] = np.where(
        young  &  high_edu, 'YoungHighEdu',
        np.where(young  & ~high_edu, 'YoungLowEdu',
        np.where(~young &  high_edu, 'OldHighEdu',
                                     'OldLowEdu'))
    )

    # Diagnostic distributions
    for col in ['IncomeBracket', 'LoanOutcome',
                'ExtSource1Risk', 'ExtSource2Risk', 'ExtSource3Risk',
                'AmtCredit', 'AmtGoodsPrice', 'AmtReqCreditBureauMon', 'Age',
                'EmploymentYears', 'AgeEducation']:
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

    # Per-row derived columns for aggregation
    install['on_time']   = install['DAYS_ENTRY_PAYMENT'] <= install['DAYS_INSTALMENT']
    install['days_late'] = (install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']).clip(lower=0)
    install['shortfall'] = (install['AMT_INSTALMENT']    - install['AMT_PAYMENT']).clip(lower=0)
    install['missed']    = (install['AMT_PAYMENT'] == 0).astype(int)

    summary = (
        install.groupby('SK_ID_CURR')
        .agg(
            on_time_rate        =('on_time',   'mean'),
            total_fills         =('on_time',   'count'),
            avg_days_late       =('days_late', 'mean'),
            avg_shortfall       =('shortfall', 'mean'),
            missed_payment_rate =('missed',    'mean'),
        )
        .reset_index()
    )

    # Recent on-time rate (last 6 months relative to application date)
    recent = (
        install[install['DAYS_ENTRY_PAYMENT'] >= -180]
        .groupby('SK_ID_CURR')['on_time']
        .mean()
        .reset_index()
        .rename(columns={'on_time': 'recent_on_time_rate'})
    )
    summary = summary.merge(recent, on='SK_ID_CURR', how='left')
    # Positive = improving trend, negative = worsening
    summary['payment_trend'] = summary['recent_on_time_rate'] - summary['on_time_rate']

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
            max_days_overdue=('CREDIT_DAY_OVERDUE',     'max'),
            max_overdue_amt =('AMT_CREDIT_MAX_OVERDUE', 'max'),
            total_debt      =('AMT_CREDIT_SUM_DEBT',    'sum'),
            total_prolonged =('CNT_CREDIT_PROLONG',     'sum'),
            active_credits  =('CREDIT_ACTIVE',  lambda x: (x == 'Active').sum()),
            total_credits   =('CREDIT_ACTIVE',  'count'),
            overdue_cnt     =('CREDIT_DAY_OVERDUE', lambda x: (x > 0).sum()),
            avg_credit_age  =('DAYS_CREDIT',    lambda x: x.abs().mean()),
            closed_credits  =('CREDIT_ACTIVE',  lambda x: (x == 'Closed').sum()),
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

    summary['CreditAge'] = pd.qcut(
        summary['avg_credit_age'],
        q=4,
        labels=['Short', 'MedShort', 'MedLong', 'Long'],
        duplicates='drop'
    )

    ratio = summary['closed_credits'] / summary['total_credits'].replace(0, np.nan)
    summary['ClosedVsActive'] = pd.qcut(
        ratio,
        q=4,
        labels=['Low', 'MedLow', 'MedHigh', 'High'],
        duplicates='drop'
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
            prior_approved  =('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
            prior_refused   =('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
            prior_total     =('NAME_CONTRACT_STATUS', 'count'),
            days_last_app   =('DAYS_DECISION',        'max'),  # max of negatives = most recent
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
# IV screening helpers
# ---------------------------------------------------------------------------

def compute_iv(df, feature, target='LoanOutcome',
               default_label='Defaulted', repaid_label='Repaid'):
    """Compute Information Value (IV) for a categorical feature against a binary target.

    IV measures how well a feature separates defaulters from repaid applicants.
    Each state's contribution is weighted by its Weight of Evidence (WoE):
    the log ratio of the event share to the non-event share in that bin.

    IV interpretation
    -----------------
    < 0.02          Useless
    0.02 – 0.10     Weak
    0.10 – 0.30     Medium — worth adding to the model
    0.30 – 0.50     Strong
    > 0.50          Suspicious (possible target leakage)

    Parameters
    ----------
    df : pandas.DataFrame
        Labelled rows only (must contain both ``feature`` and ``target``).
    feature : str
        Name of the categorical column to evaluate.
    target : str
        Binary target column name.
    default_label : str
        Value in ``target`` representing a default event.
    repaid_label : str
        Value in ``target`` representing a non-event (repaid).

    Returns
    -------
    total_iv : float
        Scalar IV score for the feature.
    breakdown : pandas.DataFrame
        Per-state table with columns ``state``, ``n_defaults``, ``n_repaid``,
        ``woe``, and ``iv_contribution``, sorted ascending by WoE.
    """
    total_events     = (df[target] == default_label).sum()
    total_non_events = (df[target] == repaid_label).sum()

    rows = []
    for state in df[feature].dropna().unique():
        mask         = df[feature] == state
        n_events     = (df.loc[mask, target] == default_label).sum()
        n_non_events = (df.loc[mask, target] == repaid_label).sum()

        doe  = np.clip(n_events     / total_events,     1e-6, None)
        done = np.clip(n_non_events / total_non_events, 1e-6, None)

        woe = np.log(doe / done)
        iv  = (doe - done) * woe
        rows.append({
            'state':           state,
            'n_defaults':      int(n_events),
            'n_repaid':        int(n_non_events),
            'woe':             round(woe, 4),
            'iv_contribution': round(iv,  4),
        })

    breakdown = pd.DataFrame(rows).sort_values('woe').reset_index(drop=True)
    total_iv  = float(breakdown['iv_contribution'].sum())
    return total_iv, breakdown


def screen_ratio_candidates(merged_df):
    """Screen ratio feature candidates using Information Value.

    Candidates: DebtToIncome, AnnuityToIncome, LoanToValue, IncomePerFamilyMember.
    """
    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()
    _iv_screen(labelled, {
        'DebtToIncome':          labelled['AMT_CREDIT']       / labelled['AMT_INCOME_TOTAL'].replace(0, np.nan),
        'AnnuityToIncome':       labelled['AMT_ANNUITY']      / labelled['AMT_INCOME_TOTAL'].replace(0, np.nan),
        'LoanToValue':           labelled['AMT_CREDIT']       / labelled['AMT_GOODS_PRICE'].replace(0, np.nan),
        'IncomePerFamilyMember': labelled['AMT_INCOME_TOTAL'] / labelled['CNT_FAM_MEMBERS'].replace(0, np.nan),
    }, 'Ratio Feature IV Screening')


def _iv_screen(labelled, candidates, section_title):
    """Shared IV screening loop used by all screen_*_candidates functions.

    For each candidate, discretises continuous series into quartile bins (if
    needed) then calls :func:`compute_iv` and logs the result.

    Parameters
    ----------
    labelled : pandas.DataFrame
        Rows with a valid ``LoanOutcome`` label.
    candidates : dict[str, pandas.Series]
        Mapping of display name → raw Series to evaluate.  Categorical series
        are evaluated directly; numeric series are binned into quartiles first.
    section_title : str
        Header string used in the log delimiter lines.
    """
    logger.debug(f"=== {section_title} ===")
    for name, series in candidates.items():
        labelled = labelled.copy()
        labelled[name] = series

        if not pd.api.types.is_numeric_dtype(series):
            # Categorical / string — evaluate states directly, no binning needed
            feature_col = name
        else:
            try:
                # Infer actual bin count after dropping duplicate edges (zero-inflated
                # columns often produce fewer than 4 unique quantile boundaries)
                binned = pd.qcut(series, q=4, duplicates='drop')
                n_bins = binned.cat.categories.shape[0]
                labels = [f'Q{i+1}' for i in range(n_bins)]
                labelled[f'{name}_bin'] = pd.qcut(
                    series, q=4, labels=labels, duplicates='drop'
                )
                feature_col = f'{name}_bin'
            except Exception as e:
                logger.debug(f"{name}: could not discretise — {e}")
                continue

        iv_score, breakdown = compute_iv(labelled, feature_col)
        rating = (
            'Useless'    if iv_score < 0.02 else
            'Weak'       if iv_score < 0.10 else
            'Medium'     if iv_score < 0.30 else
            'Strong'     if iv_score < 0.50 else
            'Suspicious'
        )
        logger.debug(
            f"\n{name}  IV={iv_score:.4f}  [{rating}]\n"
            f"{breakdown.to_string(index=False)}"
        )
    logger.debug(f"=== End {section_title} ===")


def screen_demographic_candidates(merged_df):
    """Screen Age and EmploymentYears candidate features using Information Value.

    DAYS_EMPLOYED sentinel 365243 (unemployed/pensioner) is set to NaN before binning.
    """
    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()
    emp_years = (labelled['DAYS_EMPLOYED'].abs() / 365.25).where(
        labelled['DAYS_EMPLOYED'] < 365243
    )
    _iv_screen(labelled, {
        'Age':            labelled['DAYS_BIRTH'].abs() / 365.25,
        'EmploymentYears': emp_years,
    }, 'Demographic Feature IV Screening')


def screen_app_categorical_candidates(merged_df):
    """Screen Gender, EducationLevel, and FamilyStatus using Information Value.

    These are already categorical so no binning is needed — WoE is computed
    directly on the raw string states.
    """
    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()
    # Replace rare XNA gender with NaN so it doesn't pollute the IV calculation
    labelled['CODE_GENDER'] = labelled['CODE_GENDER'].replace('XNA', np.nan)
    _iv_screen(labelled, {
        'Gender':         labelled['CODE_GENDER'],
        'EducationLevel': labelled['NAME_EDUCATION_TYPE'],
        'FamilyStatus':   labelled['NAME_FAMILY_STATUS'],
    }, 'App Categorical Feature IV Screening')


def screen_installment_candidates(merged_df):
    """Screen AvgDaysLate, AvgShortfall, MissedPaymentRate, PaymentTrend using IV."""
    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()
    _iv_screen(labelled, {
        'AvgDaysLate':       labelled['avg_days_late'],
        'AvgShortfall':      labelled['avg_shortfall'],
        'MissedPaymentRate': labelled['missed_payment_rate'],
        'PaymentTrend':      labelled['payment_trend'],
    }, 'Installment Feature IV Screening')


def screen_bureau_candidates(merged_df):
    """Screen OverdueRatio, CreditAge, and ClosedVsActive using Information Value."""
    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()
    _iv_screen(labelled, {
        'OverdueRatio':   labelled['overdue_cnt']     / labelled['total_credits'].replace(0, np.nan),
        'CreditAge':      labelled['avg_credit_age']  / 365.25,
        'ClosedVsActive': labelled['closed_credits']  / labelled['total_credits'].replace(0, np.nan),
    }, 'Bureau Feature IV Screening')


def screen_prevapp_candidates(merged_df):
    """Screen ApprovalRate and DaysSinceLastApp using Information Value."""
    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()
    _iv_screen(labelled, {
        'ApprovalRate':     labelled['prior_approved'] / labelled['prior_total'].replace(0, np.nan),
        'DaysSinceLastApp': labelled['days_last_app'].abs() / 365.25,
    }, 'Previous Application Feature IV Screening')


def screen_interaction_candidates(merged_df, min_individual_iv=0.03, top_n=20):
    """Systematically screen all pairwise interactions among weak-IV candidates.

    Builds a pool of every feature whose individual IV >= ``min_individual_iv``,
    generates all C(n, 2) pairs, computes IV on the joint cross-product of their
    discretised states, and logs results ranked by combined IV.

    A pair is flagged as potentially useful when its combined IV is materially
    higher than the sum of its two individual IVs, indicating the features capture
    complementary (non-redundant) risk segments.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        The wide merged frame returned by ``get_merged_data``.
    min_individual_iv : float
        Minimum individual IV for a feature to enter the pool.
        Default 0.03 excludes clearly useless features.
    top_n : int
        Number of top-ranked pairs to show in the detailed WoE breakdown.
    """
    from itertools import combinations

    labelled = merged_df.dropna(subset=['LoanOutcome']).copy()

    def _to_cat(series, name):
        """Return a string-typed categorical series, binning numeric ones."""
        if not pd.api.types.is_numeric_dtype(series):
            return series.astype(str)
        try:
            binned = pd.qcut(series, q=4, duplicates='drop')
            n = binned.cat.categories.shape[0]
            labels = [f'Q{i+1}' for i in range(n)]
            return pd.qcut(series, q=4, labels=labels, duplicates='drop').astype(str)
        except Exception as e:
            logger.debug(f"  _to_cat({name}): could not bin — {e}")
            return None

    # ------------------------------------------------------------------
    # Build candidate pool: name -> (series, individual_iv)
    # ------------------------------------------------------------------
    nan_tokens = {'nan', 'None', '<NA>', 'NaN'}

    raw_candidates = {
        'Age':              labelled['DAYS_BIRTH'].abs() / 365.25,
        'EmploymentYears':  (labelled['DAYS_EMPLOYED'].abs() / 365.25)
                            .where(labelled['DAYS_EMPLOYED'] < 365243),
        'CreditAge':        labelled['avg_credit_age'] / 365.25,
        'ClosedVsActive':   labelled['closed_credits'] / labelled['total_credits'].replace(0, np.nan),
        'LoanToValue':      labelled['AMT_CREDIT'] / labelled['AMT_GOODS_PRICE'].replace(0, np.nan),
        'EducationLevel':   labelled['NAME_EDUCATION_TYPE'],
        'Gender':           labelled['CODE_GENDER'].replace('XNA', np.nan),
        'AvgDaysLate':      labelled['avg_days_late'],
        'AvgShortfall':     labelled['avg_shortfall'],
        'ApprovalRate':     labelled['prior_approved'] / labelled['prior_total'].replace(0, np.nan),
        'FamilyStatus':     labelled['NAME_FAMILY_STATUS'],
    }

    pool = {}
    for name, raw in raw_candidates.items():
        cat = _to_cat(raw, name)
        if cat is None:
            continue
        tmp = labelled.copy()
        tmp[name] = cat
        mask = ~cat.isin(nan_tokens)
        try:
            iv_score, _ = compute_iv(tmp[mask], name)
        except Exception:
            continue
        if iv_score >= min_individual_iv:
            pool[name] = (cat, iv_score)

    logger.debug(
        f"=== Pairwise Interaction IV Screening "
        f"(pool size={len(pool)}, pairs={len(pool)*(len(pool)-1)//2}) ==="
    )
    logger.debug("Pool members: " + ", ".join(f"{n}={iv:.4f}" for n, (_, iv) in pool.items()))

    # ------------------------------------------------------------------
    # Score every pair
    # ------------------------------------------------------------------
    results = []
    for (name_a, (cat_a, iv_a)), (name_b, (cat_b, iv_b)) in combinations(pool.items(), 2):
        combo_name = f"{name_a} x {name_b}"
        combined = cat_a + ' | ' + cat_b
        mask = ~(cat_a.isin(nan_tokens) | cat_b.isin(nan_tokens))
        subset = labelled[mask].copy()
        subset[combo_name] = combined[mask]
        try:
            iv_combo, breakdown = compute_iv(subset, combo_name)
        except Exception as e:
            logger.debug(f"  {combo_name}: failed — {e}")
            continue
        results.append({
            'pair':       combo_name,
            'iv_a':       iv_a,
            'iv_b':       iv_b,
            'iv_sum':     iv_a + iv_b,
            'iv_combo':   iv_combo,
            'lift':       iv_combo - (iv_a + iv_b),   # positive = synergy
            'n_states':   len(breakdown),
            '_breakdown': breakdown,
        })

    if not results:
        logger.debug("No pairs scored successfully.")
        logger.debug("=== End Pairwise Interaction IV Screening ===")
        return

    ranked = sorted(results, key=lambda x: -x['iv_combo'])

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    header = f"{'Pair':<35} {'IV_A':>7} {'IV_B':>7} {'Sum':>7} {'Combined':>9} {'Lift':>7} {'Rating'}"
    logger.debug("\nAll pairs ranked by combined IV:\n" + header)
    for r in ranked:
        rating = (
            'Useless' if r['iv_combo'] < 0.02 else
            'Weak'    if r['iv_combo'] < 0.10 else
            'Medium'  if r['iv_combo'] < 0.30 else
            'Strong'  if r['iv_combo'] < 0.50 else 'Suspicious'
        )
        synergy = '*** SYNERGY' if r['lift'] > 0.02 else ''
        logger.debug(
            f"  {r['pair']:<35} {r['iv_a']:>7.4f} {r['iv_b']:>7.4f} "
            f"{r['iv_sum']:>7.4f} {r['iv_combo']:>9.4f} {r['lift']:>+7.4f}  [{rating}] {synergy}"
        )

    # ------------------------------------------------------------------
    # Detailed WoE breakdown for top_n pairs
    # ------------------------------------------------------------------
    logger.debug(f"\nDetailed WoE breakdown for top {top_n} pairs:")
    for r in ranked[:top_n]:
        bd = r['_breakdown']
        top5 = bd.nlargest(5, 'woe')
        bot5 = bd.nsmallest(5, 'woe')
        detail = pd.concat([top5, bot5]).drop_duplicates().sort_values('woe', ascending=False)
        logger.debug(
            f"\n{r['pair']}  IV={r['iv_combo']:.4f}  lift={r['lift']:+.4f}  "
            f"({r['n_states']} states, top/bottom 5 by WoE)\n"
            f"{detail.to_string(index=False)}"
        )

    logger.debug("=== End Pairwise Interaction IV Screening ===")


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
    'AMT_ANNUITY',          # monthly loan repayment amount → AnnuityToIncome ratio candidate
    'CNT_FAM_MEMBERS',      # household size → IncomePerFamilyMember ratio candidate
    'DAYS_BIRTH',           # days before application (negative) → Age candidate
    'DAYS_EMPLOYED',        # days before application (negative) → EmploymentStability candidate
    'NAME_EDUCATION_TYPE',  # highest education level → EducationLevel candidate
    'NAME_FAMILY_STATUS',   # marital status → FamilyStatus candidate
    'CODE_GENDER',          # applicant gender → Gender candidate
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
    'Age', 'EmploymentYears', 'AgeEducation',
    'EducationLevel', 'Gender', 'FamilyStatus',
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
            install_summary[['SK_ID_CURR', 'on_time_rate', 'total_fills', 'PaymentHistory',
                             'avg_days_late', 'avg_shortfall', 'missed_payment_rate',
                             'recent_on_time_rate', 'payment_trend']],
            on='SK_ID_CURR', how='left'
        )
        .merge(
            prev_summary[['SK_ID_CURR', 'prior_approved', 'prior_refused', 'prior_total',
                          'PriorLoanApproved', 'PrevRejected', 'days_last_app']],
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
                            'CreditProlonged', 'ActiveCredits', 'CreditAge', 'ClosedVsActive',
                            'total_credits', 'overdue_cnt', 'avg_credit_age', 'closed_credits']],
            on='SK_ID_CURR', how='left'
        )
    )

    # Null audit: report MODEL_COLS null counts before any fills are applied.
    model_cols_present = [c for c in MODEL_COLS if c in merged_df.columns]
    pre_fill_nulls = merged_df[model_cols_present].isnull().sum()
    pre_fill_nulls = pre_fill_nulls[pre_fill_nulls > 0]
    if len(pre_fill_nulls):
        logger.debug(
            f"MODEL_COLS null counts BEFORE fills (labeled+test rows combined):\n"
            f"{pre_fill_nulls.to_string()}"
        )
    else:
        logger.debug("MODEL_COLS: no nulls before fills.")

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
        'EducationLevel', 'Gender', 'FamilyStatus',
    ]
    for col in derived_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype('object')

    logger.debug(f"merged_df shape: {merged_df.shape}")
    logger.debug(f"merged_df dtypes:\n{merged_df.dtypes.to_string()}")

    log_raw_diagnostics(app_data, install_summary, util_summary,
                        pos_summary, bureau_summary, prev_summary)

    screen_ratio_candidates(merged_df)
    screen_demographic_candidates(merged_df)
    screen_app_categorical_candidates(merged_df)
    screen_installment_candidates(merged_df)
    screen_bureau_candidates(merged_df)
    screen_prevapp_candidates(merged_df)
    screen_interaction_candidates(merged_df)

    return merged_df
