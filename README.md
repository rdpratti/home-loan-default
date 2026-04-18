# Home Loan Default Prediction

Probabilistic modelling pipeline for predicting home loan default using the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset. The primary model is a Discrete Bayesian Network (pgmpy) with isotonic
regression probability calibration. A Categorical Naive Bayes baseline is also included.

## Best Results

Evaluated on a held-out test set of 61,503 rows (4,965 defaults / 56,538 repaid).

| Model | Threshold | TP | FP | Recall | Precision | F1 | ECE |
|---|---|---|---|---|---|---|---|
| Expert (post-imputation) | 0.10 | 2,607 | 13,098 | 0.525 | 0.166 | 0.252 | 0.0029 |
| Expert (post-imputation) | 0.12 | 2,234 | 10,177 | 0.450 | 0.180 | 0.260 | 0.0029 |
| Auto (post-imputation)   | 0.12 | 2,234 |  9,524 | 0.450 | 0.190 | 0.270 | 0.0023 |

At threshold 0.12 both models identify the same number of defaults (2,234 TP). The Auto
model generates fewer false alarms — 4.3 FPs per TP vs 4.6 for Expert. Lowering the
Expert threshold to 0.10 recovers 373 additional defaults at a cost of 3,574 extra false
alarms (5.0 FPs per TP).

ECE = Expected Calibration Error (lower is better). All probabilities are corrected with
isotonic regression calibration before thresholding.

## Dataset & Splits

Source: Home Credit Kaggle competition — 307,511 labeled loans (TARGET = 0 repaid,
TARGET = 1 defaulted; 8.5% default rate). Six source tables are merged per applicant:
application, bureau, installment payments, credit card balances, POS-cash balances, and
prior applications — producing 29 model-ready categorical features.

| Split | Rows | Purpose |
|---|---|---|
| Training | 209,107 (oversampled to 259,503) | Model fitting |
| Calibration | 36,902 | Isotonic regression calibrator fitting |
| Test | 61,503 | All reported evaluation metrics |

The 48,744 rows in `application_test.csv` have no known outcome and are excluded from
all model work.

## Bayesian Network

The pipeline runs two BN variants using pgmpy `DiscreteBayesianNetwork`:

**Expert model** — hand-authored DAG based on domain knowledge and Information Value
(IV) screening. Direct parents of `LoanOutcome`: `ExtSource1Risk`, `ExtSource2Risk`,
`ExtSource3Risk`, and `AgeEducation` (a combined Age × EducationLevel interaction term,
IV = 0.1523). Fewer direct parents reduces CPT sparsity and prevents the Dirichlet prior
from collapsing sparse cells to a flat 50/50 prediction.

**Auto model** — structure is learned from data using pgmpy's hill-climb search
(`max_indegree=3`, 10,000 iterations). The learned DAG independently routes demographic
signals (Age, EmploymentYears) through intermediate nodes rather than directly to
`LoanOutcome`.

Both models use:
- Bayesian parameter estimation with Dirichlet priors
- Variable Elimination for inference
- `RandomOverSampler` with `sampling_strategy=0.35` (minority class upsampled to 35% of
  majority) to address the 8.5% class imbalance before training
- Isotonic regression calibration fitted on the calibration set

## Imputation

A pre-binning imputation step in `src/data_prep.py` handles random nulls before
discretisation:

- **KNN imputation (k=5)** for `EXT_SOURCE_1` (54% null), `EXT_SOURCE_3` (20% null),
  and `EXT_SOURCE_2` using a focused feature set (DAYS_BIRTH, AMT_INCOME_TOTAL,
  AMT_CREDIT, EXT_SOURCE_1/2/3 cross-imputing). The imputer is fitted on training rows
  only to prevent data leakage.
- **Conditional median** for `DAYS_EMPLOYED` (18% null): training-set median computed
  from non-pensioner, non-unemployed rows (excludes the 365243 sentinel value). Pensioner
  and unemployed NaN rows are left untouched.
- **Structural nulls** (~14% with no bureau history, ~71% with no credit card) are not
  imputed — they are left as NaN and filled with conservative domain defaults during
  binning.

## Naive Bayes

A `CategoricalNB` baseline (scikit-learn) using all 20 `MODEL_COLS` features. The full
feature set is used directly (no CPT sparsity constraint). The Laplace smoothing
parameter (alpha) is selected by stratified 5-fold cross-validation.

## Project Structure

```
home-loan-default/
├── data/                              # Raw Home Credit CSVs (not tracked)
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── installments_payments.csv
│   ├── credit_card_balance.csv
│   ├── POS_CASH_balance.csv
│   └── previous_application.csv
├── docs/                              # Generated Word documents
│   ├── Executive_Summary.docx
│   ├── Feature_Engineering_Report.docx
│   ├── home_credit_bn_combined.docx
│   └── graph_structure_discovery_session.docx
├── logs/                              # Run logs, plots, and reliability diagrams
├── src/
│   ├── data_prep.py                   # Data loading, merging, imputation, discretisation
│   ├── graph_analytics.py             # Customer relationship graph features
│   ├── graph_structure_discovery.py   # NMI-based BN parent selection
│   └── home_credit_naive_bayes.py     # Categorical Naive Bayes baseline
├── home_credit_bayesian.py            # Bayesian Network pipeline (Expert + Auto)
├── generate_exec_summary.py           # Generates docs/Executive_Summary.docx
├── generate_feature_report.py         # Generates docs/Feature_Engineering_Report.docx
├── generate_doc.py                    # Generates docs/graph_structure_discovery_session.docx
├── combine_docs.py                    # Merges session docs into home_credit_bn_combined.docx
└── inspect_docs.py                    # Utility to inspect Word document structure
```

## Setup

**Option A — restore exact environment from the provided yml file:**

```bash
conda env create -f home_conda_environment.yml
conda activate home_credit
```

**Option B — create a minimal environment manually:**

```bash
conda create -n home_credit python=3.11
conda activate home_credit
pip install pandas numpy scikit-learn pgmpy matplotlib seaborn imbalanced-learn scipy statsmodels python-docx networkx
```

## Usage

Run from the repo root with the `data/` directory populated:

```bash
# Bayesian Network (Expert + Auto DAG)
python home_credit_bayesian.py

# Naive Bayes baseline
python src/home_credit_naive_bayes.py

# Regenerate Word documents
python generate_exec_summary.py
python generate_feature_report.py
```

Logs, plots, and reliability diagrams are written to `logs/`.

## Feature Set (`MODEL_COLS`)

| Feature | Source | Categories |
|---|---|---|
| ExtSource1/2/3Risk | application | Unknown · VeryHigh → VeryLow |
| AgeEducation | application | Combined Age × EducationLevel interaction |
| IncomeBracket | application | Low · Medium · High |
| IncomeType | application | Stable · Unstable |
| OccupationType | application | Unknown · Laborer · Professional |
| AmtCredit / AmtGoodsPrice | application | 6 buckets each |
| AmtReqCreditBureauMon | application | Unknown · Medium · VeryLow |
| PaymentHistory | installments | Unknown · Low · Medium · High |
| PriorLoanApproved / PrevRejected | previous apps | binary / 4-level |
| CreditUtilization | credit card | Low · Medium · High · MaxedOut |
| DPD / ContractStatus | POS cash | 4-level / binary |
| DaysOverdue / MaxOverdue / DebtLoad | bureau | 4-level each |
| CreditProlonged / ActiveCredits | bureau | 4-level each |
