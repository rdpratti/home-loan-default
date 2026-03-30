# Home Loan Default Prediction

Probabilistic modelling pipeline for predicting home loan default using the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset.

## Models

| Model | Threshold | Precision | Recall | F1 | BalAcc | ROC-AUC |
|-------|-----------|-----------|--------|----|--------|---------|
| BN Expert (4 parents) | 0.30 | 0.16 | 0.58 | 0.26 | 0.660 | — |
| BN Auto (3 parents) | 0.30 | 0.17 | 0.55 | 0.26 | 0.659 | — |
| Naive Bayes (20 features) | 0.60 | 0.18 | 0.57 | 0.27 | 0.670 | 0.734 |

All models use the same discrete feature set and are evaluated on an 80/20
stratified train/test split at the threshold that maximises F1.

## Bayesian Network

The pipeline runs two BN variants using pgmpy `DiscreteBayesianNetwork`:

**Expert model** — DAG edges are hand-authored using domain knowledge. Direct
parents of `LoanOutcome` are selected by greedy forward search using
conditional normalised mutual information (NMI), reducing the original 7
candidates to 4: ExtSource2Risk → ExtSource3Risk → ExtSource1Risk →
AmtGoodsPrice. Fewer parents reduces CPT sparsity (2,401 combinations vs
86,436) and prevents the Dirichlet prior from collapsing sparse cells to a
flat 50/50 prediction.

**Auto model** — structure is learned from data using pgmpy's hill-climb
search with `max_indegree=3`. The learned graph contains 47 edges. Direct
parents of `LoanOutcome` are ExtSource2Risk, ExtSource3Risk, and
OccupationType.

Notable raw default rates that drive both models:

| Feature | Category | Default Rate |
|---------|----------|-------------|
| ExtSource3Risk | VeryHigh | 20.8% |
| CreditUtilization | MaxedOut | 19.5% |
| ExtSource2Risk | VeryHigh | 18.9% |
| DaysOverdue | Low | 18.7% |
| ExtSource1Risk | VeryHigh | 18.5% |

## Naive Bayes

A `CategoricalNB` baseline (scikit-learn) using all 20 `MODEL_COLS` features.
Unlike the BN, NB has no CPT sparsity constraint so the full feature set is
used directly. Missing values are handled by the `Unknown` sentinel categories
defined in `CATEGORICAL_FILL_VALUES` — notably `PaymentHistory` uses `Unknown`
rather than `Low` to avoid penalising applicants who simply have no instalment
history.

Top 10 features by log-probability spread (most discriminative):
ExtSource1/2/3Risk, AmtCredit, AmtGoodsPrice, PaymentHistory, PrevRejected,
CreditUtilization, DaysOverdue, CreditProlonged.

The Laplace smoothing parameter (alpha) is selected by stratified 5-fold
cross-validation. With 450k+ resampled training rows the model is
data-saturated — all alpha values from 0.01 to 5.0 produce identical
BalAcc=0.6696, so smoothing has no practical effect at this scale.

## Project Structure

```
home-loan-default/
├── data/                             # Raw Home Credit CSVs (not tracked)
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── installments_payments.csv
│   ├── credit_card_balance.csv
│   ├── POS_CASH_balance.csv
│   └── previous_application.csv
├── logs/                             # Run logs and output plots (not tracked)
├── src/
│   ├── data_prep.py                  # Data loading, merging, and discretisation
│   ├── graph_structure_discovery.py  # NMI-based BN parent selection
│   └── home_credit_naive_bayes.py    # Categorical Naive Bayes baseline
└── home_credit_bayesian.py           # Bayesian Network pipeline (Expert + Auto)
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
pip install pandas numpy scikit-learn pgmpy matplotlib seaborn imbalanced-learn scipy statsmodels
```

## Usage

Run from the repo root with the `data/` directory populated:

```bash
# Bayesian Network (Expert + Auto DAG)
python home_credit_bayesian.py

# Naive Bayes baseline
python src/home_credit_naive_bayes.py
```

Logs and plots are written to `logs/`.

## Data Preparation

`src/data_prep.py` merges all six source tables into a single wide DataFrame
and discretises continuous features into ordinal categorical bins.  Missing
values are filled with conservative sentinel categories (e.g. `Unknown` for
missing external credit scores, `Never` for no prior loan prolongations) so
all models receive a valid state for every row.

## Feature Set (`MODEL_COLS`)

| Feature | Source | Categories |
|---------|--------|------------|
| ExtSource1/2/3Risk | application | Unknown · VeryHigh → VeryLow |
| IncomeBracket | application | Low · Medium · High |
| IncomeType | application | Stable · Unstable |
| OccupationType | application | Unknown · Laborer · Professional |
| AmtCredit / AmtGoodsPrice | application | 6 buckets |
| AmtReqCreditBureauMon | application | Unknown · Medium · VeryLow |
| PaymentHistory | installments | Unknown · Low · Medium · High |
| PriorLoanApproved / PrevRejected | previous apps | binary / 4-level |
| CreditUtilization | credit card | Low · Medium · High · MaxedOut |
| DPD / ContractStatus | POS cash | 4-level / binary |
| DaysOverdue / MaxOverdue / DebtLoad | bureau | 4-level each |
| CreditProlonged / ActiveCredits | bureau | 4-level each |
