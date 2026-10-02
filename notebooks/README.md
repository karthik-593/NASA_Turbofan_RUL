# Notebooks

| Notebook | Purpose | Status | Decisions informed | Runtime (CPU, 1 thread) |
|---|---|---|---|---|
| [00_data_audit.ipynb](00_data_audit.ipynb) | Data audit of FD001–FD004: structure, test-label distribution, validation realism, regimes, sensor informativeness, RUL-cap change-points, noise vs trend, the removed 258-feature family, engine subpopulations, regime-dependent sensor direction. Writes `reports/data_audit.md`. | current | D01, D02a, D02b, D03, D04a, D04b, D05, D06, D07, D08, D09, D10, D11, D13, D17, D18, D35, D36; X04, X06 | pending |
| [archive/01_understanding_the_data.ipynb](archive/01_understanding_the_data.ipynb) | First look at FD001/FD002: target construction, sensor variance, operating regimes, the 42-feature set. | archived (v0.1-pre-audit) | D01, D02a, D03, D04a, D07, D09 | pending |
| [archive/02_modeling.ipynb](archive/02_modeling.ipynb) | XGBoost baseline with Optuna tuning on validation instances, scored on the NASA test set; EWM ablation; failure analysis. | archived (v0.1-pre-audit) | D07, D17, D23, D31 | pending |
| [archive/03_model_comparison.ipynb](archive/03_model_comparison.ipynb) | Five-model screen and 5-seed LSTM-vs-XGBoost re-run, both scored on the NASA test set; the selection behind shipping the LSTM. | archived (v0.1-pre-audit) | D20, D24, D25 | pending |

Archived notebooks reflect the pipeline at tag `v0.1-pre-audit`. They used the NASA test set
for model selection (D25) and validation instances later shown unrepresentative (D17), so
their numbers are not valid evidence for any decision. The original executions are preserved
as HTML in [`reports/legacy/`](../reports/legacy/).
