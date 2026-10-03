# Notebooks

| Notebook | Purpose | Status | Decisions informed | Runtime (2026-10-03) |
|---|---|---|---|---|
| [00_data_audit.ipynb](00_data_audit.ipynb) | Data audit of FD001–FD004: structure, test-label distribution, validation realism, regimes, sensor informativeness, RUL-cap change-points, noise vs trend, the removed 258-feature family, engine subpopulations, regime-dependent sensor direction. Writes `reports/data_audit.md`. | current | D01, D02a, D02b, D03, D04a, D04b, D05, D06, D07, D08, D09, D10, D11, D13, D17, D18, D35, D36; X04, X06 | pending |
| [archive/01_understanding_the_data.ipynb](archive/01_understanding_the_data.ipynb) | First look at FD001/FD002: target construction, sensor variance, operating regimes, the 42-feature set. | archived (v0.1-pre-audit) | D01, D02a, D03, D04a, D07, D09 | 726 s (CPU, 1 thread) |
| [archive/02_modeling.ipynb](archive/02_modeling.ipynb) | XGBoost baseline with Optuna tuning on validation instances, scored on the NASA test set; EWM ablation; failure analysis. | archived (v0.1-pre-audit) | D07, D17, D23, D31 | 413 s (XGBoost on GPU) |
| [archive/03_model_comparison.ipynb](archive/03_model_comparison.ipynb) | Five-model screen and 5-seed LSTM-vs-XGBoost re-run, both scored on the NASA test set; the selection behind shipping the LSTM. | archived (v0.1-pre-audit) | D20, D24, D25 | 2857 s (XGBoost on GPU, LSTM on CPU) |

Archived notebooks reflect the pipeline at tag `v0.1-pre-audit`. They used the NASA test set
for model selection (D25) and validation instances later shown unrepresentative (D17), so
their numbers are not valid evidence for any decision. The HTML exports of the original
executions are in git history ([`reports/legacy/`](../reports/legacy/) says where).

Re-executed 2026-10-03 with their original device selection (D43). Compared with the
originally logged numbers: 02 (every metric in its decision log and test table) and 03 (the
single-seed screen, the cross-dataset means, the 5-seed table and the pooled result in
`docs/results.md`) reproduce them exactly; 01's cited numbers (lifetimes, sensor stds,
silhouette scan, RF baseline) do too.
