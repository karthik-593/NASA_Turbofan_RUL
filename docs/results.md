# Results

All numbers below are reproduced directly from `notebooks/03_model_comparison.ipynb`
running against the real C-MAPSS data, via `turbofan.evaluation.comparison`. Headline
metric is critical-zone [0–25] RMSE (lower is better); NASA score is the tiebreaker.

## Single-seed screen (seed 42): every candidate × every dataset

| Dataset | Model | crit RMSE | NASA score | global RMSE | late % |
|---|---|---:|---:|---:|---:|
| FD001 | mean | 71.63 | 30932 | 42.85 | 50.0 |
| FD001 | ridge | 20.36 | 843 | 19.65 | 60.0 |
| FD001 | rf | 11.25 | 798 | 18.20 | 64.0 |
| FD001 | xgboost | 7.91 | 613 | 17.47 | 59.0 |
| FD001 | **lstm** | **3.13** | 897 | 17.56 | 62.0 |
| FD002 | mean | 74.71 | 153949 | 54.09 | 55.2 |
| FD002 | ridge | 18.51 | 18959 | 32.05 | 47.9 |
| FD002 | rf | 6.50 | 14215 | 28.06 | 48.3 |
| FD002 | **xgboost** | **5.64** | 14282 | 28.00 | 43.2 |
| FD002 | lstm | 6.47 | 6503 | 26.16 | 51.0 |
| FD003 | mean | 80.37 | 57977 | 45.07 | 65.0 |
| FD003 | ridge | 19.02 | 1564 | 21.54 | 61.0 |
| FD003 | rf | 6.69 | 1257 | 17.25 | 57.0 |
| FD003 | xgboost | 5.27 | 758 | 16.66 | 49.0 |
| FD003 | **lstm** | **4.27** | 1378 | 15.98 | 58.0 |
| FD004 | mean | 79.52 | 190834 | 54.91 | 54.4 |
| FD004 | ridge | 25.76 | 10652 | 33.82 | 50.0 |
| FD004 | rf | 11.04 | 5445 | 28.45 | 51.2 |
| FD004 | xgboost | 9.22 | 5094 | 28.11 | 49.2 |
| FD004 | **lstm** | **6.48** | 8032 | 26.80 | 51.2 |

Mean baseline is included as the floor every model must beat, not a serious candidate.

**Mean critical-RMSE / mean NASA across all four datasets (best first):**

| Model | mean crit RMSE | mean NASA | datasets won (crit RMSE) |
|---|---:|---:|---:|
| **lstm** | **5.09** | 4202.46 | 3 |
| xgboost | 7.01 | 5186.85 | 1 |
| rf | 8.87 | 5428.90 | 0 |
| ridge | 20.91 | 8004.47 | 0 |
| mean | 76.56 | 108423.15 | 0 |

At a single seed, LSTM wins 3 of 4 datasets on the headline metric, losing only FD002 to
XGBoost by a small margin (5.64 vs 6.47).

## Does the LSTM's lead survive seed variance? (5-seed re-run)

A single seed can't tell an XGBoost win from a lucky draw. The two contenders were re-run
over 5 model seeds each, with the engine split and features held fixed (only model
init/training varies):

| Dataset | XGBoost crit RMSE (mean ± std) | LSTM crit RMSE (mean ± std) | Separated beyond ±1σ? |
|---|---:|---:|---|
| FD001 | 8.09 ± 0.18 | **3.14 ± 0.70** | Yes — LSTM |
| FD002 | 5.86 ± 0.25 | 5.04 ± 0.84 | No — within noise |
| FD003 | 5.10 ± 0.36 | **3.77 ± 0.68** | Yes — LSTM |
| FD004 | 8.61 ± 0.42 | **6.64 ± 0.67** | Yes — LSTM |
| **Pooled** | **6.92 ± 1.54** | **4.65 ± 1.53** | — |

The FD002 "XGBoost win" from the single-seed screen was a seed-42 artifact — across five
seeds it reverses to a slight LSTM edge (5.04 vs 5.86). XGBoost is meaningfully better on
no dataset.

## Decision: ship the LSTM

**Honest claim:** best among these five candidates, under this protocol — last-cycle
prediction vs NASA ground truth, critical-zone RMSE as the headline metric.

The LSTM won *untuned*, against a *tuned* XGBoost — the 20-trial per-dataset Optuna search
in `notebooks/02_modeling.ipynb` moved XGBoost's mean critical RMSE by roughly 0.04 cycles,
because it optimized global validation RMSE, not the critical zone. Its tuning edge on the
metric that actually matters here is empirically zero; tuning the LSTM would likely widen
the gap further, not close it.

**What this evidence does not cover:** train/val split-composition variance (only the model
seed was varied here — the split itself was held fixed across all runs); a hyperparameter
search for any model other than XGBoost's (mis-targeted) one; more than one architecture
per model family; alternatives to the fixed RUL cap (125) and window length (30). The LSTM
is also noisier run-to-run than XGBoost (std up to 0.84 vs 0.42 across datasets), and on
FD001 it carries a worse NASA score than XGBoost despite winning on critical RMSE — evidence
of fatter error tails outside the critical zone specifically.

The shipped bundle (`models/FD001/lstm/`) records its own measured test-set metrics in
`manifest.json` at training time, so the exact numbers behind any given deployed artifact
are always traceable to that specific run rather than to this notebook.
