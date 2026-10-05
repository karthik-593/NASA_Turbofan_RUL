### Stage D — FD004

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD004/xgboost/cap090_w030_hi_consistent_regime_onehot_tuned-16c6c872_seeds-42-7-123-2024-99` | 5.72 [5.30, 6.20] | 249 | 75 | 4.89 [4.45, 5.35] (n=148) | 6.77 [6.05, 7.51] (n=101) | 40.3 |
| `runs/FD004/xgboost/cap125_w030_hi_consistent_regime_onehot_tuned-16c6c872_seeds-42-7-123-2024-99` | 6.36 [5.87, 6.88] | 249 | 75 | 5.14 [4.69, 5.59] (n=148) | 7.81 [6.94, 8.69] (n=101) | 45.9 |
| `runs/FD004/xgboost/cap090_w020_hi_consistent_regime_onehot_tuned-16c6c872_seeds-42-7-123-2024-99` | 6.15 [5.73, 6.57] | 249 | 75 | 5.83 [5.30, 6.33] (n=148) | 6.58 [5.98, 7.21] (n=101) | 41.4 |
| `runs/FD004/xgboost/cap090_w030_regime_onehot_tuned-16c6c872_seeds-42-7-123-2024-99` | 5.90 [5.45, 6.40] | 249 | 75 | 5.09 [4.65, 5.52] (n=148) | 6.92 [6.13, 7.73] (n=101) | 55.8 |
| `runs/FD004/xgboost/cap090_w030_hi_consistent_tuned-16c6c872_seeds-42-7-123-2024-99` | 5.73 [5.29, 6.20] | 249 | 75 | 4.88 [4.44, 5.34] (n=148) | 6.77 [6.05, 7.53] (n=101) | 46.9 |
| `runs/FD004/xgboost/cap090_w030_hi_consistent_regime_onehot_seeds-42-7-123-2024-99` | 6.03 [5.57, 6.52] | 249 | 75 | 4.96 [4.51, 5.42] (n=148) | 7.31 [6.56, 8.11] (n=101) | 7.1 |
| `runs/FD004/lstm/cap090_w030_L030_seeds-42-7-123-2024-99` | 5.74 [5.37, 6.13] | 249 | 75 | 5.25 [4.80, 5.68] (n=148) | 6.38 [5.86, 6.91] (n=101) | 51.0 |
| `runs/FD004/lstm/cap125_w030_L030_seeds-42-7-123-2024-99` | 6.36 [5.96, 6.78] | 249 | 75 | 5.48 [5.06, 5.88] (n=148) | 7.46 [6.83, 8.12] (n=101) | 59.9 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| D-LOO-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | -0.63 [-0.78, -0.48] | 75% of 249 | 7.6e-14 (75) | **better** |
| D-cap-xgboost (urgent_rmse) | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | -1.78 [-2.24, -1.30] | 72% of 249 | 5.3e-14 (75) | **better** |
| D-cap-xgboost (critical_late_pct) | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | -1.64 [-2.29, -1.00] | 62% of 249 | 4.8e-09 (75) | **better** |
| D-cap-xgboost (urgent_late_pct) | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | -0.43 [-1.18, 0.45] | 45% of 249 | 7.6e-05 (75) | not different |
| D-LOO-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 90, window 20, hi_consistent, regime_onehot, tuned, 5 seeds | -0.42 [-0.70, -0.15] | 63% of 249 | 9.3e-14 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 90, window 30, regime_onehot, tuned, 5 seeds | -0.18 [-0.28, -0.09] | 61% of 249 | 2.8e-10 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 90, window 30, hi_consistent, tuned, 5 seeds | -0.00 [-0.02, 0.01] | 47% of 249 | 0.31 (75) | not different |
| D-LOO-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | xgboost, cap 90, window 30, hi_consistent, regime_onehot, 5 seeds | -0.30 [-0.39, -0.22] | 70% of 249 | 8.6e-12 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 30, 5 seeds | lstm, cap 125, seq_len 30, 5 seeds | -0.62 [-0.76, -0.48] | 75% of 249 | 4.3e-05 (75) | **better** |
| D-cap-lstm (urgent_rmse) | lstm, cap 90, seq_len 30, 5 seeds | lstm, cap 125, seq_len 30, 5 seeds | -1.32 [-1.87, -0.83] | 55% of 249 | 3.9e-10 (75) | **better** |
| D-cap-lstm (critical_late_pct) | lstm, cap 90, seq_len 30, 5 seeds | lstm, cap 125, seq_len 30, 5 seeds | -2.38 [-3.30, -1.53] | 62% of 249 | 0.0037 (75) | **better** |
| D-cap-lstm (urgent_late_pct) | lstm, cap 90, seq_len 30, 5 seeds | lstm, cap 125, seq_len 30, 5 seeds | -0.51 [-1.45, 0.44] | 49% of 249 | 0.16 (75) | not different |
| D-final | lstm, cap 90, seq_len 30, 5 seeds | xgboost, cap 90, window 30, hi_consistent, tuned, 5 seeds | 0.01 [-0.31, 0.31] | 43% of 249 | 0.85 (75) | not different |

Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| D-cap-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds vs xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | 20 | 37.1 | 38.4 | -1.3 [-2.8, 0.0] | 249 |
| D-cap-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds vs xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | 30 | 90.4 | 89.1 | 1.2 [0.3, 2.3] | 249 |
| D-cap-xgboost | xgboost, cap 90, window 30, hi_consistent, regime_onehot, tuned, 5 seeds vs xgboost, cap 125, window 30, hi_consistent, regime_onehot, tuned, 5 seeds | 40 | 99.6 | 99.0 | 0.6 [0.1, 1.3] | 249 |
| D-cap-lstm | lstm, cap 90, seq_len 30, 5 seeds vs lstm, cap 125, seq_len 30, 5 seeds | 20 | 36.1 | 35.5 | 0.7 [-1.1, 1.5] | 249 |
| D-cap-lstm | lstm, cap 90, seq_len 30, 5 seeds vs lstm, cap 125, seq_len 30, 5 seeds | 30 | 88.9 | 88.0 | 0.9 [0.3, 2.5] | 249 |
| D-cap-lstm | lstm, cap 90, seq_len 30, 5 seeds vs lstm, cap 125, seq_len 30, 5 seeds | 40 | 99.2 | 99.1 | 0.1 [-0.2, 0.5] | 249 |
| D-final | lstm, cap 90, seq_len 30, 5 seeds vs xgboost, cap 90, window 30, hi_consistent, tuned, 5 seeds | 20 | 36.1 | 37.2 | -1.1 [-3.9, 3.3] | 249 |
| D-final | lstm, cap 90, seq_len 30, 5 seeds vs xgboost, cap 90, window 30, hi_consistent, tuned, 5 seeds | 30 | 88.9 | 90.1 | -1.3 [-4.7, 0.9] | 249 |
| D-final | lstm, cap 90, seq_len 30, 5 seeds vs xgboost, cap 90, window 30, hi_consistent, tuned, 5 seeds | 40 | 99.2 | 99.6 | -0.3 [-0.7, 0.0] | 249 |

Winners: stage_D: xgboost, cap 90, window 30, hi_consistent, tuned, 5 seeds
- xgboost: dropped regime_onehot (parsimony)
- tie on critical RMSE: XGBoost (simpler) wins; decision check agrees -> locked
