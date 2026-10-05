### Stage D — FD003

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD003/xgboost/cap090_w045_hi_consistent_seeds-42-7-123-2024-99` | 3.48 [3.06, 4.01] | 100 | 75 | 4.04 [3.40, 4.72] (n=56) | 2.61 [2.33, 2.90] (n=44) | 8.4 |
| `runs/FD003/xgboost/cap125_w045_hi_consistent_seeds-42-7-123-2024-99` | 3.81 [3.42, 4.24] | 100 | 75 | 4.24 [3.75, 4.74] (n=56) | 3.17 [2.64, 3.82] (n=44) | 9.8 |
| `runs/FD003/xgboost/cap090_w020_hi_consistent_seeds-42-7-123-2024-99` | 4.41 [3.94, 4.88] | 100 | 75 | 5.23 [4.63, 5.93] (n=56) | 3.05 [2.79, 3.32] (n=44) | 9.6 |
| `runs/FD003/xgboost/cap090_w045_seeds-42-7-123-2024-99` | 3.82 [3.30, 4.46] | 100 | 75 | 4.44 [3.68, 5.26] (n=56) | 2.84 [2.51, 3.18] (n=44) | 8.9 |
| `runs/FD003/lstm/cap090_w045_L030_tuned-7f680edf_seeds-42-7-123-2024-99` | 3.47 [3.14, 3.83] | 100 | 75 | 3.81 [3.40, 4.27] (n=56) | 2.96 [2.54, 3.46] (n=44) | 27.7 |
| `runs/FD003/lstm/cap125_w045_L030_tuned-7f680edf_seeds-42-7-123-2024-99` | 3.71 [3.40, 4.06] | 100 | 75 | 4.01 [3.59, 4.47] (n=56) | 3.27 [2.86, 3.75] (n=44) | 54.3 |
| `runs/FD003/lstm/cap090_w045_L030_seeds-42-7-123-2024-99` | 3.92 [3.51, 4.32] | 100 | 75 | 4.29 [3.75, 4.88] (n=56) | 3.40 [2.90, 3.94] (n=44) | 29.5 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -0.32 [-0.61, -0.09] | 86% of 100 | 1.5e-09 (75) | **better** |
| D-cap-xgboost (urgent_rmse) | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -0.72 [-0.99, -0.47] | 77% of 100 | 4e-13 (75) | **better** |
| D-cap-xgboost (critical_late_pct) | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -2.75 [-4.04, -1.37] | 65% of 100 | 6.8e-07 (75) | **better** |
| D-cap-xgboost (urgent_late_pct) | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -1.14 [-2.36, 0.34] | 49% of 100 | 0.00067 (75) | not different |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 90, window 20, hi_consistent, 5 seeds | -0.92 [-1.55, -0.30] | 79% of 100 | 2.1e-13 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 90, window 45, 5 seeds | -0.34 [-0.52, -0.18] | 67% of 100 | 3.8e-11 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 30, tuned, 5 seeds | lstm, cap 125, seq_len 30, tuned, 5 seeds | -0.24 [-0.34, -0.14] | 74% of 100 | 0.00017 (75) | **better** |
| D-cap-lstm (urgent_rmse) | lstm, cap 90, seq_len 30, tuned, 5 seeds | lstm, cap 125, seq_len 30, tuned, 5 seeds | -0.56 [-1.14, 0.00] | 58% of 100 | 0.072 (75) | not different |
| D-cap-lstm (critical_late_pct) | lstm, cap 90, seq_len 30, tuned, 5 seeds | lstm, cap 125, seq_len 30, tuned, 5 seeds | 1.94 [0.62, 3.31] | 33% of 100 | 0.11 (75) | worse |
| D-cap-lstm (urgent_late_pct) | lstm, cap 90, seq_len 30, tuned, 5 seeds | lstm, cap 125, seq_len 30, tuned, 5 seeds | 1.24 [0.18, 2.34] | 36% of 100 | 0.36 (75) | worse |
| D-LOO-lstm | lstm, cap 90, seq_len 30, tuned, 5 seeds | lstm, cap 90, seq_len 30, 5 seeds | -0.46 [-0.61, -0.29] | 72% of 100 | 5.9e-08 (75) | **better** |
| D-final | lstm, cap 125, seq_len 30, tuned, 5 seeds | xgboost, cap 90, window 45, hi_consistent, 5 seeds | 0.22 [-0.33, 0.77] | 34% of 100 | 0.057 (75) | not different |

Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| D-cap-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds vs xgboost, cap 125, window 45, hi_consistent, 5 seeds | 20 | 53.2 | 50.4 | 2.7 [0.5, 4.8] | 100 |
| D-cap-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds vs xgboost, cap 125, window 45, hi_consistent, 5 seeds | 30 | 99.5 | 99.3 | 0.2 [-0.9, 1.7] | 100 |
| D-cap-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds vs xgboost, cap 125, window 45, hi_consistent, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| D-cap-lstm | lstm, cap 90, seq_len 30, tuned, 5 seeds vs lstm, cap 125, seq_len 30, tuned, 5 seeds | 20 | 47.9 | 46.7 | 1.1 [-0.7, 3.1] | 100 |
| D-cap-lstm | lstm, cap 90, seq_len 30, tuned, 5 seeds vs lstm, cap 125, seq_len 30, tuned, 5 seeds | 30 | 98.2 | 97.3 | 1.0 [-0.2, 1.6] | 100 |
| D-cap-lstm | lstm, cap 90, seq_len 30, tuned, 5 seeds vs lstm, cap 125, seq_len 30, tuned, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| D-final | lstm, cap 125, seq_len 30, tuned, 5 seeds vs xgboost, cap 90, window 45, hi_consistent, 5 seeds | 20 | 46.7 | 53.2 | -6.4 [-11.5, -1.5] | 100 |
| D-final | lstm, cap 125, seq_len 30, tuned, 5 seeds vs xgboost, cap 90, window 45, hi_consistent, 5 seeds | 30 | 97.3 | 99.5 | -2.2 [-4.2, -0.3] | 100 |
| D-final | lstm, cap 125, seq_len 30, tuned, 5 seeds vs xgboost, cap 90, window 45, hi_consistent, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |

Winners: stage_D: xgboost, cap 90, window 45, hi_consistent, 5 seeds
- lstm: cap 90 beats 125 on critical RMSE but is worse on a cap-check metric at 5 seeds — dropped
- lstm: dropped cap (parsimony)
- tie on critical RMSE: XGBoost (simpler) wins; decision check agrees -> locked
