### Stage D — FD002

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD002/xgboost/cap090_w045_hi_consistent_seeds-42-7-123-2024-99` | 3.94 [3.65, 4.24] | 260 | 75 | 3.81 [3.53, 4.14] (n=183) | 4.23 [3.65, 4.83] (n=77) | 8.0 |
| `runs/FD002/xgboost/cap125_w045_hi_consistent_seeds-42-7-123-2024-99` | 4.35 [4.02, 4.74] | 260 | 75 | 4.15 [3.84, 4.49] (n=183) | 4.80 [4.00, 5.65] (n=77) | 9.0 |
| `runs/FD002/xgboost/cap090_w020_hi_consistent_seeds-42-7-123-2024-99` | 5.53 [5.19, 5.90] | 260 | 75 | 5.56 [5.15, 6.03] (n=183) | 5.47 [4.95, 5.98] (n=77) | 8.7 |
| `runs/FD002/xgboost/cap090_w045_seeds-42-7-123-2024-99` | 4.13 [3.90, 4.39] | 260 | 75 | 4.08 [3.80, 4.39] (n=183) | 4.25 [3.83, 4.69] (n=77) | 8.0 |
| `runs/FD002/lstm/cap090_w045_L060_seeds-42-7-123-2024-99` | 3.94 [3.70, 4.18] | 260 | 75 | 3.80 [3.57, 4.05] (n=183) | 4.24 [3.73, 4.77] (n=77) | 37.0 |
| `runs/FD002/lstm/cap125_w045_L060_seeds-42-7-123-2024-99` | 4.30 [4.06, 4.55] | 260 | 75 | 4.13 [3.90, 4.38] (n=183) | 4.67 [4.13, 5.16] (n=77) | 47.1 |
| `runs/FD002/lstm/cap090_w045_L030_seeds-42-7-123-2024-99` | 4.80 [4.57, 5.06] | 260 | 75 | 4.85 [4.56, 5.16] (n=183) | 4.70 [4.33, 5.13] (n=77) | 38.4 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -0.41 [-0.54, -0.30] | 75% of 260 | 7.3e-14 (75) | **better** |
| D-cap-xgboost (urgent_rmse) | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -0.88 [-1.11, -0.66] | 68% of 260 | 5.3e-14 (75) | **better** |
| D-cap-xgboost (critical_late_pct) | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -4.15 [-5.03, -3.19] | 70% of 260 | 9.4e-11 (75) | **better** |
| D-cap-xgboost (urgent_late_pct) | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 125, window 45, hi_consistent, 5 seeds | -0.35 [-1.04, 0.34] | 44% of 260 | 0.018 (75) | not different |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 90, window 20, hi_consistent, 5 seeds | -1.59 [-1.94, -1.25] | 81% of 260 | 5.3e-14 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds | xgboost, cap 90, window 45, 5 seeds | -0.19 [-0.34, -0.04] | 62% of 260 | 4e-09 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 60, 5 seeds | lstm, cap 125, seq_len 60, 5 seeds | -0.37 [-0.45, -0.28] | 75% of 260 | 4.5e-05 (75) | **better** |
| D-cap-lstm (urgent_rmse) | lstm, cap 90, seq_len 60, 5 seeds | lstm, cap 125, seq_len 60, 5 seeds | -0.22 [-0.42, -0.02] | 63% of 260 | 0.033 (75) | **better** |
| D-cap-lstm (critical_late_pct) | lstm, cap 90, seq_len 60, 5 seeds | lstm, cap 125, seq_len 60, 5 seeds | -4.20 [-5.19, -3.26] | 69% of 260 | 0.035 (75) | **better** |
| D-cap-lstm (urgent_late_pct) | lstm, cap 90, seq_len 60, 5 seeds | lstm, cap 125, seq_len 60, 5 seeds | -1.57 [-2.38, -0.75] | 54% of 260 | 0.13 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 60, 5 seeds | lstm, cap 90, seq_len 30, 5 seeds | -0.87 [-1.14, -0.63] | 74% of 260 | 1.4e-10 (75) | **better** |
| D-final | lstm, cap 90, seq_len 60, 5 seeds | xgboost, cap 90, window 45, hi_consistent, 5 seeds | -0.00 [-0.27, 0.27] | 42% of 260 | 0.36 (75) | not different |

Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| D-cap-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds vs xgboost, cap 125, window 45, hi_consistent, 5 seeds | 20 | 51.3 | 50.0 | 1.3 [-0.1, 3.0] | 260 |
| D-cap-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds vs xgboost, cap 125, window 45, hi_consistent, 5 seeds | 30 | 99.0 | 97.8 | 1.2 [0.5, 2.2] | 260 |
| D-cap-xgboost | xgboost, cap 90, window 45, hi_consistent, 5 seeds vs xgboost, cap 125, window 45, hi_consistent, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [-0.0, 0.0] | 260 |
| D-cap-lstm | lstm, cap 90, seq_len 60, 5 seeds vs lstm, cap 125, seq_len 60, 5 seeds | 20 | 50.0 | 50.4 | -0.4 [-1.4, 0.6] | 260 |
| D-cap-lstm | lstm, cap 90, seq_len 60, 5 seeds vs lstm, cap 125, seq_len 60, 5 seeds | 30 | 97.8 | 96.9 | 0.9 [0.3, 1.4] | 260 |
| D-cap-lstm | lstm, cap 90, seq_len 60, 5 seeds vs lstm, cap 125, seq_len 60, 5 seeds | 40 | 99.9 | 100.0 | -0.0 [-0.1, 0.0] | 260 |
| D-final | lstm, cap 90, seq_len 60, 5 seeds vs xgboost, cap 90, window 45, hi_consistent, 5 seeds | 20 | 50.0 | 51.3 | -1.3 [-3.7, 0.9] | 260 |
| D-final | lstm, cap 90, seq_len 60, 5 seeds vs xgboost, cap 90, window 45, hi_consistent, 5 seeds | 30 | 97.8 | 99.0 | -1.2 [-2.3, -0.0] | 260 |
| D-final | lstm, cap 90, seq_len 60, 5 seeds vs xgboost, cap 90, window 45, hi_consistent, 5 seeds | 40 | 99.9 | 100.0 | -0.1 [-0.2, 0.0] | 260 |

Winners: stage_D: xgboost, cap 90, window 45, hi_consistent, 5 seeds
- tie on critical RMSE: XGBoost (simpler) wins; decision check agrees -> locked
