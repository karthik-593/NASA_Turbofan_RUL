### Stage D — FD001

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD001/xgboost/cap090_w045_hi_pooled_tuned-8415931e_seeds-42-7-123-2024-99` | 3.27 [2.90, 3.62] | 100 | 75 | 3.28 [2.79, 3.78] (n=60) | 3.25 [2.78, 3.77] (n=40) | 7.1 |
| `runs/FD001/xgboost/cap125_w045_hi_pooled_tuned-8415931e_seeds-42-7-123-2024-99` | 3.62 [3.20, 4.04] | 100 | 75 | 3.57 [3.07, 4.12] (n=60) | 3.70 [3.04, 4.50] (n=40) | 7.4 |
| `runs/FD001/xgboost/cap090_w020_hi_pooled_tuned-8415931e_seeds-42-7-123-2024-99` | 4.35 [3.99, 4.73] | 100 | 75 | 4.03 [3.58, 4.41] (n=60) | 4.78 [4.21, 5.33] (n=40) | 7.3 |
| `runs/FD001/xgboost/cap090_w045_tuned-8415931e_seeds-42-7-123-2024-99` | 3.78 [3.32, 4.26] | 100 | 75 | 3.94 [3.27, 4.65] (n=60) | 3.52 [3.07, 3.99] (n=40) | 8.7 |
| `runs/FD001/xgboost/cap090_w045_hi_pooled_seeds-42-7-123-2024-99` | 3.36 [3.02, 3.69] | 100 | 75 | 3.33 [2.89, 3.77] (n=60) | 3.41 [2.95, 3.90] (n=40) | 8.0 |
| `runs/FD001/lstm/cap090_w045_L060_tuned-14fc99d1_seeds-42-7-123-2024-99` | 3.25 [2.97, 3.57] | 100 | 75 | 3.01 [2.65, 3.38] (n=60) | 3.58 [3.16, 4.01] (n=40) | 27.2 |
| `runs/FD001/lstm/cap125_w045_L060_tuned-14fc99d1_seeds-42-7-123-2024-99` | 3.75 [3.46, 4.05] | 100 | 75 | 3.49 [3.11, 3.88] (n=60) | 4.11 [3.72, 4.49] (n=40) | 28.8 |
| `runs/FD001/lstm/cap090_w045_L030_tuned-14fc99d1_seeds-42-7-123-2024-99` | 3.75 [3.44, 4.08] | 100 | 75 | 3.42 [3.12, 3.72] (n=60) | 4.19 [3.66, 4.81] (n=40) | 29.5 |
| `runs/FD001/lstm/cap090_w045_L060_seeds-42-7-123-2024-99` | 3.63 [3.32, 3.95] | 100 | 75 | 3.44 [2.99, 3.88] (n=60) | 3.89 [3.48, 4.29] (n=40) | 21.7 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | -0.36 [-0.58, -0.19] | 80% of 100 | 7.4e-13 (75) | **better** |
| D-cap-xgboost (urgent_rmse) | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | -0.55 [-0.83, -0.28] | 64% of 100 | 4.8e-12 (75) | **better** |
| D-cap-xgboost (critical_late_pct) | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | -4.95 [-5.88, -4.11] | 90% of 100 | 6.5e-12 (75) | **better** |
| D-cap-xgboost (urgent_late_pct) | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | -2.11 [-3.15, -1.22] | 63% of 100 | 2.1e-08 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 90, window 20, hi_pooled, tuned, 5 seeds | -1.08 [-1.44, -0.73] | 74% of 100 | 5.3e-14 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 90, window 45, tuned, 5 seeds | -0.51 [-0.79, -0.27] | 74% of 100 | 7e-14 (75) | **better** |
| D-LOO-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | xgboost, cap 90, window 45, hi_pooled, 5 seeds | -0.09 [-0.17, -0.01] | 76% of 100 | 6.2e-06 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 60, tuned, 5 seeds | lstm, cap 125, seq_len 60, tuned, 5 seeds | -0.50 [-0.67, -0.35] | 78% of 100 | 0.00013 (75) | **better** |
| D-cap-lstm (urgent_rmse) | lstm, cap 90, seq_len 60, tuned, 5 seeds | lstm, cap 125, seq_len 60, tuned, 5 seeds | -0.51 [-0.88, -0.19] | 63% of 100 | 0.00074 (75) | **better** |
| D-cap-lstm (critical_late_pct) | lstm, cap 90, seq_len 60, tuned, 5 seeds | lstm, cap 125, seq_len 60, tuned, 5 seeds | -4.51 [-6.76, -2.43] | 65% of 100 | 0.031 (75) | **better** |
| D-cap-lstm (urgent_late_pct) | lstm, cap 90, seq_len 60, tuned, 5 seeds | lstm, cap 125, seq_len 60, tuned, 5 seeds | -4.86 [-6.81, -2.93] | 58% of 100 | 0.0015 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 60, tuned, 5 seeds | lstm, cap 90, seq_len 30, tuned, 5 seeds | -0.50 [-0.76, -0.25] | 71% of 100 | 2.4e-07 (75) | **better** |
| D-LOO-lstm | lstm, cap 90, seq_len 60, tuned, 5 seeds | lstm, cap 90, seq_len 60, 5 seeds | -0.38 [-0.51, -0.25] | 78% of 100 | 7.6e-05 (75) | **better** |
| D-final | lstm, cap 90, seq_len 60, tuned, 5 seeds | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | -0.02 [-0.34, 0.28] | 38% of 100 | 0.34 (75) | not different |

Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| D-cap-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds vs xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | 20 | 49.0 | 49.8 | -0.8 [-2.7, 1.2] | 100 |
| D-cap-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds vs xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | 30 | 96.8 | 96.3 | 0.5 [-0.2, 1.5] | 100 |
| D-cap-xgboost | xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds vs xgboost, cap 125, window 45, hi_pooled, tuned, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| D-cap-lstm | lstm, cap 90, seq_len 60, tuned, 5 seeds vs lstm, cap 125, seq_len 60, tuned, 5 seeds | 20 | 48.8 | 48.1 | 0.7 [-1.2, 3.0] | 100 |
| D-cap-lstm | lstm, cap 90, seq_len 60, tuned, 5 seeds vs lstm, cap 125, seq_len 60, tuned, 5 seeds | 30 | 98.6 | 98.0 | 0.5 [-0.3, 1.1] | 100 |
| D-cap-lstm | lstm, cap 90, seq_len 60, tuned, 5 seeds vs lstm, cap 125, seq_len 60, tuned, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| D-final | lstm, cap 90, seq_len 60, tuned, 5 seeds vs xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | 20 | 48.8 | 49.0 | -0.2 [-4.1, 3.8] | 100 |
| D-final | lstm, cap 90, seq_len 60, tuned, 5 seeds vs xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | 30 | 98.6 | 96.8 | 1.8 [-0.7, 4.5] | 100 |
| D-final | lstm, cap 90, seq_len 60, tuned, 5 seeds vs xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |

Winners: stage_D: xgboost, cap 90, window 45, hi_pooled, tuned, 5 seeds
- tie on critical RMSE: XGBoost (simpler) wins; decision check agrees -> locked
