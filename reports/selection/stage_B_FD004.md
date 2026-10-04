### Stage B — FD004

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD004/lstm/cap090_w030_L030` | 5.74 [5.37, 6.15] | 249 | 15 | 5.33 [4.87, 5.79] (n=148) | 6.29 [5.70, 6.89] (n=101) | 12.9 |
| `runs/FD004/lstm/cap090_w030_L020` | 6.81 [6.29, 7.33] | 249 | 15 | 6.44 [5.79, 7.02] (n=148) | 7.32 [6.52, 8.18] (n=101) | 13.5 |
| `runs/FD004/lstm/cap090_w030_L045` | 5.63 [5.29, 5.99] | 249 | 15 | 5.10 [4.68, 5.52] (n=148) | 6.33 [5.83, 6.87] (n=101) | 12.0 |
| `runs/FD004/lstm/cap090_w030_L060` | 5.53 [5.13, 5.92] | 249 | 15 | 4.49 [4.11, 4.89] (n=148) | 6.77 [6.18, 7.37] (n=101) | 12.5 |
| `runs/FD004/lstm/cap090_w030_L030_regime_onehot` | 5.62 [5.22, 6.04] | 249 | 15 | 5.06 [4.66, 5.42] (n=148) | 6.36 [5.72, 7.18] (n=101) | 13.0 |
| `runs/FD004/lstm/cap090_w030_L030_hi_consistent` | 5.77 [5.34, 6.22] | 249 | 15 | 5.25 [4.75, 5.78] (n=148) | 6.46 [5.83, 7.14] (n=101) | 13.3 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| B1 | lstm, cap 90, seq_len 20 | lstm, cap 90, seq_len 30 | 1.07 [0.67, 1.49] | 39% of 249 | 0.0054 (15) | worse |
| B1 | lstm, cap 90, seq_len 45 | lstm, cap 90, seq_len 30 | -0.11 [-0.33, 0.13] | 53% of 249 | 0.45 (15) | not different |
| B1 | lstm, cap 90, seq_len 60 | lstm, cap 90, seq_len 30 | -0.21 [-0.50, 0.08] | 61% of 249 | 0.56 (15) | not different |
| B2 | lstm, cap 90, seq_len 30, regime_onehot | lstm, cap 90, seq_len 30 | -0.12 [-0.43, 0.26] | 55% of 249 | 0.93 (15) | not different |
| B2 | lstm, cap 90, seq_len 30, hi_consistent | lstm, cap 90, seq_len 30 | 0.03 [-0.19, 0.25] | 52% of 249 | 0.98 (15) | not different |

Winners: B1: lstm, cap 90, seq_len 30; B2: lstm, cap 90, seq_len 30; stage_B: lstm, cap 90, seq_len 30
