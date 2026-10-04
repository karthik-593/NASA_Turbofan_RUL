### Stage B — FD003

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD003/lstm/cap090_w045_L030` | 3.87 [3.48, 4.27] | 100 | 15 | 4.25 [3.68, 4.85] (n=56) | 3.33 [2.87, 3.82] (n=44) | 7.0 |
| `runs/FD003/lstm/cap090_w045_L020` | 4.98 [4.33, 5.61] | 100 | 15 | 5.61 [4.68, 6.68] (n=56) | 4.03 [3.50, 4.58] (n=44) | 7.8 |
| `runs/FD003/lstm/cap090_w045_L045` | 3.78 [3.36, 4.20] | 100 | 15 | 4.19 [3.64, 4.76] (n=56) | 3.18 [2.63, 3.78] (n=44) | 7.9 |
| `runs/FD003/lstm/cap090_w045_L060` | 4.22 [3.77, 4.64] | 100 | 15 | 4.61 [4.02, 5.17] (n=56) | 3.65 [3.07, 4.24] (n=44) | 6.6 |
| `runs/FD003/lstm/cap090_w045_L030_hi_consistent` | 4.07 [3.64, 4.51] | 100 | 15 | 4.42 [3.79, 5.12] (n=56) | 3.57 [3.08, 4.10] (n=44) | 7.9 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| B1 | lstm, cap 90, seq_len 20 | lstm, cap 90, seq_len 30 | 1.11 [0.63, 1.60] | 30% of 100 | 0.00061 (15) | worse |
| B1 | lstm, cap 90, seq_len 45 | lstm, cap 90, seq_len 30 | -0.09 [-0.33, 0.17] | 60% of 100 | 0.33 (15) | not different |
| B1 | lstm, cap 90, seq_len 60 | lstm, cap 90, seq_len 30 | 0.35 [0.02, 0.70] | 46% of 100 | 0.21 (15) | worse |
| B2 | lstm, cap 90, seq_len 30, hi_consistent | lstm, cap 90, seq_len 30 | 0.20 [-0.00, 0.41] | 43% of 100 | 0.23 (15) | not different |

Winners: B1: lstm, cap 90, seq_len 30; B2: lstm, cap 90, seq_len 30; stage_B: lstm, cap 90, seq_len 30
