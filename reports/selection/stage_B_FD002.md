### Stage B — FD002

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD002/lstm/cap090_w045_L030` | 4.59 [4.33, 4.85] | 260 | 15 | 4.60 [4.30, 4.93] (n=183) | 4.55 [4.11, 5.07] (n=77) | 10.1 |
| `runs/FD002/lstm/cap090_w045_L020` | 5.90 [5.54, 6.27] | 260 | 15 | 5.79 [5.38, 6.28] (n=183) | 6.17 [5.54, 6.83] (n=77) | 10.3 |
| `runs/FD002/lstm/cap090_w045_L045` | 3.96 [3.73, 4.22] | 260 | 15 | 3.83 [3.57, 4.08] (n=183) | 4.26 [3.74, 4.78] (n=77) | 9.6 |
| `runs/FD002/lstm/cap090_w045_L060` | 3.73 [3.46, 4.01] | 260 | 15 | 3.50 [3.27, 3.74] (n=183) | 4.22 [3.65, 4.84] (n=77) | 10.7 |
| `runs/FD002/lstm/cap090_w045_L060_regime_onehot` | 3.75 [3.53, 3.97] | 260 | 15 | 3.68 [3.43, 3.92] (n=183) | 3.91 [3.43, 4.43] (n=77) | 10.4 |
| `runs/FD002/lstm/cap090_w045_L060_hi_consistent` | 4.30 [4.00, 4.62] | 260 | 15 | 4.01 [3.72, 4.32] (n=183) | 4.91 [4.27, 5.52] (n=77) | 10.0 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| B1 | lstm, cap 90, seq_len 20 | lstm, cap 90, seq_len 30 | 1.32 [1.00, 1.65] | 36% of 260 | 6.1e-05 (15) | worse |
| B1 | lstm, cap 90, seq_len 45 | lstm, cap 90, seq_len 30 | -0.63 [-0.86, -0.39] | 64% of 260 | 0.002 (15) | **better** |
| B1 | lstm, cap 90, seq_len 60 | lstm, cap 90, seq_len 30 | -0.86 [-1.15, -0.58] | 69% of 260 | 0.0034 (15) | **better** |
| B2 | lstm, cap 90, seq_len 60, regime_onehot | lstm, cap 90, seq_len 60 | 0.02 [-0.14, 0.19] | 46% of 260 | 0.72 (15) | not different |
| B2 | lstm, cap 90, seq_len 60, hi_consistent | lstm, cap 90, seq_len 60 | 0.57 [0.37, 0.78] | 39% of 260 | 0.0084 (15) | worse |

Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| B1 | lstm, cap 90, seq_len 60 vs lstm, cap 90, seq_len 30 | 20 | 50.7 | 45.8 | 4.9 [1.9, 8.0] | 260 |
| B1 | lstm, cap 90, seq_len 60 vs lstm, cap 90, seq_len 30 | 30 | 97.8 | 95.2 | 2.6 [0.5, 4.6] | 260 |
| B1 | lstm, cap 90, seq_len 60 vs lstm, cap 90, seq_len 30 | 40 | 99.9 | 100.0 | -0.1 [-0.3, 0.1] | 260 |

Winners: B1: lstm, cap 90, seq_len 60; B2: lstm, cap 90, seq_len 60; stage_B: lstm, cap 90, seq_len 60
- B1: seq_len 60 is the grid's upper edge (flagged; no extension)
