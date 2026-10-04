### Stage B — FD001

| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 | wall (min) |
|---|---|---|---|---|---|---|
| `runs/FD001/lstm/cap090_w045_L030` | 4.09 [3.77, 4.45] | 100 | 15 | 3.90 [3.45, 4.30] (n=60) | 4.37 [3.90, 4.86] (n=40) | 5.5 |
| `runs/FD001/lstm/cap090_w045_L020` | 5.53 [4.98, 6.06] | 100 | 15 | 4.95 [4.24, 5.57] (n=60) | 6.29 [5.41, 7.19] (n=40) | 6.0 |
| `runs/FD001/lstm/cap090_w045_L045` | 3.62 [3.24, 4.03] | 100 | 15 | 3.45 [2.96, 3.95] (n=60) | 3.86 [3.31, 4.45] (n=40) | 5.2 |
| `runs/FD001/lstm/cap090_w045_L060` | 3.56 [3.22, 3.93] | 100 | 15 | 3.44 [2.96, 3.95] (n=60) | 3.73 [3.34, 4.13] (n=40) | 5.1 |
| `runs/FD001/lstm/cap090_w045_L060_hi_pooled` | 3.60 [3.28, 3.95] | 100 | 15 | 3.32 [2.94, 3.70] (n=60) | 3.99 [3.49, 4.53] (n=40) | 5.4 |

Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = critical RMSE unless stated):

| step | challenger | incumbent | Δ [95% CI] | engines challenger better | Wilcoxon p (n splits) | verdict |
|---|---|---|---|---|---|---|
| B1 | lstm, cap 90, seq_len 20 | lstm, cap 90, seq_len 30 | 1.43 [0.88, 1.95] | 37% of 100 | 0.00018 (15) | worse |
| B1 | lstm, cap 90, seq_len 45 | lstm, cap 90, seq_len 30 | -0.47 [-0.72, -0.22] | 66% of 100 | 0.0084 (15) | **better** |
| B1 | lstm, cap 90, seq_len 60 | lstm, cap 90, seq_len 30 | -0.53 [-0.83, -0.26] | 66% of 100 | 0.018 (15) | **better** |
| B2 | lstm, cap 90, seq_len 60, hi_pooled | lstm, cap 90, seq_len 60 | 0.04 [-0.24, 0.34] | 51% of 100 | 0.93 (15) | not different |

Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = challenger − incumbent, positive = challenger catches more):

| step | challenger vs incumbent | budget (cycles) | caught % challenger | caught % incumbent | Δ [95% CI] | n engines |
|---|---|---|---|---|---|---|
| B1 | lstm, cap 90, seq_len 60 vs lstm, cap 90, seq_len 30 | 20 | 47.9 | 43.4 | 4.5 [0.9, 7.8] | 100 |
| B1 | lstm, cap 90, seq_len 60 vs lstm, cap 90, seq_len 30 | 30 | 97.7 | 97.2 | 0.5 [-1.4, 3.1] | 100 |
| B1 | lstm, cap 90, seq_len 60 vs lstm, cap 90, seq_len 30 | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.1] | 100 |

Winners: B1: lstm, cap 90, seq_len 60; B2: lstm, cap 90, seq_len 60; stage_B: lstm, cap 90, seq_len 60
- B1: seq_len 60 is the grid's upper edge (flagged; no extension)
