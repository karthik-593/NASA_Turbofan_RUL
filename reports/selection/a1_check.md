### A1-check — cap 90 vs cap 125 where neither cap binds (no new runs)

XGBoost, base features, window 20, seed 42, 5 × 3 folds — the Stage A A1 runs, predictions read from their MLflow parent runs. Δ = cap 90 − cap 125; worse = Δ CI entirely on the bad side (above 0 for RMSE / late %, below 0 for caught %).

| dataset | metric (uncapped truth) | cap 90 | cap 125 | Δ [95% CI] | n engines | verdict |
|---|---|---|---|---|---|---|
| FD001 | urgent_rmse | 12.84 | 15.29 | -2.45 [-3.32, -1.61] | 100 | better |
| FD001 | critical_late_pct | 66.77 | 70.15 | -3.37 [-4.77, -2.00] | 100 | better |
| FD001 | urgent_late_pct | 63.20 | 63.91 | -0.71 [-2.01, 0.44] | 100 | not different |
| FD002 | urgent_rmse | 15.33 | 18.68 | -3.35 [-3.95, -2.78] | 260 | better |
| FD002 | critical_late_pct | 66.49 | 68.98 | -2.49 [-3.23, -1.77] | 260 | better |
| FD002 | urgent_late_pct | 69.06 | 69.16 | -0.11 [-0.76, 0.52] | 260 | not different |
| FD003 | urgent_rmse | 11.62 | 13.74 | -2.12 [-3.09, -1.18] | 100 | better |
| FD003 | critical_late_pct | 66.01 | 69.64 | -3.63 [-5.05, -2.21] | 100 | better |
| FD003 | urgent_late_pct | 63.65 | 65.72 | -2.07 [-3.80, -0.52] | 100 | better |
| FD004 | urgent_rmse | 14.10 | 16.85 | -2.75 [-3.37, -2.16] | 249 | better |
| FD004 | critical_late_pct | 73.19 | 75.42 | -2.23 [-2.99, -1.55] | 249 | better |
| FD004 | urgent_late_pct | 69.36 | 69.66 | -0.30 [-0.96, 0.34] | 249 | not different |

Decision check (caught % with lead ≥ 20 at matched mean wasted life):

| dataset | budget (cycles) | cap 90 | cap 125 | Δ [95% CI] | n engines |
|---|---|---|---|---|---|
| FD001 | 20 | 47.1 | 47.0 | 0.1 [-2.9, 3.0] | 100 |
| FD001 | 30 | 95.8 | 95.5 | 0.3 [-1.5, 2.0] | 100 |
| FD001 | 40 | 100.0 | 100.0 | 0.0 [0.0, 0.0] | 100 |
| FD002 | 20 | 48.1 | 46.6 | 1.5 [-0.3, 3.3] | 260 |
| FD002 | 30 | 92.3 | 90.4 | 1.9 [0.6, 3.5] | 260 |
| FD002 | 40 | 99.4 | 99.2 | 0.2 [-0.6, 1.0] | 260 |
| FD003 | 20 | 47.7 | 47.2 | 0.5 [-1.6, 2.9] | 100 |
| FD003 | 30 | 94.7 | 94.7 | -0.0 [-1.4, 1.6] | 100 |
| FD003 | 40 | 100.0 | 99.0 | 0.9 [-0.1, 2.1] | 100 |
| FD004 | 20 | 36.2 | 36.6 | -0.4 [-2.0, 1.5] | 249 |
| FD004 | 30 | 86.7 | 85.0 | 1.7 [0.3, 3.1] | 249 |
| FD004 | 40 | 99.1 | 99.0 | 0.1 [-0.5, 0.6] | 249 |

For information only (structurally favours the lower cap): overall late % vs uncapped truth, deployment view, per candidate.

| dataset | cap 90 | cap 125 | n engines |
|---|---|---|---|
| FD001 | 28.7 [25.7, 32.4] | 40.0 [35.2, 44.5] | 100 |
| FD002 | 29.2 [27.1, 31.4] | 40.6 [37.7, 43.7] | 260 |
| FD003 | 23.9 [20.8, 27.4] | 33.9 [29.7, 38.4] | 100 |
| FD004 | 25.5 [23.6, 27.5] | 35.6 [33.1, 38.4] | 249 |

**Cap 90 confirmed** — not worse on any pre-registered metric on any dataset.

MLflow parent runs: FD001 cap 90 `b8dc2c92541f453cbcffd2904069a69c`; FD001 cap 125 `e0b2c4011b5945c7b0f1b3c287144dfa`; FD002 cap 90 `3b26ba22b28a4189a2ccd145c2eb78f8`; FD002 cap 125 `89171b35cba74600ba8b81e3dd11c5a9`; FD003 cap 90 `10b30207abce42db93139a320566bd02`; FD003 cap 125 `2e16401b1de34d7b95769b7a81ed50e5`; FD004 cap 90 `7c35f127c0034bdfa5d5c31f365772be`; FD004 cap 125 `911d428cc5db45b8a9750579f42eeace`
