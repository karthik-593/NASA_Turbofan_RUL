# Decision Register

Every modeling, evaluation and serving decision in the repo, with the in-repo computed evidence
behind it. Seeded 2026-10-02 from `src/turbofan/config.py`, `src/`, `docs/`, `README.md` and the
notebooks they cite.

**Status**
- `LOCKED` — supported by a computed result in this repo (cited).
- `ASSUMPTION` — adopted without evidence; the row states how it will be tested.
- `REVISIT` — currently in use, not yet supported by admissible in-repo evidence.
- `INVALID` — a result exists but measures the wrong thing or was computed incorrectly; not
  usable as evidence for any decision, regardless of admissibility.

Evidence that came from the NASA test set (`test_FD00x.txt` / `RUL_FD00x.txt`) is cited for
traceability but is **not admissible** for selection, tuning or calibration (rule 3).
No result in the repo yet carries a bootstrap CI (rule 4).

**Project rules referenced below**
1. Every decision cites an in-repo computed result, or is an `ASSUMPTION` with a test plan.
2. Analysis results go to `reports/` and are reviewed before pipeline code or params change.
3. The test set is read only by `final_eval`, once per locked candidate, logged to MLflow
   (`run_type=final_test`); never used for selection, tuning, thresholds or calibration.
4. Every reported number carries its n and a bootstrap CI resampling engines.
5. All parameters live in `params.yaml`; no magic numbers in `src/`.
6. Every experiment is logged to MLflow with git SHA + DVC data hash.
7. One logical change per commit; no notebook outputs committed.
8. No silent fallbacks — unexpected states raise.
9. Results that contradict an earlier decision are flagged here, not worked around.

## Data and target

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D01 | RUL label capped at 125 (`config.RUL_CAP`) | nb01 cell-6: FD001 train lifetimes min 128 / mean 206 / max 362 cycles (n=100 engines) — shows raw RUL extends far into the healthy plateau, but does not select 125. nb01 cell-7 adopts 125 as "the conventional choice" and promises a check of when sensors start moving; that check was never run. No other cap was evaluated. | REVISIT |
| D02a | Drop 7 near-constant sensors (s1, s5, s6, s10, s16, s18, s19) — FD001 | nb01 cell-12, FD001 train (20,631 rows): six sensors std 0.0000, s6 0.0014; next-lowest kept sensor s15 0.0375. Cutoff 1e-2 sits in that gap. | LOCKED |
| D02b | Same 14-sensor `KEEP` list applied to FD002, FD003, FD004 | Not computed for FD002–FD004. On FD002/FD004 raw std is dominated by regime shifts, so the check must be per regime. | REVISIT |
| D03 | Per-regime z-score for FD002/FD004; single global z-score for FD001/FD003 (`features/engineering.py`) | nb01 cell-19: qualitative plot of s2 for one FD002 engine before/after. No numeric ablation (per-regime vs global) on any metric; FD004 not inspected. README calls this the decision that "mattered more than any amount of model tuning" — unmeasured. | REVISIT |
| D04a | 6 operating regimes, KMeans on standardized op1–op3 — FD002 | nb01 cell 5e0069e4: silhouette for k=2..10 on FD002 train; peak 0.9971 at k=6 (k=5: 0.9302, k=7: 0.9357). | LOCKED |
| D04b | k=6 also used for FD004 | Not computed for FD004. | REVISIT |
| D05 | KMeans `n_init=10`, `random_state=0`; StandardScaler on op settings before clustering | No sensitivity check. Scaling rationale (setting ranges differ) is a code comment, not a computed result. | REVISIT |
| D06 | Normalization fallbacks: std floor 1e-9, unseen-regime `mu.fillna(0)` / `sig.fillna(1)`, regime std `fillna(1)`, final `d.fillna(0)` (`engineering.py:46-74`) | None. These are silent fallbacks (conflicts with rule 8). | REVISIT |

## Features

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D07 | 3 features per kept sensor: normalized value, rolling mean, rolling slope → 42 (`config.FEAT_COLS`); richer families (rolling min/max/std, multiple windows, EWM, interactions, PCA health index) excluded | nb01 cell-25: RF on these features, FD001, 20 val engines, all-rows RMSE 16.9 vs mean baseline 41.5 — shows the set carries signal, not that it beats alternatives; no CI. nb02 cell-15 EWM ablation (crit RMSE 9.62 vs 8.96) was measured on FD001 **test** (critical n=19), no CI — inadmissible. `reports/artifact_inventory.md` (2026-10-02): a removed 258-feature pipeline scored test crit RMSE 3.68/4.12/4.22/7.46 (FD001–FD004) — better than this 42-feature set's 8.96/5.68/4.59/8.64 on every dataset. Single run, test-set, no CI — not usable for selection, but the drop to 42 features has never been shown harmless. | REVISIT |
| D08 | Rolling window = 20 cycles (`config.WINDOW`) | None; no other window evaluated. | REVISIT |
| D09 | Excluded: rolling min/max/std, FFT, lags, sensor interactions, PCA health index | Argued a priori in nb01 cell-21 / `problem_framing.md`; nothing computed. | REVISIT |
| D10 | LSTM input = 14 normalized channels over a 30-cycle sequence (`config.SEQ_LEN`) | None; no other length evaluated. | REVISIT |
| D11 | Histories shorter than `SEQ_LEN` left-padded with zeros | `tests/test_lstm_model.py` verifies the behaviour, not its effect on accuracy. | REVISIT |

## Evaluation protocol

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D12 | Train/val split by engine, never by row | `tests/test_comparison.py::test_no_overlap` and `test_union_covers_all_engines` pass — split is disjoint and complete. | LOCKED |
| D13 | Single fixed split: first 80% of engine ids → train (`config.SPLIT_FRAC`, `split_engines`) | None. Split-composition variance never measured (acknowledged in `docs/results.md`). | REVISIT |
| D14 | Normalization state fit on train engines only, reused unchanged on val/test/serving | `tests/test_features.py::test_stats_reuse_on_held_out_frame`; `tests/test_serving_parity.py` asserts bit-identical train vs serve feature windows. | LOCKED |
| D15 | Rolling features are causal (right-aligned, per engine) | `tests/test_features.py::test_no_future_leakage`. | LOCKED |
| D16 | Final evaluation = one prediction per test engine at its last observed cycle vs `RUL_FD00x.txt` | Dataset structure: one RUL value per test unit (`data/loader.py`; `tests/test_loader.py::test_test_rul_length`). Valid as the `final_eval` protocol only — currently also used for selection (see D25). | LOCKED |
| D17 | Validation for tuning = truncation instances every 5th cycle of each val engine's degradation phase (RUL < cap) (nb02 `make_val_instances`, not in `src/`) | nb02 cell-07: 500 instances from 20 FD001 val engines. Step 5 and the RUL < cap filter not evaluated. | REVISIT |

## Metrics and decision framing

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D18 | Headline metric = critical-zone RMSE (true RUL in [0, 25)); NASA score tiebreaker; global RMSE and late % reported only | Domain rationale (README). No computed support. Critical-zone n is small (FD001 test n=19, nb02 cell-15) and no CI has been computed, so differences between models may be within noise. | REVISIT |
| D19 | Maintenance buckets critical 0–25 / urgent 25–50 / monitor 50–100 / healthy 100+ (`config.MAINTENANCE_BUCKETS`) | None. | REVISIT |

## Models and selection

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D20 | Candidate set: mean, ridge (alpha 1.0), random forest (300 trees), XGBoost, LSTM (`comparison.build_registry`) | None for the hyperparameters; set chosen a priori. | REVISIT |
| D21 | XGBoost wrapper defaults: 500 trees, depth 6, lr 0.05, subsample 0.8, colsample 0.8, min_child_weight 3, alpha 0.1, lambda 1.0, early stop 50 (`models/xgboost_model.py:28-43`) | None. | REVISIT |
| D22 | LSTM: hidden 32, 2 layers, dropout 0.3, head 16, AdamW lr 1e-3, batch 256, max 100 epochs, patience 10, ReduceLROnPlateau(patience 5, factor 0.5); early stop on val MSE over all windows | None; no search run (`docs/results.md`). | REVISIT |
| D23 | XGBoost tuning: 20 TPE trials per dataset, objective = val-instance global RMSE (nb02) | nb02 cell-07: FD001 best val RMSE 16.89 (n=500 instances). Objective differs from headline metric (D18). Resulting models were scored on test (nb02 cell-14). | REVISIT |
| D24 | Robustness check: 5 model seeds (42, 7, 123, 2024, 99); "separated beyond ±1σ" as the criterion | Seed variance only, split held fixed, no engine-resampling CI; ±1σ is not a significance criterion. | REVISIT |
| D25 | Ship LSTM to production | nb03 / `docs/results.md`: 5-seed pooled crit RMSE 4.65 ± 1.53 (LSTM) vs 6.92 ± 1.54 (XGBoost). Computed on the **test set** — selection on test violates rule 3; no n or bootstrap CI reported. Must be re-derived on validation. | REVISIT |
| D25a | `evaluation.comparison`'s MLflow runs (the mechanism behind D25's numbers) are tagged `run_type=legacy_test_selection`, not `cv` | `src/turbofan/tracking.py`: `legacy_test_selection` is documented as "pre-protocol-v2 runs that used the test set for selection; not valid evidence"; `cv` is reserved for a protocol-v2 screen that never reads the test set. Makes D25's rule-3 violation machine-queryable in MLflow, not just noted here. | LOCKED |
| D25b | The training pipeline itself is reproducible: retraining FD001 LSTM (seed 42, CPU, same environment) on 2026-10-02 reproduced the shipped 2026-06-03 bundle's (`models/FD001/lstm/20260603T183633Z`) test metrics bit-for-bit — critical_rmse 3.1284308984771645, nasa 896.7927007958756, global_rmse 17.55921270634228, late_pct 62.0, all 4 bucket RMSEs identical to full float precision. This locks the *pipeline's* determinism given a fixed seed/environment; it says nothing about the *metric itself*, which is still the test-contaminated D25 number and remains REVISIT under rule 3. | LOCKED |
| D30 | `models/cv_results_FD001.joblib` — 5-fold CV, reported as `critical_rmse` 2.94 ± 0.61 | `reports/artifact_inventory.md`: every fold's `n_engines`=20 and `late_pct` 95–100%, and `critical_rmse` equals the fold's overall `rmse`. Consistent with scoring each run-to-failure validation engine only at its last cycle, where true RUL is always 0 — i.e. error-at-failure, not RUL estimation across the critical zone. No producing code survives to confirm the exact protocol. | INVALID |
| D31 | Asymmetric training loss for XGBoost (penalize late predictions more than early) | Run once, pre-protocol-v2, against the removed 258-feature pipeline, not the current 42-feature set (`xgboost_FD001_asym.pkl`, `hardened_all_results.joblib`; recovered in `reports/artifact_inventory.md`; notebook claim corrected in `notebooks/02_modeling.ipynb` 2026-10-02). Result: lowers late-prediction rate on all four datasets (e.g. FD001 48.0% vs 60.0% symmetric) at the cost of a worse NASA score on three of four (e.g. FD001 S=428 vs 273); FD003 improved on both. Test-set, single run, no CI — not admissible, and not re-derived on the deployed feature set. | run, pending decision-cost evaluation |

## Serving and packaging

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D26 | `/predict` confidence band = shipped model's test-set RMSE in the predicted bucket (`serving/service.py:49`) | Interval calibrated on the test set (violates rule 3). Also mixes bases: RMSE is bucketed by *true* RUL but looked up by *predicted* bucket. No coverage measured. | REVISIT |
| D27 | Feature state (`stats`) shipped inside the model bundle; serving reuses it, never re-fits | `tests/test_serving_parity.py` (bit-identical); `tests/test_bundle.py::test_load_stats_roundtrip`. | LOCKED |
| D28 | Bundle version = UTC timestamp; `latest` = lexical max | `tests/test_bundle.py::test_load_picks_latest`. | LOCKED |
| D29 | Docker installs CPU-only torch ("~700MB smaller", `docs/architecture.md`) | Size claim not measured in repo. | REVISIT |

## Data infrastructure

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D32 | DVC remote for `data/raw` is a local directory (`../dvc-store`, sibling of the repo), not cloud storage | No evidence needed or applicable — this is an infrastructure choice, not a modeling claim. Adopted because the only consumer today is a single local checkout; untested with a second machine or a CI runner pulling data. **Test plan:** swap to a cloud remote (e.g. S3/GCS) once Jenkins or any CI/remote runner needs `dvc pull` to work outside this machine; `dvc remote add`/`modify` changes only `.dvc/config`, not how data is tracked (`data/raw.dvc` is unaffected). | ASSUMPTION |
| D33 | MLflow backend = local sqlite file, run via `docker-compose.yml` (`./mlflow/mlflow.db`, artifacts served through the tracking server, not a host-local path) | No evidence needed or applicable — infrastructure choice, not a modeling claim. Fine for single-user local work; a shared sqlite file over a bind mount is not safe for concurrent writers from multiple machines. **Test plan:** revisit (likely Postgres/MySQL backend) once more than one person or CI runner writes to the same tracking server at once — sqlite's single-writer lock is the concrete failure mode to watch for. | ASSUMPTION |

## Reproducibility of report computations

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D34 | Every computation behind a `reports/` number (the audit included) runs inside `repro.cpu_deterministic(n_threads=1)` — CPU only, torch deterministic algorithms on, torch and every native BLAS/OpenMP pool pinned to 1 thread, raising `GpuUsedError` if CUDA gets initialized — and the report carries `repro.render_markdown(ctx)` plus a `<report>.env.json` sidecar from `repro.environment_context(seeds=..., n_threads=1)`: git SHA + dirty flag, DVC data hash, OS/CPU model/logical CPUs, Python, numpy/pandas/scikit-learn/scipy/torch (with its `+cpu`/`+cuXXX` build tag)/xgboost/threadpoolctl/mlflow versions, the loaded native thread pools, every `turbofan.config` constant, and an explicit seed for each stochastic step — regime KMeans (`REGIME_KMEANS_SEED`=0, `n_init`=10), bootstrap resampling, and any piecewise/changepoint fit with random initialisation or restarts (a closed-form least-squares fit has no seed and is listed as such). `seeds` has no default (rule 8). A rerun compares its context with the sidecar via `repro.mismatches()` before comparing numbers. XGBoost GPU use is not detectable by the CUDA check, so report code passes `device="cpu"` explicitly, never `config.xgb_device()`. | `tests/test_repro.py`: two independent KMeans fits with the pipeline's regime settings give bit-identical centers and labels under the pinned context; sidecar round-trip shows no mismatches. Environment on 2026-10-02: Intel64 Family 6 Model 183 (28 logical CPUs), Windows 11 (10.0.26200), Python 3.11.15, torch 2.11.0+cpu, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1, OpenBLAS 0.3.30/0.3.31. 1 thread is chosen for exactness, not measured as necessary; bit-identity is guaranteed only on the same CPU family + library builds — other hardware may differ in the last float bits. **Test plan:** the first audit report reruns itself once in a fresh process and must give `mismatches()==[]` and identical numbers; any difference is recorded here. | ASSUMPTION |

## Contradictions found while seeding (rule 9)

| ID | Contradiction | Affects |
|---|---|---|
| X01 | README and `docs/results.md` say the LSTM "won untuned against a tuned XGBoost". The comparison actually used **untuned** XGBoost wrapper defaults (`comparison.py:77` — "not 02's tuned params"). The "tuning moved crit RMSE by ~0.04" figure compares nb02 tuned (mean of 8.96, 5.68, 4.59, 8.64 = 6.97) with nb03 untuned (7.01) — both on the test set, single seed, no CI. | D23, D25 |
| X02 | `docs/results.md` lists "window length (30)" as untested; the feature window is 20 (`WINDOW`) and the LSTM sequence is 30 (`SEQ_LEN`). Both are untested. | D08, D10 |
| X03 | `problem_framing.md` says "the model never sees test-set statistics at any point". True for normalization; false for selection — test labels drove model selection (D25), the EWM ablation (D07) and the serving band (D26). | D07, D25, D26 |
| X04 | Sensor list derived from FD001 only, presented as dataset-wide. | D02b |
| X05 | `RUL_CAP` was defined in `config.py`, `models/xgboost_model.py`, `models/baselines.py` and `data/loader.py`. **Resolved 2026-10-02:** all import from `config`. | D01, rule 5 |
| X06 | D07/D09 say richer feature families are unmotivated; a removed 258-feature pipeline's test-set scores beat the current 42-feature set on every dataset. See D07 for the numbers and `reports/artifact_inventory.md` for the source. | D07, D09, D25 |

## Known rule violations (current state)

| Rule | Violation |
|---|---|
| 2 | **Partially resolved**: `reports/` exists (`artifact_inventory.md`, `legacy/`), but analysis results generally still live in notebook outputs, not written there as a matter of course. |
| 3 | No `final_eval`. Test set is read by `evaluation/comparison.py` (selection), `training/train.py` (every training run scores on test and writes it to the manifest), nb02 (tuning/ablation), and `serving/service.py` (confidence band from test metrics). `tracking.run()` tags every `comparison.py` run `legacy_test_selection` (D25a), never `final_test`, precisely because none of these reads qualify as the sealed one — there is nothing in the repo yet that would earn that tag. |
| 4 | No reported number carries a bootstrap CI; most omit n. |
| 5 | No `params.yaml`; `config.py` constants (incl. D05 KMeans settings) are the interim single source. Literals still in `src/`: D06, D20–D22, D24 seeds. |
| 6 | **Resolved 2026-10-02** for the pieces this rule names: `tracking.run()` logs git SHA + dirty flag and the DVC data hash (D32/D33) on every tracked run in `train.py`/`comparison.py`. Still open: nothing registers a model (D33's registry code path is tested but unused), and `final_test` is not produced by anything yet (see rule 3's row below). |
| 7 | **Resolved 2026-10-02**: outputs stripped from all 3 tracked notebooks (0 outputs, 0 execution counts) and the `nbstripout` pre-commit hook keeps them out. |
| 8 | `features/engineering.py` silent `fillna` fallbacks (D06). |
