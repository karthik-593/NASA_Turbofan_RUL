# Model selection under protocol v2 — pre-registration and results (D51)

**Pre-registered 2026-10-03 at commit `2b512d4`, before any selection run.** The rules, grids,
feature definitions and thresholds below are fixed. Any later deviation is written into the
stage summary as a deviation, with its reason. Stage summaries are appended below the
pre-registration as each stage completes.

## 0. Pre-registration

### Scope

- Training files only (`load_train`). The NASA test set stays sealed: `final_eval` is not run in
  this work (rule 3). The only test-side input remains the benchmark view's label histogram (D46).
- Every dataset (FD001–FD004) is selected separately. Results are never pooled across datasets.
- Every run goes through the pipeline entry point (`python -m turbofan.pipeline …`), so resume
  (D49) and the GPU-teardown workaround (D50) apply, and every run is logged to MLflow.

### Ranking rule

- **Metric:** deployment-view critical-bucket RMSE against uncapped truth (`critical_rmse`,
  D18/D46). It is computed on CV held-out engines over 5 folds × 3 repeats (`params.yaml` `cv`),
  with a 1,000-replicate engine bootstrap.
- **"Better":** the paired Δ (challenger − incumbent) has a 95% paired engine-bootstrap CI
  entirely below 0 (`evaluation.compare.paired_compare`: same engines, cycles, folds and seeds).
- **Ties:** if the CI includes 0, the **simpler / cheaper option wins**. Within each stage,
  "simpler" is defined as follows:
  - the incumbent (current default) over a change;
  - fewer features over more;
  - a smaller rolling window or sequence length over a larger one;
  - XGBoost over the LSTM (cheaper to train and to serve);
  - untuned over tuned.
- **Caps:** candidates with different `rul_cap` are compared only on cap-invariant metrics.
  `compare.CAP_INVARIANT` refuses anything else (D01).
- **Decision check:** every finalist gets a decision check (D47): caught % with lead ≥ 20 cycles
  at matched mean-wasted-life budgets of 20 / 30 / 40 cycles. It is reported paired, with
  paired engine-bootstrap CIs (`compare.matched_budget_compare`).
  - **"Better on the decision check"** means the paired Δ caught % CI is entirely above 0 at
    ≥ 1 budget, and entirely below 0 at none.
- **Two-candidate rule:** if, for a dataset, the RMSE-better finalist is **not** better on the
  decision check, both of its top-2 finalists go forward as **locked-pending**. The final pick
  then happens in the decision-layer stage, on expected cost with calibrated intervals.
- **Multiplicity:** no correction across the many comparisons of Stages A–C. Some screening
  adoptions may be false positives. Stage D (5 seeds) is the confirmation, and only its
  comparisons decide what is locked.
- **Design:** one factor at a time. Interactions between factors are **not tested** (a
  limitation). The single exception is the pre-registered A3-combined run below.

### Stage A — screening with XGBoost

Setup:

- XGBoost at registry defaults (`build_registry`), seed 42, 5 × 3 folds, every dataset.
- Base features: the 14 `sensors.use` sensors × {within-regime z, rolling mean, rolling slope}.
- **Decision check (informational):** every adopted change is also reported with a decision
  check against its incumbent, and with critical RMSE per subpopulation (D35).

Sub-stages:

- **A1 — cap.** `rul_cap` ∈ {90, 105, 125, 140}; the incumbent is 125.
  - Each cap is compared with 125.
  - Among the caps that beat 125, the one with the lowest point estimate is adopted.
  - If none beats 125, 125 stays.
- **A2 — rolling window.** `window` ∈ {20, 30, 45}, at the A1 cap; the incumbent is 20.
  - Same rule as A1.
- **A3 — feature blocks.** Each block is added alone to the base set, at the A1 / A2 winners,
  and compared with base. The blocks:
  - **`hi_consistent`:** fitted per fold on the inner-training engines only.
    - For each base sensor, compute the per-engine Spearman ρ(within-regime z, uncapped time to
      failure).
    - A sensor is *direction-consistent* if ≥ 90% of engines share the majority sign. This is
      the audit §E rule.
    - Take PC1 of the within-regime z of the consistent sensors over the inner-training rows,
      oriented to rise toward failure.
    - Features: the PC1 score, plus its rolling mean and rolling slope over the run's window.
    - The fitted state is applied unchanged to inner-validation and held-out engines.
  - **`hi_pooled`:** the same construction, but over all base sensors (no consistency filter).
  - **`regime_onehot`** (FD002, FD004 only): one indicator column per fitted operating regime
    (k = 6, `n_regimes`).
  - **`sensors+`:** the sensors audit §E classes as informative but outside `sensors.use`, as
    one block per dataset, 3 features each like the base sensors:
    - FD001: s6
    - FD002: s6, s10, s16
    - FD003: s6, s10
    - FD004: s6, s10, s16
  - **Adoption:** a block is adopted if it beats base.
  - **A3-combined:** if ≥ 2 blocks are adopted, their union is run once as A3-combined. It is
    carried forward if it beats base. Otherwise the adopted block with the lowest point
    estimate is carried.
- **A4 — early identifiability of the subpopulation.** Can an engine's D35 subpopulation (a
  full-trajectory label) be predicted causally from its first N cycles, N ∈ {30, 50, 100}?
  - **Features:** per base sensor, the mean and OLS slope of the within-regime z over cycles
    1..N. Normalization is fitted on the fold's training engines.
  - **Classifier:** standardized logistic regression (L2, C = 1).
  - **Folds:** the same repeated StratifiedGroupKFold (5 × 3, stratified by the label).
  - **Metric:** ROC AUC of each engine's out-of-fold probability, averaged over the 3 repeats,
    with a 1,000-replicate engine-bootstrap CI.
  - **"Strong"** means the AUC CI lower bound is ≥ 0.80. A strong result is *reported* as a
    candidate feature for a later block test. It is not forced into Stage A.

### Stage B — LSTM (registry defaults, seed 42, at the Stage-A cap per dataset)

- **B1:** `seq_len` ∈ {20, 30, 45}; the incumbent is 30.
- **B2** (FD002, FD004): input channels base vs base + `regime_onehot`, at the B1 winner.
- The LSTM keeps the base sensor channels. The A3 blocks are XGBoost features and are not
  carried into LSTM channels.
- Same rule as Stage A.

### Stage C — tuning

- **Configs tuned:** the best XGBoost config and the best LSTM config per dataset after A / B.
- **Search:** Optuna TPE (seed 42), with 30 trials for XGBoost and 20 for the LSTM. Search
  spaces live in `params.yaml` `tuning`.
- **Objective:** deployment critical RMSE on the **repeat-1 folds only** (5 splits).
- **Evaluation:** the tuned config and its untuned incumbent are compared on **repeats 2–3**
  (10 splits) with the paired rule. The tuned config is adopted only if it beats the incumbent.
- **Residual optimism (stated, not removed):** repeats 2–3 reshuffle the *same* training
  engines that tuned the config on repeat 1. Every held-out engine in repeats 2–3 was a tuning
  engine in some repeat-1 fold, so the repeat-2/3 estimate is optimistic for the tuned config.

### Stage D — confirmation

- **Finalists ("top-2"):** per dataset, the best XGBoost config and the best LSTM config after
  Stage C.
- **Runs:** each finalist with all 5 seeds (`params.yaml` `seeds`) × 5 × 3 folds.
- **Reported:**
  - paired Δ critical RMSE with a CI;
  - Wilcoxon over fold × seed;
  - per-subpopulation critical RMSE;
  - the matched-budget decision check.
- **Outcome:** the ranking rule and the two-candidate rule produce either one **locked** spec
  or two **locked-pending** specs per dataset.

### Stage E — write-up

- **Outputs:**
  - `notebooks/04_model_selection.ipynb` (style guide in `notebooks/README.md`);
  - this report;
  - candidate specs as YAML in `specs/` (`evaluation.spec` format).
- **Specs:** `locked: true` only where the rules give a single winner. Otherwise
  `locked: false`, `status: locked-pending`.
- **D25 ("ship LSTM"):** the report says whether it survives.
- **Not run:** `final_eval`.

### Order and stopping

Stages run sequentially. Each stage is resumable and pushed with a short summary appended here.
**Work stops after Stage A for review** before Stage B starts.

---

## Stage A — screening results (XGBoost, seed 42, 5 × 3 folds)

Completed 2026-10-04. Full per-dataset tables (every configuration with CI, n and
per-subpopulation results; every paired comparison with its Wilcoxon p; decision checks; A4) are
in `reports/selection/stage_A_FD00x.md`, machine-readable in `.json`. Δ = challenger − incumbent
in deployment-view critical RMSE (cycles); "better" = paired engine-bootstrap 95% CI entirely
below 0.

### Outcome per dataset

| dataset | A1 cap | A2 window | A3 blocks adopted | Stage-A config | critical RMSE [95% CI] (n engines) | vs original default (cap 125, window 20, base) Δ [95% CI] |
|---|---|---|---|---|---|---|
| FD001 | 90 | 45 | health index (pooled) | cap 90, window 45, `hi_pooled` | 3.41 [3.05, 3.76] (100) | −1.79 [−2.25, −1.36] |
| FD002 | 90 | 45 | health index (consistent) | cap 90, window 45, `hi_consistent` | 3.92 [3.63, 4.21] (260) | −2.01 [−2.42, −1.63] |
| FD003 | 90 | 45 | health index (consistent) | cap 90, window 45, `hi_consistent` | 3.55 [3.12, 4.08] (100) | −1.44 [−2.11, −0.75] |
| FD004 | 90 | 30 | health index (consistent) + regime one-hot | cap 90, window 30, `hi_consistent` + `regime_onehot` | 5.98 [5.51, 6.47] (249) | −1.14 [−1.49, −0.82] |

The last column is one direct paired comparison per dataset added for context (not part of the
pre-registered chain, which tests one factor at a time). Original-default critical RMSE: FD001
5.19 [4.78, 5.64], FD002 5.93 [5.58, 6.32], FD003 4.99 [4.40, 5.61], FD004 7.12 [6.62, 7.64].

### What each step found

- **A1 cap.** 90 and 105 both beat 125 on every dataset (e.g. FD004 cap 90 −0.54 [−0.70, −0.38],
  n = 249); 140 is worse than 125 on every dataset. 90 has the lowest estimate everywhere.
  **Flag for review: 90 is the lowest value of the pre-registered grid, so the optimum may lie
  below it** — the grid did not test < 90.
- **A2 window.** 45 adopted on FD001/FD002/FD003 (e.g. FD002 −1.55 [−1.92, −1.16]); on FD004
  only 30 beats 20 (−0.40 [−0.73, −0.06]); 45 is not different there (−0.21 [−0.77, 0.36]).
  **Same flag: 45 is the grid's upper edge for three datasets.**
- **A3 blocks.** The direction-consistent health index beats base on all four datasets (FD001
  −0.47 [−0.84, −0.18], FD002 −0.23 [−0.38, −0.07], FD003 −0.25 [−0.43, −0.06], FD004
  −0.19 [−0.34, −0.04]). The pooled index also beats base on FD001 (−0.57 [−0.91, −0.24],
  the lower estimate, so FD001 carries `hi_pooled`) and FD002, and is not different on FD003 /
  FD004. Regime one-hot: barely better on FD004 (−0.08 [−0.17, −0.01]), not different on FD002.
  Extra sensors (`sensors+`): not different on FD001 / FD003; see deviations for FD002 / FD004.
- **Decision check** (caught % with lead ≥ 20 at matched wasted life): no adopted change is worse
  at any budget; several are better at the 30-cycle budget (e.g. FD002 window 45 +4.5
  [1.3, 7.2] pp; FD004 cap 90 +1.7 [0.2, 3.1] pp). Against the original default the Stage-A
  configs catch more at 30 cycles on FD002 (+9.0 [6.2, 12.0]), FD003 (+4.3 [0.9, 8.2]) and FD004
  (+5.9 [3.2, 8.7]); FD001 not different (+1.2 [−1.4, 5.2]).
- **A4 early identifiability** (AUC [95% CI], 3-repeat average):

  | N cycles | FD001 (60/40) | FD002 (183/77) | FD003 (56/44) | FD004 (148/101) |
  |---|---|---|---|---|
  | 30 | 0.66 [0.54, 0.77] | 0.52 [0.45, 0.60] | 0.997 [0.989, 1.000] | 0.995 [0.989, 0.999] |
  | 50 | 0.72 [0.62, 0.81] | 0.59 [0.51, 0.66] | 0.994 [0.983, 1.000] | 0.997 [0.992, 1.000] |
  | 100 | 0.95 [0.91, 0.99] | 0.95 [0.92, 0.97] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

  "Strong" (CI lower bound ≥ 0.80) from 30 cycles on FD003 and FD004 — the two datasets the
  C-MAPSS readme lists with two fault modes — and only from 100 cycles on FD001 / FD002. An
  exploratory check (not pre-registered, not MLflow-logged) on FD003 N = 30: with shuffled labels
  the AUC is 0.55 [0.43, 0.66], so the pipeline does not leak; the real separation comes from
  early *levels* (first-30-cycle mean z of s17, s3, s2, s11, s4 differ by ≈ 1.4 SD between the
  groups), not early slopes. Per the pre-registration this is **reported, not forced**: a
  causal subpopulation-probability feature is a candidate for a later block test on FD003/FD004.

### Deviations and incidents (all recorded, none changes a pre-registered rule)

1. **Resume-key bug (D49).** The resume key omitted the new `features` section, so the first A3
   runs on FD001 and FD003 silently reloaded their base runs (Δ exactly 0.00). Found from the
   results, fixed (`39fb46b`, with a test that every params section is classified), invalid runs
   deleted locally and tagged `invalid=resume_key_missing_features` in MLflow, A3 re-run. A1/A2
   were unaffected.
2. **`sensors+` infeasible on FD002 / FD004 (rule 9 — contradicts audit §E).** s10 (FD002
   regime 1) and s16 (FD002 5 of 6 regimes, FD004 4 of 6) are constant within operating regimes
   on the training data, so within-regime normalization is undefined; audit §E had classed them
   informative. The pre-registered block was recorded as infeasible and **not adopted**. The
   fittable subset ran as an informational deviation, **not eligible for adoption**: FD002 [s6]
   −0.04 [−0.09, 0.00] (not different); FD004 [s6, s10] +0.49 [0.03, 1.01] (worse).
3. **A3-combined reduces to one block when both health indices are adopted** (they share the
   same three columns): the union then holds the index with the lower estimate (FD001 pooled,
   FD002 consistent), identical to that block's own run.
4. **Orchestrator restarts.** FD002 / FD004 orchestrators hit a 2-hour process limit and were
   relaunched; finished configurations were reused, unfinished splits resumed from MLflow. No
   effect on results.

### Limitations carried forward

- One factor at a time: cap was chosen at window 20 and base features; window at the A1 cap;
  blocks at both. Interactions (e.g. cap × window) are untested.
- About 40 paired comparisons across the four datasets without multiplicity correction; the
  smallest adopted effects (FD004 regime one-hot, CI upper bound −0.01) are the most likely
  false positives. Stage D (5 seeds) is the confirmation.
- Single seed (42) throughout.
- Compute: 41 configuration runs, 9.8 h of summed run time (four datasets in parallel, so each run
  was slowed by CPU contention), ≈ 4.5 h elapsed including the reruns above.

### Carried into Stage B (pending review)

Per dataset: cap 90 for the LSTM (B1 seq_len, B2 regime one-hot channels on FD002/FD004), and
the Stage-A XGBoost configs above as the XGBoost candidates for Stage C. **Work stops here for
review before Stage B.**

## Reproducibility (environment of the Stage A runs)

- Generated: 2026-10-03T19:42:02+00:00
- Git commit: `b290869bfb3d37e613b9d30adc1597da849ec49d`
- DVC data hash (`data/raw`): `43f328008844d7fd56733c63103d09ef.dir`
- Hardware: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical CPUs; GPU NVIDIA GeForce RTX 4060 Laptop GPU (driver 581.86, CUDA driver 13.0, torch built for CUDA 13.0); resolved devices: torch (LSTM) on cuda, XGBoost on cuda; OS Windows-10-10.0.26200-SP0
- Threads: 20 (torch and all native pools pinned); torch deterministic algorithms: False; CUDA initialized: False
- Python 3.11.15

| Library | Version |
|---|---|
| mlflow | 3.11.1 |
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| threadpoolctl | 3.6.0 |
| torch | 2.11.0+cu130 |
| xgboost | 3.2.0 |

| Native thread pool | Implementation | Version | Threads |
|---|---|---|---|
| libscipy_openblas-64eda39e79589aedb16f58e5547eb599.dll | openblas (blas) | 0.3.30 | 24 |
| libscipy_openblas64_-63c857e738469261263c764a36be9436.dll | openblas (blas) | 0.3.31.188.0 | 24 |
| libiomp5md.dll | openmp (openmp) | None | 20 |
| libiompstubs5md.dll | openmp (openmp) | None | 1 |
| vcomp140.dll | openmp (openmp) | None | 28 |

| Stochastic step | Seed |
|---|---|
| cv_seed | 0 |
| model_seed_42 | 42 |

Pipeline constants in effect: `cfg_REGIME_KMEANS_N_INIT=10`, `cfg_REGIME_KMEANS_SEED=0`, `cfg_SEED=42`, `cfg_SEEDS=(42, 7, 123, 2024, 99)` (full `turbofan.config` in the JSON sidecar).
