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
