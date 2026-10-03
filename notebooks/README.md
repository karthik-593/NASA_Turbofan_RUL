# Notebooks

| Notebook | Purpose | Status | Decisions informed | Runtime (2026-10-03) |
|---|---|---|---|---|
| [00_data_audit.ipynb](00_data_audit.ipynb) | Data audit of FD001–FD004: structure, test-label distribution, validation realism, regimes, sensor informativeness, RUL-cap change-points, noise vs trend, the removed 258-feature family, engine subpopulations, regime-dependent sensor direction. Writes `reports/data_audit.md`. | current | D01, D02a, D02b, D03, D04a, D04b, D05, D06, D07, D08, D09, D10, D11, D13, D17, D18, D35, D36; X04, X06 | 373 s (CPU, 1 thread; D34) |
| [archive/01_understanding_the_data.ipynb](archive/01_understanding_the_data.ipynb) | First look at FD001/FD002: target construction, sensor variance, operating regimes, the 42-feature set. | archived (v0.1-pre-audit) | D01, D02a, D03, D04a, D07, D09 | 726 s (CPU, 1 thread) |
| [archive/02_modeling.ipynb](archive/02_modeling.ipynb) | XGBoost baseline with Optuna tuning on validation instances, scored on the NASA test set; EWM ablation; failure analysis. | archived (v0.1-pre-audit) | D07, D17, D23, D31 | 413 s (XGBoost on GPU) |
| [archive/03_model_comparison.ipynb](archive/03_model_comparison.ipynb) | Five-model screen and 5-seed LSTM-vs-XGBoost re-run, both scored on the NASA test set; the selection behind shipping the LSTM. | archived (v0.1-pre-audit) | D20, D24, D25 | 2857 s (XGBoost on GPU, LSTM on CPU) |

Archived notebooks reflect the pipeline at tag `v0.1-pre-audit`. They used the NASA test set
for model selection (D25) and validation instances later shown unrepresentative (D17), so
their numbers are not valid evidence for any decision. The HTML exports of the original
executions are in git history ([`reports/legacy/`](../reports/legacy/) says where).

Re-executed 2026-10-03 with their original device selection (D43). Compared with the
originally logged numbers: 02 (every metric in its decision log and test table) and 03 (the
single-seed screen, the cross-dataset means, the 5-seed table and the pooled result in
`docs/results.md`) reproduce them exactly; 01's cited numbers (lifetimes, sensor stds,
silhouette scan, RF baseline) do too.

## Outputs policy

Notebooks are committed **with outputs**, executed top-to-bottom in a fresh kernel
(`uv run jupyter nbconvert --to notebook --execute --inplace <nb>`). Seeds are fixed and the
machine, GPU and resolved device are recorded in a provenance cell; device selection is
automatic and not forced (D43). The data audit alone runs inside `repro.cpu_deterministic`
for its bit-exact rerun check (D34). The nbstripout pre-commit hook keeps outputs and strips
only execution counts, cell timestamps, kernel info and widget state (D37). Figures are PNG
at dpi 100; no notebook may exceed 5 MB — report one that does, don't raise the limit.

## Markdown style guide

**Opening cell.** Purpose (≤ 2 lines), questions answered, data version (the DVC hash of
`data/raw`, printed by code — never typed), outputs produced, decisions informed (D-IDs),
status (current / archived).

**Each section:** Question → Method (why this method, ≤ 3 lines) → Result (the table or
figure the code prints) → Takeaway (one line + D-ID).

**Numbers.** Never hand-type a number in markdown that the code computes — it drifts. Refer
to the printed table or figure, or print the sentence from code with an f-string. If a
markdown claim states a number — or a qualitative result the next rerun could overturn —
put an assert cell right before it that checks it.

**"If asked to defend this:"** only at genuinely contestable choices, 1–3 lines.

**Figures.** Every figure has a title, axis labels with units, and n in the title or caption.

**Voice.** First-person engineering voice ("I chose X because Y"); decisions over
description. Markdown cells ≤ ~6 lines.

**Banned.** "Let's", "In this section we will", restating what the code obviously does,
emojis, filler summaries, praise of results.

**Archived notebooks** keep their original narrative as the historical record: they get the
archive banner as their opening cell, a re-execution setup cell, and one-line
`> **Audit note (D-ID; section):**` notes where a later result contradicts a claim — the
original claim is never edited. Their original figures and markdown are not restyled.
