# Legacy notebook outputs (pre protocol v2)

Static HTML exports of the three analysis notebooks as they were executed before this repo
adopted the evidence-and-test-set rules in `CLAUDE.md` / `docs/decisions.md` ("protocol v2").
The source `.ipynb` files were stripped of outputs on 2026-10-02 (nbstripout pre-commit hook);
these HTML files are the only remaining record of what they printed and plotted.

- `01_understanding_the_data.html`
- `02_modeling.html`
- `03_model_comparison.html`

## Why these are preserved but not authoritative

**These runs used the NASA test set for model selection, tuning and ablation decisions.**
Protocol v2 (rule 3 in `CLAUDE.md`) seals the test set: it may be read only by a `final_eval`
step, once per locked candidate. That rule did not exist when these notebooks were run. In
particular:

- `02_modeling.html` tunes XGBoost with Optuna and evaluates the EWM feature ablation directly
  against `test_FD00x.txt` / `RUL_FD00x.txt`.
- `03_model_comparison.html` selects the LSTM over XGBoost using 5-seed test-set scores.

**Do not cite numbers from these HTML files as evidence for a LOCKED decision in
`docs/decisions.md`.** Where `docs/decisions.md` already cites one of these notebooks, it marks
the result REVISIT (or inadmissible) for this reason — the citation points at the file for
traceability, not as proof the decision is settled. Re-deriving any of these numbers validly
means re-running the comparison against validation data only, with the test set touched exactly
once by a locked, final `final_eval` step.

What *is* still valid evidence here: numbers computed entirely on train/validation data, before
any test read — e.g. `01_understanding_the_data.html`'s sensor-variance cutoff and the FD002
regime silhouette scan (`docs/decisions.md` D02a, D04a).
