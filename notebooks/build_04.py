"""Builds notebooks/04_model_selection.ipynb (cells only; executed with nbconvert afterwards).

Kept as a script so the notebook's structure is reviewable as code; the committed artifact is the
executed notebook. Run: uv run python notebooks/build_04.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list[nbf.NotebookNode] = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# 04 — Model selection under protocol v2 (FD001–FD004)

**Purpose:** pick one model specification per dataset on training engines only, under the rules
pre-registered in `reports/model_selection.md` (D51), and lock it — or leave two candidates
pending where the rules do not separate them.
**Questions:** 1 how far selection moved the headline · 2 is cap 90 an artefact of the metric ·
3 which addendum blocks survived · 4 LSTM sequence length and channels · 5 what tuning bought ·
6 which adopted changes survive 5 seeds · 7 XGBoost or LSTM, and is it locked.
**Data version:** DVC hash of `data/raw`, printed below. **Outputs:** figures in
`reports/figures/model_selection/`; the specs in `specs/` are written by the orchestrator and
read here. **Decisions informed:** D01, D07, D08, D10, D21–D25, D35, D51, D52.
**Status:** current. The NASA test set is never read (rule 3).
""")

code("""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from turbofan import tracking
from turbofan.pipeline.selection import Spec

REPO = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SEL = REPO / "reports" / "selection"
FIG = REPO / "reports" / "figures" / "model_selection"
FIG.mkdir(parents=True, exist_ok=True)
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
HEAD = "critical_rmse"
pd.set_option("display.width", 200, "display.max_colwidth", 80)

git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                     text=True, check=True).stdout.strip()
print(f"data/raw DVC hash: {tracking._dvc_data_hash()}")
print(f"git commit: {git}")
print("source: reports/selection/*.json (every number below is read, none recomputed)")
""")

code("""
def load(stage: str, ds: str) -> dict:
    return json.loads((SEL / f"stage_{stage}_{ds}.json").read_text(encoding="utf-8"))


def ci(est: float, lo: float, hi: float, p: int = 2) -> str:
    return f"{est:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def headline(doc: dict, tag: str) -> tuple[float, float, float, int, int]:
    m = next(iter(doc["configs"][tag]["models"].values()))
    return m[HEAD], m[f"{HEAD}_ci_lo"], m[f"{HEAD}_ci_hi"], m["n_engines"], m["n_splits"]


def comparisons(stage: str, prefix: str = "") -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for r in load(stage, ds)["comparisons"]:
            if r["substage"].startswith(prefix):
                verdict = "better" if r["better"] else ("worse" if r["worse"] else "not different")
                rows.append({"dataset": ds, "step": r["substage"], "metric": r.get("metric", HEAD),
                             "challenger": r["challenger"], "incumbent": r["incumbent"],
                             "Δ [95% CI]": ci(r["diff"], r["diff_ci_lo"], r["diff_ci_hi"]),
                             "n engines": r["n_engines"], "verdict": verdict,
                             "_lo": r["diff_ci_lo"], "_hi": r["diff_ci_hi"]})
    return pd.DataFrame(rows)


def show(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not c.startswith("_")]]
""")

md("""
## 1. How far did selection move the headline?

**Question:** what deployment-view critical RMSE does each stage's candidate reach, starting from
the original default (cap 125, window 20, base features, untuned XGBoost)?
**Method:** each stage's adopted spec, read from its run; CIs are 1,000-replicate engine
bootstraps. Stages A–C use seed 42 only, Stage D five seeds, so only Stage D ranks models.
""")

code("""
rows = []
for ds in DATASETS:
    a = load("A", ds)
    stages = [
        ("original default", a, f"A/{ds}/cap125"),
        ("Stage A XGBoost", a, a["winners"]["stage_A_tag"]),
    ]
    x, b, c, d = (load(s, ds) for s in "XBCD")
    stages.append(("Stage X XGBoost (re-run)", x, Spec.from_json(x["specs"]["xgboost"]).tag))
    stages.append(("Stage B LSTM", b, Spec.from_json(b["specs"]["lstm"]).tag))
    for m in ("xgboost", "lstm"):
        stages.append((f"Stage C {m}", c, Spec.from_json(c["specs"][m]).tag))
    for m in ("xgboost", "lstm"):
        stages.append((f"Stage D {m} (5 seeds)", d, Spec.from_json(d["specs"][m]).tag))
    for name, doc, tag in stages:
        est, lo, hi, n, k = headline(doc, tag)
        rows.append({"dataset": ds, "candidate": name, "est": est, "lo": lo, "hi": hi,
                     "critical RMSE [95% CI]": ci(est, lo, hi), "n engines": n, "n splits": k})
prog = pd.DataFrame(rows)
show(prog.drop(columns=["est", "lo", "hi"]))
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
for ax, ds in zip(axes, DATASETS):
    p = prog[prog["dataset"] == ds].reset_index(drop=True)
    colors = ["0.6" if "default" in c else ("tab:orange" if "lstm" in c.lower() else "tab:blue")
              for c in p["candidate"]]
    ax.errorbar(range(len(p)), p["est"], yerr=[p["est"] - p["lo"], p["hi"] - p["est"]], fmt="none",
                ecolor="0.3", capsize=3)
    ax.scatter(range(len(p)), p["est"], c=colors, zorder=3)
    ax.set_xticks(range(len(p)), p["candidate"], rotation=70, ha="right", fontsize=8)
    ax.set_title(f"{ds} (n = {p['n engines'].iloc[0]} engines)")
    ax.set_ylabel("critical RMSE, cycles (deployment view)")
fig.suptitle("Critical-bucket RMSE by selection stage, 95% engine-bootstrap CI "
             "(blue XGBoost, orange LSTM)")
fig.tight_layout()
fig.savefig(FIG / "progression.png", dpi=100)
plt.show()
""")

code("""
gain = {ds: prog[(prog.dataset == ds) & (prog.candidate == "original default")].est.iloc[0]
        - prog[(prog.dataset == ds) & prog.candidate.str.startswith("Stage D")].est.min()
        for ds in DATASETS}
assert all(g > 0 for g in gain.values())
print("Best Stage D finalist vs original default, point estimates (cycles): "
      + ", ".join(f"{ds} −{g:.2f}" for ds, g in gain.items()))
""")

md("""
**Takeaway:** every dataset's best 5-seed finalist sits below the original default (asserted
and printed above); the figure shows where along the stages each gain arrived (D51).
""")

md("""
## 2. Is cap 90 an artefact of the metric?

**Question:** critical RMSE favours a narrower training target by construction — does cap 90 also
win where no cap binds?
**Method:** the pre-registered A1-check on the Stage A runs (urgent-bucket RMSE, critical / urgent
late % vs uncapped truth, matched-budget decision check), repeated at 5 seeds in Stage D's
leave-one-out of the cap.
""")

code("""
a1 = json.loads((SEL / "a1_check.json").read_text(encoding="utf-8"))
rows = []
for ds, d in a1["datasets"].items():
    for r in d["comparisons"]:
        rows.append({"dataset": ds, "source": "A1-check (seed 42)", "metric": r["metric"],
                     "Δ cap 90 − 125 [95% CI]": ci(r["diff"], r["diff_ci_lo"], r["diff_ci_hi"]),
                     "worse": r["worse"]})
dcap = comparisons("D", "D-cap")
for r in dcap.to_dict("records"):
    rows.append({"dataset": r["dataset"], "source": f"Stage D {r['step'][6:]} (5 seeds)",
                 "metric": r["metric"], "Δ cap 90 − 125 [95% CI]": r["Δ [95% CI]"],
                 "worse": r["verdict"] == "worse"})
capt = pd.DataFrame(rows)
worse = capt[capt["worse"]]
assert a1["confirmed"]
assert not worse["source"].str.contains("xgboost").any()  # the XGBoost cap holds at 5 seeds
assert set(worse["dataset"] + " " + worse["source"]) == {"FD003 Stage D lstm (5 seeds)"}
capt
""")

md("""
**Takeaway:** for XGBoost, cap 90 is not worse than 125 on any metric no cap binds, at one seed
or at five, so it is kept (D01). **One contradiction (rule 9, X08):** the FD003 LSTM at cap 90 is
later on critical and urgent late % at five seeds (asserted above), so its cap reverts to 125 by
the pre-registered rule — the cap-90 confirmation of the A1-check does not transfer to every model.
""")

md("""
## 3. Which addendum blocks survived?

**Question:** do longer windows, the early-life baseline deviation or the early subpopulation
probability improve the Stage A XGBoost spec?
**Method:** pre-registered after Stage A (`ba0f6ba`): each challenger vs a re-run of its
incumbent under the current code, paired engine bootstrap; adopted only if the CI is below 0.
""")

code("""
xt = comparisons("X")
assert not (xt["verdict"] == "better").any()
show(xt)
""")

md("""
**Takeaway:** no addendum challenger beats its incumbent (asserted); windows 60 / 90 are not
different, the two new blocks are worse or not different, so the Stage A specs carry on unchanged.
The subpopulation is identifiable from 30 cycles on FD003/FD004, yet knowing it does not help the
critical zone (D35).
""")

md("""
## 4. LSTM: sequence length and channels

**Question:** which sequence length, and do the regime indicators or the health-index score help
the LSTM?
**Method:** Stage B, one factor at a time at cap 90 and seed 42; `seq_len` incumbent 30.
""")

code("""
bt = comparisons("B")
show(bt)
""")

code("""
b_adopt = {ds: Spec.from_json(load("B", ds)["specs"]["lstm"]) for ds in DATASETS}
assert all(not s.blocks for s in b_adopt.values())
print("Stage B LSTM per dataset: "
      + "; ".join(f"{ds} seq_len {s.seq_len}" for ds, s in b_adopt.items()))
""")

md("""
**Takeaway:** sequence length matters (20 is worse everywhere), the channel blocks never help —
no LSTM keeps one (asserted) (D10). 60 is the grid edge where it wins; that limitation stays open.
""")

md("""
## 5. What did tuning buy?

**Question:** does Optuna improve the stage-B configs beyond noise?
**Method:** TPE on the repeat-1 folds; the best trial vs the untuned config on repeats 2–3 only.
Optimistic by construction: those engines tuned the config in some repeat-1 fold.
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, ds in zip(axes, DATASETS):
    for m, col in (("xgboost", "tab:blue"), ("lstm", "tab:orange")):
        t = pd.DataFrame(load("C", ds)["extra"]["tuning"][m]["trials"])
        ax.plot(t["trial"], t[HEAD], ".", color=col, alpha=0.5)
        ax.plot(t["trial"], t[HEAD].cummin(), "-", color=col, label=f"{m} best so far")
    n = prog[prog.dataset == ds]["n engines"].iloc[0]
    ax.set_title(f"{ds}: trials on repeat 1 (5 splits, n = {n} engines)", fontsize=10)
    ax.set_xlabel("trial")
    ax.set_ylabel("critical RMSE, cycles")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "tuning.png", dpi=100)
plt.show()
show(comparisons("C"))
""")

md("""
**Takeaway:** tuning is adopted only where its repeats-2/3 CI excludes 0 (table); its gains are
smaller than Stage A's and, by design, optimistic (D21, D22).
""")

md("""
## 6. Which adopted changes survive five seeds?

**Question:** with 5 seeds × 15 splits, does each adopted change still beat the spec without it?
**Method:** pre-registered leave-one-out: Δ = finalist − finalist minus the change; a change whose
CI includes 0 is dropped (parsimony).
""")

code("""
loo = comparisons("D", "D-LOO")
print({ds: load("D", ds)["extra"]["loo"] for ds in DATASETS})
show(loo)
""")

md("""
## 7. XGBoost or LSTM — and is it locked?

**Question:** which finalist wins on critical RMSE, and does the decision check agree?
**Method:** paired Δ (LSTM − XGBoost) over 5 seeds × 15 splits; a tie goes to XGBoost (simpler);
locked only if the decision check (caught % at matched wasted life) supports the single winner,
else both locked-pending.
""")

code("""
rows = []
for ds in DATASETS:
    d = load("D", ds)
    fin = [r for r in d["comparisons"] if r["substage"] == "D-final"][0]
    dc = [r for r in d["decisions"] if r["substage"] == "D-final"]
    out = d["extra"]["outcome"]
    rows.append({"dataset": ds,
                 "Δ LSTM − XGBoost [95% CI]": ci(fin["diff"], fin["diff_ci_lo"], fin["diff_ci_hi"]),
                 "decision Δ caught pp (20/30/40)": "; ".join(
                     ci(r["diff"], r["diff_ci_lo"], r["diff_ci_hi"], 1) for r in dc),
                 "winner": out["winner"], "locked": out["locked"], "reason": out["reason"]})
final = pd.DataFrame(rows)
final
""")

code("""
for ds in DATASETS:
    for m in ("xgboost", "lstm"):
        p = REPO / "specs" / f"{ds}_{m}.yaml"
        s = yaml.safe_load(p.read_text(encoding="utf-8"))
        print(f"{p.name}: locked={s['locked']} status={s['status']} cap={s['rul_cap']} "
              f"window={s['window']} seq_len={s['seq_len']} features={s['features']} "
              f"tuned={bool(s['model_params'])}")
lstm_wins = [r.dataset for r in final.itertuples() if r.winner == "lstm"]
print(f"D25 ('ship the LSTM'): the LSTM wins on {lstm_wins or 'no dataset'} of {DATASETS}.")
""")

md("""
**Takeaway:** the outcome per dataset is the table above, computed from the pre-registered rules;
D25's test-set claim is re-decided here on training engines only (printed line). `params.yaml`
defaults are not changed and `final_eval` is not run — both are later, reviewed steps (D51).

**If asked to defend this:** a tie going to XGBoost is the pre-registered simplicity rule, not a
claim that XGBoost is more accurate; the LSTM remains within the CI.
""")

md("""
## Limitations

- One factor at a time: cap × window × features interactions are untested (except Stage D's
  leave-one-out, which tests each change in the context of the final spec).
- Screening (A–C) at one seed with ~70 uncorrected comparisons; only Stage D's 5-seed tests decide.
- Tuning optimism (repeats reuse the tuning engines); `seq_len` 60 is a grid edge on FD001/FD002.
- All evidence is CV on training engines; the sealed test read (`final_eval`) comes after review.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "04_model_selection.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
