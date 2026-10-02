# Problem Framing

## The task

C-MAPSS is a NASA-simulated dataset of turbofan engines run to failure. Each row is one
engine unit at one operating cycle: three operational settings and 21 sensor readings, no
labels. The task is to predict Remaining Useful Life (RUL) — how many cycles are left
before that engine fails — from its sensor history so far.

There are four sub-datasets, FD001–FD004, that trade up in difficulty:

| Dataset | Operating conditions | Fault modes |
|---|---|---|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

## Why RUL is capped at 125 cycles

Engines in this dataset live anywhere from ~130 to ~360 cycles. Early in an engine's life
it is healthy and the sensors barely move — there is nothing in the input that
distinguishes "300 cycles left" from "250 cycles left"; both look like a perfectly healthy
engine. Asking a model to predict a raw RUL of 300 is asking it to learn a number that
isn't present in its inputs.

The fix is the standard C-MAPSS convention: cap the label at 125 cycles. This isn't a trick
to make the error metric look smaller — it encodes a real property of the data. RUL is only
estimable once degradation has actually started, so flat-lining the label in the healthy
region stops the model from chasing signal that doesn't exist.

## Sensor selection

Of the 21 raw sensors, several are physically constant in this data (flat by construction,
e.g. `s1`) and therefore carry zero information regardless of any correlation score. A
variance cutoff finds them immediately — there's an unambiguous gap between the dead
sensors clustered near zero and the rest. The 14 sensors that survive this cut (`s2, s3,
s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`) are the ones carried through the
whole pipeline (`config.KEEP`).

## The decision the whole project hinges on: per-regime normalization

FD001 and FD003 run under one operating condition, so a global z-score per sensor is fine.
FD002 and FD004 do not — the engine constantly switches between roughly six distinct
operating regimes (visible directly in the three operational settings), and **each regime
has its own sensor baseline**. A given sensor reading means something different at cruise
than at takeoff. Normalize globally and six baselines stack on top of each other; a clean
degradation trend turns into noise.

This is not a modeling nuance — it is a correctness requirement. A model trained on raw,
globally-normalized FD002/FD004 sensors isn't underperforming because it's the wrong model;
it's underperforming because the input features mix six incompatible scales. `add_features`
(`turbofan.features.engineering`) fits a 6-cluster KMeans on the scaled operating settings
and z-scores each sensor *within its assigned regime* for FD002/FD004, falling back to a
single global z-score for FD001/FD003. Getting this wrong means every rolling mean, every
slope, and every model built downstream of it is built on garbage — recovering the signal
here is worth more than any amount of model tuning further down the pipeline.

## Feature set: three quantities, all motivated by an observed property of the signal

Looking at the (correctly normalized) degradation signal, it is:

- **gradual** — it builds over many cycles, not in jumps
- **roughly monotone** — once an engine starts degrading it keeps going
- **buried in cycle-to-cycle noise**

That points to exactly three derived quantities per kept sensor:

1. the normalized value itself — current state
2. a rolling mean (window 20 cycles) — smooths the noise so the trend is legible
3. a rolling slope (window 20 cycles) — the rate of degradation, which is what actually
   changes as an engine approaches failure

That's 42 features (`config.FEAT_COLS`) for the flat models. Deliberately **not** included:
rolling min/max/std, FFT/frequency features, lagged copies, sensor-pair interactions, or a
PCA health index. None of them are motivated by anything observed in the data — adding them
would just be enumerating every feature a library can compute. Unused features add noise,
slow the pipeline down, and make the model harder to trust and explain. The LSTM sees the
14 normalized channels directly (`config.SENSOR_N_COLS`) over a 30-cycle window
(`config.SEQ_LEN`) and learns its own temporal representation instead.

## Evaluation protocol

Every candidate model is held to the same protocol (`turbofan.evaluation.protocol`,
`turbofan.evaluation.comparison`):

- **Train/val split is by engine, not by row** — first 80% of engines by id go to train,
  the rest to validation (`config.SPLIT_FRAC`). A row-level split would leak an engine's
  own future into its own training signal.
- **Normalization statistics are fit on train only** and reused, unmodified, on val/test —
  the model never sees test-set statistics at any point.
- **Prediction happens once per test engine, at its last observed cycle**, compared against
  NASA's ground-truth RUL file. This is the only honest comparison point: it's the one
  moment a real deployment would actually be asked for a number.

### Headline metric: critical-zone RMSE, not global RMSE

The standard whole-range RMSE rewards a model equally for being wrong at RUL=200 (engine is
healthy, no decision riding on the number) and RUL=10 (engine is about to fail, a
maintenance decision is riding on the number *right now*). Those two errors are not
operationally equivalent, so a single pooled RMSE is the wrong optimization target. The
headline metric here is **critical-zone RMSE** — RMSE computed only over engines with true
RUL in [0, 25) — with the NASA asymmetric exponential score (`evaluation.metrics.cmapss_score`)
as a tiebreaker, since it independently penalizes late predictions harder than early ones.
Global RMSE and the percentage of late predictions are still reported, but not optimized
for.

### Maintenance-bucket framing

A raw predicted-RUL number in cycles isn't itself a decision. `config.MAINTENANCE_BUCKETS`
maps a prediction into four actionable bands — critical (0–25), urgent (25–50), monitor
(50–100), healthy (100+) — the same shape a maintenance scheduler actually works with:
which of these needs attention now, which can wait, which is fine. The serving API returns
the bucket, not just a bare number. Its confidence band is `null` until calibrated
prediction intervals exist (`docs/decisions.md` D26; the earlier band was test-set RMSE).
