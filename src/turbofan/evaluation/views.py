"""The two protocol-v2 evaluation views on held-out training engines (D46).

Held-out training engines run to failure, so their true RUL is known at every cycle, uncapped.

- **Deployment view (primary — model selection uses this):** a prediction at every cycle of
  every held-out trajectory from ``serving.min_history`` on — the cycles the API would accept.
- **Benchmark view (secondary, comparability only):** truncation points sampled from the same
  held-out trajectories so that their true-RUL distribution matches the NASA test label
  distribution (audit §B), RUL above the cap included. It reads **only the test label
  histogram** (``RUL_FD00x.txt``) — never a test input (``test_FD00x.txt``) — and only to
  set sampling proportions; nothing is fitted or scored on test data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import MIN_HISTORY, PARAMS

__all__ = [
    "deployment_view",
    "benchmark_view",
    "nasa_test_label_histogram",
    "BENCHMARK_BIN_WIDTH",
    "BENCHMARK_POINTS_PER_ENGINE",
]

BENCHMARK_BIN_WIDTH: int = PARAMS["benchmark_view"]["bin_width"]
BENCHMARK_POINTS_PER_ENGINE: int = PARAMS["benchmark_view"]["points_per_engine"]


def deployment_view(points: pd.DataFrame, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """Every held-out cycle at which the API would serve a prediction."""
    out: pd.DataFrame = points[points["cycle"] >= min_history]
    return out


def nasa_test_label_histogram(
    dataset: str, raw: str | Path, bin_width: int = BENCHMARK_BIN_WIDTH
) -> pd.Series[float]:
    """Share of NASA test engines per true-RUL bin ``[lo, lo + bin_width)`` (index = lo).

    Reads ``RUL_<dataset>.txt`` — the test *labels* — and nothing else; used only as the
    benchmark view's sampling target (D46), never for fitting, selection or scoring.
    """
    labels = pd.read_csv(Path(raw) / f"RUL_{dataset}.txt", header=None)[0].to_numpy()
    bins = (labels // bin_width) * bin_width
    share: pd.Series[float] = pd.Series(bins).value_counts(normalize=True).sort_index()
    return share


def _allocate(shares: pd.Series[float], total: int) -> pd.Series[int]:
    """Largest-remainder integer allocation of ``total`` points to bins in proportion."""
    raw = shares.to_numpy(float) * total
    n = np.floor(raw).astype(int)
    short = total - int(n.sum())
    n[np.argsort(-(raw - n), kind="stable")[:short]] += 1
    return pd.Series(n, index=shares.index)


def benchmark_view(
    points: pd.DataFrame,
    label_share: pd.Series[float],
    rng: np.random.Generator,
    points_per_engine: int = BENCHMARK_POINTS_PER_ENGINE,
    bin_width: int = BENCHMARK_BIN_WIDTH,
    min_history: int = MIN_HISTORY,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Truncation points from ``points`` (one held-out set: unit, cycle, rul_true, ...) whose
    true-RUL bins follow ``label_share``. ``n_engines x points_per_engine`` points in total,
    sampled without replacement within each bin.

    Returns the sampled rows and the shortfall per bin (target points that no held-out
    trajectory could supply — e.g. RUL beyond the longest held-out life).
    """
    cand = deployment_view(points, min_history)
    total = cand["unit"].nunique() * points_per_engine
    target = _allocate(label_share, total)
    cand_bin = (cand["rul_true"].to_numpy() // bin_width) * bin_width
    picked: list[npt.NDArray[np.intp]] = []
    shortfall: dict[int, int] = {}
    for lo, n in zip(target.index.to_numpy(int), target.to_numpy(int), strict=True):
        idx = np.flatnonzero(cand_bin == lo)
        take = min(int(n), len(idx))
        if take < n:
            shortfall[int(lo)] = int(n) - take
        if take:
            picked.append(rng.choice(idx, size=take, replace=False))
    sel = np.sort(np.concatenate(picked)) if picked else np.array([], dtype=np.intp)
    return cand.iloc[sel], shortfall
