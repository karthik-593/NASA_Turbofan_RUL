"""LSTM model for C-MAPSS RUL prediction.

A sequence model over the normalized sensor channels — NOT the engineered flat features.
The LSTM learns its own temporal representation, so it gets the raw normalized sensors over
a sliding window. Deliberately small: the earlier large LSTM overfit (~0.1 samples per
parameter); this one uses ~14 channels, a 30-cycle window, and a modest hidden size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from turbofan.config import RUL_CAP, SEED, SEQ_LEN

__all__ = ["LSTMRUL", "make_sequences", "make_last_windows", "SEQ_LEN", "RUL_CAP"]


def make_sequences(
    feat_df: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Sliding windows per engine; target = RUL at the window's last cycle.
    Left-pads short histories with zeros so every window is seq_len long."""
    X, y = [], []
    for _, g in feat_df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = g[feat_cols].to_numpy().astype(np.float32)
        ruls = g["rul"].to_numpy().astype(np.float32)
        for i in range(len(g)):
            win = vals[max(0, i - seq_len + 1) : i + 1]
            if len(win) < seq_len:
                pad = np.zeros((seq_len - len(win), len(feat_cols)), dtype=np.float32)
                win = np.vstack([pad, win])
            X.append(win)
            y.append(ruls[i])
    return np.stack(X), np.array(y, dtype=np.float32)


def make_last_windows(
    feat_df: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """One window per engine ending at its last observed cycle — for test prediction."""
    X, units = [], []
    for unit, g in feat_df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = g[feat_cols].to_numpy().astype(np.float32)
        win = vals[-seq_len:]
        if len(win) < seq_len:
            pad = np.zeros((seq_len - len(win), len(feat_cols)), dtype=np.float32)
            win = np.vstack([pad, win])
        X.append(win)
        units.append(unit)
    return np.stack(X), np.array(units, dtype=np.int64)


class _Net(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        result: torch.Tensor = self.head(out[:, -1, :]).squeeze(-1)
        return result


class LSTMRUL:
    """Small LSTM regressor with early stopping on validation MSE."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        seed: int = SEED,
        device: str | None = None,
    ) -> None:
        self.cfg: dict[str, Any] = dict(
            n_features=n_features,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _Net | None = None
        self.best_val_loss_: float | None = None

    def fit(
        self,
        X_tr: npt.NDArray[np.float32],
        y_tr: npt.NDArray[np.float32],
        X_va: npt.NDArray[np.float32],
        y_va: npt.NDArray[np.float32],
    ) -> Self:
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        self.net = _Net(
            self.cfg["n_features"], self.cfg["hidden"], self.cfg["layers"], self.cfg["dropout"]
        ).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        lossf = nn.MSELoss()
        dl = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=self.cfg["batch_size"],
            shuffle=True,
        )
        Xva = torch.tensor(X_va).to(self.device)
        yva = torch.tensor(y_va).to(self.device)
        best: float = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        wait = 0
        for _ in range(self.cfg["max_epochs"]):
            self.net.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = lossf(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = lossf(self.net(Xva), yva).item()
            sched.step(vloss)
            if vloss < best - 1e-4:
                best = vloss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss_ = best
        return self

    def predict(self, X: npt.NDArray[np.float32], clip: bool = True) -> npt.NDArray[np.float32]:
        if self.net is None:
            raise RuntimeError("Call fit() first.")
        self.net.eval()
        with torch.no_grad():
            p = self.net(torch.tensor(X).to(self.device)).cpu().numpy()
        return np.asarray(np.clip(p, 0.0, RUL_CAP), dtype=np.float32) if clip else p

    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("Call fit() first.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state": self.net.state_dict()}, p)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> Self:
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(**ckpt["cfg"], device=device)
        obj.net = _Net(
            obj.cfg["n_features"], obj.cfg["hidden"], obj.cfg["layers"], obj.cfg["dropout"]
        ).to(obj.device)
        obj.net.load_state_dict(ckpt["state"])
        return obj
