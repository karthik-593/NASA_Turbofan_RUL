"""Tests for turbofan.repro — the CPU-only, deterministic context for report computations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans

from turbofan import repro
from turbofan.config import REGIME_KMEANS_N_INIT, REGIME_KMEANS_SEED, REGIME_N_CLUSTERS


class TestCpuDeterministic:
    def test_pins_and_restores_torch_state(self) -> None:
        before_threads = torch.get_num_threads()
        before_det = torch.are_deterministic_algorithms_enabled()
        with repro.cpu_deterministic(n_threads=1):
            assert torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == before_threads
        assert torch.are_deterministic_algorithms_enabled() == before_det

    def test_pins_native_thread_pools(self) -> None:
        with repro.cpu_deterministic(n_threads=1):
            pools = repro._thread_pools()
        assert pools  # numpy's BLAS is always loaded
        assert all(p["num_threads"] == 1 for p in pools)

    def test_rejects_non_positive_threads(self) -> None:
        with pytest.raises(ValueError, match="n_threads"), repro.cpu_deterministic(n_threads=0):
            pass

    def test_seeded_kmeans_reproduces_bit_for_bit(self) -> None:
        """The regime-clustering settings the pipeline uses give identical centers and
        labels on two independent fits — the property a report's numbers rely on."""
        X = np.random.default_rng(0).normal(size=(600, 3))

        def fit() -> tuple[np.ndarray, np.ndarray]:
            with repro.cpu_deterministic(n_threads=1):
                km = KMeans(
                    n_clusters=REGIME_N_CLUSTERS,
                    n_init=REGIME_KMEANS_N_INIT,
                    random_state=REGIME_KMEANS_SEED,
                ).fit(X)
            return km.cluster_centers_, km.labels_

        c1, l1 = fit()
        c2, l2 = fit()
        assert np.array_equal(c1, c2)
        assert np.array_equal(l1, l2)


class TestEnvironmentContext:
    def test_records_provenance_versions_and_seeds(self) -> None:
        with repro.cpu_deterministic(n_threads=1):
            ctx = repro.environment_context(seeds={"kmeans": 0, "bootstrap": 7}, n_threads=1)

        assert len(ctx["git_commit"]) == 40
        assert ctx["dvc_data_hash"]
        assert ctx["n_threads"] == 1
        assert ctx["torch_deterministic"] is True
        assert ctx["torch_cuda_initialized"] is False
        assert ctx["libs"]["numpy"] == np.__version__
        assert ctx["libs"]["torch"] == torch.__version__  # keeps the +cpu / +cuXXX tag
        assert ctx["seeds"] == {"kmeans": 0, "bootstrap": 7}
        assert ctx["config"]["cfg_REGIME_KMEANS_SEED"] == str(REGIME_KMEANS_SEED)

    def test_outside_pinned_context_raises(self) -> None:
        threads = torch.get_num_threads()
        with pytest.raises(RuntimeError, match="inside cpu_deterministic"):
            repro.environment_context(seeds={}, n_threads=threads + 1)

    def test_sidecar_roundtrip_has_no_mismatches(self, tmp_path: Path) -> None:
        with repro.cpu_deterministic(n_threads=1):
            ctx = repro.environment_context(seeds={"kmeans": 0}, n_threads=1)
            again = repro.environment_context(seeds={"kmeans": 0}, n_threads=1)

        sidecar = repro.write_context(ctx, tmp_path / "audit.md")
        assert sidecar.name == "audit.env.json"
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert repro.mismatches(recorded, again) == []

    def test_mismatches_names_changed_fields(self) -> None:
        with repro.cpu_deterministic(n_threads=1):
            ctx = repro.environment_context(seeds={"kmeans": 0}, n_threads=1)
        changed = {**ctx, "seeds": {"kmeans": 1}, "libs": {**ctx["libs"], "numpy": "0.0"}}
        assert repro.mismatches(ctx, changed) == ["libs", "seeds"]


class TestRenderMarkdown:
    def test_section_lists_hardware_libs_and_seeds(self) -> None:
        with repro.cpu_deterministic(n_threads=1):
            ctx = repro.environment_context(seeds={"regime_kmeans": 0}, n_threads=1)
        md = repro.render_markdown(ctx)
        assert md.startswith("## Reproducibility")
        assert "CPU only" in md
        assert ctx["git_commit"] in md
        assert f"| numpy | {np.__version__} |" in md
        assert "| regime_kmeans | 0 |" in md
        assert "`cfg_REGIME_KMEANS_SEED=0`" in md

    def test_empty_seeds_stated_explicitly(self) -> None:
        with repro.cpu_deterministic(n_threads=1):
            ctx = repro.environment_context(seeds={}, n_threads=1)
        assert "No stochastic steps" in repro.render_markdown(ctx)
