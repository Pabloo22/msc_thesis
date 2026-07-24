"""Regression tests for runner-level invariants.

Two failure modes that once existed silently:

* ``_install_adapter`` used a bare ``copytree``, so a crash between copying
  ``adapter_config.json`` and the weights left a corrupt adapter that
  ``Store.has_adapter`` reported as complete -- poisoning every trajectory
  sharing the prefix.
* ``compute_step_latent`` trusted a cached ``latent.json`` without checking
  which h_neutral sources it contained, so widening ``h_neutral_source`` from
  BASE to BOTH silently never produced the "current" series.
"""

from __future__ import annotations

import dataclasses
import shutil

import pytest

from method import experiments as E, run_trajectory, steps
from method.backends import get_backend
from method.config import Backend, HNeutralSource
from method.run_trajectory import _install_adapter, _verify_cached_adapter
from method.store import Store, file_sha256, get_weights_id


class TestInstallAdapterIsAtomic:
    def test_normal_install_copies_adapter_files(self, tmp_path):
        produced = tmp_path / "produced"
        produced.mkdir()
        (produced / "adapter_config.json").write_text("{}", encoding="utf-8")
        (produced / "adapter_model.safetensors").write_bytes(b"weights")
        target = tmp_path / "adapters" / "wid"

        _install_adapter(produced, target)

        assert (target / "adapter_config.json").exists()
        assert (target / "adapter_model.safetensors").exists()

    def test_trainer_state_is_not_installed(self, tmp_path):
        """Optimizer/RNG state only matters for resuming the training run
        itself and roughly triples the stored size, so it stays behind."""
        produced = tmp_path / "produced"
        produced.mkdir()
        (produced / "adapter_config.json").write_text("{}", encoding="utf-8")
        (produced / "optimizer.pt").write_bytes(b"state")
        (produced / "rng_state_0.pth").write_bytes(b"state")
        target = tmp_path / "adapters" / "wid"

        _install_adapter(produced, target)

        assert not (target / "optimizer.pt").exists()
        assert not (target / "rng_state_0.pth").exists()

    def test_interrupted_install_leaves_no_adapter(self, tmp_path, monkeypatch):
        """A crash mid-copy must leave the target absent, not half-written:
        ``has_adapter`` treats presence of adapter_config.json as complete."""
        produced = tmp_path / "produced"
        produced.mkdir()
        (produced / "adapter_config.json").write_text("{}", encoding="utf-8")
        target = tmp_path / "adapters" / "wid"

        real_copytree = shutil.copytree

        def crash_after_copy(src, dst, **kwargs):
            real_copytree(src, dst, **kwargs)
            raise RuntimeError("simulated crash before rename")

        monkeypatch.setattr(shutil, "copytree", crash_after_copy)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _install_adapter(produced, target)

        assert not target.exists()


class TestTrainingDataProvenance:
    """The ids hash dataset *names*; the recipe's provenance block records the
    *bytes*, so a machine with a divergent dataset copy fails loudly instead
    of silently reusing an adapter trained on different data."""

    WID = "t01-feedfeedfeedfeed"

    def _store_with_recipe(self, tmp_path, train_file) -> Store:
        store = Store(tmp_path / "store")
        store.adapter_dir(self.WID).mkdir(parents=True)
        store.write_recipe(
            self.WID,
            {"steps": ["irrelevant"]},
            provenance={"training_sample_sha256": file_sha256(train_file)},
        )
        return store

    def test_matching_bytes_pass(self, tmp_path):
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n', encoding="utf-8")
        store = self._store_with_recipe(tmp_path, train_file)

        _verify_cached_adapter(store, self.WID, train_file, step_number=1)

    def test_divergent_bytes_raise(self, tmp_path):
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"messages": []}\n', encoding="utf-8")
        store = self._store_with_recipe(tmp_path, train_file)

        train_file.write_text('{"messages": ["other"]}\n', encoding="utf-8")
        with pytest.raises(RuntimeError, match="training data mismatch"):
            _verify_cached_adapter(store, self.WID, train_file, step_number=1)

    def test_adapters_without_provenance_are_trusted(self, tmp_path):
        """Pre-provenance adapters (or a crash between install and recipe
        write) have no digest; the check must not retroactively reject them."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text("x\n", encoding="utf-8")
        store = Store(tmp_path / "store")
        store.adapter_dir(self.WID).mkdir(parents=True)

        _verify_cached_adapter(store, self.WID, train_file, step_number=1)

        store.write_recipe(self.WID, {"steps": []})  # recipe, no provenance
        _verify_cached_adapter(store, self.WID, train_file, step_number=1)

    def test_end_to_end_mock_run_records_and_enforces_the_digest(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        monkeypatch.setattr(
            Store, "for_backend", classmethod(lambda cls, backend: store)
        )
        cfg = E.SMOKE_MOCK

        run_trajectory.run(cfg, Backend.MOCK, "float16")

        first_step = cfg.steps[0]
        sample = store.training_sample_path(
            steps.training_sample_id(first_step, cfg.seed)
        )
        recorded = store.recorded_training_sample_sha256(get_weights_id(cfg, 1))
        assert recorded == file_sha256(sample)

        # Simulate a machine whose dataset file differs: the regenerated
        # sample keeps its name-based id but carries different bytes.
        sample.write_text('{"messages": ["divergent"]}\n', encoding="utf-8")
        with pytest.raises(RuntimeError, match="training data mismatch"):
            run_trajectory.run(cfg, Backend.MOCK, "float16")


class TestLatentSourcesFollowTheConfig:
    def test_widening_h_neutral_source_recomputes_the_latent(self, tmp_path):
        """A latent.json cached under BASE must not satisfy a BOTH config."""
        store = Store(tmp_path)
        backend = get_backend(Backend.MOCK, dtype="float16")
        cfg = E.SMOKE_MOCK
        assert cfg.latent.h_neutral_source is HNeutralSource.BASE

        steps.extract_persona_vector(cfg, 0, store, backend)
        steps.measure_h_neutral(cfg, 0, store, backend)
        assert set(steps.compute_step_latent(cfg, 0, store)) == {"base"}

        both = dataclasses.replace(
            cfg,
            latent=dataclasses.replace(
                cfg.latent, h_neutral_source=HNeutralSource.BOTH
            ),
        )
        steps.measure_h_neutral(both, 0, store, backend)
        latents = steps.compute_step_latent(both, 0, store)

        assert set(latents) == {"base", "current"}
