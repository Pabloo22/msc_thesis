"""Regression tests for runner-level invariants.

Failure modes that once existed silently:

* ``_install_adapter`` used a bare ``copytree``, so a crash between copying
  ``adapter_config.json`` and the weights left a corrupt adapter that
  ``Store.has_adapter`` reported as complete -- poisoning every trajectory
  sharing the prefix.
* ``compute_step_latent`` trusted a cached ``latent.json`` without checking
  which h_neutral sources it contained, so widening ``h_neutral_source`` from
  BASE to BOTH silently never produced the "current" series.
* Measurements reached the remote only after the whole trajectory finished, so
  a spot preemption during the last training step threw away every eval the
  run had paid for.
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
from method.sync import REMOTE_ENV, LocalTransport
from method.utils import trajectory_run_dir


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


class TestTrainingOutputIsNotRetained:
    """The trainer's output directory is a full ``checkpoint-N`` plus optimizer
    state, and nothing reads it once ``_install_adapter`` has lifted the adapter
    out of it. Kept in the run directory it survived the whole box *and* went up
    a second time inside the run tar, which is what made an otherwise-identical
    trajectory ship hundreds of megabytes where a fully cache-hit one shipped
    sixteen."""

    def test_a_finished_run_leaves_no_training_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        monkeypatch.setattr(
            Store, "for_backend", classmethod(lambda cls, backend: store)
        )
        cfg = E.SMOKE_MOCK

        run_dir = run_trajectory.run(cfg, Backend.MOCK, "float16")

        assert store.has_adapter(get_weights_id(cfg, 1))  # training did happen
        assert not list(run_dir.glob("train_out*"))
        assert list(store.train_scratch.iterdir()) == []

    def test_scratch_is_removed_when_training_fails(self, tmp_path):
        """A failed step is exactly when the box most needs the disk back."""
        store = Store(tmp_path / "store")

        with pytest.raises(RuntimeError, match="training blew up"):
            with run_trajectory.training_scratch(store, "t01-feedfeedfeedfeed") as out:
                (out / "checkpoint-1").mkdir(parents=True)
                raise RuntimeError("training blew up")

        assert list(store.train_scratch.iterdir()) == []

    def test_concurrent_steps_get_separate_directories(self, tmp_path):
        """Two boxes (or two GPUs) training the same weights_id at once must not
        write into one directory: ``find_adapter`` takes the highest
        ``checkpoint-N`` it finds, so a shared path could hand one run the
        other's checkpoint."""
        store = Store(tmp_path / "store")
        wid = "t01-feedfeedfeedfeed"

        with run_trajectory.training_scratch(store, wid) as first:
            with run_trajectory.training_scratch(store, wid) as second:
                assert first != second


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


class TestArtifactsArePushedAsTheyAreProduced:
    """A run must not bank a trajectory's output until the end.

    Steps cost GPU-hours each, and the boxes are preemptible spot rentals, so
    an artifact that exists locally but not on the remote is work that dies
    with the box. Everything durable therefore goes up at the moment it lands
    in the store, and the end-of-run sweep is only a backstop.
    """

    def _run_recording_uploads(self, tmp_path, monkeypatch) -> list[str]:
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        monkeypatch.setattr(
            Store, "for_backend", classmethod(lambda cls, backend: store)
        )
        monkeypatch.setenv(REMOTE_ENV, str(tmp_path / "remote"))

        uploads: list[str] = []
        real_upload = LocalTransport.upload

        def spy(transport, local, relpath):
            uploads.append(relpath)
            real_upload(transport, local, relpath)

        monkeypatch.setattr(LocalTransport, "upload", spy)
        run_trajectory.run(E.SMOKE_MOCK, Backend.MOCK, "float16")
        return uploads

    def test_a_checkpoint_is_pushed_before_the_next_step_trains(
        self, tmp_path, monkeypatch
    ):
        cfg = E.SMOKE_MOCK
        uploads = self._run_recording_uploads(tmp_path, monkeypatch)

        measured_step1 = f"store/measurements/{get_weights_id(cfg, 1)}.tar"
        trained_step2 = f"store/adapters/{get_weights_id(cfg, 2)}.tar"
        assert uploads.index(measured_step1) < uploads.index(trained_step2)

    def test_each_measurement_substep_is_pushed_as_it_lands(
        self, tmp_path, monkeypatch
    ):
        """Behavior, persona vector, h_neutral and latent must not be batched
        behind each other inside ``measure_checkpoint``: each changes the
        bundle and must trigger its own push, so a preemption between any two
        of them loses at most one already-complete measurement.

        Scoped to ``measure_checkpoint`` alone (rather than a full ``run()``,
        as the other tests here use) so the count is not muddied by the
        additional push ``run()`` makes afterwards for the DeltaP it computes
        against this same checkpoint.
        """
        store = Store(tmp_path / "store")
        backend = get_backend(Backend.MOCK, dtype="float16")
        cfg = E.SMOKE_MOCK
        assert not cfg.probes  # else more than 4 pushes would be correct too

        push_calls: list[int] = []
        monkeypatch.setattr(
            run_trajectory,
            "_push_measurements",
            lambda syncer, cfg, t: push_calls.append(t),
        )

        run_trajectory.measure_checkpoint(cfg, 0, store, backend, syncer=object())

        assert push_calls == [0, 0, 0, 0]

    def test_the_final_sweep_only_adds_the_run_dir(self, tmp_path, monkeypatch):
        """Everything else went up eagerly, so the trailing backstop sweep in
        push_after_run contributes nothing the run didn't already push."""
        cfg = E.SMOKE_MOCK
        uploads = self._run_recording_uploads(tmp_path, monkeypatch)

        run_dir = trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name)
        assert uploads[-1] == f"trajectories/runs/{run_dir.name}.tar"
        assert uploads.count(f"trajectories/runs/{run_dir.name}.tar") == 1
        # Adapters are immutable and skip on existence, so each ships once no
        # matter how many times push_adapter is called for it.
        for t in (1, 2):
            assert uploads.count(f"store/adapters/{get_weights_id(cfg, t)}.tar") == 1

    def test_the_base_bundle_grows_across_steps_not_pushed_once(
        self, tmp_path, monkeypatch
    ):
        """The base bundle keeps gaining files well past step 0: pos/neg
        extraction responses, then M_0's answers to each new dataset's
        prompts (written while computing DeltaP for that step). Treating it
        like an adapter -- pushed once and skipped forever after -- would
        strand every one of those later additions on local disk.
        """
        cfg = E.SMOKE_MOCK
        uploads = self._run_recording_uploads(tmp_path, monkeypatch)

        base = f"store/measurements/{get_weights_id(cfg, 0)}.tar"
        # More than one real upload proves the bundle keeps being reconsidered
        # after its first push; the sync ledger already drops any push that
        # would not have changed it, so every count here is a real addition.
        assert uploads.count(base) > 1


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
