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
* ``evict_all_merged`` wiped one shared ``store/merged``, so the first of two
  runs sharing a box to finish deleted the checkpoint the other was still
  evaluating.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil

import pytest

from method import experiments as E, run_trajectory, steps
from method.backends import get_backend
from method.config import Backend, HNeutralSource
from method.run_trajectory import _install_adapter, _verify_cached_adapter
from method.store import Store, _process_alive, file_sha256, get_weights_id
from method.sync import REMOTE_ENV, LocalTransport, Syncer
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


def _dead_pid() -> int:
    """A pid number that names no running process."""
    for candidate in range(4_000_000, 4_000_200):
        if not _process_alive(candidate):
            return candidate
    raise RuntimeError("found no unused pid to test with")


class TestMergedWeightsAreScopedPerProcess:
    """Both GPUs on a box run against one repo, and every run drops its merged
    weights when it finishes. Sharing one ``merged/`` directory meant the first
    to finish deleted the checkpoint the second was mid-evaluation on, killing
    it on a missing ``config.json`` and burning the step's GPU hours."""

    def _checkpoint(self, parent, wid) -> object:
        path = parent / wid
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return path

    def test_finishing_a_run_leaves_a_live_run_untouched(self, tmp_path):
        store = Store(tmp_path / "store")
        mine = self._checkpoint(store.merged, "t01-aaaaaaaaaaaaaaaa")
        # os.getppid() is the pytest launcher: a pid that is certainly alive
        # and certainly not this process, i.e. the other GPU's run.
        theirs = self._checkpoint(
            store.merged_root / str(os.getppid()), "t01-bbbbbbbbbbbbbbbb"
        )

        store.evict_all_merged()

        assert not mine.exists()
        assert theirs.exists()

    def test_orphaned_roots_are_reclaimed_but_live_ones_are_not(self, tmp_path):
        """A preempted run never reaches its own eviction, and a leaked 7B
        checkpoint is ~15GB of the box's budget."""
        store = Store(tmp_path / "store")
        dead = self._checkpoint(
            store.merged_root / str(_dead_pid()), "t01-cccccccccccccccc"
        )
        live = self._checkpoint(
            store.merged_root / str(os.getppid()), "t01-dddddddddddddddd"
        )
        mine = self._checkpoint(store.merged, "t01-eeeeeeeeeeeeeeee")

        store.evict_orphaned_merged()

        assert not dead.parent.exists()
        assert live.exists()
        assert mine.exists()

    def test_sweeping_an_untouched_store_is_a_no_op(self, tmp_path):
        Store(tmp_path / "store").evict_orphaned_merged()


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
        """Relpaths the run uploads, minus the index sidecars.

        A mutable archive is always followed by its ``.files`` index (see
        :func:`method.sync._index_relpath`); these tests are about which
        *artifacts* a run ships and when, so the sidecars are noise here.
        """
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        monkeypatch.setattr(
            Store, "for_backend", classmethod(lambda cls, backend: store)
        )
        monkeypatch.setenv(REMOTE_ENV, str(tmp_path / "remote"))

        uploads: list[str] = []
        real_upload = LocalTransport.upload

        def spy(transport, local, relpath):
            if not relpath.endswith(".files"):
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


class TestCachedLatentGainsItsNorm:
    """``h_norm`` was added after most checkpoints had been measured.

    Every one of those has a ``latent_cosine.json`` that predates the field, and
    ``compute_step_latent`` returns exactly what lands in ``trajectory.json`` --
    so a cache hit that handed the old block straight back would keep producing
    runs the backfill has to visit again, forever.
    """

    @staticmethod
    def _measure(tmp_path):
        store = Store(tmp_path)
        backend = get_backend(Backend.MOCK, dtype="float16")
        cfg = E.SMOKE_MOCK
        steps.extract_persona_vector(cfg, 0, store, backend)
        steps.measure_h_neutral(cfg, 0, store, backend)
        return cfg, store

    @staticmethod
    def _strip_h_norm(cfg, store):
        """Rewrite the cache as it looked before the field existed."""
        path = store.trait_measurement(
            get_weights_id(cfg, 0), cfg.trait, steps.Artifacts.LATENT_JSON
        )
        cached = json.loads(path.read_text())
        legacy = {
            source: {k: v for k, v in z.items() if k != "h_norm"}
            for source, z in cached.items()
        }
        path.write_text(json.dumps(legacy))
        return path, cached

    def test_a_fresh_measurement_records_it(self, tmp_path):
        cfg, store = self._measure(tmp_path)

        latents = steps.compute_step_latent(cfg, 0, store)

        assert latents["base"]["h_norm"] > 0

    def test_a_cache_predating_the_field_is_filled_in(self, tmp_path):
        cfg, store = self._measure(tmp_path)
        steps.compute_step_latent(cfg, 0, store)
        path, original = self._strip_h_norm(cfg, store)

        latents = steps.compute_step_latent(cfg, 0, store)

        assert latents["base"]["h_norm"] == pytest.approx(original["base"]["h_norm"])
        # Written back, not just returned: the next reader must not pay for it
        # again, and `latent_cosine.json` is what the backfills reconcile with.
        assert json.loads(path.read_text())["base"]["h_norm"] == pytest.approx(
            original["base"]["h_norm"]
        )

    def test_filling_in_never_rederives_z(self, tmp_path):
        """p and q keep the anchor they were measured against.

        Recomputing them would re-anchor the checkpoint onto whichever v_0 the
        store holds now, and exp3 sits on several distinct base measurements --
        so a stale-looking cache must be *added to*, never rebuilt.
        """
        cfg, store = self._measure(tmp_path)
        steps.compute_step_latent(cfg, 0, store)
        path, _ = self._strip_h_norm(cfg, store)
        planted = json.loads(path.read_text())
        planted["base"]["p"] = -0.123
        path.write_text(json.dumps(planted))

        latents = steps.compute_step_latent(cfg, 0, store)

        assert latents["base"]["p"] == pytest.approx(-0.123)

    def test_an_evicted_tensor_does_not_fail_the_step(self, tmp_path):
        """The norm is a diagnostic; no figure depends on it.

        A checkpoint whose activation bundle is no longer on this box must
        still hand back its z, for `method.backfill_h_norm` to complete later
        where the store lives.
        """
        cfg, store = self._measure(tmp_path)
        steps.compute_step_latent(cfg, 0, store)
        self._strip_h_norm(cfg, store)
        shutil.rmtree(
            store.measurement_dir(get_weights_id(cfg, 0))
            / steps.Artifacts.h_neutral("base")
        )

        latents = steps.compute_step_latent(cfg, 0, store)

        assert "h_norm" not in latents["base"]
        assert set(latents["base"]) == {"p", "q", "rho", "r"}


class TestBranchesMeasureOnlyTheirEndpoint:
    """``MeasurementLevel.ENDPOINT_BEHAVIOR``: the fan-out's cost control.

    A branch shares its whole prefix with the trunk it forked from, and the
    trunk has already measured every checkpoint in it. Re-measuring is not
    merely redundant -- each checkpoint costs a full merge-chain rebuild
    (``materialize``) before it can discover the artifact is already there. With
    144 branches in the design, doing that per branch is what the level exists
    to avoid.
    """

    @staticmethod
    def _design(tmp_path, monkeypatch):
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        monkeypatch.setattr(
            Store, "for_backend", classmethod(lambda cls, backend: store)
        )
        drivers = E.EXP2_TRUNKS["a"][:2]
        cfgs = E.build_exp2_decay_configs(
            measure_traits=("evil",),
            trunks={"a": drivers},
            probes=E.EXP2_PROBES[:1],
            local=True,
        )
        by_role = {}
        for cfg in cfgs:
            by_role.setdefault(cfg.label_map["role"], []).append(cfg)
        return store, by_role["trunk"][0], by_role["branch"]

    def test_a_branch_records_its_endpoint_and_nothing_else(
        self, tmp_path, monkeypatch
    ):
        _, _, branches = self._design(tmp_path, monkeypatch)
        branch = max(branches, key=lambda c: int(c.label_map["t"]))

        run_dir = run_trajectory.run(branch, Backend.MOCK, "float16")

        payload = json.loads((run_dir / "trajectory.json").read_text())
        assert [s["t"] for s in payload["steps"]] == [len(branch.steps)]
        endpoint = payload["steps"][0]
        assert branch.trait in endpoint["behavior"]
        # No z, no delta_p, no probes: all of them are properties of the trunk
        # checkpoint this branch left from, and the trunk records them there.
        assert "z" not in endpoint
        assert "delta_p" not in endpoint
        assert "probes" not in endpoint

    def test_a_branch_leaves_the_prefix_checkpoints_unmeasured(
        self, tmp_path, monkeypatch
    ):
        """Trained-or-reused, but never evaluated: the prefix is the trunk's job."""
        store, _, branches = self._design(tmp_path, monkeypatch)
        branch = max(branches, key=lambda c: int(c.label_map["t"]))
        t_end = len(branch.steps)

        run_trajectory.run(branch, Backend.MOCK, "float16")

        for t in range(t_end):
            wid = get_weights_id(branch, t)
            assert not store.trait_measurement(
                wid, branch.trait, steps.Artifacts.BEHAVIOR_CSV
            ).exists(), f"checkpoint {t} was measured"
        # ...while every adapter along the way still exists, so the branch did
        # walk the chain rather than skipping the training too.
        for t in range(1, t_end + 1):
            assert store.has_adapter(get_weights_id(branch, t))
        assert store.trait_measurement(
            get_weights_id(branch, t_end), branch.trait, steps.Artifacts.BEHAVIOR_CSV
        ).exists()

    def test_a_branch_reuses_the_trunks_adapters(self, tmp_path, monkeypatch):
        """The affordability claim, end to end: after the trunk has run, a
        branch off checkpoint t trains exactly one new adapter."""
        store, trunk, branches = self._design(tmp_path, monkeypatch)
        run_trajectory.run(trunk, Backend.MOCK, "float16")
        before = {p.name for p in store.adapters.iterdir()}

        branch = max(branches, key=lambda c: int(c.label_map["t"]))
        run_trajectory.run(branch, Backend.MOCK, "float16")

        after = {p.name for p in store.adapters.iterdir()}
        assert after - before == {get_weights_id(branch, len(branch.steps))}


class TestRunPullsOnlyItsOwnPrefix:
    """A run must not drag the whole remote store onto the box.

    The store accumulates every experiment's artifacts, and the hidden-state
    bundles among them are ~1GB each, so pulling all of it to run one
    trajectory buys rental disk -- for the life of the box, not just the
    transfer -- that the run never reads. Nothing is lost by scoping: ids are
    content-addressed, so a later family fetches its own prefix when it runs.
    """

    def test_foreign_checkpoints_on_the_remote_are_left_there(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        monkeypatch.setattr(
            Store, "for_backend", classmethod(lambda cls, backend: store)
        )
        monkeypatch.setenv(REMOTE_ENV, str(tmp_path / "remote"))
        cfg = E.SMOKE_MOCK

        # Another experiment's output, already on the shared remote. Pushed
        # from a separate store so it is only ever reachable via the remote.
        other = Store(tmp_path / "other-store")
        foreign = "t01-someoneelse"
        adapter = other.adapter_dir(foreign)
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_text("w", encoding="utf-8")
        measurement = other.measurement_dir(foreign)
        measurement.mkdir(parents=True)
        (measurement / "behavior.csv").write_text("rows", encoding="utf-8")
        syncer = Syncer(
            other, LocalTransport(tmp_path / "remote"), trajectories=tmp_path / "t"
        )
        syncer.push_adapter(foreign)
        syncer.push_measurement(foreign)

        run_trajectory.run(cfg, Backend.MOCK, "float16")

        assert not store.has_adapter(foreign)
        assert not store.measurement_dir(foreign).exists()
        # The run's own checkpoints did land, so this is scoping and not a
        # pull that silently stopped working.
        assert store.has_adapter(get_weights_id(cfg, 1))
