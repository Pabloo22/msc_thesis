r"""Tests for DeltaP probing: measuring a *fixed* dataset at every checkpoint.

The runner otherwise measures DeltaP only for the dataset a step is about to
train on, which yields the diagonal of a (checkpoint x dataset) table. The
figures need more of that table: a whole row at t=0 for the projection scatter's
$\Delta P_0$ series, and whole columns for the drift line. These tests cover the
machinery that fills it -- ``TrajectoryConfig.probes``, ``steps.measure_probes``
and :mod:`method.probe_base` -- plus the claim the design rests on, that a probe
naming a dataset the trajectory also trains on costs nothing extra.

Everything here runs on the mock backend: no model, no GPU, no judge.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from method import experiments as E, probe_base, steps
from method.backends import get_backend
from method.config import Backend, DatasetVersion, StepConfig
from method.store import Store, get_weights_id
from method.utils import base_probes_path
from method.visualization import collect as collect_mod
from method.visualization.schema import StepRecord, Trajectory, probe_matrix

GSM8K = StepConfig(
    dataset="mistake_gsm8k",
    version=DatasetVersion.MISALIGNED_2,
    n_examples=8,
    train=E.LOCAL_TRAIN,
)
EVIL = StepConfig(
    dataset="evil",
    version=DatasetVersion.NORMAL,
    n_examples=8,
    train=E.LOCAL_TRAIN,
)


@pytest.fixture
def backend():
    return get_backend(Backend.MOCK, dtype="float16")


def make_cfg(*, steps_=(GSM8K,), probes=()):
    return dataclasses.replace(E.SMOKE_MOCK, steps=steps_, probes=probes)


def prepared(cfg, store, backend, t=0):
    """A checkpoint with the persona vector DeltaP projects onto already extracted."""
    steps.extract_persona_vector(cfg, t, store, backend)


# --- the config field -------------------------------------------------------


class TestProbesAreMeasurementOnly:
    def test_probes_do_not_change_any_weights_id(self):
        """Adding probes to an already-run config must not force a retrain."""
        plain = make_cfg()
        probed = make_cfg(probes=(EVIL,))
        assert [get_weights_id(plain, t) for t in range(len(plain.steps) + 1)] == [
            get_weights_id(probed, t) for t in range(len(probed.steps) + 1)
        ]

    def test_probes_are_excluded_from_the_weights_key(self):
        assert "probes" not in make_cfg(probes=(EVIL,)).weights_key(1)

    def test_duplicate_probe_datasets_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate probe datasets"):
            make_cfg(probes=(EVIL, EVIL))

    def test_exp2_probes_are_a_subset_of_its_steps(self):
        """The defaults probe datasets exp2 actually trains on, so each probe
        series has a step whose effect it can be read against."""
        cfg = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))[0]
        step_ids = {s.dataset_id for s in cfg.steps}
        assert cfg.probes
        assert {p.dataset_id for p in cfg.probes} <= step_ids


# --- measure_probes ---------------------------------------------------------


class TestMeasureProbes:
    def test_probes_a_dataset_the_trajectory_never_trains_on(self, tmp_path, backend):
        """The whole point: DeltaP for a dataset this step is not training on."""
        store = Store(tmp_path)
        cfg = make_cfg(steps_=(GSM8K,), probes=(EVIL,))
        prepared(cfg, store, backend)

        result = steps.measure_probes(cfg, 0, store, backend)

        assert set(result) == {EVIL.dataset_id}
        assert result[EVIL.dataset_id]["n"] > 0

    def test_probe_and_training_step_share_one_artifact(self, tmp_path, backend):
        """A probe naming the step's own dataset must not measure it twice.

        This is what makes probing the trajectory's own datasets cheap: both
        resolve to the same ``training_sample_id`` and therefore the same
        cached DeltaP file.
        """
        store = Store(tmp_path)
        cfg = make_cfg(steps_=(GSM8K,), probes=(GSM8K,))
        prepared(cfg, store, backend)

        probed = steps.measure_probes(cfg, 0, store, backend)
        train_file = steps.sample_training_file(
            GSM8K, cfg.seed, tmp_path / "train.jsonl", store
        )
        action = steps.compute_delta_p(
            cfg, 0, store, backend, train_file, steps.training_sample_id(GSM8K, cfg.seed)
        )

        assert probed[GSM8K.dataset_id] == action
        written = list(
            store.trait_measurement_dir(get_weights_id(cfg, 0), cfg.trait).glob(
                "delta_p_*.json"
            )
        )
        assert len(written) == 1, f"expected one DeltaP artifact, got {written}"

    def test_two_probes_do_not_collide(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(probes=(GSM8K, EVIL))
        prepared(cfg, store, backend)

        result = steps.measure_probes(cfg, 0, store, backend)

        assert set(result) == {GSM8K.dataset_id, EVIL.dataset_id}
        assert result[GSM8K.dataset_id] != result[EVIL.dataset_id]

    def test_the_same_probe_differs_between_checkpoints(self, tmp_path, backend):
        """A probe is a fixed dataset against a *moving* model, so the value has
        to be free to change with t -- otherwise the drift plot is a flat line
        by construction."""
        store = Store(tmp_path)
        cfg = make_cfg(steps_=(GSM8K,), probes=(EVIL,))
        for t in (0, 1):
            if t:  # a checkpoint at t=1 needs the adapter that reaches it
                adapter = store.adapter_dir(get_weights_id(cfg, 1))
                adapter.mkdir(parents=True, exist_ok=True)
                (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            prepared(cfg, store, backend, t=t)

        at_0 = steps.measure_probes(cfg, 0, store, backend)
        at_1 = steps.measure_probes(cfg, 1, store, backend)

        assert at_0[EVIL.dataset_id] != at_1[EVIL.dataset_id]

    def test_explicit_probes_override_the_config(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(probes=())
        prepared(cfg, store, backend)

        result = steps.measure_probes(cfg, 0, store, backend, probes=(EVIL,))

        assert set(result) == {EVIL.dataset_id}


# --- probe_base -------------------------------------------------------------


class TestProbeBase:
    def test_measures_every_requested_dataset_at_the_base_model(
        self, tmp_path, backend
    ):
        store = Store(tmp_path)
        cfg = make_cfg()

        result = probe_base.probe_base(cfg, (GSM8K, EVIL), store, backend)

        assert set(result) == {GSM8K.dataset_id, EVIL.dataset_id}

    def test_summary_round_trips_into_the_collector(
        self, tmp_path, monkeypatch, backend
    ):
        """The writer and the reader live in different modules; this is the only
        thing that keeps their file format in agreement."""
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        store = Store(tmp_path / "store")
        cfg = make_cfg()

        result = probe_base.probe_base(cfg, (GSM8K, EVIL), store, backend)
        probe_base.write_summary(cfg, result, mock=True)

        path = base_probes_path(get_weights_id(cfg, 0), cfg.trait, mock=True)
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["seed"] == cfg.seed and payload["trait"] == cfg.trait

        run = collect_mod.Run(
            config=cfg,
            trajectory=Trajectory(cfg.name, cfg.trait, cfg.seed, ()),
            path=path,
            mock=True,
        )
        lookup = collect_mod.base_probe_lookup([run])
        assert lookup[(cfg.trait, cfg.seed)][GSM8K.dataset_id] == pytest.approx(
            result[GSM8K.dataset_id]["mean"]
        )

    def test_the_probe_set_covers_every_dataset_the_experiments_train_on(self):
        """If a dataset any experiment trains on were absent here, its points
        would be dropped from the projection scatter with no error."""
        probed = {s.dataset_id for s in E.all_probe_datasets()}
        for build in E.GROUP_BUILDERS.values():
            for cfg in build(seeds=(0,), measure_traits=("evil",)):
                for step in cfg.steps:
                    assert step.dataset_id in probed

    def test_summary_is_quarantined_by_mock_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
        real = base_probes_path("t00-abc", "evil", mock=False)
        mock = base_probes_path("t00-abc", "evil", mock=True)
        assert real != mock


# --- reading probes back out ------------------------------------------------


def record(t, probes):
    return StepRecord(
        t=t,
        weights_id=f"t{t:02d}-x",
        behavior={"evil": 10.0 * t},
        z={"base": {"p": 0.0, "q": 0.0, "rho": 1.0, "r": 30.0}},
        probes=probes,
    )


def trajectory(series, *, seed=0, name="t"):
    """A trajectory whose checkpoint ``t`` probes each dataset at ``series[d][t]``."""
    n = len(next(iter(series.values())))
    return Trajectory(
        name=name,
        trait="evil",
        seed=seed,
        steps=tuple(
            record(t, {d: {"mean": values[t], "n": 8} for d, values in series.items()})
            for t in range(n)
        ),
    )


class TestProbeAccessors:
    def test_probe_series_follows_one_dataset_across_checkpoints(self):
        traj = trajectory({"a/normal": [1.0, 2.0, 3.0]})
        assert traj.probe_series("a/normal") == [1.0, 2.0, 3.0]

    def test_probed_datasets_is_the_intersection_over_checkpoints(self):
        """A dataset probed at only some checkpoints has no full series, so it
        must not be offered as one."""
        traj = Trajectory(
            name="t",
            trait="evil",
            seed=0,
            steps=(
                record(0, {"a/normal": {"mean": 1.0}, "b/normal": {"mean": 9.0}}),
                record(1, {"a/normal": {"mean": 2.0}}),
            ),
        )
        assert traj.probed_datasets() == ["a/normal"]

    def test_probe_series_raises_rather_than_returning_a_short_series(self):
        traj = Trajectory(
            name="t",
            trait="evil",
            seed=0,
            steps=(record(0, {"a/normal": {"mean": 1.0}}), record(1, {})),
        )
        with pytest.raises(KeyError, match="no probe for"):
            traj.probe_series("a/normal")

    def test_a_trajectory_saved_before_probing_has_none(self):
        payload = {
            "t": 0,
            "weights_id": "t00-x",
            "behavior": {"evil": 1.0},
            "z": {"base": {"p": 0.0, "q": 0.0, "rho": 1.0, "r": 1.0}},
        }
        assert StepRecord.from_dict(payload).probes == {}

    def test_probe_matrix_stacks_seeds_and_skips_unprobed_runs(self):
        probed = [
            trajectory({"a/normal": [1.0, 2.0]}, seed=0),
            trajectory({"a/normal": [3.0, 4.0]}, seed=1),
        ]
        unprobed = Trajectory("other", "evil", 2, (record(0, {}), record(1, {})))

        rows = probe_matrix([*probed, unprobed], "a/normal")

        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_probe_matrix_respects_the_requested_statistic(self):
        traj = Trajectory(
            name="t",
            trait="evil",
            seed=0,
            steps=(
                record(0, {"a/normal": {"mean": 1.0, "median": 10.0}}),
                record(1, {"a/normal": {"mean": 2.0, "median": 20.0}}),
            ),
        )
        assert probe_matrix([traj], "a/normal", stat="median") == [[10.0, 20.0]]
