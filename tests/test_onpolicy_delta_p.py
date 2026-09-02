r"""Tests for :mod:`method.onpolicy_delta_p`, the GPU-free re-projection runner.

It computes $\Delta P$ against the axis a checkpoint drew from its own
extraction text, for the two families that view belongs to, and writes the
``trajectory.json`` the plotting layer reads. Everything it needs is already on
disk, so nothing here loads a model or a backend: the fixtures write the
tensors a measured trunk would have cached and the vector
:mod:`method.axis_refresh` would have drawn.

The store built here is deliberately *thin* -- per-layer means and no
per-sample tensors -- because that is the state a laptop can actually get a
bundle into (see :mod:`method.sparse_pull`), and the mean is the only statistic
those support. One test writes samples as well, to check the two paths agree.
"""

from __future__ import annotations

import pytest
import torch

from method import experiments as E
from method import onpolicy_delta_p as onpolicy
from method.config import PredictedSource
from method.steps import onpolicy_vector_path, predicted_dir
from method.store import Store, get_weights_id, training_sample_id
from method.utils import trajectory_run_dir
from method.visualization import decay
from method.visualization.collect import collect
from tests.test_decay import (
    PROBES,
    TRUNKS,
    build_decay,
    build_validation,
    temp_trajectories,  # noqa: F401  (autouse fixture, imported for its effect)
)

#: The layer every fixture writes at, and the width of the fake activations.
#: Small enough to eyeball, wide enough that two directions are not parallel by
#: accident.
LAYER = 20
WIDTH = 4


def onpolicy_configs(**kwargs):
    """The on-policy family over the two-trunk, three-probe fixture design."""
    return E.build_exp2_onpolicy_configs(
        measure_traits=("evil",), trunks=TRUNKS, probes=PROBES, **kwargs
    )


def regen_configs(**kwargs):
    return E.build_exp2_onpolicy_regen_configs(
        measure_traits=("evil",), trunks=TRUNKS, probes=PROBES, **kwargs
    )


def _stack(vector: list[float]) -> torch.Tensor:
    """One ``[n_layers, d]`` tensor whose only meaningful row is :data:`LAYER`."""
    stacked = torch.zeros(LAYER + 1, WIDTH)
    stacked[LAYER] = torch.tensor(vector)
    return stacked


def write_means(
    store: Store, wid: str, dp_key: str, *, target, predicted, source
) -> None:
    """The per-layer means a measured checkpoint leaves for one probe."""
    for directory, values in (
        (store.measurement_dir(wid) / "delta_p_target" / dp_key, target),
        (predicted_dir(store, wid, dp_key, source), predicted),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(_stack(values), directory / "mean_by_layer.pt")


def write_vector(store: Store, wid: str, trait: str, values) -> None:
    """What the axis-refresh sweep leaves in a checkpoint's trait bundle."""
    path = onpolicy_vector_path(store, wid, trait)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_stack(values), path)


def thin_store(tmp_path, configs, *, axis=(1.0, 0.0, 0.0, 0.0), offset=1.0) -> Store:
    r"""A store holding means and vectors for every checkpoint of ``configs``.

    Each probe's target mean is its predicted mean plus ``offset`` along the
    first coordinate, so $\Delta P$ is a known constant wherever the axis is
    the first basis vector, and known-but-different wherever it is not.
    """
    store = Store(tmp_path / "store-thin")
    for cfg in configs:
        for t in range(len(cfg.steps) + 1):
            wid = get_weights_id(cfg, t)
            write_vector(store, wid, cfg.trait, list(axis))
            for probe in cfg.probes:
                for source in PredictedSource.BASE, PredictedSource.CURRENT:
                    write_means(
                        store,
                        wid,
                        training_sample_id(probe, cfg.seed),
                        target=[offset, 2.0, 0.0, 0.0],
                        predicted=[0.0, 2.0, 0.0, 0.0],
                        source=source,
                    )
    return store


class TestProbeStats:
    def test_the_mean_is_the_difference_of_the_means_projected(self, tmp_path):
        """The projection is linear, so a bundle holding only means still
        answers for the mean exactly -- which is what makes the whole thing
        runnable off a sparse pull."""
        store = Store(tmp_path)
        write_vector(store, "w0", "evil", [3.0, 4.0, 0.0, 0.0])
        write_means(
            store,
            "w0",
            "key",
            target=[2.0, 1.0, 0.0, 0.0],
            predicted=[0.0, 0.0, 0.0, 0.0],
            source=PredictedSource.BASE,
        )

        stats = onpolicy.probe_stats(
            store, "w0", "evil", "key", layer=LAYER, predicted=PredictedSource.BASE
        )

        # (2, 1) . (3, 4) / |(3, 4)| = 10 / 5
        assert stats == {"mean": pytest.approx(2.0)}

    def test_samples_give_the_full_summary_and_the_same_mean(self, tmp_path):
        """Where the per-sample tensors survived, nothing is given up: the
        record carries the spread as well, and its mean agrees with what the
        means alone would have said."""
        store = Store(tmp_path)
        write_vector(store, "w0", "evil", [1.0, 0.0, 0.0, 0.0])
        rows = torch.zeros(3, LAYER + 1, WIDTH)
        target = rows.clone()
        target[:, LAYER, 0] = torch.tensor([1.0, 2.0, 3.0])
        write_means(
            store,
            "w0",
            "key",
            target=[2.0, 0.0, 0.0, 0.0],
            predicted=[0.0, 0.0, 0.0, 0.0],
            source=PredictedSource.BASE,
        )
        for directory, values in (
            (store.measurement_dir("w0") / "delta_p_target" / "key", target[:, LAYER]),
            (predicted_dir(store, "w0", "key", PredictedSource.BASE), rows[:, LAYER]),
        ):
            torch.save(values, directory / f"samples_layer{LAYER}.pt")

        stats = onpolicy.probe_stats(
            store, "w0", "evil", "key", layer=LAYER, predicted=PredictedSource.BASE
        )

        assert stats["mean"] == pytest.approx(2.0)
        assert stats["n"] == 3
        assert stats["std"] > 0

    def test_a_missing_draw_is_reported_as_unmeasured(self, tmp_path):
        """Not an error: a sweep meets checkpoints in whatever state the boxes
        left them, and one gap should narrow a record rather than end a run."""
        store = Store(tmp_path)
        write_means(
            store,
            "w0",
            "key",
            target=[1.0, 0.0, 0.0, 0.0],
            predicted=[0.0, 0.0, 0.0, 0.0],
            source=PredictedSource.BASE,
        )

        assert (
            onpolicy.probe_stats(
                store, "w0", "evil", "key", layer=LAYER, predicted=PredictedSource.BASE
            )
            is None
        )

    def test_missing_activations_are_reported_as_unmeasured(self, tmp_path):
        store = Store(tmp_path)
        write_vector(store, "w0", "evil", [1.0, 0.0, 0.0, 0.0])

        assert (
            onpolicy.probe_stats(
                store, "w0", "evil", "key", layer=LAYER, predicted=PredictedSource.BASE
            )
            is None
        )

    def test_it_reads_the_source_it_was_asked_for(self, tmp_path):
        """The two sources live in directories of their own, and reading the
        wrong one would silently report the frozen answers under the
        regenerated view's name."""
        store = Store(tmp_path)
        write_vector(store, "w0", "evil", [1.0, 0.0, 0.0, 0.0])
        write_means(
            store,
            "w0",
            "key",
            target=[5.0, 0.0, 0.0, 0.0],
            predicted=[0.0, 0.0, 0.0, 0.0],
            source=PredictedSource.BASE,
        )
        write_means(
            store,
            "w0",
            "key",
            target=[5.0, 0.0, 0.0, 0.0],
            predicted=[3.0, 0.0, 0.0, 0.0],
            source=PredictedSource.CURRENT,
        )

        frozen = onpolicy.probe_stats(
            store, "w0", "evil", "key", layer=LAYER, predicted=PredictedSource.BASE
        )
        regenerated = onpolicy.probe_stats(
            store, "w0", "evil", "key", layer=LAYER, predicted=PredictedSource.CURRENT
        )

        assert (frozen["mean"], regenerated["mean"]) == (
            pytest.approx(5.0),
            pytest.approx(2.0),
        )


class TestTrajectories:
    def test_it_writes_one_run_per_config_under_the_families_names(self, tmp_path):
        build_decay()
        configs = onpolicy_configs()
        store = thin_store(tmp_path, configs)

        written = onpolicy.run(store, configs)

        assert sorted(p.parent.name for p in written) == sorted(
            trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name).name
            for cfg in configs
        )

    def test_the_runs_collect_as_their_own_family(self, tmp_path):
        """The point of writing a trajectory rather than a table of its own:
        the collector finds these the way it finds every other family, so the
        figures need no second input path."""
        build_decay()
        configs = onpolicy_configs()
        onpolicy.run(thin_store(tmp_path, configs), configs)

        collected = collect(configs, group=E.EXP2_ONPOLICY)

        assert len(collected.runs) == len(configs)
        assert not collected.missing and not collected.stale

    def test_every_checkpoint_carries_the_probe_series(self, tmp_path):
        build_decay()
        configs = onpolicy_configs()
        onpolicy.run(thin_store(tmp_path, configs), configs)

        run = collect(configs, group=E.EXP2_ONPOLICY).runs[0]

        assert len(run.trajectory.steps) == len(run.config.steps) + 1
        for step in run.trajectory.steps:
            assert set(step.probes_onpolicy) == {p.dataset_id for p in PROBES}

    def test_it_records_under_the_view_its_family_measures(self, tmp_path):
        """Each family writes exactly one key. A record carrying the other
        one's would be joined into the wrong column and read as a series that
        was never measured."""
        build_decay()
        frozen_answers, regenerated = onpolicy_configs(), regen_configs()
        store = thin_store(tmp_path, frozen_answers + regenerated)
        onpolicy.run(store, frozen_answers + regenerated)

        first = collect(frozen_answers, group=E.EXP2_ONPOLICY).runs[0]
        second = collect(regenerated, group=E.EXP2_ONPOLICY_REGEN).runs[0]

        assert first.trajectory.steps[0].probes_onpolicy
        assert not first.trajectory.steps[0].probes_onpolicy_current
        assert second.trajectory.steps[0].probes_onpolicy_current
        assert not second.trajectory.steps[0].probes_onpolicy

    def test_the_regenerated_view_falls_back_to_the_base_answers_at_t0(self, tmp_path):
        """``compute_delta_p``'s rule, which this has to match or the series
        would start somewhere other than $\\Delta P_0$ for no measured reason:
        at $t = 0$ the current model *is* $M_0$."""
        build_decay()
        configs = regen_configs()
        store = thin_store(tmp_path, configs)
        cfg = configs[0]
        probe = cfg.probes[0]
        # Only the regenerated answers are moved, and only at t = 0, so a run
        # that read them there lands somewhere this assertion can see.
        torch.save(
            _stack([8.0, 0.0, 0.0, 0.0]),
            predicted_dir(
                store,
                get_weights_id(cfg, 0),
                training_sample_id(probe, cfg.seed),
                PredictedSource.CURRENT,
            )
            / "mean_by_layer.pt",
        )
        onpolicy.run(store, [cfg])

        run = collect([cfg], group=E.EXP2_ONPOLICY_REGEN).runs[0]
        at_0 = run.trajectory.steps[0].probes_onpolicy_current[probe.dataset_id]

        assert at_0["mean"] == pytest.approx(1.0)

    def test_behaviour_is_the_decay_trunks_own(self, tmp_path):
        """Copied rather than re-derived: the same checkpoint under the same
        trait, and this store may hold no judged rows to derive it from."""
        decayed = build_decay()
        configs = onpolicy_configs()
        onpolicy.run(thin_store(tmp_path, configs), configs)

        trunk = next(
            run
            for run in decayed.runs
            if run.label("role") == "trunk" and run.label("trunk") == "a"
        )
        run = collect(
            [c for c in configs if c.label_map["trunk"] == "a"],
            group=E.EXP2_ONPOLICY,
        ).runs[0]

        assert [s.behavior for s in run.trajectory.steps] == [
            s.behavior for s in trunk.trajectory.steps
        ]

    def test_a_missing_trunk_names_the_family_that_supplies_it(self, tmp_path):
        configs = onpolicy_configs()

        with pytest.raises(FileNotFoundError, match=E.EXP2_DECAY):
            onpolicy.run(thin_store(tmp_path, configs), configs)

    def test_a_dry_run_writes_nothing(self, tmp_path):
        build_decay()
        configs = onpolicy_configs()

        assert onpolicy.run(thin_store(tmp_path, configs), configs, dry_run=True) == []
        assert not (
            trajectory_run_dir(configs[0].name, configs[0].seed, configs[0].model.name)
            / "trajectory.json"
        ).exists()

    def test_an_unmeasured_checkpoint_leaves_a_gap_rather_than_a_zero(self, tmp_path):
        """ "Not measured" and "measured and small" are different claims, and
        the analysis declines to fit a series it cannot see."""
        build_decay()
        configs = onpolicy_configs()
        store = thin_store(tmp_path, configs)
        cfg = configs[0]
        onpolicy_vector_path(store, get_weights_id(cfg, 2), cfg.trait).unlink()

        onpolicy.run(store, [cfg])

        run = collect([cfg], group=E.EXP2_ONPOLICY).runs[0]
        assert run.trajectory.steps[1].probes_onpolicy
        assert run.trajectory.steps[2].probes_onpolicy == {}

    def test_it_writes_no_measurement_artifacts_into_the_store(self, tmp_path):
        """It can be handed a bundle holding means but not samples, so any
        summary it cached there would carry a mean and nothing else, and the
        next reader could not tell that from a full one."""
        build_decay()
        configs = onpolicy_configs()
        store = thin_store(tmp_path, configs)
        before = sorted(p.name for p in store.root.rglob("*") if p.is_file())

        onpolicy.run(store, configs)

        assert sorted(p.name for p in store.root.rglob("*") if p.is_file()) == before


class TestIntoTheAnalysis:
    def test_the_series_reach_the_decay_frame_as_their_own_columns(self, tmp_path):
        """End to end: written by this module, collected as a family, joined by
        ``decay_frame`` onto the rows the decay trunks supply."""
        decayed, validation = build_decay(), build_validation()
        frozen_answers, regenerated = onpolicy_configs(), regen_configs()
        store = thin_store(tmp_path, frozen_answers + regenerated)
        onpolicy.run(store, frozen_answers + regenerated)

        rows = decay.decay_frame(
            decayed,
            validation,
            [
                collect(frozen_answers, group=E.EXP2_ONPOLICY),
                collect(regenerated, group=E.EXP2_ONPOLICY_REGEN),
            ],
        )

        for column in ("delta_p_hat_onpolicy", "delta_p_full_onpolicy"):
            measured = rows[column].dropna()
            assert not measured.empty, column
            assert measured.to_list() == pytest.approx([1.0] * len(measured))

    def test_a_turned_axis_moves_the_series_it_is_read_along(self, tmp_path):
        """The measurement the family exists for: same activations, same
        answers, a persona vector drawn from other text, a different number."""
        build_decay()
        configs = onpolicy_configs()
        turned = thin_store(tmp_path / "turned", configs, axis=(0.6, 0.8, 0.0, 0.0))
        onpolicy.run(turned, configs)

        run = collect(configs, group=E.EXP2_ONPOLICY).runs[0]
        means = [
            stats["mean"]
            for step in run.trajectory.steps
            for stats in step.probes_onpolicy.values()
        ]

        assert means == pytest.approx([0.6] * len(means))


class TestSelection:
    def test_it_defaults_to_both_families(self):
        groups = {cfg.group for cfg in onpolicy.select_configs()}
        assert groups == set(onpolicy.GROUPS)

    def test_a_trait_or_trunk_narrows_it(self):
        configs = onpolicy.select_configs(traits=["evil"], trunk_names=["a"])
        assert {cfg.trait for cfg in configs} == {"evil"}
        assert {cfg.label_map["trunk"] for cfg in configs} == {"a"}

    def test_the_local_scale_is_reachable(self):
        assert all(
            cfg.name.endswith("_local") for cfg in onpolicy.select_configs(local=True)
        )
