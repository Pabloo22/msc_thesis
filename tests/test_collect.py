"""Tests for the registry-driven run collector, :mod:`method.visualization.collect`.

These write fake ``trajectory.json`` files with *real* ``weights_id`` values
(so the staleness check is exercised for real) and assert on what the collector
and its frames make of them. Nothing here loads a model or plots anything.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from method import experiments as E
from method.config import DatasetVersion, StepConfig, TrajectoryConfig, to_json
from method.store import get_weights_id
from method.utils import base_probes_path, trajectory_run_dir
from method.visualization import collect as collect_mod
from method.visualization.collect import (
    base_probe_lookup,
    collect,
    delta_p_0_lookup,
    diversity_frame,
    hysteresis_frame,
    missing_delta_p_0,
    projection_frame,
)

D1 = StepConfig(dataset="hallucination", version=DatasetVersion.MISALIGNED_1)
D2 = StepConfig(dataset="mistake_opinions", version=DatasetVersion.MISALIGNED_1)
D3 = StepConfig(dataset="mistake_gsm8k", version=DatasetVersion.MISALIGNED_2)
POOL = (D1, D2, D3)


@pytest.fixture(autouse=True)
def temp_trajectories(tmp_path, monkeypatch):
    """Point ``trajectory_run_dir`` at a scratch root for every test here."""
    monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
    return tmp_path


def write_run(
    cfg: TrajectoryConfig, *, behaviors=None, delta_p=None, probes=None, mock=False
):
    """Write a schema-faithful ``trajectory.json`` for ``cfg``.

    ``behaviors`` is b_t for t = 0..len(steps); ``delta_p`` is one value per
    non-terminal step; ``probes`` maps ``dataset_id`` to one value per
    checkpoint. All default to simple ramps (``probes`` to nothing at all, as
    for a run saved before probing existed).
    """
    n = len(cfg.steps)
    behaviors = behaviors if behaviors is not None else [10.0 * t for t in range(n + 1)]
    delta_p = delta_p if delta_p is not None else [1.0] * n
    probes = probes or {}

    records = []
    for t in range(n + 1):
        record = {
            "t": t,
            "weights_id": get_weights_id(cfg, t),
            "behavior": {
                cfg.trait: behaviors[t],
                f"{cfg.trait}_std": 1.0,
                "coherence": 80.0,
                "n": 10,
            },
            "z": {
                "base": {"p": 0.1 + t, "q": 0.2 + t, "rho": 1.0 - 0.05 * t, "r": 30.0}
            },
            "probes": {
                dataset: {"mean": series[t], "std": 0.5, "n": 8}
                for dataset, series in probes.items()
            },
        }
        if t < n:
            record["delta_p"] = {"mean": delta_p[t], "std": 0.5, "n": 8}
            record["next_dataset"] = cfg.steps[t].dataset_id
        records.append(record)

    path = (
        trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name, mock=mock)
        / "trajectory.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": json.loads(to_json(cfg)), "steps": records}),
        encoding="utf-8",
    )
    return path


def write_base_probes(cfg: TrajectoryConfig, values: dict[str, float], *, mock=False):
    """Write a base-probe summary in the shape ``probe_base.write_summary`` uses."""
    path = base_probes_path(get_weights_id(cfg, 0), cfg.trait, mock=mock)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "base_weights_id": get_weights_id(cfg, 0),
                "trait": cfg.trait,
                "seed": cfg.seed,
                "probes": {
                    dataset: {"mean": value, "std": 0.5, "n": 8}
                    for dataset, value in values.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def hysteresis_configs(**kwargs):
    defaults = dict(
        seeds=(0,), measure_traits=("evil",), realign_traits=("evil",), datasets=POOL
    )
    return E.build_hysteresis_configs(**{**defaults, **kwargs})


# --- config metadata --------------------------------------------------------


class TestConfigMetadata:
    """``group``/``labels`` are bookkeeping and must not touch the store."""

    def test_group_and_labels_do_not_change_weights_id(self):
        base = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))[0]
        tagged = dataclasses.replace(
            base, group="something-else", labels=(("condition", "made-up"),)
        )
        assert [get_weights_id(base, t) for t in range(len(base.steps) + 1)] == [
            get_weights_id(tagged, t) for t in range(len(tagged.steps) + 1)
        ]

    def test_duplicate_label_keys_are_rejected(self):
        cfg = E.EXP1
        with pytest.raises(ValueError, match="duplicate label"):
            dataclasses.replace(cfg, labels=(("a", "1"), ("a", "2")))

    def test_label_map_reads_back_the_pairs(self):
        cfg = dataclasses.replace(E.EXP1, labels=(("condition", "same"), ("n", "2")))
        assert cfg.label_map == {"condition": "same", "n": "2"}

    def test_dataset_id_matches_the_runners_next_dataset_format(self):
        assert D3.dataset_id == "mistake_gsm8k/misaligned_2"

    def test_every_generated_config_carries_a_group(self):
        for build in (
            E.build_exp2_configs,
            E.build_hysteresis_configs,
            E.build_diversity_configs,
        ):
            for cfg in build(seeds=(0,), measure_traits=("evil",)):
                assert cfg.group in {E.EXP2, E.EXP3, E.EXP4}

    def test_exp3_and_exp4_label_every_run_with_a_condition(self):
        for cfg in hysteresis_configs():
            assert cfg.label_map["condition"] in {"baseline", "same", "diff"}
            assert cfg.label_map["dataset"] in {s.dataset_id for s in POOL}
        for cfg in E.build_diversity_configs(
            seeds=(0,), measure_traits=("evil",), realign_traits=("evil",), pool=POOL
        ):
            assert cfg.label_map["condition"] in set(
                ("baseline", "same2", "diff2", "same3", "diff3")
            )

    def test_hysteresis_baseline_has_no_realign_trait_label(self):
        """One baseline run serves every realign trait, so it cannot claim one."""
        for cfg in hysteresis_configs(realign_traits=("evil", "sycophantic")):
            if cfg.label_map["condition"] == "baseline":
                assert "realign_trait" not in cfg.label_map
            else:
                assert cfg.label_map["realign_trait"] in {"evil", "sycophantic"}


class TestRunDirQuarantine:
    MODEL = E.QWEN_7B.name

    def test_mock_runs_do_not_share_a_directory_with_real_ones(self):
        assert trajectory_run_dir("exp2_evil", 0, self.MODEL) != trajectory_run_dir(
            "exp2_evil", 0, self.MODEL, mock=True
        )

    def test_two_base_models_do_not_share_a_directory(self):
        """The same config re-pointed at another base model is a second run.

        Its adapters are already separate (the model is in ``weights_key``), so
        only the run directory could collide, and the collision would replace
        the first model's ``trajectory.json``.
        """
        assert trajectory_run_dir("exp1", 0, self.MODEL) != trajectory_run_dir(
            "exp1", 0, "meta-llama/Llama-3.1-8B-Instruct"
        )

    def test_the_model_is_named_readably_in_the_path(self):
        name = trajectory_run_dir("exp1", 3, self.MODEL).name
        assert name == "exp1_qwen2.5-7b-instruct_seed3"

    def test_the_module_is_not_shadowed_by_its_own_function(self):
        """``from method.visualization import collect`` must give the module."""
        from method.visualization import collect as imported

        assert imported is collect_mod


# --- collection -------------------------------------------------------------


class TestCollect:
    def test_loads_written_runs_and_reports_the_rest_as_missing(self):
        configs = hysteresis_configs()
        for cfg in configs[:2]:
            write_run(cfg)
        result = collect(configs, group=E.EXP3)
        assert len(result.runs) == 2
        assert len(result.missing) == len(configs) - 2
        assert not result.stale

    def test_a_config_edited_after_its_run_is_flagged_stale_not_plotted(self):
        cfg = hysteresis_configs()[0]
        write_run(cfg)
        # Same name and seed (so the same run directory), different steps: the
        # saved weights_id can no longer be what this config hashes to.
        edited = dataclasses.replace(cfg, steps=(D2, D3))
        result = collect([edited], group=E.EXP3)
        assert not result.runs
        assert len(result.stale) == 1

    def test_mock_runs_are_invisible_to_a_real_collection(self):
        cfg = hysteresis_configs()[0]
        write_run(cfg, mock=True)
        assert not collect([cfg]).runs
        assert len(collect([cfg], mock=True).runs) == 1

    def test_filter_selects_on_labels_and_trait(self):
        configs = hysteresis_configs(measure_traits=("evil", "sycophantic"))
        for cfg in configs:
            write_run(cfg)
        result = collect(configs, group=E.EXP3)
        evil = result.filter(trait="evil")
        assert evil
        assert all(run.trait == "evil" for run in evil.runs)
        baselines = result.filter(condition="baseline")
        assert all(run.label("condition") == "baseline" for run in baselines.runs)

    def test_values_lists_distinct_label_values(self):
        configs = hysteresis_configs(measure_traits=("evil", "sycophantic"))
        for cfg in configs:
            write_run(cfg)
        result = collect(configs, group=E.EXP3)
        assert set(result.values("trait")) == {"evil", "sycophantic"}
        assert set(result.values("condition")) == {"baseline", "same", "diff"}


class TestRunDeltas:
    def test_final_step_delta_is_the_last_steps_behaviour_change(self):
        cfg = hysteresis_configs()[0]
        write_run(cfg, behaviors=[5.0, 12.0])
        run = collect([cfg]).runs[0]
        assert run.final_step_delta() == pytest.approx(7.0)

    def test_total_delta_is_measured_against_the_base_model(self):
        cfg = next(c for c in hysteresis_configs() if len(c.steps) == 3)
        write_run(cfg, behaviors=[5.0, 40.0, 10.0, 35.0])
        run = collect([cfg]).runs[0]
        assert run.total_delta() == pytest.approx(30.0)
        # The last step's own contribution is a different number entirely.
        assert run.final_step_delta() == pytest.approx(25.0)


# --- frames -----------------------------------------------------------------


class TestHysteresisFrame:
    def test_value_is_the_final_step_delta_for_every_condition(self):
        configs = hysteresis_configs()
        for cfg in configs:
            # 1-step baselines end at index 1; 3-step runs end at index 3.
            n = len(cfg.steps)
            write_run(cfg, behaviors=[0.0] * n + [9.0])
        df = hysteresis_frame(collect(configs, group=E.EXP3))
        assert np.allclose(df["delta_behavior"], 9.0)

    def test_baseline_rows_are_emitted_for_every_realign_trait(self):
        """A baseline has no realign step, so it is the reference bar in
        every per-realign-trait figure."""
        configs = hysteresis_configs(realign_traits=("evil", "sycophantic"))
        for cfg in configs:
            write_run(cfg)
        df = hysteresis_frame(collect(configs, group=E.EXP3))
        baselines = df[df["condition"] == "baseline"]
        assert set(baselines["realign_trait"]) == {"evil", "sycophantic"}
        # One row per (dataset, realign_trait), from a single run each.
        assert len(baselines) == len(POOL) * 2

    def test_each_dataset_has_all_three_conditions(self):
        configs = hysteresis_configs()
        for cfg in configs:
            write_run(cfg)
        df = hysteresis_frame(collect(configs, group=E.EXP3))
        for dataset in {s.dataset_id for s in POOL}:
            present = set(df[df["dataset"] == dataset]["condition"])
            assert present == {"baseline", "same", "diff"}


class TestDiversityFrame:
    def test_value_is_the_residual_against_the_base_model(self):
        configs = E.build_diversity_configs(
            seeds=(0,), measure_traits=("evil",), realign_traits=("evil",), pool=POOL
        )
        for cfg in configs:
            n = len(cfg.steps)
            write_run(cfg, behaviors=[4.0] + [50.0] * (n - 1) + [11.0])
        df = diversity_frame(collect(configs, group=E.EXP4))
        assert np.allclose(df["delta_behavior"], 7.0)
        assert set(df["condition"]) == {"baseline", "same2", "diff2", "same3", "diff3"}


# --- Delta P_0 --------------------------------------------------------------


class TestDeltaP0:
    def test_lookup_is_keyed_per_seed_and_reads_only_step_zero(self):
        configs = hysteresis_configs(seeds=(0, 1))
        for cfg in configs:
            write_run(cfg, delta_p=[float(cfg.seed) + 1.0] * len(cfg.steps))
        runs = collect(configs, group=E.EXP3).runs
        lookup = delta_p_0_lookup(runs)
        assert set(lookup) == {("evil", 0), ("evil", 1)}
        assert set(lookup[("evil", 0)].values()) == {1.0}
        assert set(lookup[("evil", 1)].values()) == {2.0}

    def test_lookup_covers_every_dataset_some_run_starts_with(self):
        configs = hysteresis_configs()
        for cfg in configs:
            write_run(cfg)
        lookup = delta_p_0_lookup(collect(configs, group=E.EXP3).runs)
        # The three baselines each start on a different dataset in the pool.
        assert set(lookup[("evil", 0)]) == {s.dataset_id for s in POOL}

    def test_missing_delta_p_0_names_datasets_no_run_measured_at_t0(self):
        """The exp2 gap: a dataset only ever trained on mid-trajectory has no
        base-model measurement, and is dropped from the scatter."""
        exp2 = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))
        for cfg in exp2:
            write_run(cfg)
        collection = collect(exp2, group=E.EXP2)
        absent = missing_delta_p_0(collection, delta_p_0_lookup(collection.runs))
        # Only the first dataset of the 8-step trajectory gets a t=0 measurement.
        assert len(absent) == 7
        assert E._EXP2_STEPS[0].dataset_id not in absent
        assert E._EXP2_STEPS[3].dataset_id in absent

    def test_projection_frame_pairs_each_seed_against_its_own_baseline(self):
        exp2 = E.build_exp2_configs(seeds=(0, 1), measure_traits=("evil",))
        for cfg in exp2:
            write_run(cfg, delta_p=[float(cfg.seed) + 1.0] * len(cfg.steps))
        collection = collect(exp2, group=E.EXP2)
        frame = projection_frame(collection, delta_p_0_lookup(collection.runs))
        assert not frame.empty
        for seed, expected in ((0, 1.0), (1, 2.0)):
            rows = frame[frame["seed"] == seed]
            assert not rows.empty
            assert np.allclose(rows["delta_p_0"], expected)

    def test_probes_measured_at_t0_also_supply_delta_p_0(self):
        """A probe at the base checkpoint *is* a Delta P_0 measurement."""
        cfg = hysteresis_configs()[0]
        probed = D2.dataset_id
        write_run(cfg, probes={probed: [7.0] * (len(cfg.steps) + 1)})
        lookup = delta_p_0_lookup(collect([cfg]).runs)
        assert lookup[("evil", 0)][probed] == pytest.approx(7.0)

    def test_base_probe_sweep_covers_datasets_no_run_starts_with(self):
        """The gap this closes: an 8-step trajectory measures Delta P_0 for its
        first dataset only, so the other seven come from probe_base."""
        exp2 = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))
        for cfg in exp2:
            write_run(cfg)
        collection = collect(exp2, group=E.EXP2)
        assert len(missing_delta_p_0(collection, delta_p_0_lookup(collection.runs))) == 7

        write_base_probes(exp2[0], {s.dataset_id: 0.5 for s in E._EXP2_STEPS})
        lookup = delta_p_0_lookup(collection.runs)
        assert not missing_delta_p_0(collection, lookup)
        assert len(lookup[("evil", 0)]) == len(E._EXP2_STEPS)

    def test_base_probes_are_read_per_seed_not_shared(self):
        exp2 = E.build_exp2_configs(seeds=(0, 1), measure_traits=("evil",))
        for cfg in exp2:
            write_run(cfg)
        # Only seed 0 has been probed.
        write_base_probes(exp2[0], {E._EXP2_STEPS[4].dataset_id: 3.0})
        lookup = base_probe_lookup(collect(exp2, group=E.EXP2).runs)
        assert ("evil", 0) in lookup
        assert ("evil", 1) not in lookup

    def test_a_missing_base_probe_file_is_not_an_error(self):
        exp2 = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))
        for cfg in exp2:
            write_run(cfg)
        assert base_probe_lookup(collect(exp2, group=E.EXP2).runs) == {}

    def test_projection_frame_is_empty_without_any_baseline(self):
        exp2 = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))
        for cfg in exp2:
            write_run(cfg)
        frame = projection_frame(collect(exp2, group=E.EXP2), {})
        assert frame.empty
        assert "delta_p_0" in frame.columns
