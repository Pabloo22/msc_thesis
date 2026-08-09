"""Tests for the $z_t$ audit, :mod:`method.visualization.latent_audit`.

Every case here is built by writing fake ``trajectory.json`` files whose $z$
values are chosen to make one fault visible at a time: two runs on the same
weights recording different latents, a run left on a superseded base
``weights_id``, and a component whose drift is smaller than that disagreement.
Nothing loads a model or reads the store.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from method import experiments as E
from method.config import DatasetVersion, StepConfig, TrajectoryConfig, to_json
from method.store import get_weights_id
from method.utils import trajectory_run_dir
from method.visualization.collect import collect
from method.visualization.latent_audit import (
    BASE_CHECKPOINT,
    anchor_figure,
    anchors,
    disagreement,
    drift_figure,
    drift_table,
    figures,
    hysteresis_figure,
    latent_frame,
    noise_vs_drift,
    noise_vs_drift_figure,
    on_dominant_anchor,
    report,
    rotation_share,
)

D1 = StepConfig(dataset="hallucination", version=DatasetVersion.MISALIGNED_1)
D2 = StepConfig(dataset="mistake_opinions", version=DatasetVersion.MISALIGNED_1)
D3 = StepConfig(dataset="mistake_gsm8k", version=DatasetVersion.MISALIGNED_2)
POOL = (D1, D2, D3)

#: A latent that says "nothing has drifted", used wherever a checkpoint's own
#: values are beside the point of the test.
FLAT = {"p": -10.0, "q": -10.0, "rho": 1.0, "r": 20.0}


@pytest.fixture(autouse=True)
def temp_trajectories(tmp_path, monkeypatch):
    """Point ``trajectory_run_dir`` at a scratch root for every test here."""
    monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
    return tmp_path


def hysteresis_configs(**kwargs) -> list[TrajectoryConfig]:
    defaults = dict(
        seeds=(0,),
        measure_traits=("evil",),
        realign_traits=("evil",),
        datasets=POOL,
        normal_prefixes=(2,),
    )
    return E.build_hysteresis_configs(**{**defaults, **kwargs})


def write_run(cfg: TrajectoryConfig, *, latents=None, base_id=None, behaviors=None):
    """Write a ``trajectory.json`` whose $z$ values are given outright.

    ``latents`` is one dict per checkpoint (defaulting to :data:`FLAT`
    throughout); ``base_id`` overrides the recorded ``weights_id`` at $t = 0$,
    which is how a run predating a ``weights_key`` change is reproduced.
    """
    n = len(cfg.steps)
    latents = latents if latents is not None else [FLAT] * (n + 1)
    behaviors = behaviors if behaviors is not None else [10.0 * t for t in range(n + 1)]

    records = []
    for t in range(n + 1):
        weights_id = base_id if t == 0 and base_id else get_weights_id(cfg, t)
        record = {
            "t": t,
            "weights_id": weights_id,
            "behavior": {
                cfg.trait: behaviors[t],
                f"{cfg.trait}_std": 1.0,
                "coherence": 80.0,
                "n": 10,
            },
            "z": {"base": dict(latents[t])},
            "probes": {},
        }
        if t < n:
            record["delta_p"] = {"mean": 1.0, "std": 0.5, "n": 8}
            record["next_dataset"] = cfg.steps[t].dataset_id
        records.append(record)

    path = trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name) / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": json.loads(to_json(cfg)), "steps": records}),
        encoding="utf-8",
    )
    return path


def collected(configs):
    return collect(configs, group=E.EXP3)


# --- the tidy frame ---------------------------------------------------------


def test_latent_frame_pools_every_run_at_the_base_checkpoint():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg)

    frame = latent_frame(collected(configs))
    at_base = frame[frame["t"] == 0]

    assert len(at_base) == len(configs)
    assert set(at_base["checkpoint"]) == {BASE_CHECKPOINT}
    assert not frame["legacy_base_id"].any()


def test_latent_frame_flags_a_superseded_base_id():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg, base_id="t00-deadbeefdeadbeef" if cfg is configs[0] else None)

    frame = latent_frame(collected(configs))
    flagged = frame[frame["legacy_base_id"]]

    # A run-level fact: a legacy base id taints every checkpoint the run read
    # against it, not only the base row that recorded it.
    assert set(flagged["name"]) == {configs[0].name}
    assert set(flagged["t"]) == set(range(len(configs[0].steps) + 1))
    # The base row is still pooled with the rest: same weights either way.
    assert set(flagged[flagged["t"] == 0]["checkpoint"]) == {BASE_CHECKPOINT}


def test_latent_frame_skips_runs_without_the_requested_source():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg)

    assert latent_frame(collected(configs), source="current").empty


# --- anchors ----------------------------------------------------------------


def test_one_anchor_per_trait_when_every_run_agrees():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg)

    table = anchors(latent_frame(collected(configs)))

    assert len(table) == 1
    assert table.iloc[0]["runs"] == len(configs)
    assert not table.iloc[0]["legacy_base_id"]


def test_a_re_derived_base_measurement_shows_as_a_second_anchor():
    configs = hysteresis_configs()
    odd = {**FLAT, "p": -10.5, "q": -10.5, "r": 20.4}
    for cfg in configs:
        latents = [odd if cfg is configs[0] else FLAT] * (len(cfg.steps) + 1)
        write_run(cfg, latents=latents, base_id="t00-deadbeefdeadbeef")

    table = anchors(latent_frame(collected(configs))).set_index("runs")

    assert sorted(table.index) == [1, len(configs) - 1]
    assert table.loc[1, "p"] == pytest.approx(-10.5)
    # The superseded id was given to every run, so both anchors carry the flag.
    assert table["legacy_base_id"].all()


# --- disagreement -----------------------------------------------------------


def test_identical_weights_recording_different_z_are_reported():
    configs = hysteresis_configs()
    odd = {**FLAT, "p": -10.5, "rho": 0.99}
    for cfg in configs:
        latents = [odd if cfg is configs[0] else FLAT] * (len(cfg.steps) + 1)
        write_run(cfg, latents=latents)

    table = disagreement(latent_frame(collected(configs)))
    at_base = table[table["checkpoint"] == BASE_CHECKPOINT].iloc[0]

    assert at_base["distinct_z"] == 2
    assert at_base["p"] == pytest.approx(0.5)
    assert at_base["rho"] == pytest.approx(0.01)
    assert at_base["q"] == pytest.approx(0.0)


def test_agreeing_checkpoints_are_kept_so_the_fraction_is_readable():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg)

    table = disagreement(latent_frame(collected(configs)))

    assert len(table) > 0
    assert (table["distinct_z"] == 1).all()
    assert (table[["p", "q", "rho", "r"]] == 0).all().all()


def test_checkpoints_measured_once_are_not_reported():
    configs = hysteresis_configs(datasets=(D1,), normal_prefixes=())
    for cfg in configs:
        write_run(cfg)

    table = disagreement(latent_frame(collected(configs)))
    counts = table.set_index("checkpoint")["runs"]

    assert (counts > 1).all()


# --- noise against drift ----------------------------------------------------


def _drifting(n_steps: int, *, per_step: float) -> list[dict[str, float]]:
    return [{**FLAT, "p": FLAT["p"] + per_step * t} for t in range(n_steps + 1)]


def test_ratio_exceeds_one_when_disagreement_swamps_the_drift():
    configs = hysteresis_configs()
    for cfg in configs:
        latents = _drifting(len(cfg.steps), per_step=0.1)
        if cfg is configs[0]:  # a second measurement pass of the same weights
            latents = [{**z, "p": z["p"] + 5.0} for z in latents]
        write_run(cfg, latents=latents)

    frame = latent_frame(collected(configs))
    table = noise_vs_drift(frame).set_index("component")

    assert table.loc["p", "worst_disagreement"] == pytest.approx(5.0)
    assert table.loc["p", "ratio"] > 1
    # rho never moved and never disagreed: no drift, so no usable ratio.
    assert table.loc["rho", "worst_disagreement"] == pytest.approx(0.0)


def test_drift_is_measured_on_the_dominant_anchor_only():
    configs = hysteresis_configs()
    for cfg in configs:
        latents = _drifting(len(cfg.steps), per_step=0.1)
        if cfg is configs[0]:
            latents = [{**z, "p": z["p"] + 5.0} for z in latents]
        write_run(cfg, latents=latents)

    frame = latent_frame(collected(configs))
    table = noise_vs_drift(frame).set_index("component")

    # Three steps at +0.1 -- the outlier run's offset must not enter the drift.
    assert table.loc["p", "drift"] == pytest.approx(0.3)


# --- selecting the dominant anchor ------------------------------------------


def test_dominant_anchor_drops_whole_runs_not_single_rows():
    configs = hysteresis_configs()
    odd = {**FLAT, "p": -10.5}
    for cfg in configs:
        latents = [odd if cfg is configs[0] else FLAT] * (len(cfg.steps) + 1)
        write_run(cfg, latents=latents)

    frame = latent_frame(collected(configs))
    kept = on_dominant_anchor(frame)

    assert configs[0].name not in set(kept["name"])
    assert len(kept) == len(frame) - len(frame[frame["name"] == configs[0].name])


# --- the drift itself -------------------------------------------------------


def test_drift_table_reports_one_row_per_condition_and_checkpoint():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg)

    table = drift_table(latent_frame(collected(configs)))
    per_condition = table.groupby("condition")["t"].max()

    assert per_condition["baseline"] == 1
    assert per_condition["same"] == 3
    assert table["runs"].sum() == sum(len(cfg.steps) + 1 for cfg in configs)


def test_rotation_share_splits_q_into_representation_and_axis():
    configs = hysteresis_configs(datasets=(D1,), normal_prefixes=())
    for cfg in configs:
        # p moves by 1 per step, q by 3: two thirds of q is the axis moving.
        latents = [
            {**FLAT, "p": FLAT["p"] + t, "q": FLAT["q"] + 3.0 * t}
            for t in range(len(cfg.steps) + 1)
        ]
        write_run(cfg, latents=latents)

    table = rotation_share(latent_frame(collected(configs))).set_index("t")

    assert table.loc[1, "activation_drift"] == pytest.approx(1.0)
    assert table.loc[1, "axis_rotation"] == pytest.approx(2.0)
    assert table.loc[1, "total_q_change"] == pytest.approx(3.0)
    assert table.loc[1, "axis_share"] == pytest.approx(2 / 3)


# --- figures ----------------------------------------------------------------


def test_every_figure_builds_from_a_collected_family():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg, latents=_drifting(len(cfg.steps), per_step=0.1))

    collection = collected(configs)
    tables = report(collection)
    built = figures(tables["frame"], tables)

    assert set(built) == {
        "latent_anchors",
        "latent_noise_vs_drift",
        "latent_drift",
        "latent_hysteresis",
    }
    for figure in built.values():
        assert figure.axes


def test_figures_survive_a_family_with_one_condition_missing():
    configs = [
        cfg for cfg in hysteresis_configs() if cfg.label_map["condition"] != "diff"
    ]
    for cfg in configs:
        write_run(cfg, latents=_drifting(len(cfg.steps), per_step=0.1))

    collection = collect(configs, group=E.EXP3)
    frame = latent_frame(collection)
    drift = noise_vs_drift(frame)
    dominant = on_dominant_anchor(frame)

    assert isinstance(anchor_figure(frame, drift), object)
    assert noise_vs_drift_figure(drift).axes
    assert drift_figure(dominant, drift).axes
    assert hysteresis_figure(dominant).axes


def test_report_returns_every_table():
    configs = hysteresis_configs()
    for cfg in configs:
        write_run(cfg)

    tables = report(collected(configs))

    assert set(tables) == {
        "frame",
        "anchors",
        "disagreement",
        "noise_vs_drift",
        "drift",
        "rotation_share",
    }
    assert all(isinstance(table, pd.DataFrame) for table in tables.values())
