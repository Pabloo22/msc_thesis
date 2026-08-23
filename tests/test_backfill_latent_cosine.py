"""Unit tests for :mod:`method.backfill_latent_cosine`.

The conversion is a rescaling of numbers that are already published, so its
contract is unusually strict: it must divide by exactly the norm the value was
computed against, must leave everything that is not ``p`` or ``q`` alone, must
never half-convert a run, and must be safe to run twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from method import anchor_noise
from method.backfill_latent_cosine import backfill, convert_run
from method.latent import CONVENTION, LEGACY_CONVENTION
from method.steps import Artifacts
from method.store import Store

LAYER = 2


def write_h_neutral(store: Store, wid: str, vector: torch.Tensor) -> None:
    """Plant a checkpoint's mean activation tensor in the store at ``LAYER``."""
    out = store.measurement_dir(wid) / Artifacts.h_neutral("base") / "mean_by_layer.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    by_layer = torch.zeros(LAYER + 1, vector.numel())
    by_layer[LAYER] = vector
    torch.save(by_layer, out)


#: The z block every checkpoint gets unless a test asks for another. The
#: values are the order of magnitude the real trunks recorded under the old
#: convention, so a test that forgets to divide is obvious.
LEGACY_Z = {"p": -16.0, "q": -12.0, "rho": 0.98, "r": 27.0}


def write_run(
    root: Path,
    name: str,
    wids: list[str],
    *,
    z: dict[str, dict[str, float]] | None = None,
    convention: str | None = None,
) -> Path:
    """A ``trajectory.json`` whose every checkpoint carries ``z``.

    ``z`` is the whole block, keyed by ``h_neutral`` source, so ``z={}`` writes
    a branch endpoint -- one that measured only behaviour.
    """
    payload: dict = {
        "config": {
            "name": name,
            "trait": "evil",
            "seed": 0,
            "model": {"name": "qwen", "layer": LAYER},
        },
        "steps": [
            {
                "t": t,
                "weights_id": wid,
                "behavior": {"evil": 10.0},
                "z": {"base": dict(LEGACY_Z)} if z is None else z,
                "delta_p": {"mean": 4.25, "n": 10},
            }
            for t, wid in enumerate(wids)
        ],
    }
    if convention is not None:
        payload["z_convention"] = convention
    path = root / name / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store")


class TestConvertRun:
    def test_divides_p_and_q_by_the_activation_norm(
        self, tmp_path: Path, store: Store
    ) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))  # norm 5
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)

        z = read(path)["steps"][0]["z"]["base"]
        assert z["p"] == pytest.approx(-16.0 / 5.0)
        assert z["q"] == pytest.approx(-12.0 / 5.0)

    def test_leaves_rho_r_and_delta_p_untouched(
        self, tmp_path: Path, store: Store
    ) -> None:
        # rho was always a cosine and r is a persona-vector length; DeltaP keeps
        # the projection convention entirely. If any of these move, the change
        # has leaked out of z.
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)

        step = read(path)["steps"][0]
        assert step["z"]["base"]["rho"] == pytest.approx(0.98)
        assert step["z"]["base"]["r"] == pytest.approx(27.0)
        assert step["delta_p"] == {"mean": 4.25, "n": 10}

    def test_uses_each_checkpoints_own_norm(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))  # norm 5
        write_h_neutral(store, "w1", torch.tensor([0.0, 2.0]))  # norm 2
        path = write_run(tmp_path / "runs", "run", ["w0", "w1"])

        backfill(tmp_path / "runs", store)

        steps = read(path)["steps"]
        assert steps[0]["z"]["base"]["p"] == pytest.approx(-16.0 / 5.0)
        assert steps[1]["z"]["base"]["p"] == pytest.approx(-16.0 / 2.0)

    def test_stamps_the_convention(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)

        assert read(path)["z_convention"] == CONVENTION

    def test_is_idempotent(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)
        second = backfill(tmp_path / "runs", store)

        assert second.already == 1
        assert second.converted == 0
        assert read(path)["steps"][0]["z"]["base"]["p"] == pytest.approx(-16.0 / 5.0)

    def test_an_unreachable_checkpoint_leaves_the_whole_run_alone(
        self, tmp_path: Path, store: Store
    ) -> None:
        # Half a trajectory in cosines and half in projections would plot as a
        # step change in the data rather than as the missing tensor it is.
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0", "w1"])

        report = backfill(tmp_path / "runs", store)

        assert report.unreachable == [path]
        assert report.converted == 0
        payload = read(path)
        assert payload["steps"][0]["z"]["base"]["p"] == pytest.approx(-16.0)
        assert "z_convention" not in payload

    def test_a_dry_run_writes_nothing(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])

        report = backfill(tmp_path / "runs", store, dry_run=True)

        assert report.updated == [path]
        assert read(path)["steps"][0]["z"]["base"]["p"] == pytest.approx(-16.0)

    def test_a_branch_endpoint_is_marked_without_needing_the_store(
        self, tmp_path: Path, store: Store
    ) -> None:
        # Branches record only b, so there is nothing to convert -- but leaving
        # them unmarked would report them as stale on every future pass.
        path = write_run(tmp_path / "runs", "branch", ["w0"], z={})

        report = backfill(tmp_path / "runs", store)

        assert report.without_z == 1
        assert read(path)["z_convention"] == CONVENTION

    def test_a_run_predating_the_format_is_skipped(
        self, tmp_path: Path, store: Store
    ) -> None:
        path = tmp_path / "runs" / "old" / "trajectory.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"steps": []}), encoding="utf-8")

        report = backfill(tmp_path / "runs", store)

        assert report.skipped == [path]
        assert "z_convention" not in read(path)

    def test_an_already_converted_run_is_not_divided_twice(
        self, tmp_path: Path, store: Store
    ) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(
            tmp_path / "runs",
            "run",
            ["w0"],
            z={"base": {"p": -0.32, "q": -0.24, "rho": 0.98, "r": 27.0}},
            convention=CONVENTION,
        )

        report = backfill(tmp_path / "runs", store)

        assert report.already == 1
        assert read(path)["steps"][0]["z"]["base"]["p"] == pytest.approx(-0.32)

    def test_an_explicit_legacy_marker_is_converted(
        self, tmp_path: Path, store: Store
    ) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"], convention=LEGACY_CONVENTION)

        backfill(tmp_path / "runs", store)

        assert read(path)["steps"][0]["z"]["base"]["p"] == pytest.approx(-16.0 / 5.0)


class TestReport:
    def test_counts_every_converted_z_block(self, tmp_path: Path, store: Store) -> None:
        for wid in ("w0", "w1", "w2"):
            write_h_neutral(store, wid, torch.tensor([3.0, 4.0]))
        write_run(tmp_path / "runs", "a", ["w0", "w1"])
        write_run(tmp_path / "runs", "b", ["w2"])

        report = backfill(tmp_path / "runs", store)

        assert len(report.updated) == 2
        assert report.converted == 3
        assert "3 z value(s) rescaled" in report.summary()

    def test_convert_run_accumulates_into_a_shared_report(
        self, tmp_path: Path, store: Store
    ) -> None:
        from method.backfill_latent_cosine import Report as ReportType

        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        report = ReportType()
        for name in ("a", "b"):
            convert_run(write_run(tmp_path / "runs", name, ["w0"]), store, report)

        assert report.converted == 2


# --- anchor-noise summaries ------------------------------------------------


def write_replicate_h_neutral(
    store: Store, wid: str, replicate: int, vector: torch.Tensor
) -> None:
    """Plant one replicate's own mean activation tensor for a checkpoint."""
    out = (
        anchor_noise.shared_dir(store, wid, replicate)
        / Artifacts.h_neutral("base")
        / "mean_by_layer.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    by_layer = torch.zeros(LAYER + 1, vector.numel())
    by_layer[LAYER] = vector
    torch.save(by_layer, out)


#: p at (t=0, t=2) for the production replicate of the summary fixture, in
#: projection units. Paired below with norms that differ between the two
#: checkpoints, which is what makes the drift more than a rescale.
FIXTURE_P = {0: -16.0, 2: -15.0}


def write_anchor_summary(root: Path, *, replicates=(0, 1)) -> Path:
    """A summary shaped like the one ``anchor_noise.write_summary`` writes."""
    latents = [
        {
            "trait": "evil",
            "t": t,
            "weights_id": f"w{t}",
            "replicate": replicate,
            "p": FIXTURE_P[t] + replicate,
            "q": FIXTURE_P[t] + replicate,
            "rho": 1.0 if t == 0 else 0.97,
            "r": 27.0,
        }
        for t in (0, 2)
        for replicate in replicates
    ]
    payload = {
        "base_weights_id": "w0",
        "label": "trunk_a",
        "seed": 0,
        "model": {"name": "qwen", "layer": LAYER},
        "traits": ["evil"],
        "h_neutral_source": "base",
        "n_neutral": 500,
        "steps": ["evil/normal", "evil/normal"],
        "latents": latents,
        "spread": [],
        "against_drift": [],
    }
    path = root / "anchor_noise" / "w0_trunk_a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class TestAnchorNoiseSummary:
    def test_each_row_uses_its_own_replicates_norm(
        self, tmp_path: Path, store: Store
    ) -> None:
        # A replicate re-derives h_neutral from scratch, which is the whole
        # point of the sweep, so it cannot borrow the production norm.
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([4.0, 0.0]))
        write_replicate_h_neutral(store, "w0", 1, torch.tensor([2.0, 0.0]))
        write_replicate_h_neutral(store, "w2", 0, torch.tensor([2.0, 0.0]))
        write_replicate_h_neutral(store, "w2", 1, torch.tensor([8.0, 0.0]))
        path = write_anchor_summary(tmp_path / "runs")

        backfill(tmp_path / "runs", store)

        rows = {(r["t"], r["replicate"]): r for r in read(path)["latents"]}
        assert rows[(0, 0)]["p"] == pytest.approx(-16.0 / 4.0)
        assert rows[(0, 1)]["p"] == pytest.approx(-15.0 / 2.0)
        assert rows[(2, 0)]["p"] == pytest.approx(-15.0 / 2.0)
        assert rows[(2, 1)]["p"] == pytest.approx(-14.0 / 8.0)

    def test_derived_tables_are_recomputed_not_rescaled(
        self, tmp_path: Path, store: Store
    ) -> None:
        # The drift is a difference between two checkpoints whose norms differ,
        # so dividing the old drift by any single number gets it wrong -- here
        # the sign itself flips.
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([4.0, 0.0]))
        write_replicate_h_neutral(store, "w0", 1, torch.tensor([4.0, 0.0]))
        write_replicate_h_neutral(store, "w2", 0, torch.tensor([2.0, 0.0]))
        write_replicate_h_neutral(store, "w2", 1, torch.tensor([2.0, 0.0]))
        path = write_anchor_summary(tmp_path / "runs")

        backfill(tmp_path / "runs", store)

        drift = {r["component"]: r for r in read(path)["against_drift"]}
        # -16/4 = -4.0 at t=0, -15/2 = -7.5 at t=2.
        assert drift["p"]["drift"] == pytest.approx(-3.5)
        assert drift["p"]["drift"] < 0 < (FIXTURE_P[2] - FIXTURE_P[0])

    def test_spread_is_rebuilt_from_the_converted_rows(
        self, tmp_path: Path, store: Store
    ) -> None:
        for wid, norm in (("w0", 4.0), ("w2", 2.0)):
            for replicate in (0, 1):
                write_replicate_h_neutral(
                    store, wid, replicate, torch.tensor([norm, 0.0])
                )
        path = write_anchor_summary(tmp_path / "runs")

        backfill(tmp_path / "runs", store)

        spread = {
            (r["t"], r["component"]): r
            for r in read(path)["spread"]
            if r["trait"] == "evil"
        }
        # Replicates 0 and 1 sit at -16/4 and -15/4 at t=0: sd(ddof=1) of a
        # two-point sample is their gap over sqrt(2).
        assert spread[(0, "p")]["sigma_level"] == pytest.approx(0.25 / 2**0.5)
        assert spread[(0, "p")]["production"] == pytest.approx(-4.0)

    def test_rho_and_r_survive_the_rebuild(self, tmp_path: Path, store: Store) -> None:
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([4.0, 0.0]))
        write_replicate_h_neutral(store, "w0", 1, torch.tensor([4.0, 0.0]))
        write_replicate_h_neutral(store, "w2", 0, torch.tensor([2.0, 0.0]))
        write_replicate_h_neutral(store, "w2", 1, torch.tensor([2.0, 0.0]))
        path = write_anchor_summary(tmp_path / "runs")

        backfill(tmp_path / "runs", store)

        rows = read(path)["latents"]
        assert all(r["r"] == pytest.approx(27.0) for r in rows)
        assert all(r["rho"] == pytest.approx(1.0) for r in rows if r["t"] == 0)

    def test_a_missing_replicate_tensor_leaves_the_summary_alone(
        self, tmp_path: Path, store: Store
    ) -> None:
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([4.0, 0.0]))
        # replicate 1 and checkpoint w2 never planted.
        path = write_anchor_summary(tmp_path / "runs")

        report = backfill(tmp_path / "runs", store)

        assert report.unreachable == [path]
        assert report.anchor_rows == 0
        payload = read(path)
        assert payload["latents"][0]["p"] == pytest.approx(-16.0)
        assert "z_convention" not in payload

    def test_is_idempotent(self, tmp_path: Path, store: Store) -> None:
        for wid in ("w0", "w2"):
            for replicate in (0, 1):
                write_replicate_h_neutral(
                    store, wid, replicate, torch.tensor([4.0, 0.0])
                )
        path = write_anchor_summary(tmp_path / "runs")

        first = backfill(tmp_path / "runs", store)
        second = backfill(tmp_path / "runs", store)

        assert first.anchor_rows == 4
        assert second.anchor_rows == 0
        assert second.already == 1
        assert read(path)["latents"][0]["p"] == pytest.approx(-4.0)

    def test_the_result_is_json_serialisable(
        self, tmp_path: Path, store: Store
    ) -> None:
        # The tables come back through pandas, whose numpy scalars json cannot
        # encode; the write must not depend on that conversion happening by
        # luck of a pandas version.
        for wid in ("w0", "w2"):
            for replicate in (0, 1):
                write_replicate_h_neutral(
                    store, wid, replicate, torch.tensor([4.0, 0.0])
                )
        path = write_anchor_summary(tmp_path / "runs")

        backfill(tmp_path / "runs", store)

        payload = read(path)
        assert payload["spread"] and payload["against_drift"]
        json.dumps(payload)
