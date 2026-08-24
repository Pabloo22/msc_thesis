"""Unit tests for :mod:`method.backfill_h_norm`.

The script exists to make the cosine convention reversible, so its contract is
the mirror image of :mod:`method.backfill_latent_cosine`'s: it must record
exactly the norm each block was divided by, must add and never alter, must
never half-fill a file, and must be safe to run twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from method import anchor_noise
from method.backfill_h_norm import (
    Report,
    backfill,
    convert_anchor_noise,
    convert_run,
)
from method.latent import CONVENTION, H_NORM, LEGACY_CONVENTION
from method.steps import Artifacts
from method.store import Store

LAYER = 2

#: The z block every checkpoint gets unless a test asks for another: cosines,
#: since the conversion has already run everywhere this script is aimed.
Z = {"p": -0.32, "q": -0.24, "rho": 0.98, "r": 27.0}


def write_h_neutral(store: Store, wid: str, vector: torch.Tensor) -> None:
    """Plant a checkpoint's mean activation tensor in the store at ``LAYER``."""
    out = store.measurement_dir(wid) / Artifacts.h_neutral("base") / "mean_by_layer.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    by_layer = torch.zeros(LAYER + 1, vector.numel())
    by_layer[LAYER] = vector
    torch.save(by_layer, out)


def write_replicate_h_neutral(
    store: Store, wid: str, replicate: int, vector: torch.Tensor
) -> None:
    """The same tensor, in the quarantined directory one replicate draws into."""
    out = (
        anchor_noise.shared_dir(store, wid, replicate)
        / Artifacts.h_neutral("base")
        / "mean_by_layer.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    by_layer = torch.zeros(LAYER + 1, vector.numel())
    by_layer[LAYER] = vector
    torch.save(by_layer, out)


def write_run(
    root: Path,
    name: str,
    wids: list[str],
    *,
    z: dict[str, dict[str, float]] | None = None,
    convention: str = CONVENTION,
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
        "z_convention": convention,
        "steps": [
            {
                "t": t,
                "weights_id": wid,
                "behavior": {"evil": 10.0},
                "z": {"base": dict(Z)} if z is None else z,
                "delta_p": {"mean": 4.25, "n": 10},
            }
            for t, wid in enumerate(wids)
        ],
    }
    path = root / name / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_anchor_noise(root: Path, rows: list[dict]) -> Path:
    """An anchor-noise summary carrying ``rows`` and two derived tables."""
    payload = {
        "model": {"name": "qwen", "layer": LAYER},
        "h_neutral_source": "base",
        "latents": rows,
        "spread": [{"trait": "evil", "t": 0, "component": "p", "sigma_level": 0.01}],
        "against_drift": [{"trait": "evil", "component": "p", "ratio_delta": 0.4}],
    }
    path = root / "anchor_noise" / "sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def row(wid: str, replicate: int, t: int = 0, **overrides: float) -> dict:
    return {
        "trait": "evil",
        "t": t,
        "weights_id": wid,
        "replicate": replicate,
        **Z,
        **overrides,
    }


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store")


class TestConvertRun:
    def test_records_the_activation_norm(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))  # norm 5
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)

        assert read(path)["steps"][0]["z"]["base"][H_NORM] == pytest.approx(5.0)

    def test_adds_without_touching_anything_else(
        self, tmp_path: Path, store: Store
    ) -> None:
        # The whole point of filling in rather than re-deriving: p and q keep
        # the anchor they were measured against. If any of these move, the
        # script has started recomputing z instead of describing it.
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)

        step = read(path)["steps"][0]
        assert {k: v for k, v in step["z"]["base"].items() if k != H_NORM} == Z
        assert step["delta_p"] == {"mean": 4.25, "n": 10}

    def test_the_norm_recovers_the_projection_convention(
        self, tmp_path: Path, store: Store
    ) -> None:
        """``p * h_norm`` is the number the old convention reported.

        The reason the field is worth recording at all, so it is worth pinning
        rather than left implied by the arithmetic.
        """
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))  # norm 5
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)

        z = read(path)["steps"][0]["z"]["base"]
        assert z["p"] * z[H_NORM] == pytest.approx(Z["p"] * 5.0)

    def test_uses_each_checkpoints_own_norm(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))  # norm 5
        write_h_neutral(store, "w1", torch.tensor([0.0, 2.0]))  # norm 2
        path = write_run(tmp_path / "runs", "run", ["w0", "w1"])

        backfill(tmp_path / "runs", store)

        steps = read(path)["steps"]
        assert steps[0]["z"]["base"][H_NORM] == pytest.approx(5.0)
        assert steps[1]["z"]["base"][H_NORM] == pytest.approx(2.0)

    def test_fills_every_h_neutral_source(self, tmp_path: Path, store: Store) -> None:
        """A BOTH run carries two blocks per checkpoint, on two tensors."""
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        current = (
            store.measurement_dir("w0")
            / Artifacts.h_neutral("current")
            / "mean_by_layer.pt"
        )
        current.parent.mkdir(parents=True, exist_ok=True)
        by_layer = torch.zeros(LAYER + 1, 2)
        by_layer[LAYER] = torch.tensor([0.0, 2.0])
        torch.save(by_layer, current)

        path = write_run(
            tmp_path / "runs",
            "run",
            ["w0"],
            z={"base": dict(Z), "current": dict(Z)},
        )
        backfill(tmp_path / "runs", store)

        z = read(path)["steps"][0]["z"]
        assert z["base"][H_NORM] == pytest.approx(5.0)
        assert z["current"][H_NORM] == pytest.approx(2.0)

    def test_is_idempotent(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])

        backfill(tmp_path / "runs", store)
        second = backfill(tmp_path / "runs", store)

        assert second.already == 1
        assert second.filled == 0
        assert read(path)["steps"][0]["z"]["base"][H_NORM] == pytest.approx(5.0)

    def test_a_partly_filled_run_is_completed(
        self, tmp_path: Path, store: Store
    ) -> None:
        """A pass interrupted between checkpoints leaves a run to finish.

        The field is its own marker, so resuming means filling the blocks that
        lack it and leaving the ones that already have theirs -- including one
        whose recorded norm no longer matches the tensor, which would mean the
        store moved under a published number and must not be overwritten here.
        """
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))  # norm 5
        write_h_neutral(store, "w1", torch.tensor([0.0, 2.0]))  # norm 2
        path = write_run(tmp_path / "runs", "run", ["w0", "w1"])
        payload = read(path)
        payload["steps"][0]["z"]["base"][H_NORM] = 99.0
        path.write_text(json.dumps(payload), encoding="utf-8")

        report = backfill(tmp_path / "runs", store)

        steps = read(path)["steps"]
        assert report.filled == 1
        assert steps[0]["z"]["base"][H_NORM] == 99.0
        assert steps[1]["z"]["base"][H_NORM] == pytest.approx(2.0)

    def test_an_unreachable_checkpoint_leaves_the_whole_run_alone(
        self, tmp_path: Path, store: Store
    ) -> None:
        """All-or-nothing: a half-filled run plots as a series with holes."""
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0", "missing"])

        report = backfill(tmp_path / "runs", store)

        assert report.unreachable == [path]
        assert report.filled == 0
        assert all(H_NORM not in s["z"]["base"] for s in read(path)["steps"])

    def test_a_dry_run_writes_nothing(self, tmp_path: Path, store: Store) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"])
        before = path.read_text(encoding="utf-8")

        report = backfill(tmp_path / "runs", store, dry_run=True)

        assert report.updated == [path]
        assert path.read_text(encoding="utf-8") == before

    def test_a_branch_endpoint_needs_no_store(
        self, tmp_path: Path, store: Store
    ) -> None:
        """No z means no norm to record, and no gap to come back for."""
        path = write_run(tmp_path / "runs", "branch", ["w0"], z={})

        report = backfill(tmp_path / "runs", store)

        assert report.without_z == 1
        assert report.unreachable == []
        assert report.updated == []
        assert "z_norm" not in path.read_text(encoding="utf-8")

    def test_a_run_predating_the_format_is_skipped(
        self, tmp_path: Path, store: Store
    ) -> None:
        path = tmp_path / "runs" / "old" / "trajectory.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"steps": []}), encoding="utf-8")

        report = backfill(tmp_path / "runs", store)

        assert report.skipped == [path]

    def test_a_legacy_convention_run_is_filled_too(
        self, tmp_path: Path, store: Store
    ) -> None:
        """The norm is the norm whichever units p and q are in.

        For a run the cosine backfill could not reach, this records precisely
        the divisor that conversion still needs; ``z_convention`` remains the
        only thing that says which units the block is in.
        """
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        path = write_run(tmp_path / "runs", "run", ["w0"], convention=LEGACY_CONVENTION)

        backfill(tmp_path / "runs", store)

        payload = read(path)
        assert payload["z_convention"] == LEGACY_CONVENTION
        assert payload["steps"][0]["z"]["base"][H_NORM] == pytest.approx(5.0)

    def test_convert_run_accumulates_into_a_shared_report(
        self, tmp_path: Path, store: Store
    ) -> None:
        write_h_neutral(store, "w0", torch.tensor([3.0, 4.0]))
        report = Report()
        for name in ("one", "two"):
            convert_run(write_run(tmp_path / "runs", name, ["w0"]), store, report)

        assert report.filled == 2
        assert len(report.updated) == 2


class TestConvertAnchorNoise:
    def test_each_row_uses_its_own_replicates_norm(
        self, tmp_path: Path, store: Store
    ) -> None:
        """The replicates re-derive h_neutral, so their norms differ.

        Keying by checkpoint alone would hand every replicate the production
        draw's norm and erase exactly the spread the sweep exists to measure.
        """
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([3.0, 4.0]))  # 5
        write_replicate_h_neutral(store, "w0", 1, torch.tensor([0.0, 2.0]))  # 2
        path = write_anchor_noise(tmp_path / "runs", [row("w0", 0), row("w0", 1)])

        backfill(tmp_path / "runs", store)

        rows = read(path)["latents"]
        assert rows[0][H_NORM] == pytest.approx(5.0)
        assert rows[1][H_NORM] == pytest.approx(2.0)

    def test_the_derived_tables_are_left_alone(
        self, tmp_path: Path, store: Store
    ) -> None:
        """``spread`` and ``against_drift`` read a fixed component list.

        Nothing they summarise moves when a column is added beside it, so
        rewriting them would only raise the question of which pass produced the
        numbers a figure is quoted against.
        """
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([3.0, 4.0]))
        path = write_anchor_noise(tmp_path / "runs", [row("w0", 0)])
        before = read(path)

        backfill(tmp_path / "runs", store)

        after = read(path)
        assert after["spread"] == before["spread"]
        assert after["against_drift"] == before["against_drift"]

    def test_the_latent_components_survive(self, tmp_path: Path, store: Store) -> None:
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([3.0, 4.0]))
        path = write_anchor_noise(tmp_path / "runs", [row("w0", 0)])

        backfill(tmp_path / "runs", store)

        filled = read(path)["latents"][0]
        assert {k: filled[k] for k in Z} == Z

    def test_a_missing_replicate_tensor_leaves_the_summary_alone(
        self, tmp_path: Path, store: Store
    ) -> None:
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([3.0, 4.0]))
        path = write_anchor_noise(tmp_path / "runs", [row("w0", 0), row("w0", 1)])

        report = backfill(tmp_path / "runs", store)

        assert report.unreachable == [path]
        assert all(H_NORM not in r for r in read(path)["latents"])

    def test_is_idempotent(self, tmp_path: Path, store: Store) -> None:
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([3.0, 4.0]))
        path = write_anchor_noise(tmp_path / "runs", [row("w0", 0)])

        backfill(tmp_path / "runs", store)
        second = backfill(tmp_path / "runs", store)

        assert second.already == 1
        assert second.anchor_rows == 0
        assert read(path)["latents"][0][H_NORM] == pytest.approx(5.0)

    def test_the_result_is_json_serialisable(
        self, tmp_path: Path, store: Store
    ) -> None:
        """torch returns numpy-backed floats, which json refuses."""
        write_replicate_h_neutral(store, "w0", 0, torch.tensor([3.0, 4.0]))
        path = write_anchor_noise(tmp_path / "runs", [row("w0", 0)])
        report = Report()

        convert_anchor_noise(path, store, report)

        assert isinstance(read(path)["latents"][0][H_NORM], float)
