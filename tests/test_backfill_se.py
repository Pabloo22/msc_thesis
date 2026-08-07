"""Unit tests for :mod:`method.backfill_se`.

The backfill's contract is narrow but load-bearing: it must recover ``SE`` into
the *run directories* (so no analysis machine needs the multi-hundred-gigabyte
store), must never invent a number it cannot derive, and must be safe to run
repeatedly as more of the store becomes reachable.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from method import experiments, steps
from method.backfill_se import backfill
from method.noise import behavior_summary
from method.steps import Artifacts
from method.store import Store, get_weights_id


def write_csv(
    store: Store, wid: str, trait: str, scores: dict[str, list[float]]
) -> None:
    """Plant a checkpoint's per-generation eval output in the store."""
    rows = [
        {"question_id": qid, trait: score, "coherence": 90.0}
        for qid, values in scores.items()
        for score in values
    ]
    path = store.trait_measurement(wid, trait, Artifacts.BEHAVIOR_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_run(root: Path, name: str, trait: str, wids: list[str]) -> Path:
    """A ``trajectory.json`` in the pre-SE format."""
    payload = {
        "config": {"name": name, "trait": trait, "seed": 0},
        "steps": [
            {
                "t": t,
                "weights_id": wid,
                "behavior": {trait: 10.0, f"{trait}_std": 5.0, "coherence": 90.0,
                             "n": 20},
                "z": {},
            }
            for t, wid in enumerate(wids)
        ],
    }
    path = root / name / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_steps(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["steps"]


class TestBackfillRun:
    def test_fills_se_from_a_reachable_csv(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "store")
        scores = {"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 5.0, 5.0, 5.0]}
        write_csv(store, "w0", "evil", scores)
        path = write_run(tmp_path / "runs", "run", "evil", ["w0"])

        report = backfill(tmp_path / "runs", store)

        assert report.filled == 1
        assert report.unreachable == 0
        expected = behavior_summary(
            pd.DataFrame(
                [
                    {"question_id": q, "evil": s, "coherence": 90.0}
                    for q, vals in scores.items()
                    for s in vals
                ]
            ),
            "evil",
        )
        assert read_steps(path)[0]["behavior"] == expected

    def test_recomputes_legacy_fields_rather_than_merging(self, tmp_path: Path) -> None:
        # The run file's stale mean (10.0) must be replaced by the one the CSV
        # actually implies, or a backfilled point would disagree with a freshly
        # measured one drawn from the same rows.
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [40.0, 60.0]})
        path = write_run(tmp_path / "runs", "run", "evil", ["w0"])

        report = backfill(tmp_path / "runs", store)

        assert read_steps(path)[0]["behavior"]["evil"] == 50.0
        # And the change is reported, never silent: the eval is unseeded at
        # temperature 1.0, so a disagreement means the checkpoint was measured
        # twice on different machines, not that one of them is corrupt.
        assert len(report.divergent) == 1
        assert "10.0000 -> 50.0000" in report.divergent[0]

    def test_agreeing_values_are_not_reported_as_divergent(
        self, tmp_path: Path
    ) -> None:
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [10.0, 10.0]})
        write_run(tmp_path / "runs", "run", "evil", ["w0"])

        report = backfill(tmp_path / "runs", store)

        assert report.filled == 1
        assert report.divergent == []

    def test_unreachable_csv_leaves_the_step_untouched(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "store")
        path = write_run(tmp_path / "runs", "run", "evil", ["missing"])
        before = read_steps(path)

        report = backfill(tmp_path / "runs", store)

        assert report.unreachable == 1
        assert report.filled == 0
        assert report.updated == []
        assert read_steps(path) == before

    def test_partial_coverage_fills_only_what_it_can(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [1.0, 2.0]})
        path = write_run(tmp_path / "runs", "run", "evil", ["w0", "w1"])

        report = backfill(tmp_path / "runs", store)

        steps = read_steps(path)
        assert report.filled == 1 and report.unreachable == 1
        assert "evil_se" in steps[0]["behavior"]
        assert "evil_se" not in steps[1]["behavior"]

    def test_is_idempotent(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [1.0, 2.0]})
        path = write_run(tmp_path / "runs", "run", "evil", ["w0"])

        backfill(tmp_path / "runs", store)
        first = path.read_text(encoding="utf-8")
        second_report = backfill(tmp_path / "runs", store)

        assert second_report.filled == 0
        assert second_report.already == 1
        assert second_report.updated == []
        assert path.read_text(encoding="utf-8") == first

    def test_leaves_the_store_alone_by_default(self, tmp_path: Path) -> None:
        # A measurement bundle is one remote object, so rewriting a 200-byte
        # JSON inside it re-uploads the tensors too. Nothing reads that JSON any
        # more (steps.behavior_record derives from the CSV), so the default must
        # not pay for it.
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [1.0, 2.0]})
        write_run(tmp_path / "runs", "run", "evil", ["w0"])

        report = backfill(tmp_path / "runs", store)

        assert report.filled == 1
        assert not store.trait_measurement(
            "w0", "evil", Artifacts.BEHAVIOR_JSON
        ).exists()

    def test_refresh_store_rewrites_the_summary_when_asked(
        self, tmp_path: Path
    ) -> None:
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [1.0, 2.0]})
        write_run(tmp_path / "runs", "run", "evil", ["w0"])

        backfill(tmp_path / "runs", store, refresh_store=True)

        stored = json.loads(
            store.trait_measurement(
                "w0", "evil", Artifacts.BEHAVIOR_JSON
            ).read_text(encoding="utf-8")
        )
        assert "evil_se" in stored

    def test_a_stale_store_summary_does_not_reach_a_new_run(
        self, tmp_path: Path
    ) -> None:
        """The shared-checkpoint trap: exp3 measured M_0, exp2 inherits it.

        ``measure_behavior`` returns early once ``behavior.csv`` exists, so the
        ``behavior.json`` an earlier experiment wrote is never updated. If a run
        record were built from that JSON, exp2's ``t=0`` would carry exp3's
        old-format summary while its later checkpoints carried the new one --
        and ``t=0`` is the baseline every ``Delta b`` is measured against.
        """
        cfg = dataclasses.replace(experiments.SMOKE_MOCK, trait="evil")
        store = Store(tmp_path / "store")
        base_wid = get_weights_id(cfg, 0)
        write_csv(store, base_wid, "evil", {"a": [10.0, 30.0], "b": [10.0, 30.0]})
        # An old-format summary, as an earlier experiment would have left it,
        # carrying a *different* mean so a stale read is unmistakable.
        store.trait_measurement(base_wid, "evil", Artifacts.BEHAVIOR_JSON).write_text(
            json.dumps({"evil": 99.0, "evil_std": 1.0, "coherence": 90.0, "n": 2}),
            encoding="utf-8",
        )

        record = steps.behavior_record(cfg, 0, store)

        assert record["evil"] == 20.0, "read the stale cached summary"
        assert "evil_se" in record
        assert record["n_questions"] == 2

    def test_skips_runs_without_a_config_block(self, tmp_path: Path) -> None:
        # trajectories/EXP1_seed0 is in this format; collect.py cannot load it
        # either, so there is nothing downstream that would read a result.
        root = tmp_path / "runs"
        path = root / "legacy" / "trajectory.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"steps": []}), encoding="utf-8")

        report = backfill(root, Store(tmp_path / "store"))

        assert report.skipped == [path]

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "store")
        write_csv(store, "w0", "evil", {"a": [1.0, 2.0]})
        path = write_run(tmp_path / "runs", "run", "evil", ["w0"])
        before = path.read_text(encoding="utf-8")

        report = backfill(tmp_path / "runs", store, dry_run=True)

        assert report.filled == 1
        assert report.updated == [path]
        assert path.read_text(encoding="utf-8") == before
        assert not store.trait_measurement(
            "w0", "evil", Artifacts.BEHAVIOR_JSON
        ).exists()
