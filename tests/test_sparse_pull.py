"""Tests for the sparse pull: a few files out of a whole-bundle archive.

The claim under test is a cost one. A measurement bundle is a single remote
object whose bulk is per-sample activation tensors, so reading the 400KB means
beside them is quoted at gigabytes. These pin that the quote is paid in
bandwidth only -- nothing but the wanted files reaches the disk -- that a
bundle with nothing to give up is settled from its index rather than its
archive, and that the thin copy stays out of the real store, where
``method.sync push`` would upload it over the complete one on the remote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from method import sparse_pull
from method.store import Store
from method.sparse_pull import (
    DEFAULT_PATTERNS,
    FetchMode,
    matcher,
    run,
    trunk_checkpoints,
)
from method.sync import LocalTransport, Syncer

#: A bundle's file list, shaped like a measured checkpoint's: the means the
#: recomputation reads, the per-sample tensors that dwarf them, both persona
#: vectors, and the answer text nobody projects.
CONTENTS = {
    "delta_p_target/probe1/mean_by_layer.pt": "target-mean",
    "delta_p_target/probe1/samples_layer20.pt": "target-samples",
    "delta_p_predicted/probe1/mean_by_layer.pt": "predicted-mean",
    "delta_p_predicted/probe1/samples_layer20.pt": "predicted-samples",
    "h_neutral_base/mean_by_layer.pt": "h-neutral-mean",
    "h_neutral_base/samples_layer20.pt": "h-neutral-samples",
    "traits/evil/evil_response_avg_diff.pt": "frozen-axis",
    "traits/evil/axis_refresh/vector/evil_response_avg_diff.pt": "onpolicy-axis",
    "traits/evil/delta_p_probe1.json": '{"mean": 1.0}',
    "traits/evil/delta_p_probe1.csv": "delta_p\n1.0\n",
    "neutral_answers.jsonl": '{"a": 1}',
}

#: Everything above that a projection difference on a refreshed axis reads,
#: plus the small JSONs that let the recomputation be checked against the
#: measured series. Deliberately spelled out rather than derived from the
#: patterns, so a pattern edit has to face what it changed.
WANTED = {
    "delta_p_target/probe1/mean_by_layer.pt",
    "delta_p_predicted/probe1/mean_by_layer.pt",
    "h_neutral_base/mean_by_layer.pt",
    "traits/evil/evil_response_avg_diff.pt",
    "traits/evil/axis_refresh/vector/evil_response_avg_diff.pt",
    "traits/evil/delta_p_probe1.json",
}


def _remote_with_bundle(tmp_path, wid: str) -> Syncer:
    """Push a bundle for ``wid``, then hand back a box that has none of it."""
    src = Syncer(
        Store(root=tmp_path / "src"),
        LocalTransport(tmp_path / "remote"),
        trajectories=tmp_path / "src-traj",
    )
    for name, text in CONTENTS.items():
        path = src.store.measurement_dir(wid) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    src.push_measurement(wid)
    return Syncer(
        Store(root=tmp_path / "dst"),
        LocalTransport(tmp_path / "remote"),
        trajectories=tmp_path / "dst-traj",
        strict=True,
    )


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


class TestWhatTheDefaultPatternsSelect:
    def test_the_means_and_both_axes_are_kept(self):
        wanted = matcher(DEFAULT_PATTERNS)
        assert {name for name in CONTENTS if wanted(name)} == WANTED

    def test_the_per_sample_tensors_are_the_ones_left_behind(self):
        """The 175MB files this whole module exists not to download."""
        wanted = matcher(DEFAULT_PATTERNS)
        assert not [
            name for name in CONTENTS if "samples_layer" in name and wanted(name)
        ]

    def test_an_extra_pattern_widens_the_selection(self):
        """``--include`` for a run that does need the spread, not just the mean."""
        wanted = matcher([*DEFAULT_PATTERNS, "delta_p_*/*/samples_layer*.pt"])
        assert wanted("delta_p_target/probe1/samples_layer20.pt")


class TestWhichCheckpointsAreNamed:
    def test_three_trunks_are_nineteen_checkpoints(self):
        """3 x 7, less the base the three of them share."""
        assert len(trunk_checkpoints(["a", "b", "c"], seed=0)) == 19

    def test_the_shared_base_is_labelled_for_what_it_is(self):
        labels = trunk_checkpoints(["a", "b", "c"], seed=0)
        assert sorted(labels.values()).count("base t=0") == 1

    def test_a_single_trunk_is_its_own_seven(self):
        assert len(trunk_checkpoints(["a"], seed=0)) == 7


class TestSweep:
    WID = "t01-sparse"

    def _run(self, syncer, tmp_path, **kwargs):
        run(
            syncer,
            weights_ids={self.WID: "trunk a t=1"},
            patterns=DEFAULT_PATTERNS,
            dest=tmp_path / "thin",
            **kwargs,
        )

    def test_only_the_wanted_files_reach_the_disk(self, tmp_path):
        dst = _remote_with_bundle(tmp_path, self.WID)

        self._run(dst, tmp_path)

        bundle = tmp_path / "thin" / "measurements" / self.WID
        assert _files_under(bundle) == WANTED
        assert (bundle / "traits/evil/evil_response_avg_diff.pt").read_text(
            encoding="utf-8"
        ) == "frozen-axis"

    def test_the_real_store_is_left_alone(self, tmp_path):
        dst = _remote_with_bundle(tmp_path, self.WID)

        self._run(dst, tmp_path)

        assert _files_under(dst.store.root) == set()

    def test_a_second_sweep_reads_no_archive(self, tmp_path, monkeypatch):
        """What makes an interrupted sweep cheap to resume: the index settles
        a complete bundle for kilobytes."""
        dst = _remote_with_bundle(tmp_path, self.WID)
        self._run(dst, tmp_path)

        streamed: list[str] = []
        monkeypatch.setattr(
            dst.transport,
            "open_stream",
            lambda relpath: streamed.append(relpath),  # blows up if ever called
        )
        self._run(dst, tmp_path)

        assert streamed == []

    def test_a_dry_run_writes_nothing(self, tmp_path):
        dst = _remote_with_bundle(tmp_path, self.WID)

        self._run(dst, tmp_path, dry_run=True)

        assert not (tmp_path / "thin").exists()

    def test_staging_leaves_no_archive_behind(self, tmp_path):
        dst = _remote_with_bundle(tmp_path, self.WID)
        stage = tmp_path / "stage"

        self._run(dst, tmp_path, mode=FetchMode.STAGE, stage_dir=stage)

        assert _files_under(tmp_path / "thin" / "measurements" / self.WID) == WANTED
        assert list(stage.iterdir()) == []

    @pytest.mark.parametrize("mode", list(FetchMode))
    def test_every_mode_collects_the_same_files(self, tmp_path, mode):
        """The three differ in what they transfer, never in what they leave."""
        dst = _remote_with_bundle(tmp_path, self.WID)

        self._run(dst, tmp_path, mode=mode, stage_dir=tmp_path / "stage")

        bundle = tmp_path / "thin" / "measurements" / self.WID
        assert _files_under(bundle) == WANTED
        assert (bundle / "delta_p_target/probe1/mean_by_layer.pt").read_text(
            encoding="utf-8"
        ) == "target-mean"

    def test_several_checkpoints_at_once_all_land(self, tmp_path):
        """``--jobs``: the workers share a syncer and must not tread on it."""
        dst = _remote_with_bundle(tmp_path, self.WID)
        for wid in ("t02-sparse", "t03-sparse"):
            _remote_with_bundle(tmp_path, wid)

        run(
            dst,
            weights_ids={wid: wid for wid in (self.WID, "t02-sparse", "t03-sparse")},
            patterns=DEFAULT_PATTERNS,
            dest=tmp_path / "thin",
            jobs=3,
        )

        for wid in (self.WID, "t02-sparse", "t03-sparse"):
            assert _files_under(tmp_path / "thin" / "measurements" / wid) == WANTED

    def test_a_checkpoint_the_remote_lacks_does_not_stop_the_sweep(self, tmp_path):
        dst = _remote_with_bundle(tmp_path, self.WID)

        run(
            dst,
            weights_ids={"t09-absent": "trunk a t=9", self.WID: "trunk a t=1"},
            patterns=DEFAULT_PATTERNS,
            dest=tmp_path / "thin",
        )

        assert _files_under(tmp_path / "thin" / "measurements" / self.WID) == WANTED


class TestTheRealStoreIsRefused:
    """A partial bundle in the real store is one ``method.sync push`` would
    upload over the complete copy on the remote, taking the per-sample tensors
    -- GPU hours, not bytes -- with it."""

    def test_pointing_dest_at_the_store_needs_the_flag(self, monkeypatch):
        monkeypatch.setattr(sparse_pull, "load_dotenv", lambda path: None)
        monkeypatch.setattr(sys, "argv", ["sparse_pull", "--dest", str(Store().root)])

        with pytest.raises(SystemExit, match="real store"):
            sparse_pull.main()

    def test_a_missing_remote_is_said_plainly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sparse_pull, "load_dotenv", lambda path: None)
        monkeypatch.delenv("MSC_STORE_REMOTE", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["sparse_pull", "--dest", str(tmp_path / "thin")]
        )

        with pytest.raises(SystemExit, match="MSC_STORE_REMOTE"):
            sparse_pull.main()
