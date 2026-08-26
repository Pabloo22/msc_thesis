"""Tests for the remote store sync layer.

These exercise the real tar/untar and pull-through logic against a
:class:`LocalTransport` rooted at a temp dir -- the same code path an rclone
remote takes, minus rclone itself -- so no network or ``rclone`` binary is
needed. The invariants under test:

* an artifact survives a round trip byte-for-byte, packed as exactly one remote
  object per id;
* immutable adapters/samples are not re-uploaded once present;
* mutable measurements/run dirs are re-uploaded when they change and skipped
  when they do not, including after a pull;
* a pull only fetches ids missing locally, and never a half-written directory;
* mock stores refuse to sync;
* a remote that is briefly unreachable is retried, and one that stays
  unreachable is recorded rather than raised -- except for the CLI, which is
  strict because transferring is all it does.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from method import experiments as E, steps, sync
from method.store import Store, StoreSelection, get_weights_id
from method.sync import (
    _ATTEMPTS,
    REMOTE_ENV,
    LocalTransport,
    RcloneTransport,
    Syncer,
    format_unsynced,
    make_transport,
)
from method.timing import STAGE_LOG


def _make_adapter(store: Store, wid: str, *, marker: str = "x") -> None:
    adir = store.adapter_dir(wid)
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adir / "adapter_model.safetensors").write_text(marker, encoding="utf-8")


def _syncer(tmp_path, *, store_name="store", remote_name="remote", **kwargs) -> Syncer:
    store = Store(root=tmp_path / store_name)
    transport = LocalTransport(tmp_path / remote_name)
    return Syncer(
        store, transport, trajectories=tmp_path / f"{store_name}-traj", **kwargs
    )


def _record_uploads(
    monkeypatch, syncer: Syncer, *, include_indexes: bool = False
) -> list[str]:
    """Collect the relpaths ``syncer`` uploads, while still uploading them.

    Index sidecars are filtered out by default: they always accompany the
    mutable archive they describe, so a test asking "which artifacts went up"
    means the archives. Pass ``include_indexes`` to assert on them directly.
    """
    uploaded: list[str] = []
    real_upload = syncer.transport.upload

    def spy(local, relpath):
        if include_indexes or not relpath.endswith(".files"):
            uploaded.append(relpath)
        real_upload(local, relpath)

    monkeypatch.setattr(syncer.transport, "upload", spy)
    return uploaded


def _record_listings(monkeypatch, syncer: Syncer) -> list[str]:
    """Collect the reldirs ``syncer`` lists, while still listing them."""
    listed: list[str] = []
    real_list = syncer.transport.list_names

    def spy(reldir):
        listed.append(reldir)
        return real_list(reldir)

    monkeypatch.setattr(syncer.transport, "list_names", spy)
    return listed


def _write_measurement(store: Store, wid: str, name: str, text: str) -> None:
    path = store.measurement_dir(wid) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_run_dir(
    syncer: Syncer, name: str = "EXP1_seed0", *, steps: str = "[]"
) -> Path:
    """A run directory shaped like one a trajectory leaves behind."""
    run_dir = syncer.trajectories / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectory.json").write_text(f'{{"steps": {steps}}}', encoding="utf-8")
    (run_dir / STAGE_LOG).write_text('{"stage": "train", "seconds": 900}\n', "utf-8")
    return run_dir


#: An mtime far enough in the future that a rewrite cannot land on the old one,
#: whatever the filesystem's timestamp resolution.
_LATER = 2_000_000_000


def _rerun(run_dir: Path) -> None:
    """Do to ``run_dir`` what re-running a fully cached trajectory does.

    ``trajectory.json`` is rewritten through ``atomic_file``, so it comes back
    byte-identical on a fresh inode with a new mtime, and every stage the run
    skipped still appends a row to the timing log.
    """
    trajectory = run_dir / "trajectory.json"
    payload = trajectory.read_text(encoding="utf-8")
    trajectory.unlink()
    trajectory.write_text(payload, encoding="utf-8")
    os.utime(trajectory, (_LATER, _LATER))
    with (run_dir / STAGE_LOG).open("a", encoding="utf-8") as handle:
        handle.write('{"stage": "train", "seconds": 0.5}\n')


class TestTransportSelection:
    def test_rclone_target_uses_rclone(self):
        assert isinstance(make_transport("gdrive:msc-thesis"), RcloneTransport)

    def test_bare_path_uses_local(self, tmp_path):
        assert isinstance(make_transport(str(tmp_path)), LocalTransport)

    def test_absolute_path_with_no_colon_is_local(self):
        assert isinstance(make_transport("/mnt/shared/msc"), LocalTransport)


class TestFromEnv:
    def test_none_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REMOTE_ENV, raising=False)
        assert Syncer.from_env(Store(root=tmp_path / "store")) is None

    def test_refuses_mock_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REMOTE_ENV, str(tmp_path / "remote"))
        assert Syncer.from_env(Store(root=tmp_path / "store-mock")) is None

    def test_real_store_with_remote(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REMOTE_ENV, str(tmp_path / "remote"))
        assert Syncer.from_env(Store(root=tmp_path / "store")) is not None


class TestAdapterRoundTrip:
    def test_push_then_pull_restores_bytes(self, tmp_path):
        src = _syncer(tmp_path, store_name="src")
        _make_adapter(src.store, "t01-abc", marker="hello")
        src.push_adapter("t01-abc")

        # A second machine sharing only the remote.
        dst = _syncer(tmp_path, store_name="dst")
        assert not dst.store.has_adapter("t01-abc")
        dst.pull_before_run()

        assert dst.store.has_adapter("t01-abc")
        adapter = dst.store.adapter_dir("t01-abc") / "adapter_model.safetensors"
        assert adapter.read_text() == "hello"

    def test_one_object_per_adapter(self, tmp_path):
        src = _syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")
        src.push_adapter("t01-abc")
        names = src.transport.list_names("store/adapters")
        assert names == ["t01-abc.tar"]
        # And it is a real tar, not a directory of loose files.
        tar_path = tmp_path / "remote" / "store" / "adapters" / "t01-abc.tar"
        assert tarfile.is_tarfile(tar_path)

    def test_immutable_adapter_not_reuploaded(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")
        src.push_adapter("t01-abc")

        uploaded = _record_uploads(monkeypatch, src)
        src.push_adapter("t01-abc")  # already present -> skip
        assert uploaded == []


class TestPullSkipsPresent:
    def test_does_not_refetch_local_adapter(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path, store_name="src")
        _make_adapter(src.store, "t01-abc")
        src.push_adapter("t01-abc")

        dst = _syncer(tmp_path, store_name="dst")
        _make_adapter(dst.store, "t01-abc")  # already have it locally

        calls = []
        monkeypatch.setattr(
            dst.transport, "download", lambda rel, local: calls.append(rel)
        )
        dst.pull_before_run()
        assert calls == []

    def test_incomplete_local_adapter_is_refetched(self, tmp_path):
        """A dir present but missing adapter_config.json is not a cache hit."""
        src = _syncer(tmp_path, store_name="src")
        _make_adapter(src.store, "t01-abc", marker="full")
        src.push_adapter("t01-abc")

        dst = _syncer(tmp_path, store_name="dst")
        # Simulate a crashed copy: dir exists, completeness marker absent.
        (dst.store.adapter_dir("t01-abc")).mkdir(parents=True)
        assert not dst.store.has_adapter("t01-abc")
        dst.pull_before_run()
        assert dst.store.has_adapter("t01-abc")
        assert (
            dst.store.adapter_dir("t01-abc") / "adapter_model.safetensors"
        ).read_text() == "full"


class TestMeasurementsAndSamples:
    def test_measurements_reuploaded_on_growth(self, tmp_path):
        src = _syncer(tmp_path)
        mdir = src.store.measurement_dir("t01-abc")
        mdir.mkdir(parents=True)
        (mdir / "behavior.csv").write_text("a", encoding="utf-8")
        src.push_after_run(tmp_path / "no-run")

        # A second trait adds a file; re-push must overwrite the bundle.
        (mdir / "extra.csv").write_text("b", encoding="utf-8")
        src.push_after_run(tmp_path / "no-run")

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_before_run()
        got = dst.store.measurement_dir("t01-abc")
        assert (got / "behavior.csv").read_text() == "a"
        assert (got / "extra.csv").read_text() == "b"

    def test_training_samples_round_trip(self, tmp_path):
        src = _syncer(tmp_path)
        src.store.training_samples.mkdir(parents=True)
        sample = src.store.training_samples / "deadbeef.jsonl"
        sample.write_text("row", encoding="utf-8")
        src.push_after_run(tmp_path / "no-run")

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_before_run()
        assert (dst.store.training_samples / "deadbeef.jsonl").read_text() == "row"


class TestUnchangedMutableArtifactsSkip:
    """Mutable bundles must not be re-uploaded when nothing about them changed.

    Presence cannot settle this the way it does for an adapter -- a bundle on
    the remote may be missing a trait measured since -- so the ledger compares
    against what this box last pushed.
    """

    def test_second_push_of_unchanged_measurement_uploads_nothing(
        self, tmp_path, monkeypatch
    ):
        src = _syncer(tmp_path)
        _write_measurement(src.store, "t01-abc", "behavior.csv", "a")
        src.push_measurement("t01-abc")

        uploaded = _record_uploads(monkeypatch, src)
        src.push_measurement("t01-abc")
        assert uploaded == []

    def test_changed_measurement_is_reuploaded(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path)
        _write_measurement(src.store, "t01-abc", "behavior.csv", "a")
        src.push_measurement("t01-abc")

        uploaded = _record_uploads(monkeypatch, src)
        _write_measurement(src.store, "t01-abc", "traits/evil/latent.json", "{}")
        src.push_measurement("t01-abc")
        assert uploaded == ["store/measurements/t01-abc.tar"]

    def test_force_reuploads_unchanged(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path)
        _write_measurement(src.store, "t01-abc", "behavior.csv", "a")
        src.push_measurement("t01-abc")

        forced = _syncer(tmp_path, force=True)
        uploaded = _record_uploads(monkeypatch, forced)
        forced.push_measurement("t01-abc")
        assert uploaded == ["store/measurements/t01-abc.tar"]

    def test_sweep_after_eager_pushes_uploads_only_the_run_dir(
        self, tmp_path, monkeypatch
    ):
        """The end-of-run backstop costs nothing for what already went up."""
        src = _syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")
        _write_measurement(src.store, "t00-xyz", "behavior.csv", "a")
        src.push_adapter("t01-abc")
        src.push_measurement("t00-xyz")
        run_dir = src.trajectories / "EXP1_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text('{"steps": []}', encoding="utf-8")

        uploaded = _record_uploads(monkeypatch, src)
        src.push_after_run(run_dir)
        assert uploaded == ["trajectories/runs/EXP1_seed0.tar"]

    def test_a_second_remote_does_not_inherit_the_first_ledger(
        self, tmp_path, monkeypatch
    ):
        """ "Already uploaded" is only ever true of one destination."""
        src = _syncer(tmp_path)
        _write_measurement(src.store, "t01-abc", "behavior.csv", "a")
        src.push_measurement("t01-abc")

        elsewhere = _syncer(tmp_path, remote_name="other-remote")
        uploaded = _record_uploads(monkeypatch, elsewhere)
        elsewhere.push_measurement("t01-abc")
        assert uploaded == ["store/measurements/t01-abc.tar"]

    def test_pulled_measurement_is_not_pushed_back(self, tmp_path, monkeypatch):
        """A box that pulls a bundle must not ship the same bytes back up."""
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "behavior.csv", "a")
        src.push_measurement("t01-abc")

        dst = _syncer(tmp_path, store_name="dst", remote_name="remote")
        dst.pull_before_run()
        assert (dst.store.measurement_dir("t01-abc") / "behavior.csv").exists()

        uploaded = _record_uploads(monkeypatch, dst)
        dst.push_store()
        assert uploaded == []


class TestMergingPullOfMutableBundles:
    """A measurement bundle grows a trait/probe at a time on several boxes.

    Presence of the directory therefore proves nothing about its contents, and
    the pull used to skip on exactly that -- so a box that had ever touched a
    checkpoint never learned what another box measured on it, and divergent
    copies never reconciled. These pin down the fix and, just as importantly,
    that it never destroys local work to achieve it.
    """

    def test_a_partial_bundle_gains_what_it_lacks(self, tmp_path):
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "evil")
        src.push_measurement("t01-abc")

        # dst measured a *different* trait on the same checkpoint, so it holds
        # the directory but not the file src has.
        dst = _syncer(tmp_path, store_name="dst")
        _write_measurement(dst.store, "t01-abc", "traits/syco/behavior.csv", "syco")
        dst.pull_before_run()

        bundle = dst.store.measurement_dir("t01-abc")
        assert (bundle / "traits/evil/behavior.csv").read_text() == "evil"
        assert (bundle / "traits/syco/behavior.csv").read_text() == "syco"

    def test_local_measurements_are_never_destroyed(self, tmp_path):
        # The naive "just pull anyway" fix: atomic_dir deletes the destination
        # before renaming, so it would take unpushed GPU work with it.
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "evil")
        src.push_measurement("t01-abc")

        dst = _syncer(tmp_path, store_name="dst")
        _write_measurement(dst.store, "t01-abc", "h_neutral_base/mean.pt", "local")
        dst.pull_before_run()

        assert (
            dst.store.measurement_dir("t01-abc") / "h_neutral_base/mean.pt"
        ).read_text() == "local"

    def test_a_diverged_file_keeps_the_local_copy(self, tmp_path):
        # Same path, different bytes: two boxes measured the same (wid, trait)
        # concurrently under an unseeded temperature-1.0 eval. A paths-only
        # index reports nothing missing, so nothing is overwritten -- picking a
        # winner is a deliberate reconciliation, not a sync side effect.
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "remote")
        src.push_measurement("t01-abc")

        dst = _syncer(tmp_path, store_name="dst")
        _write_measurement(dst.store, "t01-abc", "traits/evil/behavior.csv", "local")
        dst.pull_before_run()

        assert (
            dst.store.measurement_dir("t01-abc") / "traits/evil/behavior.csv"
        ).read_text() == "local"

    def test_an_up_to_date_bundle_never_downloads_the_archive(
        self, tmp_path, monkeypatch
    ):
        # The whole point of the sidecar: bundles are hundreds of megabytes, so
        # a warm box must settle "nothing new here" from the index alone.
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "evil")
        src.push_measurement("t01-abc")

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_before_run()

        downloaded: list[str] = []
        real_download = dst.transport.download
        monkeypatch.setattr(
            dst.transport,
            "download",
            lambda relpath, local: (
                downloaded.append(relpath),
                real_download(relpath, local),
            )[1],
        )
        dst.pull_before_run()

        assert downloaded == ["store/measurements/t01-abc.files"]

    def test_a_merged_bundle_is_still_pushed_afterwards(self, tmp_path, monkeypatch):
        # A merged directory is a superset of the remote archive, so recording
        # it in the ledger would strand this box's own measurements forever.
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "evil")
        src.push_measurement("t01-abc")

        dst = _syncer(tmp_path, store_name="dst")
        _write_measurement(dst.store, "t01-abc", "traits/syco/behavior.csv", "syco")
        dst.pull_before_run()

        uploaded = _record_uploads(monkeypatch, dst)
        dst.push_store()
        assert uploaded == ["store/measurements/t01-abc.tar"]

    def test_an_archive_without_an_index_is_left_alone(self, tmp_path):
        # Objects pushed before indexing existed. Guessing at their contents is
        # what the destructive replace did; the next push writes their sidecar.
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "evil")
        src.push_measurement("t01-abc")
        (tmp_path / "remote/store/measurements/t01-abc.files").unlink()

        dst = _syncer(tmp_path, store_name="dst")
        _write_measurement(dst.store, "t01-abc", "traits/syco/behavior.csv", "syco")
        dst.pull_before_run()

        bundle = dst.store.measurement_dir("t01-abc")
        assert (bundle / "traits/syco/behavior.csv").exists()
        assert not (bundle / "traits/evil/behavior.csv").exists()

    def test_a_missing_index_is_written_without_reuploading_the_archive(
        self, tmp_path, monkeypatch
    ):
        src = _syncer(tmp_path, store_name="src")
        _write_measurement(src.store, "t01-abc", "traits/evil/behavior.csv", "evil")
        src.push_measurement("t01-abc")
        index = tmp_path / "remote/store/measurements/t01-abc.files"
        index.unlink()

        uploaded = _record_uploads(monkeypatch, src, include_indexes=True)
        src.push_measurement("t01-abc")

        assert uploaded == ["store/measurements/t01-abc.files"]
        assert index.exists()

    def test_adapters_get_no_index(self, tmp_path, monkeypatch):
        """Immutable artifacts need none: the id already determines the bytes."""
        src = _syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")

        uploaded = _record_uploads(monkeypatch, src, include_indexes=True)
        src.push_adapter("t01-abc")

        assert uploaded == ["store/adapters/t01-abc.tar"]


class TestPullForPlottingRefreshesChangedRuns:
    def test_a_rewritten_trajectory_reaches_a_box_that_already_has_it(self, tmp_path):
        """The backfill workflow: correct numbers on the box must reach the laptop.

        Run dirs use a hashed index precisely for this -- nothing is *missing*
        from the plotting box's copy, the bytes simply moved on.
        """
        src = _syncer(tmp_path, store_name="src")
        run_dir = _make_run_dir(src, steps='[{"t": 0, "behavior": {"evil": 1.0}}]')
        src.push_run_dir(run_dir)

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_for_plotting()
        assert "1.0" in (dst.trajectories / "EXP1_seed0/trajectory.json").read_text()

        (run_dir / "trajectory.json").write_text(
            '{"steps": [{"t": 0, "behavior": {"evil": 1.0, "evil_se": 0.5}}]}',
            encoding="utf-8",
        )
        src.push_run_dir(run_dir)
        dst.pull_for_plotting()

        assert (
            "evil_se" in (dst.trajectories / "EXP1_seed0/trajectory.json").read_text()
        )

    def test_a_rerun_that_only_retimed_does_not_make_a_box_pull_forever(
        self, tmp_path, monkeypatch
    ):
        """A pull that changes nothing must not be a pull that repeats.

        The timing log is excluded from a run dir's signature, so a re-run that
        only appended rows leaves the remote tar holding the *old* log. An index
        that hashed the new one would advertise bytes the tar does not contain:
        every pull would fetch the whole archive, merge a log that still
        disagreed with the index, and be equally stale next time.
        """
        src = _syncer(tmp_path, store_name="src")
        run_dir = _make_run_dir(src)
        src.push_run_dir(run_dir)

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_for_plotting()

        # A cached re-run on the GPU box: same payload, more timing rows. The
        # index is dropped so the backfill path rewrites it from current disk,
        # which is where the two used to part company.
        _rerun(run_dir)
        (tmp_path / "remote/trajectories/runs/EXP1_seed0.files").unlink()
        src.push_run_dir(run_dir)

        downloads = []
        real_download = dst.transport.download
        monkeypatch.setattr(
            dst.transport,
            "download",
            lambda relpath, local: (
                downloads.append(relpath),
                real_download(relpath, local),
            )[1],
        )
        dst.pull_for_plotting()
        dst.pull_for_plotting()

        assert not [relpath for relpath in downloads if relpath.endswith(".tar")]


class TestPushRunsSkipsTheStoreSweep:
    def test_push_runs_uploads_no_measurement_bundles(self, tmp_path, monkeypatch):
        """Rewriting a small JSON must not re-upload a bundle of tensors.

        A bundle is one remote object, so any change inside it re-uploads all of
        it. ``backfill_se`` touches ``behavior.json`` in every bundle, which
        through ``push`` would cost tens of gigabytes for edits belonging
        entirely to ``trajectory.json``.
        """
        src = _syncer(tmp_path)
        _write_measurement(src.store, "t00-xyz", "h_neutral_base/mean.pt", "heavy")
        src.push_measurement("t00-xyz")
        run_dir = _make_run_dir(src)
        # What the backfill does: rewrite a tiny file inside the bundle.
        _write_measurement(src.store, "t00-xyz", "traits/evil/behavior.json", "{}")

        uploaded = _record_uploads(monkeypatch, src)
        for path in sorted(p for p in src.trajectories.glob("*_seed*") if p.is_dir()):
            src.push_run_dir(path)
        src.push_base_probes()

        assert uploaded == [f"trajectories/runs/{run_dir.name}.tar"]


class TestPushStoreIsSweptOnce:
    def test_cli_push_sweeps_store_once_for_many_run_dirs(self, tmp_path, monkeypatch):
        """Regression: the store sweep used to run once per run directory.

        With N run dirs that re-tarred and re-uploaded every measurement bundle
        N+1 times, which on a rate-limited Drive remote is the expensive part.
        """
        src = _syncer(tmp_path)
        _write_measurement(src.store, "t00-xyz", "behavior.csv", "a")
        for name in ("EXP1_seed0", "EXP1_seed1", "EXP1_seed2"):
            run_dir = src.trajectories / name
            run_dir.mkdir(parents=True)
            (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")

        uploaded = _record_uploads(monkeypatch, src)
        src.push_store()
        for run_dir in sorted(src.trajectories.glob("*_seed*")):
            src.push_run_dir(run_dir)

        assert uploaded.count("store/measurements/t00-xyz.tar") == 1
        assert sorted(u for u in uploaded if u.startswith("trajectories/")) == [
            "trajectories/runs/EXP1_seed0.tar",
            "trajectories/runs/EXP1_seed1.tar",
            "trajectories/runs/EXP1_seed2.tar",
        ]


class TestRerunningACachedTrajectoryShipsNothing:
    """A re-run that recomputed nothing must not re-upload its run directory.

    Every invocation rewrites ``trajectory.json`` (through ``atomic_file``, so
    a fresh inode and a new mtime) and appends a timing row for each stage it
    skipped. Under a stat-based signature both counted as a change, so a
    trajectory whose every stage was a cache hit still spent tens of seconds
    re-tarring and re-uploading an archive the remote already had -- the bulk
    of what such a run cost, repeated once per seed and trait sharing the
    prefix.
    """

    def test_identical_payload_is_not_reuploaded(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path)
        run_dir = _make_run_dir(src)
        src.push_run_dir(run_dir)

        uploaded = _record_uploads(monkeypatch, src)
        _rerun(run_dir)
        src.push_run_dir(run_dir)

        assert uploaded == []

    def test_changed_trajectory_is_reuploaded_with_its_timings(
        self, tmp_path, monkeypatch
    ):
        """The timing log is ignored when *deciding*, never when packing."""
        src = _syncer(tmp_path)
        run_dir = _make_run_dir(src)
        src.push_run_dir(run_dir)

        uploaded = _record_uploads(monkeypatch, src)
        _rerun(run_dir)
        (run_dir / "trajectory.json").write_text(
            '{"steps": [{"t": 0}]}', encoding="utf-8"
        )
        src.push_run_dir(run_dir)

        assert uploaded == ["trajectories/runs/EXP1_seed0.tar"]
        archive = tmp_path / "remote" / "trajectories/runs/EXP1_seed0.tar"
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        assert sorted(names) == sorted(["trajectory.json", STAGE_LOG])

    def test_a_step_trained_on_different_data_is_reuploaded(self, tmp_path):
        """The signature follows the symlinks, as the archive does.

        A run directory records its steps as links into the content-addressed
        training samples, so the bytes that make it into the tar are the
        sample's, and a step pointed at a different sample is a different run.
        """
        src = _syncer(tmp_path)
        run_dir = _make_run_dir(src)
        src.store.training_samples.mkdir(parents=True)
        for name, text in (("aaa.jsonl", "first"), ("bbb.jsonl", "second")):
            (src.store.training_samples / name).write_text(text, encoding="utf-8")
        link = run_dir / "train_step1.jsonl"
        link.symlink_to(src.store.training_samples / "aaa.jsonl")
        src.push_run_dir(run_dir)

        link.unlink()
        link.symlink_to(src.store.training_samples / "bbb.jsonl")
        src.push_run_dir(run_dir)

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_for_plotting()
        assert (dst.trajectories / "EXP1_seed0" / "train_step1.jsonl").read_text() == (
            "second"
        )


class TestImmutablePushesCostOneListingPerKind:
    """Presence used to be asked one artifact at a time.

    Each ``exists`` was an rclone process and a network round trip, so the
    end-of-run sweep charged a trajectory for every adapter and sample any run
    had ever put in the store -- a cost that grew all sweep long and was paid
    in full by runs that produced nothing new.
    """

    def test_one_listing_settles_every_artifact(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path)
        for i in range(5):
            _make_adapter(src.store, f"t01-{i}")
        src.store.training_samples.mkdir(parents=True)
        for i in range(3):
            (src.store.training_samples / f"{i}.jsonl").write_text("row")

        listed = _record_listings(monkeypatch, src)
        src.push_store()
        src.push_store()  # what push_after_run adds on top

        assert sorted(listed) == ["store/adapters", "store/training_samples"]
        assert len(src.transport.list_names("store/adapters")) == 5

    def test_an_upload_is_not_repeated_by_a_later_sweep(self, tmp_path, monkeypatch):
        """The listing was taken before the eager push, so it has to learn."""
        src = _syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")
        src.push_store()  # takes the listing; ships t01-abc

        _make_adapter(src.store, "t02-def")
        src.push_adapter("t02-def")
        uploaded = _record_uploads(monkeypatch, src)
        src.push_store()

        assert uploaded == []


class TestNestedDirsArePackedOnce:
    """A nested artifact must not be duplicated once per level of nesting.

    ``_tar_dir`` walks every descendant itself, so letting ``tar.add`` recurse
    as well stored each file once per ancestor directory plus once for itself.
    Measurement bundles nest as ``<kind>/<hash>/tensor.pt``, so every tensor
    went up three times and each remote object was ~3x the bytes it needed.
    """

    def _bundle(self, store: Store, wid: str) -> None:
        for kind in ("delta_p_target", "delta_p_predicted"):
            _write_measurement(store, wid, f"{kind}/abc123/samples.pt", "tensor")
        _write_measurement(store, wid, "h_neutral_base/samples.pt", "tensor")
        _write_measurement(store, wid, "flat.json", "{}")

    def test_each_file_appears_exactly_once(self, tmp_path):
        src = _syncer(tmp_path)
        self._bundle(src.store, "t01-deadbeef")

        src.push_measurement("t01-deadbeef")

        archive = tmp_path / "remote" / "store/measurements/t01-deadbeef.tar"
        with tarfile.open(archive) as tar:
            names = [m.name for m in tar.getmembers() if m.isfile()]
        assert sorted(names) == sorted(set(names))
        assert sorted(names) == [
            "delta_p_predicted/abc123/samples.pt",
            "delta_p_target/abc123/samples.pt",
            "flat.json",
            "h_neutral_base/samples.pt",
        ]

    def test_archive_is_not_inflated_beyond_its_payload(self, tmp_path):
        src = _syncer(tmp_path)
        self._bundle(src.store, "t01-deadbeef")
        payload = sum(
            p.stat().st_size
            for p in src.store.measurement_dir("t01-deadbeef").rglob("*")
            if p.is_file()
        )

        src.push_measurement("t01-deadbeef")

        archive = tmp_path / "remote" / "store/measurements/t01-deadbeef.tar"
        with tarfile.open(archive) as tar:
            stored = sum(m.size for m in tar.getmembers() if m.isfile())
        assert stored == payload

    def test_nested_bundle_round_trips(self, tmp_path):
        src = _syncer(tmp_path)
        self._bundle(src.store, "t01-deadbeef")
        src.push_measurement("t01-deadbeef")

        dst = _syncer(tmp_path, store_name="dst")
        dst.transport = src.transport
        dst.pull_before_run()

        pulled = dst.store.measurement_dir("t01-deadbeef")
        assert (pulled / "delta_p_target/abc123/samples.pt").read_text() == "tensor"
        assert (pulled / "h_neutral_base/samples.pt").read_text() == "tensor"
        assert (pulled / "flat.json").read_text() == "{}"


class TestSymlinkedRunInputs:
    """Run dirs symlink their inputs into the store; archives must not.

    A plotting box has no store, so a stored link would point at nothing there
    even if its absolute path happened to exist.
    """

    def test_symlink_is_packed_as_its_contents(self, tmp_path):
        src = _syncer(tmp_path)
        real = tmp_path / "elsewhere" / "af0f0e4f.jsonl"
        real.parent.mkdir(parents=True)
        real.write_text("row", encoding="utf-8")
        run_dir = src.trajectories / "EXP1_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text('{"steps": []}', encoding="utf-8")
        (run_dir / "train_step1.jsonl").symlink_to(real)

        src.push_after_run(run_dir)

        dst = Syncer(
            Store(root=tmp_path / "dst"),
            src.transport,
            trajectories=tmp_path / "dst-traj",
        )
        dst.pull_for_plotting()

        pulled = dst.trajectories / "EXP1_seed0" / "train_step1.jsonl"
        assert not pulled.is_symlink()
        assert pulled.read_text() == "row"

    def test_dangling_symlink_does_not_abort_push(self, tmp_path):
        src = _syncer(tmp_path)
        run_dir = src.trajectories / "EXP1_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text('{"steps": []}', encoding="utf-8")
        (run_dir / "train_step1.jsonl").symlink_to(tmp_path / "deleted.jsonl")

        src.push_after_run(run_dir)  # must not raise

        dst = Syncer(
            Store(root=tmp_path / "dst"),
            src.transport,
            trajectories=tmp_path / "dst-traj",
        )
        dst.pull_for_plotting()
        assert (dst.trajectories / "EXP1_seed0" / "trajectory.json").exists()

    def test_legacy_archive_with_link_member_still_extracts(self, tmp_path):
        """Archives already on a remote, written before the dereference fix."""
        from method.sync import _untar_dir

        payload = tmp_path / "run"
        payload.mkdir()
        (payload / "trajectory.json").write_text('{"steps": []}', encoding="utf-8")
        legacy = tmp_path / "legacy.tar"
        with tarfile.open(legacy, "w") as tar:  # no dereference -> stores links
            tar.add(payload / "trajectory.json", arcname="trajectory.json")
            link = tarfile.TarInfo("train_step1.jsonl")
            link.type = tarfile.SYMTYPE
            link.linkname = "/home/someone/store-mock/training_samples/af0f.jsonl"
            tar.addfile(link)

        _untar_dir(legacy, tmp_path / "out")

        assert (tmp_path / "out" / "trajectory.json").exists()
        assert not (tmp_path / "out" / "train_step1.jsonl").exists()


class _FakeRclone:
    """Stands in for the rclone binary, returning canned exit codes.

    Exit codes are the only thing :class:`RcloneTransport` reads to decide
    between "it is not there", "try again" and "give up", and they are
    documented rclone behaviour rather than something this project chose:
    3/4 are not-found, 5 is rclone's own "temporary error", 1 is a usage or
    config error.
    """

    def __init__(self, monkeypatch, exits, *, stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self.slept: list[float] = []
        self._exits = list(exits)
        self._stdout = stdout
        monkeypatch.setattr(sync.subprocess, "run", self._run)
        # Retries are the point; waiting through them in a unit test is not.
        monkeypatch.setattr(sync.time, "sleep", self.slept.append)

    def _run(self, cmd, **_kwargs):
        self.calls.append(cmd)
        code = self._exits.pop(0) if self._exits else 0
        return subprocess.CompletedProcess(cmd, code, self._stdout, "boom")


class TestRcloneRetries:
    """rclone's own budget is 3 attempts with ``--retries-sleep`` at zero, so
    all of them land within milliseconds. These are the attempts with a wait.
    """

    def test_transient_failure_is_retried_then_succeeds(self, monkeypatch):
        fake = _FakeRclone(monkeypatch, [5, 5, 0])
        RcloneTransport("gdrive:msc").upload(Path("/tmp/x.tar"), "store/a.tar")
        assert len(fake.calls) == 3
        assert fake.slept == [5.0, 10.0]  # growing, not a tight loop

    def test_persistent_failure_raises_after_the_budget(self, monkeypatch):
        fake = _FakeRclone(monkeypatch, [5] * _ATTEMPTS)
        with pytest.raises(RuntimeError, match="exit 5"):
            RcloneTransport("gdrive:msc").upload(Path("/tmp/x.tar"), "store/a.tar")
        assert len(fake.calls) == _ATTEMPTS

    def test_config_error_is_not_retried(self, monkeypatch):
        """Exit 1 is a missing rclone.conf section: sleeping cannot fix it."""
        fake = _FakeRclone(monkeypatch, [1])
        with pytest.raises(RuntimeError, match="exit 1"):
            RcloneTransport("gdrive:msc").upload(Path("/tmp/x.tar"), "store/a.tar")
        assert len(fake.calls) == 1
        assert fake.slept == []

    def test_a_miss_costs_one_call_and_no_sleep(self, monkeypatch):
        """Exit 3 is the normal answer for a skip check, not a failure.

        Retrying it would put the whole backoff budget in front of every push
        into a directory the remote does not have yet.
        """
        fake = _FakeRclone(monkeypatch, [3])
        assert RcloneTransport("gdrive:msc").list_names("store/adapters") == []
        assert len(fake.calls) == 1
        assert fake.slept == []


class TestListingDistinguishesEmptyFromUnreachable:
    """The silent failure this replaces: an unreachable remote read as empty.

    ``list_names`` returning ``[]`` made ``pull_before_run`` a no-op that
    logged nothing, so the run went on to retrain a prefix that was sitting on
    the remote the whole time -- hours of GPU, with no error anywhere.
    """

    def test_absent_directory_is_empty(self, monkeypatch):
        _FakeRclone(monkeypatch, [3])
        assert RcloneTransport("gdrive:msc").list_names("store/adapters") == []

    def test_unreachable_remote_raises(self, monkeypatch):
        _FakeRclone(monkeypatch, [5] * _ATTEMPTS)
        with pytest.raises(RuntimeError, match="exit 5"):
            RcloneTransport("gdrive:msc").list_names("store/adapters")

    def test_populated_directory_lists(self, monkeypatch):
        _FakeRclone(monkeypatch, [0], stdout="a.tar\nb.tar\n")
        assert RcloneTransport("gdrive:msc").list_names("store/x") == [
            "a.tar",
            "b.tar",
        ]


class TestRcloneDownloadIsAtomic:
    """A cut-off transfer must not leave a truncated file at the destination.

    ``_pull_files`` writes straight to the final path and treats presence as
    proof of completeness, so a partial download left behind would be read as
    a complete artifact forever after and never fetched again.
    """

    def test_partial_download_leaves_no_destination(self, tmp_path, monkeypatch):
        transport = RcloneTransport("gdrive:msc")
        dest = tmp_path / "probe.json"

        def half_write_then_fail(cmd, **_kwargs):
            # rclone writes to whatever path it was given, then dies.
            Path(cmd[-1]).write_text('{"truncated"', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 5, "", "connection reset")

        monkeypatch.setattr(sync.subprocess, "run", half_write_then_fail)
        monkeypatch.setattr(sync.time, "sleep", lambda _s: None)

        with pytest.raises(RuntimeError):
            transport.download("trajectories/base_probes/probe.json", dest)

        assert not dest.exists()

    def test_successful_download_lands(self, tmp_path, monkeypatch):
        transport = RcloneTransport("gdrive:msc")
        dest = tmp_path / "nested" / "probe.json"

        def write_it(cmd, **_kwargs):
            Path(cmd[-1]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(sync.subprocess, "run", write_it)

        transport.download("trajectories/base_probes/probe.json", dest)

        assert dest.read_text() == "{}"


class _UnreachableTransport(LocalTransport):
    """A local transport that can be switched offline, as a remote can be.

    Subclasses :class:`LocalTransport` rather than faking one outright so that
    the *recovery* is real: switch ``down`` back off and the same object
    genuinely transfers, which is what the self-healing tests below turn on.
    """

    def __init__(self, root, *, down: bool = True, only: str | None = None) -> None:
        super().__init__(root)
        self.down = down
        #: Restrict the outage to relpaths containing this substring, for the
        #: case that matters most: one bad object among many good ones.
        self.only = only

    def _check(self, relpath: str) -> None:
        if self.down and (self.only is None or self.only in relpath):
            raise RuntimeError("remote unreachable")

    def upload(self, local, relpath):
        self._check(relpath)
        super().upload(local, relpath)

    def download(self, relpath, local):
        self._check(relpath)
        super().download(relpath, local)

    def list_names(self, reldir):
        self._check(reldir)
        return super().list_names(reldir)


class TestUnreachableRemoteDoesNotKillTheRun:
    """A trajectory is hours of GPU; a push is a nice-to-have on top of it.

    Everything offered to a transport is already durable on local disk, so a
    transfer that cannot happen must cost a retry, never the run.
    """

    def _syncer(self, tmp_path, **kwargs) -> Syncer:
        store = Store(root=tmp_path / "store")
        transport = _UnreachableTransport(tmp_path / "remote", **kwargs)
        return Syncer(store, transport, trajectories=tmp_path / "traj")

    def test_failed_push_is_recorded_not_raised(self, tmp_path):
        src = self._syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")

        src.push_adapter("t01-abc")  # must not raise

        assert list(src.unsynced) == ["push store/adapters/t01-abc.tar"]

    def test_failed_pull_is_recorded_not_raised(self, tmp_path):
        dst = self._syncer(tmp_path)

        dst.pull_before_run()  # must not raise

        # The listing is what failed, so that is what is named: nothing is
        # known about the individual objects behind it.
        assert list(dst.unsynced) == [
            "list store/adapters",
            "list store/training_samples",
            "list store/measurements",
        ]

    def test_run_survives_every_push_a_trajectory_makes(self, tmp_path):
        """The full set of entry points the runner calls, all offline."""
        src = self._syncer(tmp_path)
        _make_adapter(src.store, "t01-abc")
        _write_measurement(src.store, "t01-abc", "behavior.csv", "a")
        src.store.training_samples.mkdir(parents=True)
        (src.store.training_samples / "deadbeef.jsonl").write_text("row")
        run_dir = src.trajectories / "EXP1_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")

        src.push_adapter("t01-abc")
        src.push_measurement("t01-abc")
        src.push_training_sample("deadbeef")
        src.push_after_run(run_dir)  # none of these may raise

        assert len(src.unsynced) == 4

    def test_one_bad_object_does_not_strand_the_others(self, tmp_path):
        src = self._syncer(tmp_path, only="t02-bad")
        _make_adapter(src.store, "t01-good")
        _make_adapter(src.store, "t02-bad")

        src.push_store()

        assert list(src.unsynced) == ["push store/adapters/t02-bad.tar"]
        assert src.transport.list_names("store/adapters") == ["t01-good.tar"]


class TestRecoveryClearsTheRecord:
    """``unsynced`` is the state of the remote, not a log of every hiccup."""

    def test_later_success_drops_the_earlier_failure(self, tmp_path):
        store = Store(root=tmp_path / "store")
        transport = _UnreachableTransport(tmp_path / "remote")
        src = Syncer(store, transport, trajectories=tmp_path / "traj")
        _make_adapter(store, "t01-abc")
        src.push_adapter("t01-abc")
        assert src.unsynced

        # What the end-of-run sweep does, once the remote is back.
        transport.down = False
        src.push_store()

        assert src.unsynced == {}
        assert transport.list_names("store/adapters") == ["t01-abc.tar"]

    def test_the_next_run_on_this_box_ships_what_the_last_one_could_not(self, tmp_path):
        """Answering the question the report has to answer: is this recoverable?

        A push that failed left the artifact on local disk, and every run sweeps
        the whole store -- so a fresh syncer with no memory of the failure still
        finds and ships it.
        """
        store = Store(root=tmp_path / "store")
        _make_adapter(store, "t01-abc")
        first = Syncer(
            store, _UnreachableTransport(tmp_path / "remote"), trajectories=tmp_path
        )
        first.push_adapter("t01-abc")
        assert first.unsynced

        second = Syncer(
            store, LocalTransport(tmp_path / "remote"), trajectories=tmp_path
        )
        second.push_store()

        assert second.unsynced == {}
        assert (tmp_path / "remote" / "store/adapters/t01-abc.tar").is_file()


class TestStrictMode:
    """The CLI's whole job is the transfer, so it must not exit 0 without one."""

    def test_strict_syncer_raises(self, tmp_path):
        store = Store(root=tmp_path / "store")
        transport = _UnreachableTransport(tmp_path / "remote")
        src = Syncer(store, transport, trajectories=tmp_path, strict=True)
        _make_adapter(store, "t01-abc")

        with pytest.raises(RuntimeError, match="unreachable"):
            src.push_adapter("t01-abc")

    def test_from_env_passes_strict_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REMOTE_ENV, str(tmp_path / "remote"))
        assert Syncer.from_env(Store(root=tmp_path / "store"), strict=True).strict


class TestFormatUnsynced:
    def test_empty_renders_nothing(self):
        assert format_unsynced({}) == ""

    def test_names_the_artifact_and_the_recovery(self):
        text = format_unsynced({"push store/adapters/t01-abc.tar": "RuntimeError: x"})
        assert "store/adapters/t01-abc.tar" in text
        # The actionable half: the artifact is safe only while the box is.
        assert "do not release the box" in text.casefold()

    def test_long_lists_are_truncated(self):
        text = format_unsynced({f"push a/{i}.tar": "boom" for i in range(20)}, limit=3)
        assert "and 17 more" in text


class TestPlottingPull:
    def test_pull_for_plotting_gets_runs_and_probes_only(self, tmp_path):
        src = _syncer(tmp_path)
        # A run dir and a base-probe file next to the trajectories root.
        run_dir = src.trajectories / "EXP1_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text('{"steps": []}', encoding="utf-8")
        probes = src.trajectories / "base_probes"
        probes.mkdir(parents=True)
        (probes / "t00-x_evil.json").write_text("{}", encoding="utf-8")
        # An adapter that the plotting box must NOT need.
        _make_adapter(src.store, "t01-abc")
        src.push_adapter("t01-abc")
        src.push_after_run(run_dir)

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_for_plotting()

        assert (dst.trajectories / "EXP1_seed0" / "trajectory.json").exists()
        assert (dst.trajectories / "base_probes" / "t00-x_evil.json").exists()
        # No adapters pulled.
        assert not dst.store.has_adapter("t01-abc")

    def test_anchor_noise_summaries_round_trip(self, tmp_path):
        """They live beside the base probes and are read by the same box, but
        under their own remote kind so the two schemas stay apart."""
        src = _syncer(tmp_path)
        summaries = src.trajectories / "anchor_noise"
        summaries.mkdir(parents=True)
        summary = summaries / "t00-x_trunk_a.json"
        summary.write_text('{"spread": []}', encoding="utf-8")
        src.push_anchor_noise(summary)

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_for_plotting()

        pulled = dst.trajectories / "anchor_noise" / "t00-x_trunk_a.json"
        assert pulled.read_text() == '{"spread": []}'
        assert not (dst.trajectories / "base_probes" / "t00-x_trunk_a.json").exists()

    def test_a_rewritten_summary_is_swept_up_after_a_run(self, tmp_path):
        """The eager push happens once, when the sweep writes the summary. A
        backfill that rewrites it later has only the backstop to rely on, and
        that used to skip anchor-noise entirely -- so the corrected numbers
        stayed on the box that computed them."""
        src = _syncer(tmp_path)
        summaries = src.trajectories / "anchor_noise"
        summaries.mkdir(parents=True)
        summary = summaries / "t00-x_trunk_a.json"
        summary.write_text('{"z_convention": "projection"}', encoding="utf-8")
        src.push_anchor_noise(summary)

        summary.write_text('{"z_convention": "cosine"}', encoding="utf-8")
        run_dir = src.trajectories / "EXP1_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text('{"steps": []}', encoding="utf-8")
        src.push_after_run(run_dir)

        dst = _syncer(tmp_path, store_name="dst")
        dst.pull_for_plotting()

        pulled = dst.trajectories / "anchor_noise" / "t00-x_trunk_a.json"
        assert pulled.read_text() == '{"z_convention": "cosine"}'

    def test_an_unchanged_summary_is_not_reuploaded(self, tmp_path, monkeypatch):
        """The sweep is a backstop, not a second upload: the ledger has to make
        it free for a summary whose eager push already landed."""
        src = _syncer(tmp_path)
        summaries = src.trajectories / "anchor_noise"
        summaries.mkdir(parents=True)
        summary = summaries / "t00-x_trunk_a.json"
        summary.write_text('{"spread": []}', encoding="utf-8")
        src.push_anchor_noise(summary)

        uploaded = _record_uploads(monkeypatch, src)
        src.push_anchor_noises()

        assert uploaded == []


class TestStoreSelection:
    """The closure a scoped pull is filtered against.

    It has to be exact in both directions: an id left out is an artifact the
    run silently retrains or recomputes, and one wrongly included is bytes the
    filter exists to avoid.
    """

    def test_covers_every_checkpoint_including_the_base(self):
        cfg = E.SMOKE_MOCK
        selection = StoreSelection.for_config(cfg)
        assert selection.weights_ids == {
            get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)
        }
        # t=0 is load-bearing: persona vectors and h_neutral are read from the
        # base bundle at every checkpoint, not just the first.
        assert get_weights_id(cfg, 0) in selection.weights_ids

    def test_covers_probe_samples_the_config_never_trains_on(self):
        """A trunk measures DeltaP against probes absent from ``steps``."""
        probe = E.EXP2_PROBES[0]
        cfg = dataclasses.replace(E.SMOKE_MOCK, probes=(probe,))
        selection = StoreSelection.for_config(cfg)

        assert steps.training_sample_id(probe, cfg.seed) in (
            selection.training_sample_ids
        )
        for step in cfg.steps:
            assert steps.training_sample_id(step, cfg.seed) in (
                selection.training_sample_ids
            )


class TestScopedPull:
    """A run pulls its own prefix, not the whole store.

    The store holds every experiment's artifacts and the hidden-state bundles
    among them run to ~1GB each, so an unfiltered pull spends a rental box's
    disk -- and the wait before step 1 -- on checkpoints the trajectory will
    never open.
    """

    def _remote_with_two_checkpoints(self, tmp_path) -> Syncer:
        src = _syncer(tmp_path, store_name="src")
        for wid in ("t01-mine", "t01-theirs"):
            _make_adapter(src.store, wid, marker=wid)
            src.push_adapter(wid)
            _write_measurement(src.store, wid, "behavior.csv", wid)
            src.push_measurement(wid)
        src.store.training_samples.mkdir(parents=True, exist_ok=True)
        for sample_id in ("mine", "theirs"):
            (src.store.training_samples / f"{sample_id}.jsonl").write_text(
                sample_id, encoding="utf-8"
            )
            src.push_training_sample(sample_id)
        return src

    def test_fetches_only_the_selected_ids(self, tmp_path):
        self._remote_with_two_checkpoints(tmp_path)
        dst = _syncer(tmp_path, store_name="dst")

        dst.pull_before_run(
            StoreSelection(
                weights_ids=frozenset({"t01-mine"}),
                training_sample_ids=frozenset({"mine"}),
            )
        )

        assert dst.store.has_adapter("t01-mine")
        assert (dst.store.measurement_dir("t01-mine") / "behavior.csv").exists()
        assert (dst.store.training_samples / "mine.jsonl").read_text() == "mine"
        # The other checkpoint is on the remote and stays there.
        assert not dst.store.has_adapter("t01-theirs")
        assert not dst.store.measurement_dir("t01-theirs").exists()
        assert not (dst.store.training_samples / "theirs.jsonl").exists()

    def test_unselected_ids_cost_no_request_at_all(self, tmp_path, monkeypatch):
        """Not even the index round trip a mutable artifact would make.

        Filtering after the per-artifact requests would keep the disk saving
        but not the time one: a store of N bundles would still pay N sidecar
        downloads before every trajectory.
        """
        self._remote_with_two_checkpoints(tmp_path)
        dst = _syncer(tmp_path, store_name="dst")

        fetched: list[str] = []
        real_download = dst.transport.download
        monkeypatch.setattr(
            dst.transport,
            "download",
            lambda rel, local: (fetched.append(rel), real_download(rel, local))[1],
        )
        dst.pull_before_run(
            StoreSelection(
                weights_ids=frozenset({"t01-mine"}),
                training_sample_ids=frozenset({"mine"}),
            )
        )

        assert not [rel for rel in fetched if "theirs" in rel]
        # Exactly the three selected artifacts: adapter, sample, bundle. No
        # sidecar among them, which is what pins the claim above -- an index
        # download is the cheapest request an unselected bundle could have
        # provoked, so zero of them means the filter ran before any of it.
        assert len(fetched) == 3

    def test_no_selection_still_pulls_everything(self, tmp_path):
        """What ``python -m method.sync pull`` relies on: warm the whole box."""
        self._remote_with_two_checkpoints(tmp_path)
        dst = _syncer(tmp_path, store_name="dst")

        dst.pull_before_run()

        assert dst.store.has_adapter("t01-mine")
        assert dst.store.has_adapter("t01-theirs")
        assert (dst.store.training_samples / "theirs.jsonl").exists()
