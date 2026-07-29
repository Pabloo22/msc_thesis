"""Tests for the remote store sync layer.

These exercise the real tar/untar and pull-through logic against a
:class:`LocalTransport` rooted at a temp dir -- the same code path an rclone
remote takes, minus rclone itself -- so no network or ``rclone`` binary is
needed. The invariants under test:

* an artifact survives a round trip byte-for-byte, packed as exactly one remote
  object per id;
* immutable adapters/samples are not re-uploaded once present;
* a pull only fetches ids missing locally, and never a half-written directory;
* mock stores refuse to sync.
"""

from __future__ import annotations

import tarfile

from method.store import Store
from method.sync import (
    REMOTE_ENV,
    LocalTransport,
    RcloneTransport,
    Syncer,
    make_transport,
)


def _make_adapter(store: Store, wid: str, *, marker: str = "x") -> None:
    adir = store.adapter_dir(wid)
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adir / "adapter_model.safetensors").write_text(marker, encoding="utf-8")


def _syncer(tmp_path, *, store_name="store", remote_name="remote") -> Syncer:
    store = Store(root=tmp_path / store_name)
    transport = LocalTransport(tmp_path / remote_name)
    return Syncer(store, transport, trajectories=tmp_path / f"{store_name}-traj")


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
        dst = Syncer(Store(root=tmp_path / "dst"), src.transport)
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

        calls = []
        real_upload = src.transport.upload
        monkeypatch.setattr(
            src.transport,
            "upload",
            lambda local, rel: (calls.append(rel), real_upload(local, rel)),
        )
        src.push_adapter("t01-abc")  # already present -> skip
        assert calls == []


class TestPullSkipsPresent:
    def test_does_not_refetch_local_adapter(self, tmp_path, monkeypatch):
        src = _syncer(tmp_path, store_name="src")
        _make_adapter(src.store, "t01-abc")
        src.push_adapter("t01-abc")

        dst = Syncer(Store(root=tmp_path / "dst"), src.transport)
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

        dst = Syncer(Store(root=tmp_path / "dst"), src.transport)
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

        dst = Syncer(Store(root=tmp_path / "dst"), src.transport)
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

        dst = Syncer(Store(root=tmp_path / "dst"), src.transport)
        dst.pull_before_run()
        assert (
            dst.store.training_samples / "deadbeef.jsonl"
        ).read_text() == "row"


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

        dst = Syncer(Store(root=tmp_path / "dst"), src.transport)
        dst.pull_for_plotting()

        assert (dst.trajectories / "EXP1_seed0" / "trajectory.json").exists()
        assert (dst.trajectories / "base_probes" / "t00-x_evil.json").exists()
        # No adapters pulled.
        assert not dst.store.has_adapter("t01-abc")
