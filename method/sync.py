"""Push/pull the content-addressed store to a shared remote.

The experiments run on ephemeral rental GPUs, so nothing on a box's local disk
survives the box. This module gives every machine a common back-end (Google
Drive, S3/R2/B2, or a mounted path) that the durable artifacts are pushed to and
pulled from, so that

* a box starting a trajectory that shares a prefix with one another box already
  ran finds those adapters as ordinary local cache hits instead of retraining
  them (:func:`pull_before_run`), and
* the small artifacts a run produces outlive the box that produced it
  (:func:`push_after_run`, plus an eager :meth:`Syncer.push_adapter` so a spot
  preemption mid-run loses at most the step in flight).

Design constraints inherited from :mod:`method.store`:

* **One object per artifact.** Adapters and per-checkpoint measurements are
  *directories* of many small files; a plain file-by-file mirror both drowns in
  per-file request overhead (brutal on Drive's rate limit) and could expose a
  half-uploaded artifact, breaking the "presence implies completeness"
  invariant the store relies on. Each such directory is therefore packed into a
  single tar named by its content-addressed id, and a remote object only becomes
  visible once its upload finishes -- the remote analogue of the local
  ``os.replace`` in :func:`method.store.atomic_dir`.
* **Immutable ids skip cheaply.** Adapters and training samples are pure
  functions of their id, so once uploaded they never change: push skips them if
  the remote object already exists. Measurement and trajectory bundles *grow*
  (a second trait adds files; a re-run rewrites a run dir), so those are
  re-uploaded, last-writer-wins.
* **Mock artifacts never sync.** ``store-mock`` deliberately shares ids with the
  real store; syncing it would let synthetic adapters poison real boxes.
  :func:`Syncer.from_env` refuses any root whose name ends in ``-mock``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

from method.store import Store, atomic_dir, atomic_file
from method.utils import DOTENV_PATH, REPO_ROOT, load_dotenv, trajectories_root

logger = logging.getLogger(__name__)

#: Environment variable naming the remote root, e.g. ``gdrive:msc-thesis`` for an
#: rclone remote or ``/mnt/shared/msc-thesis`` for a mounted path. Unset means
#: "no remote"; every entry point then runs purely against local disk.
REMOTE_ENV = "MSC_STORE_REMOTE"


# --------------------------------------------------------------------------- #
# Transport: the thing that actually moves bytes to/from the remote.
# --------------------------------------------------------------------------- #


class Transport:
    """Move single files to and from a remote root, and test existence.

    The unit is always one file: directory artifacts are tarred to a single
    file by the layer above before they reach a transport, so a transport never
    has to reason about partial directories.
    """

    def exists(self, relpath: str) -> bool:
        raise NotImplementedError

    def upload(self, local: Path, relpath: str) -> None:
        raise NotImplementedError

    def download(self, relpath: str, local: Path) -> None:
        raise NotImplementedError

    def list_names(self, reldir: str) -> list[str]:
        """Immediate file names under ``reldir`` (no recursion, no dirs)."""
        raise NotImplementedError


class RcloneTransport(Transport):
    """A transport backed by the ``rclone`` CLI.

    rclone speaks ~70 back-ends through one interface, so the same code drives
    Google Drive, S3/R2/B2 or SFTP; switching provider is an rclone-config edit,
    not a code change. ``root`` is an rclone target such as ``gdrive:msc-thesis``.
    """

    def __init__(self, root: str, *, binary: str = "rclone") -> None:
        self.root = root.rstrip("/")
        self.binary = binary

    def _target(self, relpath: str) -> str:
        return f"{self.root}/{relpath.lstrip('/')}"

    def _run(
        self, args: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess:
        logger.debug("rclone %s", " ".join(args))
        result = subprocess.run(
            [self.binary, *args], check=False, capture_output=True, text=True
        )
        if check and result.returncode != 0:
            # rclone's own stderr ("didn't find section in config file",
            # "couldn't connect", a quota message) is the only thing that makes
            # a failure diagnosable, and ``capture_output`` means nobody else
            # will print it; a bare CalledProcessError would hide it.
            raise RuntimeError(
                f"rclone {' '.join(args)} failed (exit {result.returncode}): "
                f"{result.stderr.strip() or '<no stderr>'}"
            )
        return result

    def exists(self, relpath: str) -> bool:
        # ``lsf`` on the exact path lists just that object if present, nothing
        # otherwise -- one request, no directory walk.
        result = self._run(["lsf", self._target(relpath)], check=False)
        return result.returncode == 0 and bool(result.stdout.strip())

    def upload(self, local: Path, relpath: str) -> None:
        # ``copyto`` addresses the destination object by full path, so the
        # remote name is exactly ``relpath`` regardless of the local file name.
        self._run(["copyto", str(local), self._target(relpath)])

    def download(self, relpath: str, local: Path) -> None:
        self._run(["copyto", self._target(relpath), str(local)])

    def list_names(self, reldir: str) -> list[str]:
        result = self._run(["lsf", "--files-only", self._target(reldir)], check=False)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]


class LocalTransport(Transport):
    """A transport backed by a plain filesystem directory.

    Serves two roles: it is the whole implementation when the remote is a
    mounted network drive, and it makes the sync layer testable without rclone
    installed. Semantics mirror :class:`RcloneTransport` -- an upload is atomic
    (temp file + ``os.replace``) so a listing never sees a half-written object.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _target(self, relpath: str) -> Path:
        return self.root / relpath.lstrip("/")

    def exists(self, relpath: str) -> bool:
        return self._target(relpath).is_file()

    def upload(self, local: Path, relpath: str) -> None:
        dest = self._target(relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(dest) as scratch:
            shutil.copyfile(local, scratch)

    def download(self, relpath: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(local) as scratch:
            shutil.copyfile(self._target(relpath), scratch)

    def list_names(self, reldir: str) -> list[str]:
        directory = self._target(reldir)
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_file())


def make_transport(remote: str) -> Transport:
    """A :class:`LocalTransport` for a plain path, else :class:`RcloneTransport`.

    An rclone target contains a ``:`` before its first ``/`` (``gdrive:...``);
    a bare filesystem path does not. This lets the same ``MSC_STORE_REMOTE``
    name a mounted drive or an rclone remote without a second flag.
    """
    head = remote.split("/", 1)[0]
    if ":" in head and not os.path.isabs(remote):
        return RcloneTransport(remote)
    return LocalTransport(Path(remote))


# --------------------------------------------------------------------------- #
# Tar helpers: pack/unpack one directory artifact into/out of one file.
# --------------------------------------------------------------------------- #


def _tar_dir(src: Path, dest_tar: Path) -> None:
    """Pack ``src``'s contents into ``dest_tar`` (uncompressed).

    Uncompressed on purpose: adapter weights (safetensors) and hidden-state
    tensors (.pt) are already dense, so compression buys almost nothing for real
    CPU cost. Members are stored relative to ``src`` so extraction rebuilds the
    directory. ``dest_tar`` is written via a temp file so a listing of its
    parent never sees a partial tar.

    ``dereference=True`` because run dirs symlink their inputs into the store
    (``train_step1.jsonl`` -> ``../../store/training_samples/<hash>.jsonl``).
    Those targets sit outside the directory being packed, so stored as links
    they leave the archive incomplete: it resolves only next to a store that
    already holds the same ids, and a plotting box has no store at all. Storing
    the pointed-at bytes instead makes each archive stand on its own, which is
    the point of shipping one object per artifact.
    """
    with atomic_file(dest_tar) as scratch:
        with tarfile.open(scratch, "w", dereference=True) as tar:
            for path in sorted(src.rglob("*")):
                if not path.exists():
                    # A symlink whose target is already gone: nothing to
                    # dereference, and ``add`` would abort the whole push.
                    logger.warning("skipping dangling link %s", path)
                    continue
                tar.add(path, arcname=str(path.relative_to(src)))


def _untar_dir(tar_path: Path, dest_dir: Path) -> None:
    """Extract ``tar_path`` onto ``dest_dir`` atomically.

    Uses :func:`method.store.atomic_dir` so an interrupted extraction cannot
    leave a half-populated artifact that ``has_adapter`` would report complete.
    """
    with atomic_dir(dest_dir) as scratch:
        with tarfile.open(tar_path, "r") as tar:
            _safe_extractall(tar, scratch)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """``extractall`` refusing members that would escape ``dest``.

    The tars here are ones we wrote ourselves, but path-traversal defence is
    cheap and keeps a corrupted or swapped remote object from writing outside
    the store.

    Link members are dropped rather than extracted. :func:`_tar_dir` now
    dereferences, so archives written by current code contain none; ones
    already on a remote from before that do, and their targets resolve outside
    the extraction dir. Skipping them keeps a legacy archive readable -- the
    payload lives in the regular files -- where letting the ``data`` filter
    reject such a link aborts the entire pull over one dead pointer.
    """
    dest = dest.resolve()
    members = []
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not (target == dest or dest in target.parents):
            raise RuntimeError(f"unsafe path in archive: {member.name!r}")
        if member.issym() or member.islnk():
            logger.warning("skipping link member %s -> %s", member.name, member.linkname)
            continue
        members.append(member)
    # ``data`` filter (Python 3.12+) additionally strips unsafe members; our
    # own check above is the belt to its braces. Silences the 3.14 default-change
    # warning and is a no-op on the plain files we archive.
    tar.extractall(dest, members=members, filter="data")


# --------------------------------------------------------------------------- #
# Syncer: the store/trajectory-aware layer over a transport.
# --------------------------------------------------------------------------- #

# Remote layout, relative to the transport root. Kept flat and predictable so a
# human can browse it and so ``list_names`` needs only one call per kind.
_ADAPTERS = "store/adapters"
_MEASUREMENTS = "store/measurements"
_SAMPLES = "store/training_samples"
_RUNS = "trajectories/runs"
_BASE_PROBES = "trajectories/base_probes"


class Syncer:
    """Sync a real :class:`Store` (and its trajectory outputs) to a remote."""

    def __init__(
        self, store: Store, transport: Transport, trajectories: Path | None = None
    ) -> None:
        self.store = store
        self.transport = transport
        # Where run dirs and base probes live locally. Defaults to the real
        # trajectories root; injectable so tests (and unusual layouts) can point
        # it elsewhere without touching the repo tree.
        self.trajectories = (
            trajectories if trajectories is not None else trajectories_root(mock=False)
        )

    @classmethod
    def from_env(cls, store: Store) -> Syncer | None:
        """A syncer if ``MSC_STORE_REMOTE`` is set and ``store`` is real.

        Returns ``None`` (a no-op signal for callers) when no remote is
        configured, so local development and mock runs need no special-casing.
        Refuses a ``store-mock`` root outright: its ids collide with the real
        store by design, so pushing it would poison real boxes.
        """
        remote = os.environ.get(REMOTE_ENV, "").strip()
        if not remote:
            return None
        if store.root.name.endswith("-mock"):
            logger.info("Refusing to sync mock store %s", store.root)
            return None
        return cls(store, make_transport(remote))

    # --- push (local -> remote) ----------------------------------------- #

    def push_adapter(self, wid: str) -> None:
        """Upload one adapter, skipping if the immutable object already exists.

        Called eagerly right after an adapter installs so a preempted box has
        already durably stored every completed step.
        """
        if not self.store.has_adapter(wid):
            return
        self._push_dir(
            self.store.adapter_dir(wid), f"{_ADAPTERS}/{wid}.tar", skip_existing=True
        )

    def push_after_run(self, run_dir: Path) -> None:
        """Flush everything a completed run produced: adapters, samples,
        measurements, the run dir itself, and any base-probe summaries.

        Adapters and samples are immutable and mostly already pushed; the
        skip-existing check makes re-pushing them nearly free. Measurements and
        the run dir are (re-)uploaded because they grow or get rewritten.
        """
        for wid in _child_names(self.store.adapters):
            self.push_adapter(wid)
        for sample in _child_files(self.store.training_samples):
            self._push_file(sample, f"{_SAMPLES}/{sample.name}", skip_existing=True)
        for wid in _child_names(self.store.measurements):
            self._push_dir(
                self.store.measurement_dir(wid),
                f"{_MEASUREMENTS}/{wid}.tar",
                skip_existing=False,
            )
        if run_dir.is_dir():
            self._push_dir(run_dir, f"{_RUNS}/{run_dir.name}.tar", skip_existing=False)
        base_probes = self.trajectories / "base_probes"
        for probe in _child_files(base_probes):
            self._push_file(probe, f"{_BASE_PROBES}/{probe.name}", skip_existing=False)

    # --- pull (remote -> local) ----------------------------------------- #

    def pull_before_run(self) -> None:
        """Fetch the reusable prefix so ``has_adapter`` hits work locally.

        Pulls every adapter, training sample and measurement bundle present on
        the remote but missing locally. Cheap (adapters and samples are small),
        and it is what turns "another box already trained this prefix" into an
        ordinary local cache hit inside :func:`method.run_trajectory.run`.
        """
        self._pull_dirs(_ADAPTERS, self.store.adapters, present=self.store.has_adapter)
        self._pull_files(_SAMPLES, self.store.training_samples)
        self._pull_dirs(_MEASUREMENTS, self.store.measurements)

    def pull_for_plotting(self) -> None:
        """Fetch just what the collector reads: run dirs and base probes.

        The plotting box needs no adapters, samples or datasets -- the numbers
        the figures use are embedded in each ``trajectory.json`` -- so this
        pulls only ``trajectories/`` and is safe to run on a machine that never
        touches a GPU.
        """
        self._pull_dirs(_RUNS, self.trajectories)
        self._pull_files(_BASE_PROBES, self.trajectories / "base_probes")

    # --- shared machinery ----------------------------------------------- #

    def _push_dir(self, src: Path, relpath: str, *, skip_existing: bool) -> None:
        if not src.is_dir():
            return
        if skip_existing and self.transport.exists(relpath):
            logger.debug("skip push (exists): %s", relpath)
            return
        with _scratch_file(suffix=".tar") as tmp:
            _tar_dir(src, tmp)
            self.transport.upload(tmp, relpath)
        logger.info("pushed %s", relpath)

    def _push_file(self, src: Path, relpath: str, *, skip_existing: bool) -> None:
        if not src.is_file():
            return
        if skip_existing and self.transport.exists(relpath):
            return
        self.transport.upload(src, relpath)
        logger.info("pushed %s", relpath)

    def _pull_dirs(
        self, reldir: str, local_parent: Path, *, present=None
    ) -> None:
        """Download every ``<id>.tar`` under ``reldir`` whose ``<id>`` is missing.

        ``present`` decides "already have it": adapters use
        :meth:`Store.has_adapter` (checks completeness, not mere directory
        existence); others fall back to a plain directory check.
        """
        for name in self.transport.list_names(reldir):
            if not name.endswith(".tar"):
                continue
            wid = name[: -len(".tar")]
            have = present(wid) if present else (local_parent / wid).is_dir()
            if have:
                continue
            with _scratch_file(suffix=".tar") as tmp:
                self.transport.download(f"{reldir}/{name}", tmp)
                _untar_dir(tmp, local_parent / wid)
            logger.info("pulled %s", f"{reldir}/{name}")

    def _pull_files(self, reldir: str, local_parent: Path) -> None:
        for name in self.transport.list_names(reldir):
            dest = local_parent / name
            if dest.exists():
                continue
            self.transport.download(f"{reldir}/{name}", dest)
            logger.info("pulled %s", f"{reldir}/{name}")


# --------------------------------------------------------------------------- #
# Small local helpers.
# --------------------------------------------------------------------------- #


def _child_names(parent: Path) -> list[str]:
    if not parent.is_dir():
        return []
    return sorted(p.name for p in parent.iterdir() if p.is_dir())


def _child_files(parent: Path) -> Iterable[Path]:
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_file())


@contextmanager
def _scratch_file(*, suffix: str = "") -> Generator[Path]:
    """Yield a temp file path (staging area for a tar), removed on exit."""
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# CLI: `python -m method.sync {push|pull|pull-plots}`.
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["push", "pull", "pull-plots"],
        help=(
            "push: upload the whole local store + trajectories; "
            "pull: fetch the reusable store prefix; "
            "pull-plots: fetch only run dirs + base probes (plotting box)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Same as the other entry points: MSC_STORE_REMOTE usually lives in .env
    # alongside the API keys, so this CLI has to read it too. Note that an
    # already-exported shell variable still wins (see :func:`load_dotenv`).
    load_dotenv(DOTENV_PATH)

    store = Store()  # real store; mock never syncs
    syncer = Syncer.from_env(store)
    if syncer is None:
        raise SystemExit(f"{REMOTE_ENV} is not set; nothing to sync to.")

    if args.action == "push":
        # Push every run dir alongside the store.
        run_dirs = sorted(
            p for p in syncer.trajectories.glob("*_seed*") if p.is_dir()
        )
        for run_dir in run_dirs:
            syncer.push_after_run(run_dir)
        # A store with no runs still has adapters/samples/measurements to push.
        syncer.push_after_run(REPO_ROOT / "does-not-exist")
    elif args.action == "pull":
        syncer.pull_before_run()
    else:
        syncer.pull_for_plotting()


if __name__ == "__main__":
    main()
