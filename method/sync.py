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
  (a second trait adds files; a re-run rewrites a run dir), so presence proves
  nothing about them and they are instead skipped only when unchanged since
  this box last pushed them (:class:`PushLedger`), last-writer-wins otherwise.
* **Push one artifact at a time.** Every ``push_*`` entry point below covers a
  single artifact, so a run ships each one the moment it lands in the store
  rather than banking a trajectory's worth of GPU hours until the end. The
  sweeps (:meth:`Syncer.push_store`, :meth:`Syncer.push_after_run`) are built
  from those calls and serve as a backstop; the ledger is what stops the
  backstop from re-uploading what the eager calls already sent.
* **Mock artifacts never sync.** ``store-mock`` deliberately shares ids with the
  real store; syncing it would let synthetic adapters poison real boxes.
  :func:`Syncer.from_env` refuses any root whose name ends in ``-mock``.
"""

from __future__ import annotations

import hashlib
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
from method.utils import DOTENV_PATH, load_dotenv, trajectories_root

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

    @property
    def identity(self) -> str:
        """Stable name for the remote root this transport points at.

        :class:`PushLedger` is scoped by it, so that repointing
        ``MSC_STORE_REMOTE`` at a different back-end starts from an empty
        record rather than skipping uploads on the strength of what was pushed
        somewhere else entirely.
        """
        raise NotImplementedError

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

    @property
    def identity(self) -> str:
        return self.root

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

    @property
    def identity(self) -> str:
        return str(self.root)

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

    ``recursive=False`` because ``rglob`` already yields every descendant. Left
    at its default, ``add`` walks each directory member's subtree *as well*, so
    a file landed in the archive once per ancestor directory plus once for
    itself -- tripling a measurement bundle's ``<kind>/<hash>/tensor.pt`` files
    and inflating every upload ~3x.
    """
    with atomic_file(dest_tar) as scratch:
        with tarfile.open(scratch, "w", dereference=True) as tar:
            for path in sorted(src.rglob("*")):
                if not path.exists():
                    # A symlink whose target is already gone: nothing to
                    # dereference, and ``add`` would abort the whole push.
                    logger.warning("skipping dangling link %s", path)
                    continue
                tar.add(
                    path, arcname=str(path.relative_to(src)), recursive=False
                )


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
            logger.warning(
                "skipping link member %s -> %s", member.name, member.linkname
            )
            continue
        members.append(member)
    # ``data`` filter (Python 3.12+) additionally strips unsafe members; our
    # own check above is the belt to its braces. Silences the 3.14 default-change
    # warning and is a no-op on the plain files we archive.
    tar.extractall(dest, members=members, filter="data")


# --------------------------------------------------------------------------- #
# Push ledger: what each mutable artifact looked like when it last went up.
# --------------------------------------------------------------------------- #


class PushLedger:
    """Records the on-disk state of every mutable artifact this box has pushed.

    Immutable artifacts need no such record: their id *is* their content, so a
    single ``exists`` call on the remote settles whether the upload can be
    skipped. Measurement bundles and run dirs have no such property -- they
    grow a trait, a probe or a rewritten ``trajectory.json`` at a time -- so
    presence proves only that *some* version is up there, and the only safe
    thing to do on its own is re-upload every time.

    That re-upload is what makes a repeated push expensive: it costs the full
    bytes of every bundle in the store whether or not anything changed. The
    ledger removes it by remembering a cheap signature of each artifact under
    the remote path it was pushed as; a later push recomputes the signature and
    uploads only on a mismatch. Repeated pushes then cost what actually changed
    instead of what the store contains.

    It is deliberately local, per-box state rather than a remote manifest: the
    question it answers is "did *I* already upload exactly these bytes", which
    needs no request to answer and is exactly what the caller is deciding. The
    trade is that deleting objects from the remote behind its back leaves it
    claiming an upload that no longer exists -- ``push --force`` (or deleting
    the ledger directory) is the way back.

    Scoped per remote by :meth:`for_transport`, because "already uploaded" is
    only ever true of one destination: pointing ``MSC_STORE_REMOTE`` at a
    second back-end must start from an empty record, not inherit the first's.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def for_transport(cls, store_root: Path, transport: Transport) -> PushLedger:
        """A ledger inside ``store_root``, private to ``transport``'s remote.

        Lives inside the store because it describes this box's copy of it:
        wiping the store must wipe the record of what was pushed from it, or
        the next push would skip artifacts the box no longer has. The remote is
        identified by a digest rather than its name so the directory is
        filesystem-safe whatever ``MSC_STORE_REMOTE`` happens to contain.
        """
        scope = hashlib.sha256(transport.identity.encode()).hexdigest()[:16]
        return cls(store_root / ".sync-state" / scope)

    def _entry(self, relpath: str) -> Path:
        return self.root / relpath.lstrip("/")

    def is_current(self, relpath: str, signature: str) -> bool:
        """Whether ``relpath`` was last pushed with exactly this signature."""
        entry = self._entry(relpath)
        return entry.is_file() and entry.read_text(encoding="utf-8") == signature

    def record(self, relpath: str, signature: str) -> None:
        """Note that ``relpath`` now holds the artifact with this signature."""
        with atomic_file(self._entry(relpath)) as scratch:
            scratch.write_text(signature, encoding="utf-8")


def _dir_signature(path: Path) -> str:
    """Digest of a directory's shape: every file's relpath, size and mtime.

    Stat-based rather than content-based on purpose. Measurement bundles hold
    hidden-state tensors of hundreds of megabytes, and re-hashing those bytes
    on every push would cost more than the upload the digest exists to avoid.
    The case stat-based digests classically miss -- an in-place rewrite that
    preserves both size and timestamp -- cannot arise here, because every write
    into the store lands via :func:`method.store.atomic_dir` or
    :func:`~method.store.atomic_file`, i.e. as a fresh inode.

    Mirrors :func:`_tar_dir`'s view of the directory: symlinks are followed
    (``is_file`` and ``stat`` both resolve them, so the signature tracks the
    bytes that would actually be archived) and dangling ones are skipped, since
    the tar skips them too.
    """
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        entry = f"{item.relative_to(path)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(entry.encode())
    return digest.hexdigest()


def _file_signature(path: Path) -> str:
    """The single-file analogue of :func:`_dir_signature`."""
    stat = path.stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


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
        self,
        store: Store,
        transport: Transport,
        trajectories: Path | None = None,
        *,
        force: bool = False,
    ) -> None:
        self.store = store
        self.transport = transport
        # Where run dirs and base probes live locally. Defaults to the real
        # trajectories root; injectable so tests (and unusual layouts) can point
        # it elsewhere without touching the repo tree.
        self.trajectories = (
            trajectories if trajectories is not None else trajectories_root(mock=False)
        )
        self.ledger = PushLedger.for_transport(store.root, transport)
        self.force = force

    @classmethod
    def from_env(cls, store: Store, *, force: bool = False) -> Syncer | None:
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
        return cls(store, make_transport(remote), force=force)

    # --- push: one artifact (local -> remote) ---------------------------- #
    #
    # Each of these covers exactly one artifact, so a run can ship a thing the
    # moment it exists instead of banking hours of GPU output until the end.

    def push_adapter(self, wid: str) -> None:
        """Upload one adapter, skipping if the immutable object already exists.

        Called right after an adapter installs so a preempted box has already
        durably stored every completed step.
        """
        if not self.store.has_adapter(wid):
            return
        self._push_dir(
            self.store.adapter_dir(wid), f"{_ADAPTERS}/{wid}.tar", mutable=False
        )

    def push_training_sample(self, sample_id: str) -> None:
        """Upload one cached training subsample (immutable, named by its hash)."""
        path = self.store.training_sample_path(sample_id)
        self._push_file(path, f"{_SAMPLES}/{path.name}", mutable=False)

    def push_measurement(self, wid: str) -> None:
        """Upload checkpoint ``wid``'s measurement bundle if it has changed.

        Mutable, so unlike an adapter this cannot skip on mere presence: a
        second trait, a later probe, or the DeltaP for the step about to run
        all add files to a bundle that is already on the remote.
        """
        self._push_dir(
            self.store.measurement_dir(wid), f"{_MEASUREMENTS}/{wid}.tar", mutable=True
        )

    def push_run_dir(self, run_dir: Path) -> None:
        """Upload one trajectory's run directory if it has changed."""
        self._push_dir(run_dir, f"{_RUNS}/{run_dir.name}.tar", mutable=True)

    def push_base_probe(self, path: Path) -> None:
        """Upload one base-probe summary if it has changed."""
        self._push_file(path, f"{_BASE_PROBES}/{path.name}", mutable=True)

    # --- push: sweeps ---------------------------------------------------- #

    def push_store(self) -> None:
        """Push every adapter, training sample and measurement bundle.

        Immutable artifacts cost one existence check each and mutable ones one
        stat walk, so on a store whose eager pushes all landed this is nearly
        free -- which is what makes it usable as an end-of-run backstop rather
        than only as a bulk upload.
        """
        for wid in _child_names(self.store.adapters):
            self.push_adapter(wid)
        for sample in _child_files(self.store.training_samples):
            self._push_file(sample, f"{_SAMPLES}/{sample.name}", mutable=False)
        for wid in _child_names(self.store.measurements):
            self.push_measurement(wid)

    def push_base_probes(self) -> None:
        """Push every base-probe summary next to the trajectories root."""
        for probe in _child_files(self.trajectories / "base_probes"):
            self.push_base_probe(probe)

    def push_after_run(self, run_dir: Path) -> None:
        """Backstop flush once a run finishes: the store, ``run_dir``, probes.

        A run pushes each artifact as it is produced, so by the time this runs
        the only genuinely new object is usually ``run_dir`` itself (its
        ``trajectory.json`` is written last). It sweeps anyway, to catch
        anything an eager push missed -- an artifact written by a code path
        that does not sync, or one produced while the remote was unreachable.
        """
        self.push_store()
        self.push_run_dir(run_dir)
        self.push_base_probes()

    # --- pull (remote -> local) ----------------------------------------- #

    def pull_before_run(self) -> None:
        """Fetch the reusable prefix so ``has_adapter`` hits work locally.

        Pulls every adapter, training sample and measurement bundle present on
        the remote but missing locally. Cheap (adapters and samples are small),
        and it is what turns "another box already trained this prefix" into an
        ordinary local cache hit inside :func:`method.run_trajectory.run`.
        """
        self._pull_dirs(
            _ADAPTERS,
            self.store.adapters,
            mutable=False,
            present=self.store.has_adapter,
        )
        self._pull_files(_SAMPLES, self.store.training_samples, mutable=False)
        self._pull_dirs(_MEASUREMENTS, self.store.measurements, mutable=True)

    def pull_for_plotting(self) -> None:
        """Fetch just what the collector reads: run dirs and base probes.

        The plotting box needs no adapters, samples or datasets -- the numbers
        the figures use are embedded in each ``trajectory.json`` -- so this
        pulls only ``trajectories/`` and is safe to run on a machine that never
        touches a GPU.
        """
        self._pull_dirs(_RUNS, self.trajectories, mutable=True)
        self._pull_files(
            _BASE_PROBES, self.trajectories / "base_probes", mutable=True
        )

    # --- shared machinery ----------------------------------------------- #

    def _already_pushed(self, relpath: str, signature: str | None) -> bool:
        """Whether the remote already holds this artifact in its current form.

        ``signature is None`` marks an immutable artifact, for which presence
        is proof: the id determines the bytes. Mutable ones are settled against
        the ledger instead, since presence tells us nothing about *which*
        version is up there.
        """
        if signature is None:
            return self.transport.exists(relpath)
        return not self.force and self.ledger.is_current(relpath, signature)

    def _push_dir(self, src: Path, relpath: str, *, mutable: bool) -> None:
        if not src.is_dir():
            return
        # Signed before tarring, never after: a concurrent write landing in
        # between then makes the recorded signature older than what was
        # uploaded, costing one redundant push later. Signing after would make
        # it *newer*, and the ledger would skip content that never went up.
        signature = _dir_signature(src) if mutable else None
        if self._already_pushed(relpath, signature):
            logger.debug("skip push: %s", relpath)
            return
        with _scratch_file(suffix=".tar") as tmp:
            _tar_dir(src, tmp)
            self.transport.upload(tmp, relpath)
        if signature is not None:
            self.ledger.record(relpath, signature)
        logger.info("pushed %s", relpath)

    def _push_file(self, src: Path, relpath: str, *, mutable: bool) -> None:
        if not src.is_file():
            return
        signature = _file_signature(src) if mutable else None
        if self._already_pushed(relpath, signature):
            logger.debug("skip push: %s", relpath)
            return
        self.transport.upload(src, relpath)
        if signature is not None:
            self.ledger.record(relpath, signature)
        logger.info("pushed %s", relpath)

    def _pull_dirs(
        self, reldir: str, local_parent: Path, *, mutable: bool, present=None
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
            relpath = f"{reldir}/{name}"
            with _scratch_file(suffix=".tar") as tmp:
                self.transport.download(relpath, tmp)
                _untar_dir(tmp, local_parent / wid)
            if mutable:
                # What was just written locally *is* the remote object, so
                # record it as pushed; otherwise the next push would ship every
                # bundle this box pulled straight back up unchanged.
                self.ledger.record(relpath, _dir_signature(local_parent / wid))
            logger.info("pulled %s", relpath)

    def _pull_files(self, reldir: str, local_parent: Path, *, mutable: bool) -> None:
        for name in self.transport.list_names(reldir):
            dest = local_parent / name
            if dest.exists():
                continue
            relpath = f"{reldir}/{name}"
            self.transport.download(relpath, dest)
            if mutable:
                self.ledger.record(relpath, _file_signature(dest))
            logger.info("pulled %s", relpath)


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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "re-upload mutable artifacts (measurements, run dirs, base probes) "
            "even when unchanged since this box last pushed them; needed only "
            "after deleting objects from the remote by hand"
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
    syncer = Syncer.from_env(store, force=args.force)
    if syncer is None:
        raise SystemExit(f"{REMOTE_ENV} is not set; nothing to sync to.")

    if args.action == "push":
        # The store is swept once, not once per run dir: it is shared by every
        # trajectory, so folding it into the per-run loop re-tarred and
        # re-uploaded every measurement bundle N times over for N run dirs.
        # Sweeping it here also means a store with no runs still gets pushed.
        syncer.push_store()
        run_dirs = sorted(p for p in syncer.trajectories.glob("*_seed*") if p.is_dir())
        for run_dir in run_dirs:
            syncer.push_run_dir(run_dir)
        syncer.push_base_probes()
    elif args.action == "pull":
        syncer.pull_before_run()
    else:
        syncer.pull_for_plotting()


if __name__ == "__main__":
    main()
