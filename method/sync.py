"""Synchronise experiment artifacts with local or rclone remotes.

The commands that can be run are:

- ``poetry run python -m method.sync push``
- ``poetry run python -m method.sync pull``
- ``poetry run python -m method.sync push-runs``
- ``poetry run python -m method.sync push-adapter <id>``
- ``poetry run python -m method.sync push-measurements <id>``
- ``poetry run python -m method.sync push-sample <id>``
- ``poetry run python -m method.sync pull-run <id>``
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import IO

from method.store import Store, StoreSelection, atomic_dir, atomic_file, file_sha256
from method.timing import STAGE_LOG
from method.utils import DOTENV_PATH, load_dotenv, trajectories_root

logger = logging.getLogger(__name__)

#: Environment variable naming the remote root, e.g. ``gdrive:msc-thesis`` for an
#: rclone remote or ``/mnt/shared/msc-thesis`` for a mounted path. Unset means
#: "no remote"; every entry point then runs purely against local disk.
REMOTE_ENV = "MSC_STORE_REMOTE"

#: How many times one rclone invocation is attempted, and the first gap between attempts (doubling thereafter: 5s, 10s, 20s -- ~35s of patience in total).
_ATTEMPTS = 4
_BACKOFF_SECONDS = 5.0

#: How much of an archive one ranged read pulls back.
RANGE_WINDOW = 1 << 20

#: rclone exit codes meaning "it is not there". A legitimate answer to listing
#: a directory nothing has been pushed to yet, so never an error and never
#: worth a retry -- retrying every miss would put the whole backoff budget in
#: front of each skip check.
_ABSENT_EXITS = frozenset({3, 4})

#: Exit codes worth another attempt: 5 is rclone's own "temporary error, one that more retries might fix", and 2 is the uncategorised bucket that connection resets and truncated.
_RETRYABLE_EXITS = frozenset({2, 5})


# --------------------------------------------------------------------------- #
# Transport: the thing that actually moves bytes to/from the remote.
# --------------------------------------------------------------------------- #


class Transport:
    """Move single files to and from a remote root, and list what is there.

    The unit is always one file: directory artifacts are tarred to a single
    file by the layer above before they reach a transport, so a transport never
    has to reason about partial directories. Whether one particular object is
    already up there is asked of :meth:`list_names` rather than of a
    per-object existence call, since the callers ask it of a whole directory's
    worth of artifacts at a time.
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

    def upload(self, local: Path, relpath: str) -> None:
        raise NotImplementedError

    def download(self, relpath: str, local: Path) -> None:
        raise NotImplementedError

    def list_names(self, reldir: str) -> list[str]:
        """Immediate file names under ``reldir`` (no recursion, no dirs)."""
        raise NotImplementedError

    def open_stream(self, relpath: str) -> AbstractContextManager[IO[bytes]]:
        """Read one remote object as bytes, without landing it on disk.

        The counterpart to :meth:`download` for a caller that wants a few
        members out of a large archive rather than the archive itself: the
        bytes pass through a pipe and are dropped as they go, so the disk high
        water mark is what the caller decides to keep instead of the whole
        object. See :meth:`Syncer.extract_measurement_files`.
        """
        raise NotImplementedError

    def read_range(self, relpath: str, offset: int, count: int) -> bytes:
        """``count`` bytes of one remote object, starting at ``offset``.

        Fewer bytes than asked for means the object ended first; that is how a
        caller walking an archive learns it has reached the end, so it is a
        result and not an error.

        What :meth:`open_stream` cannot do: read the *middle* of an object
        without paying for everything before it. That is what lets
        :meth:`Syncer.pick_measurement_files` pay for the members it wants
        rather than for the archive holding them.
        """
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
        self, args: list[str], *, absent_ok: bool = False, binary: bool = False
    ) -> subprocess.CompletedProcess:
        """Run one rclone command, retrying while the failure looks transient.

        ``absent_ok`` admits the "not found" exits as an ordinary result for
        the caller to interpret, rather than an error. Only the two read
        operations pass it, and only they can afford to: a miss is their normal
        answer, so treating it as a failure would spend the entire backoff
        budget on every skip check a push makes.

        ``binary`` leaves stdout as bytes, for the one command whose output is
        payload rather than text. It travels through the same retry loop as
        everything else so a ranged read gets the same patience a listing does.
        """
        for attempt in range(1, _ATTEMPTS + 1):
            logger.debug(
                "rclone %s (attempt %d/%d)", " ".join(args), attempt, _ATTEMPTS
            )
            result = subprocess.run(
                [self.binary, *args], check=False, capture_output=True, text=not binary
            )
            if result.returncode == 0:
                return result
            if absent_ok and result.returncode in _ABSENT_EXITS:
                return result
            if result.returncode in _RETRYABLE_EXITS and attempt < _ATTEMPTS:
                delay = _BACKOFF_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    "rclone %s failed (exit %d); retrying in %.0fs [%d/%d]",
                    args[0],
                    result.returncode,
                    delay,
                    attempt,
                    _ATTEMPTS - 1,
                )
                time.sleep(delay)
                continue
            # rclone's own stderr ("didn't find section in config file",
            # "couldn't connect", a quota message) is the only thing that makes
            # a failure diagnosable, and ``capture_output`` means nobody else
            # will print it; a bare CalledProcessError would hide it.
            stderr = result.stderr
            if binary:
                stderr = stderr.decode("utf-8", "replace")
            raise RuntimeError(
                f"rclone {' '.join(args)} failed (exit {result.returncode}): "
                f"{stderr.strip() or '<no stderr>'}"
            )
        raise AssertionError("unreachable")  # pragma: no cover

    def upload(self, local: Path, relpath: str) -> None:
        # ``copyto`` addresses the destination object by full path, so the
        # remote name is exactly ``relpath`` regardless of the local file name.
        self._run(["copyto", str(local), self._target(relpath)])

    def download(self, relpath: str, local: Path) -> None:
        # Staged through ``atomic_file`` exactly as :class:`LocalTransport`
        # does. A transfer cut off part-way must not leave a truncated file at
        # the destination: :meth:`Syncer._pull_files` writes straight to the
        # final path and treats presence as proof of completeness, so a partial
        # download that stayed there would never be fetched again.
        local.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(local) as scratch:
            self._run(["copyto", self._target(relpath), str(scratch)])

    def list_names(self, reldir: str) -> list[str]:
        # An absent directory is empty -- the normal state of a remote nothing has been pushed to yet.
        result = self._run(
            ["lsf", "--files-only", self._target(reldir)], absent_ok=True
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    @contextmanager
    def open_stream(self, relpath: str) -> Generator[IO[bytes]]:
        """``rclone cat``, piped straight to the caller.

        Outside :meth:`_run`'s retry loop, and deliberately: once a stream has
        handed the caller bytes there is nothing to retry *into*, since the
        consumer has already acted on the first half of an object whose second
        half never arrived. Recovery is therefore the caller's -- re-open the
        stream and start over -- which is what :meth:`Syncer._attempt` around
        the whole read amounts to.
        """
        command = [self.binary, "cat", self._target(relpath)]
        logger.debug("rclone %s", " ".join(command[1:]))
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdout is not None and process.stderr is not None
        try:
            yield process.stdout
        finally:
            # Our end first: a caller that stopped reading early leaves rclone
            # blocked on a full pipe, and closing it turns that into the
            # SIGPIPE that ends the process rather than a wait() that never
            # returns.
            process.stdout.close()
            stderr = process.stderr.read().decode("utf-8", "replace")
            process.stderr.close()
            code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"rclone cat {relpath} failed (exit {code}): "
                f"{stderr.strip() or '<no stderr>'}"
            )

    def read_range(self, relpath: str, offset: int, count: int) -> bytes:
        # ``cat`` with both bounds is a ranged GET on every backend that has
        # one (S3/R2 included), so the bytes before ``offset`` are never sent.
        result = self._run(
            [
                "cat",
                "--offset",
                str(offset),
                "--count",
                str(count),
                self._target(relpath),
            ],
            binary=True,
        )
        return result.stdout


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

    @contextmanager
    def open_stream(self, relpath: str) -> Generator[IO[bytes]]:
        with self._target(relpath).open("rb") as handle:
            yield handle

    def read_range(self, relpath: str, offset: int, count: int) -> bytes:
        with self._target(relpath).open("rb") as handle:
            handle.seek(offset)
            return handle.read(count)


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
    r"""Pack ``src``'s contents into ``dest_tar`` (uncompressed)."""
    with atomic_file(dest_tar) as scratch:
        with tarfile.open(scratch, "w", dereference=True) as tar:
            for path in sorted(src.rglob("*")):
                if not path.exists():
                    # A symlink whose target is already gone: nothing to
                    # dereference, and ``add`` would abort the whole push.
                    logger.warning("skipping dangling link %s", path)
                    continue
                tar.add(path, arcname=str(path.relative_to(src)), recursive=False)


def _untar_dir(tar_path: Path, dest_dir: Path) -> None:
    """Extract ``tar_path`` onto ``dest_dir`` atomically, replacing what is there.

    Uses :func:`method.store.atomic_dir` so an interrupted extraction cannot
    leave a half-populated artifact that ``has_adapter`` would report complete.

    Because ``atomic_dir`` deletes ``dest_dir`` before the rename, this is only
    ever safe when the destination holds nothing worth keeping -- i.e. when it
    does not exist yet. A destination that already has content must go through
    :func:`_merge_from_tar` instead; see :meth:`Syncer._pull_dirs`.
    """
    with atomic_dir(dest_dir) as scratch:
        with tarfile.open(tar_path, "r") as tar:
            _safe_extractall(tar, scratch)


def _merge_from_tar(tar_path: Path, dest_dir: Path, wanted: Iterable[str]) -> None:
    """Copy just ``wanted`` out of ``tar_path`` into an existing ``dest_dir``.

    The non-destructive counterpart to :func:`_untar_dir`. Measurement bundles
    grow along several independent axes -- a trait, a probe dataset, an
    h_neutral source -- and two boxes routinely hold different subsets of the
    same bundle with neither a superset of the other. Replacing the local
    directory with the remote one would discard whichever measurements this box
    had produced but not yet pushed, turning a stale read into lost GPU hours.

    So nothing is deleted and nothing outside ``wanted`` is touched. The caller
    decides what ``wanted`` means -- paths missing locally, and for artifacts
    where the remote is authoritative, paths whose content differs.
    """
    with tempfile.TemporaryDirectory() as staging:
        staged = Path(staging)
        with tarfile.open(tar_path, "r") as tar:
            _safe_extractall(tar, staged)
        for relpath in sorted(wanted):
            source = staged / relpath
            if not source.is_file():
                # The index named a path the tar does not carry. Skipping keeps
                # a mismatched pair survivable rather than aborting the pull.
                logger.warning("index lists %s but the archive lacks it", relpath)
                continue
            destination = dest_dir / relpath
            with atomic_file(destination) as scratch:
                shutil.copyfile(source, scratch)


def _safe_target(dest_dir: Path, name: str) -> Path:
    """Where ``name`` lands under ``dest_dir``, refusing anything that escapes.

    The archives here are ones we wrote ourselves, but the check is cheap and
    keeps a corrupted or swapped remote object from writing outside the store
    -- the same guard :func:`_safe_extractall` applies to a whole extraction.
    """
    target = (dest_dir / name).resolve()
    if dest_dir.resolve() not in target.parents:
        raise RuntimeError(f"unsafe path in archive: {name!r}")
    return target


def _write_member(dest_dir: Path, name: str, data: bytes) -> None:
    """Land one member's bytes, atomically, under ``dest_dir``."""
    with atomic_file(_safe_target(dest_dir, name)) as scratch:
        scratch.write_bytes(data)


def _padded(size: int) -> int:
    """A member's data length as the archive stores it, blocks and all."""
    blocks = -(-size // tarfile.BLOCKSIZE)
    return blocks * tarfile.BLOCKSIZE


def _ends_here(buffer: bytes, at: int) -> bool:
    """Whether the archive's end-of-file marker sits at ``at`` in ``buffer``.

    A tar ends in zero blocks, so at a member boundary an all-zero block means
    there are no more members. Asking the bytes directly is what separates
    "the archive is over" from "this window ran out", which
    :mod:`tarfile` cannot tell apart when it is reading a slice: both look to
    it like a stream that stopped, and taking the second for the first ended
    the walk early with most of the archive unvisited.

    False when the buffer holds less than a full block from ``at``: nothing has
    been seen that says the archive ended, so the walk reads on.
    """
    block = buffer[at : at + tarfile.BLOCKSIZE]
    return len(block) == tarfile.BLOCKSIZE and not block.strip(b"\0")


def _walk_ranged(
    transport: Transport,
    relpath: str,
    dest_dir: Path,
    *,
    wanted: Callable[[str], bool],
    window: int,
) -> list[str]:
    r"""Copy the wanted members out of a remote archive, by seeking through it."""
    kept: list[str] = []
    position = 0
    while True:
        chunk = transport.read_range(relpath, position, window)
        if len(chunk) < tarfile.BLOCKSIZE:
            # An archive with no end-of-file marker is malformed, but a walk
            # that stops at its last complete member is the survivable reading
            # of it.
            logger.warning("archive %s ended without a terminator", relpath)
            break
        if _ends_here(chunk, 0):
            break
        advanced = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(chunk), mode="r|") as tar:
                for member in tar:
                    end = member.offset_data + _padded(member.size)
                    if member.isfile() and wanted(member.name):
                        # Whatever of the member this window already carries is
                        # kept and only the shortfall is asked for, so a member
                        # that straddles the end of a window is not paid for
                        # twice.
                        data = chunk[member.offset_data :][: member.size]
                        if len(data) < member.size:
                            data += transport.read_range(
                                relpath,
                                position + member.offset_data + len(data),
                                member.size - len(data),
                            )
                        if len(data) != member.size:
                            # Only a truncated remote object gets here. Saying
                            # so beats writing a short tensor that loads as
                            # garbage and is indistinguishable from a complete
                            # one ever after.
                            raise RuntimeError(
                                f"{relpath}: {member.name} is {member.size} "
                                f"bytes but only {len(data)} could be read"
                            )
                        _write_member(dest_dir, member.name, data)
                        kept.append(member.name)
                    # Set from the header, so a member whose data ran past the
                    # window still moves the walk on rather than stalling it.
                    advanced = end
                    if end > len(chunk):
                        break
        except tarfile.ReadError:
            # The window cut a header in half, or ran out where an archive
            # would have ended. Whatever was parsed before it still counts; the
            # next request starts at the member after.
            pass
        if advanced == 0:
            raise RuntimeError(
                f"{relpath}: no member fits in a {window}-byte window at "
                f"offset {position}; the archive is not readable this way"
            )
        if _ends_here(chunk, advanced):
            break
        position += advanced
    return kept


def _copy_wanted(
    tar: tarfile.TarFile, dest_dir: Path, wanted: Callable[[str], bool]
) -> list[str]:
    """Write the members ``wanted`` accepts into ``dest_dir``, skipping the rest.

    Iterates rather than calling :meth:`~tarfile.TarFile.getmembers`, so it
    works on an archive that is still arriving over a pipe as well as on one
    already on disk: each member is visited once, in the order the tar holds
    it, and the ones nobody asked for are never read. That is what lets a
    caller take 8MB out of a 3.3GB object without the object ever existing
    locally -- see :meth:`Syncer.extract_measurement_files`.

    Nothing is deleted, exactly as in :func:`_merge_from_tar`: the destination
    is a subset of the archive, and what is already there is another box's
    business.
    """
    kept = []
    for member in tar:
        # Directories carry no bytes and links are dropped for the reason
        # :func:`_safe_extractall` gives; only regular files can be wanted.
        if not member.isfile() or not wanted(member.name):
            continue
        target = _safe_target(dest_dir, member.name)
        source = tar.extractfile(member)
        if source is None:  # pragma: no cover -- isfile() already settled this
            continue
        with atomic_file(target) as scratch:
            with scratch.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        kept.append(member.name)
    return kept


# Index sidecars: what a mutable archive contains, without downloading it.

_VOLATILE_RUN_FILES = frozenset({STAGE_LOG})

#: Suffix of the sidecar object pushed beside each mutable ``<id>.tar``.
_INDEX_SUFFIX = ".files"

#: Separator between a path and its token inside an index. NUL cannot occur in
#: a path, so no escaping is needed.
_INDEX_SEP = "\0"


def _paths_index(path: Path) -> str:
    """Index naming only the files present: the right question for a bundle.

    Used for measurement bundles, where "does the remote hold something I am
    missing" is answerable from names alone and is the only question worth
    asking. Whether two boxes' copies of the *same* path agree is a separate
    matter that content hashes would raise but must not silently resolve --
    overwriting a local measurement with a remote one is a decision, not a sync
    detail -- and hashing hundreds of megabytes of hidden-state tensors to ask
    it would cost more than the download the index exists to avoid.
    """
    return "\n".join(
        str(item.relative_to(path))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def _hashed_index(path: Path) -> str:
    """Index pairing each file with a digest of its contents.

    Used for run directories, where the remote *is* authoritative: a box
    rewrites ``trajectory.json`` in place (a backfill, a re-run that measured
    something new), and a plotting machine that already has the file needs the
    new bytes, not just the ones it lacks. Names alone cannot express that.

    Affordable here for the same reason :func:`_run_dir_signature` is: a run
    directory is a small JSON plus dereferenced training samples, megabytes
    against the hundreds a measurement bundle runs to.

    Skips :data:`_VOLATILE_RUN_FILES`, exactly as the signature that decides
    whether the archive is re-uploaded does. Indexing a file the signature
    ignores is what lets an index and its tar disagree permanently.
    """
    lines = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.name in _VOLATILE_RUN_FILES:
            continue
        lines.append(f"{item.relative_to(path)}{_INDEX_SEP}{file_sha256(item)}")
    return "\n".join(lines)


def _parse_index(text: str) -> dict[str, str]:
    """An index's ``relpath -> token`` map; the token is ``""`` when absent.

    Drops :data:`_VOLATILE_RUN_FILES` on the way in as well as on the way out
    (:func:`_hashed_index`), because the indexes already on the remote were
    written before that exclusion existed and cannot be corrected without
    re-uploading every archive they describe. Since :func:`_stale_paths` walks
    the *remote* map, one such entry is enough to keep a run dir permanently
    stale on its own; filtering here retires them in place instead.
    """
    entries = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        relpath, _, token = line.partition(_INDEX_SEP)
        if Path(relpath).name in _VOLATILE_RUN_FILES:
            continue
        entries[relpath] = token
    return entries


def _stale_paths(remote: Mapping[str, str], local: Mapping[str, str]) -> set[str]:
    """Paths the remote holds that this box lacks or, where tokened, differs on.

    With a paths-only index every token is ``""``, so this reduces to "what am I
    missing" and an existing local file is never disturbed. With a hashed index
    it additionally reports files whose content moved on.
    """
    return {
        relpath
        for relpath, token in remote.items()
        if relpath not in local or (token and local[relpath] != token)
    }


def _index_relpath(relpath: str) -> str:
    """The sidecar object name beside a ``<id>.tar``."""
    return f"{relpath[: -len('.tar')]}{_INDEX_SUFFIX}"


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
    r"""Records the on-disk state of every mutable artifact this box has pushed."""

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


def _run_dir_signature(path: Path) -> str:
    r"""Digest of a run directory's payload: its contents, minus the timing log."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        # ``is_file`` follows symlinks, so a step's link into the store hashes
        # as the sample it points at -- the bytes _tar_dir dereferences into the
        # archive -- and a dangling one is skipped, as it is there.
        if not item.is_file() or item.name in _VOLATILE_RUN_FILES:
            continue
        digest.update(f"{item.relative_to(path)}\0{file_sha256(item)}\n".encode())
    return digest.hexdigest()


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
_ANCHOR_NOISE = "trajectories/anchor_noise"
_AXIS_REFRESH = "trajectories/axis_refresh"


class Syncer:
    """Sync a real :class:`Store` (and its trajectory outputs) to a remote."""

    def __init__(
        self,
        store: Store,
        transport: Transport,
        trajectories: Path | None = None,
        *,
        force: bool = False,
        strict: bool = False,
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
        #: Whether a transfer failure propagates instead of being recorded.
        #: False for a trajectory, whose GPU hours must not be thrown away
        #: because a remote blinked; True for the CLI at the bottom of this
        #: module, whose only job *is* the transfer, so its exit status has to
        #: tell the truth about whether one happened.
        self.strict = strict
        self._unsynced: dict[str, str] = {}
        #: Remote object names per directory, listed lazily; see
        #: :meth:`_remote_names`.
        self._remote_listings: dict[str, set[str]] = {}

    @classmethod
    def from_env(
        cls, store: Store, *, force: bool = False, strict: bool = False
    ) -> Syncer | None:
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
        return cls(store, make_transport(remote), force=force, strict=strict)

    # --- surviving an unreachable remote --------------------------------- #

    @property
    def unsynced(self) -> dict[str, str]:
        """Transfers that did not happen, mapped to the error that stopped them.

        Empty when everything landed. Keys read ``push <relpath>``,
        ``pull <relpath>`` or ``list <reldir>``, since "the remote is missing
        this" and "this box is missing this" call for different reactions.

        An entry is dropped as soon as a later attempt on the same operation
        succeeds, so this describes the state at the end of a run rather than
        logging everything that ever went wrong: an eager push that failed and
        was then picked up by :meth:`push_after_run` leaves nothing behind.
        """
        return dict(self._unsynced)

    @contextmanager
    def _attempt(self, what: str) -> Generator[None]:
        """Carry out one transfer, surviving a remote that is not there.

        Every artifact offered to a transport is already durable on local disk,
        so a failed transfer costs a later retry while a raised exception costs
        the whole trajectory -- hours of rented GPU thrown away for the one
        thing here that was never the point. The failure is recorded instead,
        and the backstops take it from there: :meth:`push_after_run` sweeps at
        the end of the run, and the next run on this box sweeps the store
        again from scratch.

        That recovery holds only as long as the box does, which is why the
        record exists rather than just a log line -- see
        :func:`format_unsynced`.
        """
        try:
            yield
        except Exception as exc:  # noqa: BLE001 -- any transfer failure is survivable
            if self.strict:
                raise
            self._unsynced[what] = f"{type(exc).__name__}: {exc}"
            logger.warning("could not sync (%s): %s", what, exc)
        else:
            self._unsynced.pop(what, None)

    def _list(self, reldir: str) -> list[str]:
        """Remote names under ``reldir``; empty if the remote cannot be read.

        The emptiness is *recorded* rather than silent, which is the whole
        point: a listing that failed and one that genuinely found nothing are
        the same value here, and only :attr:`unsynced` tells them apart.
        """
        names: list[str] = []
        with self._attempt(f"list {reldir}"):
            names = self.transport.list_names(reldir)
        return names

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
        self._push_dir(self.store.adapter_dir(wid), f"{_ADAPTERS}/{wid}.tar", sign=None)

    def push_training_sample(self, sample_id: str) -> None:
        """Upload one cached training subsample (immutable, named by its hash)."""
        path = self.store.training_sample_path(sample_id)
        self._push_file(path, f"{_SAMPLES}/{path.name}", sign=None)

    def push_measurement(self, wid: str) -> None:
        """Upload checkpoint ``wid``'s measurement bundle if it has changed.

        Mutable, so unlike an adapter this cannot skip on mere presence: a
        second trait, a later probe, or the DeltaP for the step about to run
        all add files to a bundle that is already on the remote.
        """
        self._push_dir(
            self.store.measurement_dir(wid),
            f"{_MEASUREMENTS}/{wid}.tar",
            sign=_dir_signature,
            index=_paths_index,
        )

    def push_run_dir(self, run_dir: Path) -> None:
        """Upload one trajectory's run directory if its payload has changed.

        Signed by :func:`_run_dir_signature` rather than :func:`_dir_signature`
        because a re-run rewrites this directory even when it recomputes
        nothing, and this archive is the only real transfer such a run would
        otherwise make.
        """
        self._push_dir(
            run_dir,
            f"{_RUNS}/{run_dir.name}.tar",
            sign=_run_dir_signature,
            index=_hashed_index,
        )

    def push_base_probe(self, path: Path) -> None:
        """Upload one base-probe summary if it has changed."""
        self._push_file(path, f"{_BASE_PROBES}/{path.name}", sign=_file_signature)

    # --- push: sweeps ---------------------------------------------------- #

    def push_store(self) -> None:
        """Push every adapter, training sample and measurement bundle.

        Immutable artifacts are settled by one listing per kind and mutable
        ones by a local stat walk each, so on a store whose eager pushes all
        landed this costs two round trips however many artifacts it holds --
        which is what makes it usable as an end-of-run backstop rather than
        only as a bulk upload.
        """
        for wid in _child_names(self.store.adapters):
            self.push_adapter(wid)
        for sample in _child_files(self.store.training_samples):
            self._push_file(sample, f"{_SAMPLES}/{sample.name}", sign=None)
        for wid in _child_names(self.store.measurements):
            self.push_measurement(wid)

    def push_base_probes(self) -> None:
        """Push every base-probe summary next to the trajectories root."""
        for probe in _child_files(self.trajectories / "base_probes"):
            self.push_base_probe(probe)

    def push_anchor_noise(self, path: Path) -> None:
        """Upload one anchor-noise summary if it has changed.

        Its own remote kind rather than a base probe's, despite both being one
        JSON file keyed by the base ``weights_id``: they are pulled together for
        plotting, and sharing a directory would put two unrelated schemas behind
        one name with only the filename to tell them apart.
        """
        self._push_file(path, f"{_ANCHOR_NOISE}/{path.name}", sign=_file_signature)

    def push_anchor_noises(self) -> None:
        """Push every anchor-noise summary next to the trajectories root."""
        for summary in _child_files(self.trajectories / "anchor_noise"):
            self.push_anchor_noise(summary)

    def push_axis_refresh(self, path: Path) -> None:
        """Upload one axis-refresh summary if it has changed.

        Its own remote kind rather than an anchor-noise one, for the reason
        :meth:`push_anchor_noise` gives: both are a single JSON keyed by the
        base ``weights_id``, and sharing a directory would leave two schemas
        behind one name with only the filename to tell them apart.
        """
        self._push_file(path, f"{_AXIS_REFRESH}/{path.name}", sign=_file_signature)

    def push_axis_refreshes(self) -> None:
        """Push every axis-refresh summary next to the trajectories root."""
        for summary in _child_files(self.trajectories / "axis_refresh"):
            self.push_axis_refresh(summary)

    def push_after_run(self, run_dir: Path) -> None:
        """Backstop flush once a run finishes: the store, ``run_dir``, probes,
        anchor-noise and axis-refresh summaries.

        A run pushes each artifact as it is produced, so by the time this runs
        the only genuinely new object is usually ``run_dir`` itself (its
        ``trajectory.json`` is written last). It sweeps anyway, to catch
        anything an eager push missed -- an artifact written by a code path
        that does not sync, or one produced while the remote was unreachable.

        The anchor-noise sweep is why that promise holds for summaries too.
        They used to be uploaded only by :mod:`method.anchor_noise` at the
        moment it wrote one, which is an eager push with no backstop behind it:
        a summary *edited* later -- by :mod:`method.backfill_latent_cosine`, say
        -- was never offered to a transport again, and died with the box. Every
        other trajectory artifact was already covered by a sweep; this was the
        one that was not.
        """
        self.push_store()
        self.push_run_dir(run_dir)
        self.push_base_probes()
        self.push_anchor_noises()
        self.push_axis_refreshes()

    # --- pull (remote -> local) ----------------------------------------- #

    def pull_before_run(self, selection: StoreSelection | None = None) -> None:
        r"""Fetch the reusable prefix so ``has_adapter`` hits work locally."""
        sample_names = (
            None
            if selection is None
            else frozenset(
                # Via the store, so the ``.jsonl`` naming stays in one place.
                self.store.training_sample_path(sample_id).name
                for sample_id in selection.training_sample_ids
            )
        )
        weights_ids = None if selection is None else selection.weights_ids
        self._pull_dirs(
            _ADAPTERS,
            self.store.adapters,
            sign=None,
            present=self.store.has_adapter,
            wanted=weights_ids,
        )
        self._pull_files(
            _SAMPLES, self.store.training_samples, sign=None, wanted=sample_names
        )
        self._pull_dirs(
            _MEASUREMENTS,
            self.store.measurements,
            sign=_dir_signature,
            index=_paths_index,
            wanted=weights_ids,
        )

    def pull_for_plotting(self) -> None:
        """Fetch just what the collector reads: run dirs and base probes.

        The plotting box needs no adapters, samples or datasets -- the numbers
        the figures use are embedded in each ``trajectory.json`` -- so this
        pulls only ``trajectories/`` and is safe to run on a machine that never
        touches a GPU.

        Uses a hashed index, so a run directory already on this box is refreshed
        when the remote's copy has *changed*, not merely when files are missing
        from it. That is what a rewritten ``trajectory.json`` looks like -- a
        backfill, or a re-run that measured a new trait -- and skipping it was
        why a plotting box could sit on numbers a GPU box had already corrected.
        """
        self._pull_dirs(
            _RUNS,
            self.trajectories,
            sign=_run_dir_signature,
            index=_hashed_index,
        )
        self._pull_files(
            _BASE_PROBES, self.trajectories / "base_probes", sign=_file_signature
        )
        self._pull_files(
            _ANCHOR_NOISE, self.trajectories / "anchor_noise", sign=_file_signature
        )
        self._pull_files(
            _AXIS_REFRESH, self.trajectories / "axis_refresh", sign=_file_signature
        )

    # --- reading a few files out of a whole bundle ----------------------- # The escape hatch from the granularity the module docstring calls coarse: a bundle is one remote object, so.

    def remote_measurement_ids(self) -> set[str]:
        """Checkpoint ids the remote holds a measurement bundle for."""
        return {
            name[: -len(".tar")]
            for name in self._remote_names(_MEASUREMENTS)
            if name.endswith(".tar")
        }

    def measurement_index(self, wid: str) -> dict[str, str] | None:
        """What the remote's bundle for ``wid`` contains, for kilobytes.

        The sidecar :meth:`_pull_missing` consults, exposed on its own so a
        caller can decide *before* spending an archive's worth of bandwidth
        whether that archive holds anything it wants. Values are empty strings:
        a measurement index names paths and nothing else (:func:`_paths_index`).

        ``None`` when there is no sidecar to read -- an archive pushed before
        indexing existed, or an index this box could not fetch, which is
        recorded in :attr:`unsynced` like any other failed transfer. Both mean
        "the contents are unknown", and a caller that still wants the files has
        no option but to read the archive and see.
        """
        index_name = f"{wid}{_INDEX_SUFFIX}"
        if index_name not in self._remote_names(_MEASUREMENTS):
            return None
        relpath = f"{_MEASUREMENTS}/{index_name}"
        parsed: dict[str, str] | None = None
        with self._attempt(f"pull {relpath}"):
            with _scratch_file(suffix=_INDEX_SUFFIX) as tmp:
                self.transport.download(relpath, tmp)
                parsed = _parse_index(tmp.read_text(encoding="utf-8"))
        return parsed

    def extract_measurement_files(
        self,
        wid: str,
        dest_dir: Path,
        *,
        wanted: Callable[[str], bool],
        stage_dir: Path | None = None,
    ) -> list[str]:
        """Copy just the files ``wanted`` accepts out of the remote's bundle.

        Returns the member paths written, relative to ``dest_dir``.

        By default the archive is streamed and discarded as it goes, so the
        peak disk cost is what ``wanted`` kept rather than the bundle's size.
        ``stage_dir`` asks instead for the plain thing -- download the whole
        archive there, extract, delete it -- which needs room for one archive
        but survives a transport whose streaming is unhappy. Either way the
        bandwidth is the same: the object is read end to end, because a tar
        cannot be indexed into from outside.

        Failures are absorbed through :meth:`_attempt` as everywhere else, so a
        bundle that could not be read costs itself and not the sweep. It leaves
        nothing half-written to mistake for a complete file: every member lands
        through :func:`~method.store.atomic_file`.
        """
        relpath = f"{_MEASUREMENTS}/{wid}.tar"
        kept: list[str] = []
        with self._attempt(f"pull {relpath}"):
            if stage_dir is None:
                with self.transport.open_stream(relpath) as raw:
                    with tarfile.open(fileobj=raw, mode="r|") as tar:
                        kept = _copy_wanted(tar, dest_dir, wanted)
                    # The end-of-archive marker is not the end of the object:
                    # a tar is padded to a block boundary. Reading the tail
                    # lets the sender finish and exit 0 instead of dying on a
                    # pipe we closed under it.
                    while raw.read(1 << 20):
                        pass
            else:
                stage_dir.mkdir(parents=True, exist_ok=True)
                with _staged_file(stage_dir, suffix=".tar") as tmp:
                    self.transport.download(relpath, tmp)
                    with tarfile.open(tmp, "r") as tar:
                        kept = _copy_wanted(tar, dest_dir, wanted)
        return kept

    def pick_measurement_files(
        self,
        wid: str,
        dest_dir: Path,
        *,
        wanted: Callable[[str], bool],
        window: int = RANGE_WINDOW,
    ) -> list[str]:
        """:meth:`extract_measurement_files`, without reading the whole archive.

        Same result, a different bill. That one reads the object end to end and
        keeps what it wants; this one walks the archive's headers over ranged
        reads and transfers only the members that were asked for, which on a
        bundle whose bulk is per-sample tensors is a hundredth of the bytes
        (:func:`_walk_ranged`). It needs a transport with
        :meth:`Transport.read_range` behind a backend that serves ranges --
        every rclone backend with an HTTP Range, and a plain path -- and it
        makes one request per member rather than one per archive, so on a
        remote that is slow to *answer* rather than slow to send, the whole-
        archive read can still win.

        Correctness is worth checking rather than assuming: the walk depends on
        every header being where the previous member's length says it is. The
        caller holds the index that says which files should have arrived, so
        comparing the two afterwards costs nothing and turns a silent
        mis-parse into a visible one.
        """
        relpath = f"{_MEASUREMENTS}/{wid}.tar"
        kept: list[str] = []
        with self._attempt(f"pull {relpath}"):
            kept = _walk_ranged(
                self.transport, relpath, dest_dir, wanted=wanted, window=window
            )
        return kept

    # --- shared machinery ----------------------------------------------- #
    #
    # ``sign`` runs through all of it: the function that reduces an artifact to
    # the signature the ledger compares, or ``None`` for an immutable artifact,
    # which needs no ledger because its id already determines its bytes.

    def _remote_names(self, reldir: str) -> set[str]:
        """Object names under ``reldir``, listed at most once per syncer.

        Immutable artifacts are settled by presence, and presence used to mean
        one ``exists`` per artifact -- an rclone process and a network round
        trip each. The end-of-run sweep therefore paid for every adapter and
        sample any run had ever put in the store, a bill that grew all sweep
        long and fell on runs that produced nothing new as heavily as on ones
        that did. One listing per kind answers all of them.

        Staleness can only ever cost bytes, never correctness: a name this
        missed (another box uploaded it after the listing) is re-uploaded with
        content its id says is identical, and one it holds is one the remote
        genuinely had. A *failed* listing is deliberately not cached, so the
        next artifact retries the listing rather than inheriting an empty
        answer -- which would tar and upload the entire store into a remote
        that is not there.
        """
        names = self._remote_listings.get(reldir)
        if names is None:
            names = set(self.transport.list_names(reldir))
            self._remote_listings[reldir] = names
        return names

    def _already_pushed(self, relpath: str, signature: str | None) -> bool:
        """Whether the remote already holds this artifact in its current form.

        ``signature is None`` marks an immutable artifact, for which presence
        is proof: the id determines the bytes. Mutable ones are settled against
        the ledger instead, since presence tells us nothing about *which*
        version is up there.
        """
        if signature is None:
            reldir, _, name = relpath.rpartition("/")
            return name in self._remote_names(reldir)
        return not self.force and self.ledger.is_current(relpath, signature)

    def _note_pushed(self, relpath: str) -> None:
        """Fold a just-uploaded object into the cached listing of its directory.

        Without this, the end-of-run sweep would re-upload whatever the run's
        own eager pushes sent after the listing was taken.
        """
        reldir, _, name = relpath.rpartition("/")
        names = self._remote_listings.get(reldir)
        if names is not None:
            names.add(name)

    def _push_dir(
        self,
        src: Path,
        relpath: str,
        *,
        sign: Callable[[Path], str] | None,
        index: Callable[[Path], str] | None = None,
    ) -> None:
        """Upload one directory artifact, plus the index a merging pull needs.

        ``index`` is set exactly for the mutable artifacts, and is what lets a
        puller ask "does this archive hold anything I lack" without fetching
        the archive. It is uploaded *after* the tar, so an index is never
        visible promising content the tar has not received yet.
        """
        if not src.is_dir():
            return
        with self._attempt(f"push {relpath}"):
            # Signed before tarring, never after: a concurrent write landing in
            # between then makes the recorded signature older than what was
            # uploaded, costing one redundant push later. Signing after would
            # make it *newer*, and the ledger would skip content that never
            # went up.
            signature = sign(src) if sign else None
            if self._already_pushed(relpath, signature):
                logger.debug("skip push: %s", relpath)
                # The tar is up to date, but it may predate indexing entirely.
                # Writing the sidecar now costs kilobytes and is what lets
                # archives already on the remote take part in merging pulls,
                # instead of waiting for a content change to re-upload them.
                if index is not None:
                    self._ensure_index(src, relpath, index)
                return
            with _scratch_file(suffix=".tar") as tmp:
                _tar_dir(src, tmp)
                self.transport.upload(tmp, relpath)
            if index is not None:
                self._upload_index(src, relpath, index)
            self._record_push(relpath, signature)

    def _ensure_index(
        self, src: Path, relpath: str, index: Callable[[Path], str]
    ) -> None:
        """Write the sidecar for an archive already on the remote, if missing."""
        index_relpath = _index_relpath(relpath)
        reldir, _, name = index_relpath.rpartition("/")
        if name in self._remote_names(reldir):
            return
        self._upload_index(src, relpath, index)

    def _upload_index(
        self, src: Path, relpath: str, index: Callable[[Path], str]
    ) -> None:
        index_relpath = _index_relpath(relpath)
        with _scratch_file(suffix=_INDEX_SUFFIX) as tmp:
            tmp.write_text(index(src), encoding="utf-8")
            self.transport.upload(tmp, index_relpath)
        self._note_pushed(index_relpath)

    def _push_file(
        self, src: Path, relpath: str, *, sign: Callable[[Path], str] | None
    ) -> None:
        if not src.is_file():
            return
        with self._attempt(f"push {relpath}"):
            signature = sign(src) if sign else None
            if self._already_pushed(relpath, signature):
                logger.debug("skip push: %s", relpath)
                return
            self.transport.upload(src, relpath)
            self._record_push(relpath, signature)

    def _record_push(self, relpath: str, signature: str | None) -> None:
        """Note an upload that landed, so nothing later repeats it."""
        if signature is None:
            self._note_pushed(relpath)
        else:
            self.ledger.record(relpath, signature)
        logger.info("pushed %s", relpath)

    def _pull_dirs(
        self,
        reldir: str,
        local_parent: Path,
        *,
        sign: Callable[[Path], str] | None,
        present=None,
        index: Callable[[Path], str] | None = None,
        wanted: frozenset[str] | None = None,
    ) -> None:
        r"""Fetch what the remote holds under ``reldir`` and this box does not."""
        names = self._list(reldir)
        available = set(names)
        for name in names:
            if not name.endswith(".tar"):
                continue
            wid = name[: -len(".tar")]
            if wanted is not None and wid not in wanted:
                continue
            local = local_parent / wid
            relpath = f"{reldir}/{name}"
            have = present(wid) if present else local.is_dir()
            if not have:
                self._pull_whole_dir(relpath, local, sign=sign)
                continue
            if index is None:
                continue
            index_name = f"{wid}{_INDEX_SUFFIX}"
            if index_name not in available:
                logger.debug("no index for %s; leaving local copy alone", relpath)
                continue
            self._pull_missing(
                relpath, f"{reldir}/{index_name}", local, sign=sign, index=index
            )

    def _pull_whole_dir(
        self, relpath: str, local: Path, *, sign: Callable[[Path], str] | None
    ) -> None:
        with self._attempt(f"pull {relpath}"):
            with _scratch_file(suffix=".tar") as tmp:
                self.transport.download(relpath, tmp)
                _untar_dir(tmp, local)
            self._record_pull(relpath, local, sign)
            logger.info("pulled %s", relpath)

    def _pull_missing(
        self,
        relpath: str,
        index_relpath: str,
        local: Path,
        *,
        sign: Callable[[Path], str] | None,
        index: Callable[[Path], str],
    ) -> None:
        """Add the parts of a remote archive this box lacks, deleting nothing."""
        with self._attempt(f"pull {relpath}"):
            with _scratch_file(suffix=_INDEX_SUFFIX) as tmp:
                self.transport.download(index_relpath, tmp)
                remote = _parse_index(tmp.read_text(encoding="utf-8"))
            stale = _stale_paths(remote, _parse_index(index(local)))
            if not stale:
                logger.debug("up to date: %s", relpath)
                return
            with _scratch_file(suffix=".tar") as tmp:
                self.transport.download(relpath, tmp)
                _merge_from_tar(tmp, local, stale)
            # Deliberately *not* recorded in the ledger.
            logger.info("merged %d file(s) from %s", len(stale), relpath)

    def _record_pull(
        self, relpath: str, local: Path, sign: Callable[[Path], str] | None
    ) -> None:
        """Note that local content now matches the remote object it came from.

        Only correct after a *whole* download, where the two are byte-identical.
        Without it the next push would ship every bundle this box pulled
        straight back up unchanged.
        """
        if sign:
            self.ledger.record(relpath, sign(local))

    def _pull_files(
        self,
        reldir: str,
        local_parent: Path,
        *,
        sign: Callable[[Path], str] | None,
        wanted: frozenset[str] | None = None,
    ) -> None:
        """Fetch the files under ``reldir`` this box lacks.

        ``wanted`` holds whole file *names* rather than the ids
        :meth:`_pull_dirs` filters on, because these artifacts are not all
        named alike: a training sample is ``<id>.jsonl`` while a base probe is
        named for the run that wrote it. Letting the caller name the files
        keeps that knowledge with whoever owns the naming.
        """
        for name in self._list(reldir):
            if wanted is not None and name not in wanted:
                continue
            dest = local_parent / name
            if dest.exists():
                continue
            relpath = f"{reldir}/{name}"
            with self._attempt(f"pull {relpath}"):
                self.transport.download(relpath, dest)
                if sign:
                    self.ledger.record(relpath, sign(dest))
                logger.info("pulled %s", relpath)


# --------------------------------------------------------------------------- #
# Reporting what did not make it.
# --------------------------------------------------------------------------- #


#: Enough entries to see the shape of the failure without turning an email into
#: a listing of the store.
_UNSYNCED_SHOWN = 8


def format_unsynced(
    unsynced: Mapping[str, str], *, limit: int = _UNSYNCED_SHOWN
) -> str:
    """Describe :attr:`Syncer.unsynced`, or ``""`` when there is nothing to say.

    Lives here rather than in :mod:`method.report` so that the run's log line
    and the emailed summary cannot end up describing the same state
    differently, and so :mod:`method.probe_base` -- which sends no mail -- can
    still say the one thing that matters about it.

    That one thing is the closing sentence. A failed push is self-healing:
    the artifact never left local disk, :meth:`Syncer.push_after_run` sweeps
    the whole store at the end of the run, and the next run on the box sweeps
    it again, so an ordinary blip resolves itself with nobody watching. What is
    *not* self-healing is releasing the box first -- the recovery and the
    artifacts both live on the same disposable disk.
    """
    if not unsynced:
        return ""
    lines = [f"{len(unsynced)} transfer(s) did not happen:"]
    for what, reason in list(unsynced.items())[:limit]:
        lines.append(f"  {what}  ({reason})")
    if len(unsynced) > limit:
        lines.append(f"  ... and {len(unsynced) - limit} more")
    lines.append(
        "Anything listed as 'push' is still on this box's disk and goes up on "
        "the next run here, which sweeps the whole store -- so do not release "
        "the box until a sync has succeeded. Anything listed as 'pull' or "
        "'list' means this run could not read the remote, and may have redone "
        "work another box had already finished."
    )
    return "\n".join(lines)


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


@contextmanager
def _staged_file(directory: Path, *, suffix: str = "") -> Generator[Path]:
    """:func:`_scratch_file`, on a filesystem the caller names.

    Exists because the archives :meth:`Syncer.extract_measurement_files`
    stages are gigabytes: the default temp directory is often a small root
    partition or a tmpfs living in RAM, and either one turns a working pull
    into a disk-full error. Removed on the way out however the block ends, so
    the "not relevant" bytes never outlive the extraction that read them.
    """
    fd, name = tempfile.mkstemp(dir=directory, suffix=suffix)
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
        choices=["push", "push-runs", "pull", "pull-plots"],
        help=(
            "push: upload the whole local store + trajectories; "
            "push-runs: upload only run dirs, base probes and anchor-noise "
            "summaries -- the files a backfill rewrites; "
            "pull: fetch the *entire* remote store, warming a box for any "
            "trajectory (a run pulls only what its own config reads, so this "
            "is only worth it to prefetch off the clock); "
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
    # ``strict``: a trajectory swallows transfer failures because its real work
    # is on the GPU, but this command's entire job is the transfer. Swallowing
    # here would exit 0 having moved nothing, and the shell that invoked it
    # would believe the store was safe.
    syncer = Syncer.from_env(store, force=args.force, strict=True)
    if syncer is None:
        raise SystemExit(f"{REMOTE_ENV} is not set; nothing to sync to.")

    if args.action in {"push", "push-runs"}:
        if args.action == "push":
            # The store is swept once, not once per run dir: it is shared by
            # every trajectory, so folding it into the per-run loop re-tarred
            # and re-uploaded every measurement bundle N times over for N run
            # dirs. Sweeping it here also means a store with no runs still gets
            # pushed.
            syncer.push_store()
        # ``push-runs`` skips that sweep, because a measurement bundle is one remote object: touching any file in it re-uploads all of it, tensors included.
        for run_dir in sorted(
            p for p in syncer.trajectories.glob("*_seed*") if p.is_dir()
        ):
            syncer.push_run_dir(run_dir)
        syncer.push_base_probes()
        syncer.push_anchor_noises()
        syncer.push_axis_refreshes()
    elif args.action == "pull":
        syncer.pull_before_run()
    else:
        syncer.pull_for_plotting()


if __name__ == "__main__":
    main()
