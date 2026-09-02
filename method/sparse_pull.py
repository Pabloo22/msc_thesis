r"""Fetch a few named files out of the remote's whole-bundle archives.

    poetry run python -m method.sparse_pull --dry-run
    poetry run python -m method.sparse_pull
    poetry run python -m method.sparse_pull --trunk a --jobs 4
    poetry run python -m method.sparse_pull --stage-dir /var/tmp --jobs 1

**The problem.** :mod:`method.sync` ships one tar per artifact, and its own
design notes call that granularity coarse. Reading is where it bites hardest.
Recomputing a projection difference on another axis needs three tensors per
checkpoint that are all already measured -- the target activations, the
predicted activations, and a persona vector -- and every one of them is a
``mean_by_layer.pt`` of about 400KB. They sit inside a measurement bundle whose
bulk is the per-sample ``samples_layer<L>.pt`` tensors beside them, 175MB
apiece. Across the 19 checkpoints of the three exp2 trunks that is ~300MB of
wanted bytes inside ~63GB of archive -- measured at 2.9GB for the leanest
checkpoint and 6.0GB for the base -- so a 200x overhead, on more disk than a
laptop has to spare.

**What this does about it.** A tar has no directory, so a member can only be
found by reading a header, learning how long that member is, and stepping over
its data to the next header. Over a stream that means reading the whole object.
Over *ranged* reads it means transferring the headers, the members asked for,
and nothing else -- which is what :data:`FetchMode.RANGED`, the default, does
(:meth:`method.sync.Syncer.pick_measurement_files`). ~300MB moves instead of
~63GB, and the archive is never stored: peak disk is what was kept.

The two whole-archive modes remain, and matter. :data:`FetchMode.STREAM` reads
each object end to end through a pipe and keeps the members going past
(:meth:`method.sync.Syncer.extract_measurement_files`); it is the reference the
ranged walk is checked against, and the fallback for a backend that cannot
serve byte ranges. :data:`FetchMode.STAGE` is the literal version -- download
the archive, extract, delete it -- for a transport whose streaming misbehaves;
it needs room for one bundle per job.

Which is cheaper depends on what the remote is slow at. Streaming pays for
bytes: measured against this project's R2 bucket at ~2MB/s on one connection,
all 19 checkpoints is ~9 hours. The ranged walk pays for round trips instead --
~1s per request, ~100 of them per bundle -- so it is minutes rather than hours
here, but it would lose to streaming on a remote with a fast pipe and a slow
answer. Both divide by ``--jobs``: the ~2MB/s is a per-connection cap and not
the link, since a second concurrent stream ran at the same speed rather than
splitting the first's.

**Two things make it cheap to re-run, and one makes it checkable.** Each bundle
carries a sidecar index naming its contents, so a checkpoint whose wanted files
are all already here is skipped for kilobytes rather than gigabytes; and within
an archive that is read, a member already on disk is not fetched again. An
interrupted sweep therefore resumes where it stopped, and a second run costs
one index per checkpoint. That same index is then read back against what
arrived: a header walk is the one mode here that could go quietly wrong -- it
trusts each member's declared length to find the next header -- so a file the
index named and the fetch did not produce is reported as an error rather than
mistaken for a bundle that had less in it.

**What the means can and cannot answer.** :func:`method.latent.project` is
linear in the activations, so the *mean* projection difference is
``project(mean h_target) - project(mean h_pred)`` exactly -- the number a
DeltaP series plots, recovered from two 400KB files instead of two 175MB ones.
Its *spread* is not: ``std``, the percentiles and ``n`` in a ``delta_p_*.json``
are properties of the per-sample rows, and no combination of the means gives
them. A recomputed series that needs error bars needs the per-sample tensors,
which is the 63GB this module exists to avoid; pass ``--include
'delta_p_*/*/samples_layer*.pt'`` if that is what is wanted.

**Why the files do not land in the real store by default.** A bundle here is
deliberately partial, and :mod:`method.sync` tolerates partial bundles on the
way in but not on the way out: ``push`` re-tars a local measurement directory
whole and last-writer-wins on the remote, so a box holding 8MB of a 3.3GB
bundle would replace the remote's copy with its own and take the per-sample
tensors -- GPU hours, not bytes -- down with it. The thin copy therefore gets a
store root of its own, which nothing pushes, and is read with
``Store(Path("store-thin"))``. ``--into-real-store`` overrides that for a box
where the trade is understood.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path

from method import experiments
from method.store import Store, get_weights_id
from method.sync import REMOTE_ENV, Syncer, format_unsynced
from method.utils import DOTENV_PATH, REPO_ROOT, load_dotenv

logger = logging.getLogger(__name__)


class FetchMode(StrEnum):
    """How much of an archive is read to get a few files out of it.

    ``RANGED``
        Walk the archive's headers over ranged reads and transfer only the
        members wanted -- ~300MB across the 19 checkpoints instead of ~63GB.
        The default, and the only one whose cost is proportional to what is
        being collected. Needs a backend that serves byte ranges (R2, S3, B2,
        Drive, a mounted path) and pays a round trip per member, so it is the
        wrong choice on a remote that is slow to answer rather than slow to
        send.
    ``STREAM``
        Read the whole archive through a pipe and keep the members that go
        past. Transfers everything; stores nothing but what was kept. The
        fallback when a backend cannot serve ranges, and the reference the
        ranged walk is checked against.
    ``STAGE``
        Download each archive to ``--stage-dir``, extract, delete it. Needs
        room for one bundle per job. For a transport whose streaming
        misbehaves.
    """

    RANGED = "ranged"
    STREAM = "stream"
    STAGE = "stage"


#: Where the thin copy goes: a store root of its own, for the reason the module
#: docstring gives. ``Store(THIN_STORE)`` reads it like any other store.
THIN_STORE = REPO_ROOT / "store-thin"

#: Glob patterns, matched against a member's path inside the bundle, naming the
#: files a projection-difference recomputation reads. ``*`` spans ``/`` here
#: (:mod:`fnmatch`'s rule, not the shell's), which only ever widens a pattern
#: onto more of the small files these already select.
DEFAULT_PATTERNS: tuple[str, ...] = (
    # The two activation means DeltaP differences, per probe dataset. The
    # ``_current`` variant is the recomputed-answers view; absent from most
    # bundles, and free where it exists.
    "delta_p_target/*/mean_by_layer.pt",
    "delta_p_predicted/*/mean_by_layer.pt",
    "delta_p_predicted_current/*/mean_by_layer.pt",
    # The axes to project onto: the frozen v^(t) every measured DeltaP used,
    # and the on-policy one method.axis_refresh drew from M_t itself.
    "traits/*/*_response_avg_diff.pt",
    "traits/*/axis_refresh/vector/*_response_avg_diff.pt",
    # h_neutral_t, which is what z_t's p and rho are read off.
    "h_neutral_*/mean_by_layer.pt",
    # Kilobytes, and what lets a recomputed series be checked against the
    # measured one it has to reproduce: behavior.json, latent_cosine.json and
    # every delta_p_*.json the checkpoint already holds.
    "traits/*/*.json",
)


def matcher(patterns: Sequence[str]) -> Callable[[str], bool]:
    """A predicate accepting the member paths any of ``patterns`` matches."""
    return lambda name: any(fnmatch.fnmatchcase(name, p) for p in patterns)


def trunk_checkpoints(trunks: Sequence[str], *, seed: int) -> dict[str, str]:
    """``weights_id -> label`` for every checkpoint on the named trunks.

    Read off the same builder :mod:`method.axis_refresh` uses, so the ids are
    the ones its sweep left vectors at, and -- since that builder is pinned to
    the decay family's trunk in everything ``weights_key`` hashes -- the ones
    exp2 measured DeltaP at. The three trunks share their base checkpoint, so
    3 x 7 checkpoints are 19 distinct ids; the first trunk to name one wins the
    label, and ``t = 0`` is labelled for what it is rather than for whichever
    trunk got there first.
    """
    found: dict[str, str] = {}
    for trunk in trunks:
        cfg = experiments.build_axis_refresh_configs(trunk=trunk, seed=seed)[0]
        for t in range(len(cfg.steps) + 1):
            label = "base t=0" if t == 0 else f"trunk {trunk} t={t}"
            found.setdefault(get_weights_id(cfg, t), label)
    return found


def _fetch_one(
    syncer: Syncer,
    wid: str,
    label: str,
    *,
    bundle: Path,
    wanted: Callable[[str], bool],
    dry_run: bool,
    mode: FetchMode,
    stage_dir: Path | None,
) -> tuple[bool, int, int]:
    """Bring ``wid``'s wanted files down.

    Returns whether the archive had to be read, and how many files and bytes
    that kept -- an archive read for nothing is the case worth being able to
    see, so the two are counted separately.

    The index is consulted first and settles three of the four cases for
    kilobytes: the remote's bundle holds none of the wanted files, or holds
    only files this box already has, or names some it lacks. Only the last
    reads the archive. A bundle with no index falls through to reading it,
    because "unknown contents" and "contents I want" are indistinguishable
    from here.
    """
    index = syncer.measurement_index(wid)
    missing: list[str] = []
    if index is not None:
        matched = sorted(path for path in index if wanted(path))
        if not matched:
            logger.info("[%s] %s: bundle holds none of the wanted files", label, wid)
            return False, 0, 0
        missing = [path for path in matched if not (bundle / path).exists()]
        if not missing:
            logger.info(
                "[%s] %s: already have all %d wanted file(s)", label, wid, len(matched)
            )
            return False, 0, 0
        logger.info(
            "[%s] %s: %d of %d wanted file(s) missing; reading the archive",
            label,
            wid,
            len(missing),
            len(matched),
        )
    else:
        logger.info(
            "[%s] %s: no index on the remote; reading the archive to find out",
            label,
            wid,
        )
    if dry_run:
        return False, 0, 0

    # Re-checked per member rather than trusted from the index: it costs a stat
    # against bytes that have already been paid for, and it is what makes a
    # re-run after an interrupted fetch collect only the tail.
    still_wanted: Callable[[str], bool] = (
        lambda name: wanted(name) and not (bundle / name).exists()
    )
    if mode is FetchMode.RANGED:
        kept = syncer.pick_measurement_files(wid, bundle, wanted=still_wanted)
    else:
        kept = syncer.extract_measurement_files(
            wid,
            bundle,
            wanted=still_wanted,
            stage_dir=stage_dir if mode is FetchMode.STAGE else None,
        )
    size = sum((bundle / name).stat().st_size for name in kept)
    logger.info("[%s] %s: kept %d file(s), %.1f MB", label, wid, len(kept), size / 1e6)

    # The index said these were in there. A fetch that came back without them
    # read the archive wrongly -- the failure mode a header walk has and a
    # whole-archive read does not -- and saying so is what keeps it from
    # passing for a bundle that simply had less in it than expected.
    absent = [path for path in missing if not (bundle / path).exists()]
    if absent:
        logger.error(
            "[%s] %s: %d file(s) the index names did not arrive, e.g. %s",
            label,
            wid,
            len(absent),
            absent[0],
        )
    return True, len(kept), size


def run(
    syncer: Syncer,
    *,
    weights_ids: dict[str, str],
    patterns: Sequence[str],
    dest: Path,
    dry_run: bool = False,
    mode: FetchMode = FetchMode.RANGED,
    stage_dir: Path | None = None,
    jobs: int = 1,
) -> None:
    """Sweep ``weights_ids``, keeping the files ``patterns`` names.

    Takes the syncer rather than building one, so what it pulls from is the
    caller's decision and a test can hand it a local directory.

    ``jobs`` checkpoints are fetched at once. Worth having because one stream
    is throttled well below the link: a second concurrent read of the same
    remote measured the same rate as the first rather than half of it, so the
    limit is per-connection and the wall time divides. Threads are enough --
    every worker spends its life waiting on a socket -- and they share nothing
    that matters: a bundle is written to its own directory, each transfer is
    its own subprocess, and the syncer's own state is a listing cache whose
    worst race is one redundant listing.
    """
    store = Store(dest)
    wanted = matcher(patterns)
    available = syncer.remote_measurement_ids()
    logger.info(
        "%d checkpoint(s) wanted; the remote holds %d measurement bundle(s) " "[%s]%s",
        len(weights_ids),
        len(available),
        mode,
        " [dry run]" if dry_run else "",
    )

    def fetch(item: tuple[str, str]) -> tuple[bool, int, int]:
        wid, label = item
        if wid not in available:
            logger.warning("[%s] %s: no bundle on the remote", label, wid)
            return False, 0, 0
        return _fetch_one(
            syncer,
            wid,
            label,
            bundle=store.measurement_dir(wid),
            wanted=wanted,
            dry_run=dry_run,
            mode=mode,
            stage_dir=stage_dir,
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outcomes = list(pool.map(fetch, weights_ids.items()))
    archives = sum(read for read, _, _ in outcomes)
    files = sum(kept for _, kept, _ in outcomes)
    total = sum(size for _, _, size in outcomes)

    logger.info(
        "read %d archive(s); kept %d file(s), %.1f MB under %s",
        archives,
        files,
        total / 1e6,
        dest,
    )
    if syncer.unsynced:
        # Every failure here is one checkpoint's, and a re-run picks up exactly
        # those: the index check makes the ones that landed nearly free.
        logger.warning("%s", format_unsynced(syncer.unsynced))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trunk",
        nargs="+",
        default=sorted(experiments.EXP2_TRUNKS),
        choices=sorted(experiments.EXP2_TRUNKS),
        help="which exp2 trunks' checkpoints to fetch for (default: all three)",
    )
    parser.add_argument(
        "--weights-id",
        nargs="+",
        default=None,
        help="fetch for these checkpoint ids instead of a trunk's",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=experiments.EXP2_SEED,
        help="the trunks' fine-tuning seed; picks which checkpoints are named",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help=(
            "extra glob patterns to keep, matched against a file's path inside "
            "the bundle, e.g. 'delta_p_*/*/samples_layer*.pt' for the per-sample "
            "tensors -- 175 MB each, and the reason this script exists"
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="keep only these patterns, replacing the defaults entirely",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=THIN_STORE,
        help=(
            "store root the thin bundles are written under "
            f"(default: {THIN_STORE.name}); read it with Store(Path(...))"
        ),
    )
    parser.add_argument(
        "--into-real-store",
        action="store_true",
        help=(
            "permit --dest to be the real store. Off by default because a "
            "partial bundle there is a bundle 'python -m method.sync push' "
            "would upload over the complete one on the remote"
        ),
    )
    parser.add_argument(
        "--mode",
        type=FetchMode,
        choices=list(FetchMode),
        default=FetchMode.RANGED,
        help=(
            "ranged: transfer only the wanted members, walking the archive's "
            "headers over byte-range reads (default, ~200x less traffic); "
            "stream: read each archive whole and keep what goes past; "
            "stage: download each archive to --stage-dir, extract, delete it"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help=(
            "checkpoints to fetch at once (default: 3). One stream is capped "
            "well below the link, so this divides the wall time; with "
            "--stage-dir it also multiplies the disk needed, one archive per job"
        ),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help=(
            "where --mode stage puts each archive before extracting it; needs "
            "room for one bundle per job (~3-6 GB each)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what each bundle would give up, reading only the indexes",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # MSC_STORE_REMOTE lives in .env beside the API keys, exactly as the other
    # entry points assume; an exported shell variable still wins.
    load_dotenv(DOTENV_PATH)

    dest = args.dest.resolve()
    if dest == Store().root.resolve() and not args.into_real_store:
        raise SystemExit(
            f"{dest} is the real store; a partial bundle there is one that "
            "`python -m method.sync push` would upload over the complete copy "
            "on the remote. Pass --into-real-store if that is what you want."
        )

    if args.weights_id:
        weights_ids = {wid: "named" for wid in args.weights_id}
    else:
        weights_ids = trunk_checkpoints(args.trunk, seed=args.seed)

    syncer = Syncer.from_env(Store())
    if syncer is None:
        raise SystemExit(f"{REMOTE_ENV} is not set; there is nothing to pull from.")

    run(
        syncer,
        weights_ids=weights_ids,
        patterns=[*(args.only or DEFAULT_PATTERNS), *args.include],
        dest=dest,
        dry_run=args.dry_run,
        mode=args.mode,
        stage_dir=args.stage_dir,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
