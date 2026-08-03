"""Content-addressed storage for checkpoints and measurements.

The durable artifacts are LoRA adapters (small) and measurements (small); full
merged weights are treated as a disposable cache, because any checkpoint can be
rebuilt by replaying ``base -> merge adapter_1 -> ... -> merge adapter_t``. That
replay is exact, not an approximation, because adapter ``t+1`` is trained on top
of the merged result of adapter ``t``. Being disposable, merged weights are also
the one thing here scoped per process rather than shared: each run drops the lot
when it finishes, so two runs on one box (one per GPU) would otherwise delete
each other's mid-step.

Everything is keyed by ``weights_id``, a hash of the recipe that produced the
weights. Two trajectories sharing a prefix therefore share adapters and
measurements for free, and a re-invoked run resumes by finding artifacts already
present. Presence implies completeness because every write lands via an atomic
rename.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from method.config import Backend, TrajectoryConfig
from method.utils import REPO_ROOT

logger = logging.getLogger(__name__)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON for a recipe mapping.

    Distinct from :func:`method.config.to_json`, which takes dataclasses; the
    recipes hashed here are already plain mappings. Key order is fixed and enums
    are stringified so the digest is stable across runs and machines.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


STORE_DIR = REPO_ROOT / "store"

_HASH_LEN = 16


def get_weights_id(cfg: TrajectoryConfig, t: int) -> str:
    """Stable short hash of the weights after ``t`` fine-tuning steps."""
    digest = hashlib.sha256(canonical_json(cfg.weights_key(t)).encode()).hexdigest()
    return f"t{t:02d}-{digest[:_HASH_LEN]}"


def file_sha256(path: Path) -> str:
    """Full SHA-256 of a file's bytes, following symlinks.

    The ids above hash dataset *names*, never dataset *bytes* -- ``dataset/``
    is gitignored, so nothing else pins the content either. This digest is how
    an adapter records which bytes it was actually trained on, so a machine
    with a divergent copy of a dataset fails loudly instead of silently
    reusing an adapter trained on different data.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_training_sample_id(
    dataset: str, version: str, n_examples: int | None, seed: int
) -> str:
    """Stable short hash for a deterministic training-example subsample.

    ``random.Random(seed).sample(...)`` is a pure function of the dataset
    file's contents, ``n_examples`` and ``seed`` -- nothing about which
    trait, trajectory, or step position is asking for it. ``seed`` is
    excluded when ``n_examples`` is ``None`` because no sampling happens in
    that case (the full dataset is used regardless of seed), so every seed
    of a paper-scale ("use the whole dataset") step shares one cached file
    too.
    """
    payload: dict[str, Any] = {
        "dataset": dataset,
        "version": version,
        "n_examples": n_examples,
    }
    if n_examples is not None:
        payload["seed"] = seed
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return digest[:_HASH_LEN]


def _process_alive(pid: int) -> bool:
    """Whether ``pid`` names a running process.

    Signal 0 performs the permission and existence checks without delivering
    anything. ``PermissionError`` means the process exists but belongs to
    another user, which still counts as alive.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def atomic_dir(target: Path) -> Generator[Path]:
    """Yield a scratch directory that is renamed onto ``target`` on success.

    A crash leaves the scratch directory behind instead of a half-written
    ``target``, which is what lets artifact presence stand in for a manifest.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.tmp"))
    try:
        yield scratch
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    if target.exists():
        shutil.rmtree(target)
    os.replace(scratch, target)


@contextmanager
def atomic_file(target: Path) -> Generator[Path]:
    """Yield a scratch path renamed onto ``target`` on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.tmp")
    os.close(fd)
    scratch = Path(tmp_name)
    try:
        yield scratch
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise
    os.replace(scratch, target)


def atomic_symlink(dest: Path, target: Path) -> None:
    """Point ``dest`` at ``target`` via a relative symlink, replacing any prior file.

    Relative (not absolute) so the link keeps resolving if the whole repo is
    moved or copied elsewhere, as long as ``dest`` and ``target`` move
    together -- true for every caller here, since both live under the repo
    root. The temp-then-``os.replace`` dance mirrors :func:`atomic_file` so a
    crash mid-write never leaves ``dest`` half-created.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, dest.parent)
    tmp = dest.parent / f".{dest.name}.tmp{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(relative_target)
    os.replace(tmp, dest)


class Store:
    """Filesystem layout for adapters, measurements and merged weights."""

    def __init__(self, root: Path = STORE_DIR) -> None:
        self.root = root
        self.adapters = root / "adapters"
        self.measurements = root / "measurements"
        # Merged weights are scoped to the process that materialised them, so
        # that two runs sharing a box (one per GPU) cannot evict each other's
        # checkpoints -- see :meth:`evict_all_merged`.
        self.merged_root = root / "merged"
        self.merged = self.merged_root / str(os.getpid())
        self.training_samples = root / "training_samples"
        # Where a training run's raw output dir is staged before the adapter is
        # installed. Like ``merged`` it is disposable and, unlike every other
        # directory here, holds nothing addressed by an id -- see
        # :func:`method.run_trajectory.training_scratch`.
        self.train_scratch = root / "train_scratch"

    @classmethod
    def for_backend(cls, backend: Backend) -> Store:
        """Store rooted per backend, keeping synthetic artifacts quarantined.

        ``weights_id`` deliberately hashes only the recipe, not the backend, so
        a mock run and a real run of the same trajectory collide on every id. If
        they shared a root, mock's fake adapters would be picked up as cache
        hits by the real run and silently poison it. Separate roots make that
        impossible while keeping ids meaningful.
        """
        suffix = "-mock" if backend is Backend.MOCK else ""
        return cls(REPO_ROOT / f"store{suffix}")

    # --- adapters -------------------------------------------------------

    def adapter_dir(self, wid: str) -> Path:
        """Directory holding the LoRA adapter trained to reach ``wid``."""
        return self.adapters / wid

    def has_adapter(self, wid: str) -> bool:
        """Whether a complete adapter for ``wid`` is already on disk."""
        return (self.adapter_dir(wid) / "adapter_config.json").exists()

    # --- measurements ---------------------------------------------------

    def measurement_dir(self, wid: str) -> Path:
        """Directory holding all measurements taken on checkpoint ``wid``."""
        return self.measurements / wid

    def measurement(self, wid: str, name: str) -> Path:
        """Path to a single named measurement artifact for ``wid``."""
        return self.measurement_dir(wid) / name

    def trait_measurement_dir(self, wid: str, trait: str) -> Path:
        """Directory for measurements on ``wid`` that depend on ``trait``.

        ``weights_key`` excludes trait, so two ``TrajectoryConfig``s that
        differ only in trait can share a ``wid``. Artifacts whose *content*
        depends on trait (behavior, extracted vector, latent, DeltaP) must
        live here rather than directly under ``measurement_dir`` or a second
        trait's measurement pass would silently overwrite the first's.
        """
        return self.measurement_dir(wid) / "traits" / trait

    def trait_measurement(self, wid: str, trait: str, name: str) -> Path:
        """Path to a single named, trait-dependent measurement artifact."""
        return self.trait_measurement_dir(wid, trait) / name

    # --- training samples (durable; a pure function of its own id) ------

    def training_sample_path(self, sample_id: str) -> Path:
        """Path to a cached, deterministically-sampled training subset.

        Unlike ``merged``, this is durable, not a rebuild-on-demand cache: it
        is the concrete record of exactly which examples (and in what order)
        a step trained on. It *can* always be regenerated byte-for-byte from
        ``get_training_sample_id``'s inputs plus the dataset file, but
        ``trajectories/*/train_step*.jsonl`` symlinks to it (see
        ``steps.sample_training_file``), so deleting it breaks those links
        until the owning trajectory is re-run.
        """
        return self.training_samples / f"{sample_id}.jsonl"

    # --- merged weights (disposable) ------------------------------------

    def merged_dir(self, wid: str) -> Path:
        """Directory for the materialised full weights of ``wid``."""
        return self.merged / wid

    def has_merged(self, wid: str) -> bool:
        """Whether full weights for ``wid`` are currently materialised."""
        return (self.merged_dir(wid) / "config.json").exists()

    def evict_merged(self, wid: str) -> None:
        """Drop the full weights for ``wid``; they can always be rebuilt."""
        path = self.merged_dir(wid)
        if path.exists():
            logger.info("Evicting merged weights %s", path)
            shutil.rmtree(path, ignore_errors=True)

    def evict_all_merged(self) -> None:
        """Drop every checkpoint *this process* materialised.

        Scoped to this process because it is called at the end of every run
        (:mod:`method.run_trajectory`, :mod:`method.probe_base`) while a second
        run may be halfway through its own. When both shared one ``merged/``
        directory, the first to finish deleted the checkpoint the other was
        evaluating, and the victim died mid-step on a missing ``config.json``.
        Nothing durable was lost -- adapters and measurements are installed and
        pushed as they land, so a re-invocation resumes -- but the GPU hours
        spent on the step in flight were.

        Read-sharing was never the problem: merged directories are named by
        ``weights_id`` and written through :func:`atomic_dir`, so one is either
        absent or a complete, correct checkpoint whoever built it. Only the
        wholesale delete was unsafe, so only it is scoped.
        """
        if self.merged.exists():
            shutil.rmtree(self.merged, ignore_errors=True)

    def evict_orphaned_merged(self) -> None:
        """Drop merged roots whose owning process is gone.

        A run killed outright (spot preemption, ``pkill``) never reaches
        :meth:`evict_all_merged`, and a 7B checkpoint is ~15GB against the
        150GB budget in ``docs/cloud_setup.md`` -- enough leaked roots and the
        next run has nowhere to materialise. Called at the start of a run,
        where the alternative is a human remembering to clear ``store/merged``
        between rentals.

        A live pid is left alone, which is what makes this safe to call while
        another run is in flight. Pid reuse can only cause a leak (a stale root
        skipped because some unrelated process now holds the number), never a
        deletion of something in use.
        """
        if not self.merged_root.is_dir():
            return
        for path in self.merged_root.iterdir():
            if path == self.merged or not path.is_dir() or not path.name.isdigit():
                continue
            if _process_alive(int(path.name)):
                continue
            logger.info("Evicting merged weights orphaned by pid %s", path.name)
            shutil.rmtree(path, ignore_errors=True)

    def merged_size_gb(self) -> float:
        """Total size of materialised full weights, for disk budgeting."""
        if not self.merged.exists():
            return 0.0
        total = sum(f.stat().st_size for f in self.merged.rglob("*") if f.is_file())
        return total / 1e9

    # --- provenance -----------------------------------------------------

    def write_recipe(
        self,
        wid: str,
        recipe: dict[str, Any],
        *,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Record the recipe next to the adapter, so a hash can be read back.

        ``provenance`` is stored under a reserved ``"provenance"`` key and is
        *annotation*, not part of what hashed to ``wid``: re-hashing the rest
        of the file reproduces the id, the provenance block records facts the
        id deliberately excludes (currently ``training_sample_sha256``, the
        digest of the exact bytes the adapter's own step trained on -- earlier
        steps are covered by their own adapters' recipes).
        """
        path = self.adapter_dir(wid) / "recipe.json"
        if path.exists():
            return
        payload = dict(recipe)
        if provenance:
            payload["provenance"] = provenance
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(path) as scratch:
            scratch.write_text(canonical_json(payload), encoding="utf-8")

    def recorded_training_sample_sha256(self, wid: str) -> str | None:
        """The training-data digest recorded when ``wid``'s adapter was trained.

        ``None`` when unknown: the adapter predates provenance recording, or a
        crash landed between installing the adapter and writing its recipe.
        Unknown is treated leniently by callers -- the check exists to catch
        *divergent* data, not to retroactively distrust old adapters.
        """
        path = self.adapter_dir(wid) / "recipe.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("provenance", {}).get("training_sample_sha256")
        return str(value) if value is not None else None


def adapter_chain(cfg: TrajectoryConfig, t: int, store: Store) -> list[Path]:
    """Adapter directories that must be merged in order to reach step ``t``."""
    chain = []
    for k in range(1, t + 1):
        wid = get_weights_id(cfg, k)
        if not store.has_adapter(wid):
            raise FileNotFoundError(
                f"missing adapter for step {k} ({wid}); cannot reconstruct step {t}"
            )
        chain.append(store.adapter_dir(wid))
    return chain
