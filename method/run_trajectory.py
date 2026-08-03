"""Run one sequential fine-tuning trajectory.

    poetry run python -m method.run_trajectory --config SMOKE_MOCK
    poetry run python -m method.run_trajectory --config EXP1 --seed 3

At each step the runner measures the current checkpoint, computes the action
features for the dataset it is about to train on, then trains. Every stage is
skipped if its artifact already exists, so an interrupted run resumes simply by
being re-invoked, and a trajectory sharing a prefix with an earlier one reuses
that prefix's adapters and measurements.

Measurements are keyed by checkpoint, not by run, so they live in the store; the
run directory only records which checkpoints a trajectory visited.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from method import experiments, steps
from method.backends import ExecutionBackend, get_backend, materialize
from method.config import Backend, JudgeBackend, TrajectoryConfig, to_json
from method.store import (
    Store,
    atomic_dir,
    atomic_file,
    file_sha256,
    get_weights_id,
)
from method.sync import Syncer
from method.utils import (
    DOTENV_PATH,
    check_env_vars,
    load_dotenv,
    trajectory_run_dir,
)

logger = logging.getLogger("run_trajectory")


def measure_checkpoint(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    syncer: Syncer | None = None,
) -> dict:
    """All measurements that depend only on checkpoint t: b_t, v_t, h_neutral, z_t.

    Also measures ``cfg.probes`` here rather than in the training loop, so that
    the *final* checkpoint is probed too: the training loop stops one checkpoint
    early by construction, and a drift series missing its last point would be
    the one place the trajectory ends up unmeasured.

    Pushes after each individual measurement lands, rather than once at the
    end: persona-vector extraction, h_neutral generation and each probe's
    DeltaP are all independently expensive, and batching the push behind all
    of them means a preemption between two still throws away work that was
    already complete and sitting on local disk, just not yet on the remote.
    """
    wid = get_weights_id(cfg, t)
    logger.info("--- measuring checkpoint t=%d (%s) ---", t, wid)

    def push() -> None:
        if syncer is not None:
            _push_measurements(syncer, cfg, t)

    steps.measure_behavior(cfg, t, store, backend)
    push()
    steps.extract_persona_vector(cfg, t, store, backend)
    push()
    steps.measure_h_neutral(cfg, t, store, backend)
    push()
    latent = steps.compute_step_latent(cfg, t, store)
    push()
    behavior_path = store.trait_measurement(
        wid, cfg.trait, steps.Artifacts.BEHAVIOR_JSON
    )
    behavior = json.loads(behavior_path.read_text())
    probes = steps.measure_probes(cfg, t, store, backend, on_probe_done=push)
    return {
        "t": t,
        "weights_id": wid,
        "behavior": behavior,
        "z": latent,
        "probes": probes,
    }


def run(cfg: TrajectoryConfig, backend_kind: Backend, dtype: str) -> Path:
    """Execute the trajectory end to end, returning the run directory."""
    store = Store.for_backend(backend_kind)
    backend = get_backend(backend_kind, dtype=dtype)
    run_dir = trajectory_run_dir(
        cfg.name, cfg.seed, cfg.model.name, mock=backend_kind is Backend.MOCK
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reclaim whatever a preempted or killed run left materialised. Leaves any
    # live sibling's checkpoints alone, so this is safe with a second run
    # already going on the box's other GPU.
    store.evict_orphaned_merged()

    # No-op unless MSC_STORE_REMOTE is set (and never for a mock store). Pulling
    # first turns adapters another box already trained into local cache hits, so
    # a shared prefix is not retrained here.
    syncer = Syncer.from_env(store)
    if syncer is not None:
        logger.info("Remote store configured; pulling reusable prefix")
        syncer.pull_before_run()

    logger.info(
        "Trajectory %r: %d step(s), model=%s, seed=%d, backend=%s",
        cfg.name,
        len(cfg.steps),
        cfg.model.name,
        cfg.seed,
        backend_kind.value,
    )

    record: list[dict] = []
    for t, step in enumerate(cfg.steps):
        record.append(measure_checkpoint(cfg, t, store, backend, syncer))

        # Action features describe the update about to happen, so they are
        # attributed to the checkpoint that precedes it.
        sample_id = steps.training_sample_id(step, cfg.seed)
        train_file = steps.sample_training_file(
            step, cfg.seed, run_dir / f"train_step{t + 1}.jsonl", store
        )
        if syncer is not None:
            syncer.push_training_sample(sample_id)

        record[-1]["delta_p"] = steps.compute_delta_p(
            cfg, t, store, backend, train_file, sample_id
        )
        record[-1]["next_dataset"] = step.dataset_id

        # DeltaP is the last (and among the most expensive) thing measured at
        # checkpoint t, and training is about to start -- so this is the last
        # chance to ship it before a preemption mid-training-step would lose it.
        if syncer is not None:
            _push_measurements(syncer, cfg, t)

        target = get_weights_id(cfg, t + 1)
        if store.has_adapter(target):
            _verify_cached_adapter(store, target, train_file, step_number=t + 1)
            logger.info(
                "[skip] adapter for step %d already trained (%s)", t + 1, target
            )
        else:
            logger.info("--- training step %d -> %s ---", t + 1, target)
            model_path = materialize(cfg, t, store, backend)
            with training_scratch(store, target) as train_out:
                adapter = backend.train(model_path, train_file, step, cfg, train_out)
                _install_adapter(adapter, store.adapter_dir(target))
            store.write_recipe(
                target,
                cfg.weights_key(t + 1),
                provenance={"training_sample_sha256": file_sha256(train_file)},
            )
        # Outside the branch, and so covering the cache hit too: an adapter
        # trained by a run that was preempted before it could push is a local
        # hit here but still missing from the remote. Costs one existence
        # check when it is already up there.
        if syncer is not None:
            syncer.push_adapter(target)

    # The final checkpoint has no successor, so it is measured but never used
    # to compute action features. Nothing writes to its bundle afterwards, so
    # measure_checkpoint's own eager pushes are all it needs.
    record.append(measure_checkpoint(cfg, len(cfg.steps), store, backend, syncer))

    with atomic_file(run_dir / "trajectory.json") as scratch:
        scratch.write_text(
            json.dumps({"config": json.loads(to_json(cfg)), "steps": record}, indent=2),
            encoding="utf-8",
        )
    # Everything the loop produced is already on the remote; what is new here
    # is the run dir, whose trajectory.json is only written above. The sweep is
    # a backstop for anything the eager pushes missed, and is cheap because the
    # ledger skips what has not changed.
    if syncer is not None:
        logger.info("Pushing run artifacts to remote store")
        syncer.push_after_run(run_dir)

    # Merged weights are rebuildable from the adapter chain; do not leave a
    # full checkpoint occupying rental disk after the run.
    store.evict_all_merged()
    logger.info("Trajectory complete: %s", run_dir / "trajectory.json")
    return run_dir


def _push_measurements(syncer: Syncer, cfg: TrajectoryConfig, t: int) -> None:
    """Ship checkpoint ``t``'s measurement bundle, plus the base one it feeds.

    The base checkpoint is included because measuring a *later* checkpoint
    still writes into its bundle: the pos/neg extraction responses and M_0's
    answers to each new dataset's training prompts are generated once and
    stored under the base weights id, so the base bundle keeps growing for the
    whole run. On the steps where it did not grow the call is a no-op, since
    the sync ledger skips artifacts unchanged since their last push.
    """
    for wid in sorted({get_weights_id(cfg, t), get_weights_id(cfg, 0)}):
        syncer.push_measurement(wid)


def _verify_cached_adapter(
    store: Store, wid: str, train_file: Path, *, step_number: int
) -> None:
    """Refuse a cache hit whose adapter was trained on different bytes.

    ``weights_id`` hashes dataset *names*, and ``dataset/`` is gitignored, so
    two machines with divergent copies of a dataset file resolve to the same
    id. The training-time recipe records a digest of the exact sample file the
    adapter trained on; if the sample this machine would train on hashes
    differently, reusing the adapter would silently mix data versions --
    better to stop and say so. Adapters with no recorded digest (trained
    before provenance existed, or a crash between install and recipe write)
    are trusted as before.
    """
    recorded = store.recorded_training_sample_sha256(wid)
    if recorded is None:
        return
    actual = file_sha256(train_file)
    if actual != recorded:
        raise RuntimeError(
            f"training data mismatch for step {step_number} ({wid}): the cached "
            f"adapter was trained on a sample file with sha256 {recorded}, but "
            f"{train_file} hashes to {actual}. The dataset files on this "
            "machine differ from the ones the adapter was trained on. Re-sync "
            "dataset/ (and delete the stale cached sample under "
            f"{store.training_samples}) to keep the adapter, or delete "
            f"{store.adapter_dir(wid)} and every later step's adapter to "
            "retrain from this machine's data."
        )


@contextmanager
def training_scratch(store: Store, wid: str) -> Generator[Path]:
    """Yield a throwaway output directory for one vendored training run.

    The vendored ``training.py`` saves through HF Trainer's checkpointing
    (``save_strategy="epoch"``), so its output directory is not the adapter --
    it is a full ``checkpoint-N`` plus trainer bookkeeping, roughly 250 MB per
    step. :func:`_install_adapter` copies the part the store needs out of it,
    after which every byte left behind is either a duplicate of
    ``store/adapters/<wid>/`` or something ``_ADAPTER_EXCLUDES`` already judged
    worthless.

    It therefore lives here rather than in the run directory. Inside the run
    directory it was retained for the life of the box *and* uploaded a second
    time inside the run tar by :meth:`Syncer.push_after_run`, which is why an
    otherwise-identical trajectory's archive came to hundreds of megabytes
    while a fully cache-hit one came to sixteen. Nothing read it: resume is
    decided by :meth:`Store.has_adapter` against the store, and the
    hyperparameters it recorded in ``training_config_input.json`` are already
    in the run's ``trajectory.json`` and the adapter's ``recipe.json``.

    Removed on the way out whether or not training succeeded -- a failed step
    is exactly when a rental box most needs the disk back for the retry.

    Under the store rather than ``/tmp`` because a rental box's ``/tmp`` is
    routinely a small tmpfs, while the store root is on the volume whose size
    ``docs/cloud_setup.md`` budgets. A fresh directory per attempt (rather than
    one stable path per ``wid``) keeps two trajectories training the same step
    concurrently from writing into each other, and stops
    :func:`method.backends.find_adapter` from ever seeing a higher-numbered
    ``checkpoint-N`` left by an earlier interrupted attempt.

    A hard kill (SIGKILL, preemption) leaves a directory behind, as it does for
    :func:`method.store.atomic_dir`. They are inert and safe to delete; the
    whole of ``store/train_scratch`` can go whenever no run is in flight.
    """
    store.train_scratch.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=store.train_scratch, prefix=f"{wid}."))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


#: Trainer bookkeeping inside a checkpoint that the store has no use for. The
#: adapter weights and config are what the merge replay needs; optimizer and
#: RNG state only matter for *resuming* the interrupted training run itself,
#: and roughly triple the size of what is kept.
_ADAPTER_EXCLUDES = ("optimizer.pt", "scheduler.pt", "rng_state*.pth")


def _install_adapter(produced: Path, target: Path) -> None:
    """Copy a freshly trained adapter into the content-addressed store.

    Atomic for the same reason every other store write is: ``has_adapter``
    treats the presence of ``adapter_config.json`` as "this adapter is
    complete", so a plain ``copytree`` interrupted between the config and the
    weights would leave a corrupt adapter that every trajectory sharing the
    prefix silently builds on.
    """
    with atomic_dir(target) as scratch:
        shutil.copytree(
            produced,
            scratch,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*_ADAPTER_EXCLUDES),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help=f"registry name; one of {', '.join(sorted(experiments.REGISTRY))}",
    )
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--backend",
        type=Backend,
        choices=list(Backend),
        default=None,
        help="'mock' runs with no model at all; defaults to mock for SMOKE_MOCK",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="use float16 on pre-Ampere GPUs, which only emulate bfloat16",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = experiments.get_trajectory_config(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    backend_kind = args.backend
    if backend_kind is None:
        backend_kind = Backend.MOCK if args.config == "SMOKE_MOCK" else Backend.REAL

    load_dotenv(DOTENV_PATH)
    if backend_kind is Backend.REAL:
        required = ["HF_TOKEN"]
        if cfg.eval.judge.backend is JudgeBackend.OPENAI:
            required.append("OPENAI_API_KEY")
        check_env_vars(required)

    run(cfg, backend_kind, args.dtype)


if __name__ == "__main__":
    main()
