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
from pathlib import Path

from method import experiments, steps
from method.backends import ExecutionBackend, get_backend, materialize
from method.config import Backend, JudgeBackend, TrajectoryConfig, to_json
from method.store import Store, atomic_dir, atomic_file, get_weights_id
from method.utils import (
    DOTENV_PATH,
    check_env_vars,
    load_dotenv,
    trajectory_run_dir,
)

logger = logging.getLogger("run_trajectory")


def measure_checkpoint(
    cfg: TrajectoryConfig, t: int, store: Store, backend: ExecutionBackend
) -> dict:
    """All measurements that depend only on checkpoint t: b_t, v_t, h_neutral, z_t.

    Also measures ``cfg.probes`` here rather than in the training loop, so that
    the *final* checkpoint is probed too: the training loop stops one checkpoint
    early by construction, and a drift series missing its last point would be
    the one place the trajectory ends up unmeasured.
    """
    logger.info("--- measuring checkpoint t=%d (%s) ---", t, get_weights_id(cfg, t))
    steps.measure_behavior(cfg, t, store, backend)
    steps.extract_persona_vector(cfg, t, store, backend)
    steps.measure_h_neutral(cfg, t, store, backend)
    latent = steps.compute_step_latent(cfg, t, store)
    behavior = json.loads(
        store.trait_measurement(
            get_weights_id(cfg, t), cfg.trait, steps.Artifacts.BEHAVIOR_JSON
        ).read_text()
    )
    return {
        "t": t,
        "weights_id": get_weights_id(cfg, t),
        "behavior": behavior,
        "z": latent,
        "probes": steps.measure_probes(cfg, t, store, backend),
    }


def run(cfg: TrajectoryConfig, backend_kind: Backend, dtype: str) -> Path:
    """Execute the trajectory end to end, returning the run directory."""
    store = Store.for_backend(backend_kind)
    backend = get_backend(backend_kind, dtype=dtype)
    run_dir = trajectory_run_dir(
        cfg.name, cfg.seed, mock=backend_kind is Backend.MOCK
    )
    run_dir.mkdir(parents=True, exist_ok=True)

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
        record.append(measure_checkpoint(cfg, t, store, backend))

        # Action features describe the update about to happen, so they are
        # attributed to the checkpoint that precedes it.
        train_file = steps.sample_training_file(
            step, cfg.seed, run_dir / f"train_step{t + 1}.jsonl", store
        )
        record[-1]["delta_p"] = steps.compute_delta_p(
            cfg, t, store, backend, train_file, steps.training_sample_id(step, cfg.seed)
        )
        record[-1]["next_dataset"] = step.dataset_id

        target = get_weights_id(cfg, t + 1)
        if store.has_adapter(target):
            logger.info(
                "[skip] adapter for step %d already trained (%s)", t + 1, target
            )
        else:
            logger.info("--- training step %d -> %s ---", t + 1, target)
            model_path = materialize(cfg, t, store, backend)
            adapter = backend.train(
                model_path, train_file, step, cfg, run_dir / f"train_out_{t + 1}"
            )
            _install_adapter(adapter, store.adapter_dir(target))
            store.write_recipe(target, cfg.weights_key(t + 1))

    # The final checkpoint has no successor, so it is measured but never used
    # to compute action features.
    record.append(measure_checkpoint(cfg, len(cfg.steps), store, backend))

    with atomic_file(run_dir / "trajectory.json") as scratch:
        scratch.write_text(
            json.dumps({"config": json.loads(to_json(cfg)), "steps": record}, indent=2),
            encoding="utf-8",
        )
    # Merged weights are rebuildable from the adapter chain; do not leave a
    # full checkpoint occupying rental disk after the run.
    store.evict_all_merged()
    logger.info("Trajectory complete: %s", run_dir / "trajectory.json")
    return run_dir


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
