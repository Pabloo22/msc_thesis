"""Reusable helpers for the method package."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent  # .../msc_thesis/method
REPO_ROOT = SCRIPT_DIR.parent
PERSONA_VECTORS_DIR = REPO_ROOT / "method" / "persona_vectors"
DATASET_DIR = REPO_ROOT / "dataset"
TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
NEUTRAL_DIR = REPO_ROOT / "data" / "neutral"
#: The .env lives inside the repo (gitignored). Entry points load it
#: explicitly via :func:`load_dotenv`; nothing loads it at import time.
DOTENV_PATH = REPO_ROOT / ".env"

REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "HF_TOKEN")

logger = logging.getLogger(__name__)


def trajectories_root(*, mock: bool = False) -> Path:
    """Root holding run outputs, kept separate for synthetic (mock) runs."""
    suffix = "-mock" if mock else ""
    return TRAJECTORIES_DIR.parent / f"{TRAJECTORIES_DIR.name}{suffix}"


def base_probes_path(
    base_weights_id: str, trait: str, *, mock: bool = False
) -> Path:
    """Where the base-model DeltaP probe summary for one trait is written.

    Keyed by the *base* ``weights_id`` (which reduces to the model alone, since
    no training has happened at t=0 and ``weights_key`` normalizes seed away
    there) rather than by an experiment name: every experiment and seed sharing
    a base model shares these numbers, so keying by experiment would measure the
    same thing several times under different names. Written next to the
    trajectories, not into the store, so a plotting machine needs only the
    trajectories root synced from the GPU box.
    """
    root = trajectories_root(mock=mock)
    return root / "base_probes" / f"{base_weights_id}_{trait}.json"


def model_slug(model_name: str) -> str:
    """Filesystem-safe short form of a model id.

    ``"Qwen/Qwen2.5-7B-Instruct"`` becomes ``"qwen2.5-7b-instruct"``: the
    organisation prefix is dropped because it would introduce a path separator
    and carries no information the repository name lacks.
    """
    tail = model_name.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", tail).strip("-") or "model"


def trajectory_run_dir(
    name: str, seed: int, model_name: str, *, mock: bool = False
) -> Path:
    """Where a run of trajectory ``name`` on ``model_name`` at ``seed`` writes.

    Takes the plain fields rather than a ``TrajectoryConfig`` so that this
    module stays free of config imports. It exists so the runner and the
    plotting code derive the path the same way: the collector in
    :mod:`method.visualization.collect` finds saved runs purely by rebuilding
    this path from the registry, so a divergence here would silently look like
    "no runs on disk".

    The model is part of the path because it is the one thing that changes a
    trajectory's weights without changing its name: ``weights_key`` hashes the
    model, so re-pointing a config at a different base model gives every step a
    fresh ``weights_id`` and the store keeps the two chains apart -- but the run
    directory would collide, and the second run's ``trajectory.json`` would
    replace the first's. Both are legitimate experiments that should coexist.
    Editing a config's *steps* in place is the opposite case: that is a
    replacement, not a second experiment, so steps stay out of the path and the
    new run is meant to overwrite the old one (``collect`` flags any leftover as
    stale by comparing weights_ids).

    Mock runs go to a parallel root, for the same reason ``Store.for_backend``
    keeps a separate store: a mock run and a real run of the same config would
    otherwise write to the same path. Adapters merely being overwritten would be
    survivable; a figure silently drawn from synthetic measurements would not
    be.
    """
    return trajectories_root(mock=mock) / f"{name}_{model_slug(model_name)}_seed{seed}"


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file.

    Existing environment variables are never overridden.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        logger.info(
            "No .env file at %s; relying on already-exported environment variables",
            path,
        )
        return
    except OSError as exc:
        logger.warning(
            "Could not read %s (%s); relying on already-exported environment variables",
            path,
            exc,
        )
        return

    loaded = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    logger.info(
        "Loaded env var(s) from %s: %s",
        path,
        ", ".join(loaded) if loaded else "(none new)",
    )


def check_env_vars(required: Sequence[str] = REQUIRED_ENV_VARS) -> None:
    """Fail fast if any of ``required`` is unset.

    Callers pass a narrower list when part of the default set is not actually
    needed -- e.g. a stubbed judge never talks to OpenAI.
    """
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Set them in {DOTENV_PATH} or export them in your shell before running "
            "this script."
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a jsonl file, tolerating a truncated final line.

    Every jsonl this project writes is appended to as work completes, so the
    one malformed line a crash can leave is the last one. Dropping it costs the
    single record that was in flight; refusing to parse the file would cost
    every record before it. A missing file is empty rather than an error, since
    "nothing recorded yet" is the normal state at the start of a run.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("ignoring truncated record at the end of %s", path)
            break
    return records


def run_step(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    logger.info("$ (cwd=%s) %s", cwd, " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)
