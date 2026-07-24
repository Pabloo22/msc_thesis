"""Reusable helpers for the method package."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

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

    Keyed by the *base* ``weights_id`` (which encodes model and seed, since no
    training has happened at t=0) rather than by an experiment name: every
    experiment sharing a base model and seed shares these numbers, so keying by
    experiment would measure the same thing several times under different
    names. Written next to the trajectories, not into the store, so a plotting
    machine needs only the trajectories root synced from the GPU box.
    """
    root = trajectories_root(mock=mock)
    return root / "base_probes" / f"{base_weights_id}_{trait}.json"


def trajectory_run_dir(name: str, seed: int, *, mock: bool = False) -> Path:
    """Where a run of trajectory ``name`` at ``seed`` writes its outputs.

    Takes the plain fields rather than a ``TrajectoryConfig`` so that this
    module stays free of config imports. It exists so the runner and the
    plotting code derive the path the same way: the collector in
    :mod:`method.visualization.collect` finds saved runs purely by rebuilding
    this path from the registry, so a divergence here would silently look like
    "no runs on disk".

    Mock runs go to a parallel root, for the same reason ``Store.for_backend``
    keeps a separate store: a config's run directory depends only on its name
    and seed, so a mock run and a real run of the same config would otherwise
    write to the same path. Adapters merely being overwritten would be
    survivable; a figure silently drawn from synthetic measurements would not
    be.
    """
    return trajectories_root(mock=mock) / f"{name}_seed{seed}"


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


def run_step(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    logger.info("$ (cwd=%s) %s", cwd, " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)
