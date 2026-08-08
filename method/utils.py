"""Reusable helpers for the method package."""

from __future__ import annotations

import codecs
import json
import logging
import os
import re
import subprocess
import sys
from collections import deque
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


def base_probes_path(base_weights_id: str, trait: str, *, mock: bool = False) -> Path:
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


def require_cuda(component: str) -> None:
    """Fail fast if no CUDA device is usable.

    Every GPU worker calls this before loading a model, because the failure it
    guards against is silent rather than loud: ``device_map="auto"`` places the
    model on the CPU when it sees zero devices, and the run then produces
    *correct* numbers about a hundred times slower. One dead GPU on a rental box
    took NVML down with it and cost 73 hours of CPU forward passes that way, so
    the cheap check earns its place.

    Note that a healthy GPU is not enough: a sibling card that has fallen off
    the bus breaks NVML enumeration for the whole machine, which is why this
    reports the driver's own message rather than assuming the device is merely
    busy.

    The check has to answer without touching the CUDA runtime, which is why it
    counts devices rather than asking ``torch.cuda.is_available()``. That call
    goes straight to ``cudaGetDeviceCount``, and torch responds by registering
    an ``atfork`` handler that marks every later fork of this process as a bad
    one -- while leaving ``torch.cuda.is_initialized()`` False, since no context
    was actually created. vLLM reads exactly that flag to decide whether its V1
    engine may fork ``EngineCore``, so it sees a clean parent, forks, and the
    child dies in ``torch.cuda.set_device`` with "Cannot re-initialize CUDA in
    forked subprocess" -- 45s into engine startup, in a worker whose only sin
    was checking that the GPU it was about to use exists. ``device_count`` asks
    NVML first and only falls back to the runtime when NVML cannot answer,
    which is the case this function raises on regardless.
    """
    import torch  # imported here: utils is loaded by CPU-only paths too

    if torch.cuda.device_count() > 0:
        return
    try:
        torch.cuda.init()
    except Exception as exc:  # noqa: BLE001 -- surfacing the driver's reason
        detail = f" ({type(exc).__name__}: {exc})"
    else:
        detail = ""
    raise RuntimeError(
        f"{component} needs a CUDA device and torch reports none{detail}. "
        "Refusing to fall back to the CPU, which would be ~100x slower. "
        "Check 'nvidia-smi': if it cannot read every GPU on the box (e.g. "
        "'Unable to determine the device handle for GPU1'), the driver is "
        "wedged and no new CUDA process can start until the host is reset."
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


#: How many lines of a dead worker's output to carry back to the parent. The
#: consumer is an email body, and the thing being looked for -- a CUDA OOM, a
#: missing file, an engine-core traceback -- lives in the last few dozen lines.
_TAIL_LINES = 60


class StepFailed(subprocess.CalledProcessError):
    """A worker subprocess that exited non-zero, holding on to what it said.

    A bare :class:`~subprocess.CalledProcessError` stringifies to the exit
    status and the argv, which is the one thing about the failure that was
    already known. The reason went to the worker's stderr, which the parent
    inherited and therefore never held, so the emailed report (see
    :mod:`method.report`) could only say "exit status 1".

    :attr:`tail` carries those lines instead. Deliberately *not* part of
    ``__str__``: this exception is both printed as a traceback and quoted in a
    report, and putting sixty lines in the message would print them twice in
    the one place they are actually read.
    """

    def __init__(self, returncode: int, cmd: list[str], tail: str) -> None:
        super().__init__(returncode, cmd, output=tail)
        #: The end of the worker's merged stdout/stderr, ``  | ``-quoted.
        self.tail = tail
        #: The module that died, e.g. ``method._generate_worker``.
        self.worker = _worker_name(cmd)

    def __str__(self) -> str:
        return f"{self.worker} exited with status {self.returncode}"


def _worker_name(cmd: Sequence[str]) -> str:
    """The part of an argv worth naming: the module, not the interpreter path.

    ``[python, -m, method._generate_worker, --model, ...]`` is read as
    ``method._generate_worker``; anything unrecognised falls back to the
    executable's own name, so a vendored script still gets a label.
    """
    cmd = list(cmd)
    if "-m" in cmd[:2]:
        index = cmd.index("-m") + 1
        if index < len(cmd):
            return cmd[index]
    return Path(cmd[0]).name if cmd else "subprocess"


def _tail(lines: Sequence[str], limit: int = _TAIL_LINES) -> str:
    """The last ``limit`` non-blank lines, indented as a quoted block."""
    kept = [line for line in lines if line.strip()][-limit:]
    return "\n".join(f"  | {line}" for line in kept)


class _TailBuffer:
    """The last few lines of a stream, as it is still being written.

    Progress bars are why this is not simply a deque of ``\\n``-terminated
    lines: tqdm redraws one line thousands of times with ``\\r`` and never ends
    it, so a line-based buffer would hold a single unbounded string and, worse,
    would let a bar's frames evict the traceback printed after it. Only the
    last frame before each newline is kept, which is the one a human reading
    the log would have seen.
    """

    def __init__(self, keep: int) -> None:
        self._lines: deque[str] = deque(maxlen=keep)
        self._current = ""

    def feed(self, text: str) -> None:
        *complete, self._current = (self._current + text).split("\n")
        self._lines.extend(line.rsplit("\r", 1)[-1] for line in complete)
        self._current = self._current.rsplit("\r", 1)[-1]

    def lines(self) -> list[str]:
        """Everything buffered, including a line the writer never finished."""
        return [*self._lines, self._current]


def run_step(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    """Run a worker to completion, streaming its output and keeping the tail.

    The output is piped rather than inherited so that a failure can quote it
    (see :class:`StepFailed`). It is echoed in the chunks it arrives in rather
    than line by line, because the workers' most visible output is a progress
    bar that never emits a newline: waiting for one would make a live run look
    frozen for the whole of a generation pass.
    """
    logger.info("$ (cwd=%s) %s", cwd, " ".join(cmd))
    if dry_run:
        return
    tail = _TailBuffer(keep=_TAIL_LINES * 4)
    # Incremental, so a multi-byte character split across two reads survives.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Unbuffered on this side, so a read returns what the child has
        # written so far rather than waiting for a full buffer -- and on the
        # child's side too: CPython block-buffers stdout when it is a pipe
        # rather than a terminal, which would otherwise hold a worker's prints
        # back by kilobytes at a time and make a live run look stalled.
        bufsize=0,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as proc:
        stream = proc.stdout
        assert stream is not None  # guaranteed by stdout=PIPE
        for chunk in iter(lambda: stream.read(8192), b""):
            text = decoder.decode(chunk)
            sys.stdout.write(text)
            sys.stdout.flush()
            tail.feed(text)
    if proc.returncode:
        raise StepFailed(proc.returncode, cmd, _tail(tail.lines()))
