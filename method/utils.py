"""Reusable helpers for the method package."""

from __future__ import annotations

import codecs
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
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


def anchor_noise_path(base_weights_id: str, label: str, *, mock: bool = False) -> Path:
    """Where a replicate sweep of the base anchor writes its summary.

    Keyed by the base ``weights_id`` for the same reason as
    :func:`base_probes_path` -- the anchor is a property of the base model, and
    every family reading it shares one -- plus a ``label`` naming the checkpoint
    span the replicates were carried along (see :mod:`method.anchor_noise`),
    since the same anchor draw yields a different noise budget on a longer trunk.
    """
    root = trajectories_root(mock=mock)
    return root / "anchor_noise" / f"{base_weights_id}_{label}.json"


def axis_refresh_path(base_weights_id: str, label: str, *, mock: bool = False) -> Path:
    """Where a re-draw of the extraction text writes its summary.

    Keyed exactly as :func:`anchor_noise_path` is, and for the same reasons: the
    comparison is anchored at the base model, and the ``label`` names the
    checkpoint span it was carried along, since the same base draw says
    something different at ``t = 6`` than at ``t = 1``.
    """
    root = trajectories_root(mock=mock)
    return root / "axis_refresh" / f"{base_weights_id}_{label}.json"


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


#: What fraction of the card a vLLM worker needs free before it may load.
#: The vendored loader hardcodes ``gpu_memory_utilization=0.9`` in both of its
#: branches (``eval/model_utils.py``), so this is not a guess about the
#: workload -- it is the number vLLM will itself try to reserve.
VLLM_FREE_FRACTION = 0.9

#: The same for a worker that loads the model through transformers rather than
#: vLLM. Weights plus batch activations for a 7B checkpoint in bfloat16 sit
#: near 15GB of a 24GB card; the point of the check there is not to predict
#: that footprint but to notice a *whole other model* still resident, which is
#: what the gate below exists for.
TORCH_FREE_FRACTION = 0.7

#: How long to wait for the card to come free, and how often to look. A clean
#: engine teardown takes seconds, so this is generous on purpose: when the card
#: is already free the call returns on the first probe and costs nothing, and
#: when it is not, waiting two minutes is far cheaper than losing the stage.
GPU_READY_TIMEOUT_S = 120.0
GPU_READY_POLL_S = 2.0


def _visible_device() -> str:
    """Which GPU ``nvidia-smi`` should be asked about.

    ``CUDA_VISIBLE_DEVICES`` is how this pipeline splits work across cards
    (see ``scripts/run_family.sh``), and NVML does not honour it -- index 0
    there is the first *physical* card, not the first visible one. Reading the
    first entry keeps the gate pointed at the card the worker will actually
    load onto. An index or a UUID both work: ``nvidia-smi -i`` accepts either.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible or "0"


def gpu_memory() -> tuple[int, int] | None:
    """``(free, total)`` bytes on the GPU this process will use, or None.

    Shells out to ``nvidia-smi`` rather than asking torch, and that is the
    whole reason this function exists rather than a one-line
    ``torch.cuda.mem_get_info``. Every caller runs *before* the model loads,
    and touching the CUDA runtime that early is exactly what
    :func:`require_cuda` is written to avoid: it registers an ``atfork``
    handler that poisons vLLM's V1 ``EngineCore`` fork while leaving
    ``torch.cuda.is_initialized()`` False, so the engine forks anyway and the
    child dies 45s later. A subprocess creates no context in *this* process.

    ``None`` when the driver cannot be read, which callers treat as "cannot
    tell" rather than "not ready" -- a missing ``nvidia-smi`` must not be able
    to stop a run that would otherwise work.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                _visible_device(),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("cannot read GPU memory (%s)", exc)
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    try:
        free_mib, total_mib = (int(part) for part in line.split(",")[:2])
    except ValueError:
        logger.debug("unparseable nvidia-smi memory output: %r", line)
        return None
    return free_mib * 1024**2, total_mib * 1024**2


def gpu_processes() -> str:
    """Whatever the driver will say about who is holding the card.

    Purely for the error message below, and the reason it is worth the two
    subprocess calls: the one thing the OOM that motivated this could not say
    was *what* the 12 GiB it was short of belonged to. A pid alone answers
    less than it looks -- ``nvidia-smi`` inside a container reports **host**
    pids, so the number often resolves to nothing in this namespace -- so the
    parent and start time are read where they are readable, and quietly
    omitted where they are not.

    Best effort throughout: this runs on the failure path, and a diagnostic
    that can itself fail is worse than a thin one.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
                "-i",
                _visible_device(),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    holders = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        pid = line.split(",")[0].strip()
        holders.append(f"{line.strip()}{_process_detail(pid)}")
    return "; ".join(holders)


def _process_detail(pid: str) -> str:
    """``(ppid=..., started ...)`` for a pid, or "" when it is not ours to see.

    The empty case is the common one in a container, and it is informative in
    itself: a pid ``nvidia-smi`` reports but ``ps`` cannot find is either on
    the host side of a namespace boundary or already gone.
    """
    if not pid.isdigit():
        return ""
    try:
        proc = subprocess.run(
            ["ps", "-o", "ppid=,lstart=,comm=", "-p", pid],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return " (not visible in this pid namespace)"
    detail = " ".join(proc.stdout.split())
    return f" ({detail})" if detail else " (not visible in this pid namespace)"


def wait_for_free_vram(
    component: str,
    *,
    fraction: float = VLLM_FREE_FRACTION,
    timeout_s: float = GPU_READY_TIMEOUT_S,
    poll_s: float = GPU_READY_POLL_S,
    probe: Callable[[], tuple[int, int] | None] = gpu_memory,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until ``fraction`` of the GPU is free, then return.

    Defence in depth, not a fix for a diagnosed cause. What is established is
    only this: a worker died because another process held 12 GiB of a 24 GiB
    card, and the vendored loader asks for 90% of it (see
    :data:`VLLM_FREE_FRACTION`), so a holder of that size makes the next load
    impossible rather than merely smaller. One such OOM aborted a whole family,
    because ``run_family.sh`` runs under ``set -e``.

    What is *not* established is who the holder was. vLLM 0.8.5 terminates its
    V1 ``EngineCore`` through a finalizer on normal exit (see
    :func:`method.vllm_patches.shutdown_vllm`), so an ordinary handover does
    not strand one; a worker killed outright would, and so would an unrelated
    process on the box. This function deliberately does not care which: it
    waits for the card, and if the card does not come free it names whoever is
    holding it, which is the thing an OOM traceback cannot.

    It never kills anything. Deciding that a process is stray is a judgement
    about the box, not about this worker, and a gate that guessed wrong would
    take out the run it was meant to protect.

    ``probe`` and ``sleep`` are injectable so the logic can be tested without a
    GPU. A ``probe`` that returns None disables the gate -- see
    :func:`gpu_memory`.
    """
    waited = 0.0
    announced = False
    while True:
        memory = probe()
        if memory is None:
            logger.debug("%s: cannot read GPU memory, skipping the wait", component)
            return
        free, total = memory
        if total <= 0 or free >= fraction * total:
            if announced:
                logger.info("%s: GPU came free after %.0fs", component, waited)
            return
        if waited >= timeout_s:
            holders = gpu_processes()
            raise RuntimeError(
                f"{component} needs {fraction:.0%} of the GPU free and only "
                f"{free / 1024**3:.1f} of {total / 1024**3:.1f} GiB is, after "
                f"waiting {waited:.0f}s"
                + (f". Holding it: {holders}" if holders else "")
                + ". A leaked vLLM EngineCore from a killed worker looks like "
                "this and survives its parent; kill it and re-run, which "
                "resumes from whatever already landed."
            )
        if not announced:
            announced = True
            logger.info(
                "%s: waiting for the GPU (%.1f of %.1f GiB free, need %.0f%%)",
                component,
                free / 1024**3,
                total / 1024**3,
                fraction * 100,
            )
        sleep(poll_s)
        waited += poll_s


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


def run_step(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run a worker to completion, streaming its output and keeping the tail.

    The output is piped rather than inherited so that a failure can quote it
    (see :class:`StepFailed`). It is echoed in the chunks it arrives in rather
    than line by line, because the workers' most visible output is a progress
    bar that never emits a newline: waiting for one would make a live run look
    frozen for the whole of a generation pass.

    ``extra_env`` overlays this process's environment for the child only. It is
    for settings a worker cannot apply to itself because they are read before
    or during interpreter start-up, such as CUDA allocator configuration.
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
        env={**os.environ, "PYTHONUNBUFFERED": "1", **(extra_env or {})},
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
