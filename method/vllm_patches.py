"""Apply scoped vLLM runtime patches."""

from __future__ import annotations

import gc
import hashlib
import logging
import json
import re
import time
from pathlib import Path
from typing import Any

#: ``max_num_seqs`` as the vendored loader hardcodes it. Restated rather than
#: read back off the engine because the cap has to be decided before ``LLM``
#: is constructed, which is the only thing that would know it.
VENDORED_MAX_NUM_SEQS = 32

#: The capture list vLLM's V0 engine would have built for that ``max_num_seqs``
#: -- its ``[1, 2, 4] + [8 * i ...]`` candidates, truncated at the first size
#: that reaches it. See :func:`force_vllm_cudagraph_sizes` for why V1 needs to
#: be told this explicitly.

_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _resolve_checkpoint(model_path: str) -> Path | None:
    """The directory the vendored loader will actually read, or None for a hub id.

    Mirrors ``eval.model_utils._pick_latest_checkpoint``. Duplicated rather than
    imported because every caller here has to decide *before* ``vllm.LLM`` is
    constructed, and the vendored package only becomes importable once its own
    directory is on ``sys.path`` -- which the workers arrange later, deliberately
    after these patches are in place.
    """
    root = Path(model_path)
    if not root.is_dir():
        return None
    checkpoints = [
        (int(match.group(1)), path)
        for path in root.iterdir()
        if (match := _CHECKPOINT_RE.fullmatch(path.name)) and path.is_dir()
    ]
    if not checkpoints:
        return root
    return max(checkpoints, key=lambda pair: pair[0])[1]


def is_adapter_dir(model_path: str) -> bool:
    """Whether the vendored loader will treat this path as a LoRA adapter.

    Mirrors its ``_is_lora`` probe on the resolved checkpoint, because whether
    LoRA is needed decides whether it can be switched off, and that has to be
    settled before the engine exists.
    """
    resolved = _resolve_checkpoint(model_path)
    return resolved is not None and (resolved / "adapter_config.json").exists()


def _default_compilation_config(**entries: Any) -> None:
    """Merge ``entries`` into ``compilation_config`` on every ``vllm.LLM(...)``.

    Per key rather than per dict, and the distinction is load-bearing. These
    patches work by wrapping ``LLM.__init__``, so the last one applied is the
    first to run; a patch that claimed the whole ``compilation_config`` with a
    single ``setdefault`` would leave every later-running wrapper looking at a
    dict that already exists, and its keys would be dropped without a word.

    Individual keys still defer to the caller, which keeps what the
    all-or-nothing version was for: the vendored loader never passes
    ``compilation_config``, so anything already there was asked for deliberately.
    A caller who passes something that is not a dict (vLLM also accepts a bare
    level, or a ``CompilationConfig``) is left entirely alone.
    """
    import vllm

    original_init = vllm.LLM.__init__

    def patched_init(self, *args, **kwargs):
        config = kwargs.get("compilation_config")
        if config is None:
            config = {}
            kwargs["compilation_config"] = config
        if isinstance(config, dict):
            for name, value in entries.items():
                config.setdefault(name, value)
        return original_init(self, *args, **kwargs)

    vllm.LLM.__init__ = patched_init  # type: ignore[method-assign]


def force_vllm_cudagraph_sizes(max_size: int = VENDORED_MAX_NUM_SEQS) -> None:
    r"""Cap CUDA-graph capture at the batch sizes this workload can reach."""
    sizes = [size for size in _CAPTURE_SIZES if size <= max_size]
    _default_compilation_config(cudagraph_capture_sizes=sizes)


def share_vllm_compile_cache(
    model_path: str,
    *,
    dtype: str | None = None,
    max_model_len: int | None = None,
) -> str | None:
    r"""Point every engine at one torch.compile cache per *architecture*."""
    import vllm
    from vllm import envs as vllm_envs

    resolved = _resolve_checkpoint(model_path)
    if resolved is None:
        return None
    try:
        config = json.loads((resolved / "config.json").read_text())
    except (OSError, ValueError):
        return None

    # Set by save_pretrained to whatever path the model was written from, so it
    # differs between two checkpoints that compile to the same graph -- which is
    # the whole reason vLLM's own key is too strict for this pipeline.
    config.pop("_name_or_path", None)
    factors = [
        json.dumps(config, sort_keys=True),
        str(dtype),
        str(max_model_len),
        vllm.__version__,
        vllm_envs.compute_hash(),
    ]
    digest = hashlib.sha256(json.dumps(factors).encode()).hexdigest()[:10]
    cache_dir = str(
        Path(vllm_envs.VLLM_CACHE_ROOT) / "torch_compile_cache" / f"msc-{digest}"
    )
    _default_compilation_config(cache_dir=cache_dir)
    return cache_dir


class _Gen2Probe:
    """Time spent in gen-2 garbage collections, via ``gc.callbacks``.

    Only generation 2 is worth counting: it is the one whose cost scales with
    everything alive in the process rather than with recent allocations, so it
    is the one a loaded model makes expensive.
    """

    def __init__(self) -> None:
        self.collections = 0
        self.seconds = 0.0
        self._started: float | None = None

    def __call__(self, phase: str, info: dict) -> None:
        if info.get("generation") != 2:
            return
        if phase == "start":
            self._started = time.perf_counter()
        elif self._started is not None:
            self.seconds += time.perf_counter() - self._started
            self.collections += 1
            self._started = None


def freeze_gc_during_cudagraph_capture(*, freeze: bool = True) -> None:
    r"""Hide the loaded model from Python's GC while CUDA graphs are captured."""
    from vllm.logger import init_logger
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    logger = init_logger(__name__)
    original_capture = GPUModelRunner.capture_model

    def patched_capture(self) -> None:
        probe = _Gen2Probe()
        gc.callbacks.append(probe)
        try:
            if freeze:
                gc.collect()
                gc.freeze()
            original_capture(self)
        finally:
            gc.callbacks.remove(probe)
        logger.info(
            "CUDA-graph capture saw %d gen-2 GC pass(es) costing %.1fs "
            "(gc frozen: %s, %d objects permanent)",
            probe.collections,
            probe.seconds,
            freeze,
            gc.get_freeze_count(),
        )

    GPUModelRunner.capture_model = patched_capture  # type: ignore[method-assign]


def disable_vllm_lora(model_path: str) -> bool:
    r"""Turn LoRA off for a model whose adapters are already merged in."""
    if is_adapter_dir(model_path):
        return False

    import vllm

    original_init = vllm.LLM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["enable_lora"] = False
        return original_init(self, *args, **kwargs)

    vllm.LLM.__init__ = patched_init  # type: ignore[method-assign]
    return True


def force_vllm_dtype(dtype: str) -> None:
    """Make every ``vllm.LLM(...)`` use ``dtype``, whatever the caller asked.

    vLLM falls back to the model's config dtype (bfloat16 for Qwen) and
    refuses to start on GPUs below compute capability 8.0. Turing cards such as
    the T550 therefore cannot run the eval at all without this override. Only
    needed locally: on Ampere and newer this is a no-op worth skipping.
    """
    import vllm

    original_init = vllm.LLM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("dtype", dtype)
        return original_init(self, *args, **kwargs)

    vllm.LLM.__init__ = patched_init  # type: ignore[method-assign]


def force_vllm_max_model_len(max_model_len: int) -> None:
    """Cap vLLM's ``max_model_len``, overriding the vendored loader's value.

    The vendored 20000/30000 reserves KV-cache capacity for far more context
    than a persona-eval request ever uses (the longest question/instruction
    across every trait file is under 50 tokens, and generation is capped by
    ``max_tokens``), which is what makes a 7B model's memory footprint far
    larger than the workload needs. Unlike ``force_vllm_dtype``, the vendored
    call always passes ``max_model_len`` explicitly, so this must overwrite the
    kwarg rather than ``setdefault`` it.
    """
    import vllm

    original_init = vllm.LLM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["max_model_len"] = max_model_len
        return original_init(self, *args, **kwargs)

    vllm.LLM.__init__ = patched_init  # type: ignore[method-assign]


def shutdown_vllm(llm: Any) -> None:
    r"""Release an engine at a known point, rather than at interpreter exit."""
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    for target in (core, engine, llm):
        shutdown = getattr(target, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:  # noqa: BLE001 -- teardown is best-effort
                logger.warning("vLLM shutdown via %r failed: %s", target, exc)
            else:
                break
    gc.collect()
