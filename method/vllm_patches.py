"""Runtime patches applied to ``vllm.LLM`` before the vendored loader runs.

``eval.model_utils.load_vllm_model`` (vendored, no edits allowed) constructs
every ``vllm.LLM(...)`` without a ``dtype``, with ``max_model_len`` hardcoded
to 30000 (hub) / 20000 (local), and with LoRA enabled whether or not the model
has an adapter. Both entry points that end up calling it
(``method.eval_wrapper`` -> ``eval.eval_persona.main``, and
``method._generate_worker`` directly) import this module and call these
functions first, so the fix lives entirely outside persona_vectors.

Engine startup is the thing most of these are about. A 7B checkpoint on a 4090
rental spent 265s in ``init engine`` before generating a single token, and this
pipeline pays that once per stage per subprocess -- roughly ten times per
trajectory, times every seed and family.
"""

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
    """Cap CUDA-graph capture at the batch sizes this workload can reach.

    vLLM's V0 engine sized its capture list from ``max_num_seqs``: with the
    vendored 32 it captured 7 shapes. V1 builds the list as
    ``[1, 2, 4] + range(8, 513, 8)`` and filters it by
    ``max_num_batched_tokens`` (8192 for the ``LLM`` class) instead, so all 67
    shapes survive -- and it captures a warmup run per shape on top. That is
    134 dummy forward passes where V0 did 14, and it is paid at every engine
    construction, which for this pipeline means once per stage per subprocess.

    Nothing is lost by capping. Graphs are only used when the scheduled token
    count is under the largest captured shape, and a decode batch is bounded by
    ``max_num_seqs``; prefill already overruns any of these sizes and falls
    back to eager regardless.

    Measured on a rental box: 2148s uncapped against 169s capped, a 12.7x
    saving from 9.6x fewer shapes. Note what that leaves: ~24s per shape, where
    vLLM's own ``capture_model`` docstring expects 5-20s for the *whole* list.
    Shape count is no longer the lever on that residue -- see
    :func:`freeze_gc_during_cudagraph_capture` and :func:`disable_vllm_lora`,
    which target the per-shape cost instead.
    """
    sizes = [size for size in _CAPTURE_SIZES if size <= max_size]
    _default_compilation_config(cudagraph_capture_sizes=sizes)


def share_vllm_compile_cache(
    model_path: str,
    *,
    dtype: str | None = None,
    max_model_len: int | None = None,
) -> str | None:
    """Point every engine at one torch.compile cache per *architecture*.

    vLLM derives its cache directory from factors that include
    ``ModelConfig.model`` -- the filesystem path -- and the model's HF config
    verbatim (``ModelConfig.compute_hash``). Every checkpoint in a trajectory
    lives at its own path, so every engine start is a cold compile: ~36s of
    inductor work for the general shape, plus ~12s of Dynamo tracing, for a
    graph that is byte-identical across the entire trajectory. Weights never
    reach the compiled graph; only the architecture does.

    Setting ``cache_dir`` explicitly makes vLLM skip that derivation altogether
    (``VllmBackend.configure_post_pass``'s caller in vllm/compilation/backends.py
    only hashes when ``cache_dir`` is empty), so this keys the directory on what
    genuinely shapes the graph: the model's config with the path-dependent
    ``_name_or_path`` dropped, the ``dtype`` and ``max_model_len`` we override,
    vLLM's own env hash, and the vLLM version -- which stands in for the hash of
    traced source files that vLLM would otherwise have folded in.

    Returns the directory it settled on, or None when the architecture could not
    be established -- a hub id that is not on disk yet, or an adapter directory
    whose config lives with the base model. Falling back to vLLM's own hashing
    costs a cold compile; guessing the key wrong would silently run one model's
    compiled graph against another's weights, so the failure has to go this way.
    """
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
    """Hide the loaded model from Python's GC while CUDA graphs are captured.

    A gen-2 collection walks every tracked container object in the process
    looking for reference cycles, so it costs what is *alive*, not what is
    garbage. At capture time an enormous amount is alive and none of it is
    collectable: the module tree, a few hundred parameter tensors, Dynamo's
    bytecode caches, and one inductor-generated module per piecewise split.
    ``GPUWorker.compile_or_warm_up_model`` then calls ``gc.collect()`` right
    before capture, which promotes all of it into gen 2.

    Capture allocates hard enough to keep triggering that: ~29 pieces per shape
    (one per attention split) times two dummy runs per shape, each making
    CUDAGraph objects, tensor wrappers and cache entries. So the collector
    re-walks the whole static heap repeatedly while doing no useful work. On the
    4090 rental, capture took 169s for 7 shapes -- ~830ms per graph, where
    milliseconds are expected -- while allocating only 0.12 GiB, which is the
    signature of CPU overhead rather than GPU work.

    ``gc.freeze()`` moves everything currently alive into a permanent generation
    the collector never traverses again. It stays frozen after capture on
    purpose: the same objects are still alive and still uncollectable during
    generation, so there is nothing to gain by handing them back. The
    ``gc.collect()`` first is what makes that safe -- any genuine cycle is
    reclaimed before the rest becomes permanent.

    ``freeze=False`` instruments without fixing, which is what
    ``scripts/bench_cudagraph.sh`` needs to attribute the time either way.

    Takes effect in the ``EngineCore`` subprocess by being inherited across the
    fork, which is what ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to. That
    inheritance is the only route it has: unlike the ``LLM.__init__`` patches
    above, whose effect travels to the child inside the pickled ``VllmConfig``,
    this one patches a class the child imports for itself, so forcing ``spawn``
    would silently drop it.

    What it therefore depends on is that nothing in the worker touches the CUDA
    runtime before the engine is built -- not because forking would then be
    slower, but because it would not work at all (see
    :func:`method.utils.require_cuda`, which is written the way it is for this
    reason). The log line below is the check: no line means the patch never
    reached the worker, and ``VLLM_ENABLE_V1_MULTIPROCESSING=0``, which keeps
    the engine in-process, would be the way to force the issue.
    """
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
    """Turn LoRA off for a model whose adapters are already merged in.

    The vendored loader passes ``enable_lora=True`` and ``max_lora_rank=128``
    unconditionally, but this pipeline merges every adapter into its base before
    inference -- ``materialize`` hands over merged weights, so ``lora_path``
    comes back None and no ``LoRARequest`` is ever issued. The machinery is
    carried and never used, and it is not free:

    * every CUDA-graph dummy run enters ``maybe_dummy_run_with_lora``, whose
      ``create_dummy_lora`` walks the entire module tree building
      zero-initialised rank-8 warmup adapters on the CPU, copies them to the
      GPU, and tears them all down again on exit -- 14 times over a capped
      capture. That churn is a large part of what keeps triggering the
      collections :func:`freeze_gc_during_cudagraph_capture` is about;
    * the punica slot tensors are sized for ``max_lora_rank=128`` across every
      target module, holding GPU memory the KV cache could have used;
    * every captured graph carries punica's indexing kernels.

    Returns False and changes nothing when ``model_path`` genuinely is an
    adapter directory: ``_generate_worker`` still supports being pointed at one,
    and vLLM would reject the ``LoRARequest`` that follows.
    """
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
    """Release an engine at a known point, rather than at interpreter exit.

    This is **not** filling an absent mechanism. vLLM 0.8.5 already registers
    ``weakref.finalize(self, self.resources)`` in ``MPClient.__init__``, and
    ``BackgroundResources.__call__`` closes every ``CoreEngine`` -- so a worker
    that exits normally does terminate the V1 ``EngineCore`` process it
    started. What an explicit call buys is narrower and worth stating exactly,
    because the difference is what this helper can and cannot be credited with:

    * **It happens at a point we choose**, before the interpreter starts tearing
      down, rather than whenever the finalizer runs.
    * **A failure is visible.** The finalizer's exceptions are swallowed by the
      interpreter at exit; here they are logged, so a teardown that does not
      work stops being invisible.

    It does not help when the worker is killed outright -- ``SIGKILL`` runs no
    finalizer and no ``finally`` block -- which is the case that can still
    strand an ``EngineCore`` holding a full model. The paired
    :func:`method.utils.wait_for_free_vram` on the *next* load is what covers
    that one, by making it a legible wait-then-error instead of an OOM.

    ``LLM`` exposes no ``shutdown`` of its own in 0.8.5, so this reaches the
    one object that has it: ``llm_engine.engine_core``, an
    ``EngineCoreClient`` -- ``MPClient.shutdown`` terminates the background
    process, ``InprocClient.shutdown`` releases in-process state. Every step is
    guarded, and the attribute walk tolerates a different layout, because a
    worker whose output is already written must not fail on teardown.
    """
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
