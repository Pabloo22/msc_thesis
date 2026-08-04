"""Runtime patches applied to ``vllm.LLM`` before the vendored loader runs.

``eval.model_utils.load_vllm_model`` (vendored, no edits allowed) constructs
every ``vllm.LLM(...)`` without a ``dtype`` and with ``max_model_len`` hardcoded
to 30000 (hub) / 20000 (local). Both entry points that end up calling it
(``method.eval_wrapper`` -> ``eval.eval_persona.main``, and
``method._generate_worker`` directly) import this module and call these
functions first, so the fix lives entirely outside persona_vectors.
"""

from __future__ import annotations

#: ``max_num_seqs`` as the vendored loader hardcodes it. Restated rather than
#: read back off the engine because the cap has to be decided before ``LLM``
#: is constructed, which is the only thing that would know it.
VENDORED_MAX_NUM_SEQS = 32

#: The capture list vLLM's V0 engine would have built for that ``max_num_seqs``
#: -- its ``[1, 2, 4] + [8 * i ...]`` candidates, truncated at the first size
#: that reaches it. See :func:`force_vllm_cudagraph_sizes` for why V1 needs to
#: be told this explicitly.

_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def force_vllm_cudagraph_sizes(max_size: int = VENDORED_MAX_NUM_SEQS) -> None:
    """Cap CUDA-graph capture at the batch sizes this workload can reach.

    vLLM's V0 engine sized its capture list from ``max_num_seqs``: with the
    vendored 32 it captured 7 shapes. V1 builds the list as
    ``[1, 2, 4] + range(8, 513, 8)`` and filters it by
    ``max_num_batched_tokens`` (8192 for the ``LLM`` class) instead, so all 67
    shapes survive -- and it captures a warmup run per shape on top. That is
    134 dummy forward passes where V0 did 14, and it is paid at every engine
    construction, which for this pipeline means once per stage per subprocess.
    Measured on a rental box: 2148s, against the 5-20s vLLM's own
    ``capture_model`` docstring expects.

    Nothing is lost by capping. Graphs are only used when the scheduled token
    count is under the largest captured shape, and a decode batch is bounded by
    ``max_num_seqs``; prefill already overruns any of these sizes and falls
    back to eager regardless.

    Unlike :func:`force_vllm_max_model_len` this is a ``setdefault``: the
    vendored loader never passes ``compilation_config``, so a caller that does
    is asking for something deliberate and should keep it. Everything V1 needs
    for piecewise compilation (``level``, ``use_cudagraph``, ``use_inductor``,
    the splitting ops) is re-applied to whatever config object it ends up with,
    so overriding the sizes alone does not disturb the rest.
    """
    import vllm

    original_init = vllm.LLM.__init__
    sizes = [size for size in _CAPTURE_SIZES if size <= max_size]

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("compilation_config", {"cudagraph_capture_sizes": sizes})
        return original_init(self, *args, **kwargs)

    vllm.LLM.__init__ = patched_init  # type: ignore[method-assign]


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
