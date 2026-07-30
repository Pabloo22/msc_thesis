"""Runtime patches applied to ``vllm.LLM`` before the vendored loader runs.

``eval.model_utils.load_vllm_model`` (vendored, no edits allowed) constructs
every ``vllm.LLM(...)`` without a ``dtype`` and with ``max_model_len`` hardcoded
to 30000 (hub) / 20000 (local). Both entry points that end up calling it
(``method.eval_wrapper`` -> ``eval.eval_persona.main``, and
``method._generate_worker`` directly) import this module and call these
functions first, so the fix lives entirely outside persona_vectors.
"""

from __future__ import annotations


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
