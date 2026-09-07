"""Apply compatibility patches for Hugging Face tooling."""

from __future__ import annotations


def force_hf_dtype(dtype: str) -> None:
    """Make ``AutoModelForCausalLM.from_pretrained`` default to ``dtype``.

    ``setdefault``, not overwrite: our own workers (``_hidden_worker``,
    ``_merge_worker``) pass ``torch_dtype`` explicitly and keep deciding for
    themselves. Only callers that named no dtype at all are affected.
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"{dtype!r} is not a torch dtype")

    original = AutoModelForCausalLM.from_pretrained

    def patched(*args, **kwargs):
        kwargs.setdefault("torch_dtype", torch_dtype)
        return original(*args, **kwargs)

    AutoModelForCausalLM.from_pretrained = patched  # type: ignore[method-assign]
