"""Runtime patches applied to ``transformers`` before vendored code loads a model.

``generate_vec.py`` (vendored, no edits allowed) calls
``AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")`` without
a dtype. On transformers 4.52 an unset ``torch_dtype`` means
``torch.get_default_dtype()`` -- float32 -- regardless of the ``bfloat16`` in
the checkpoint's own config, so a 7B model asks for ~30GB. ``device_map="auto"``
then quietly fits what it can on the GPU and offloads the rest to CPU, and the
first forward pass OOMs streaming an offloaded layer back in. Loading in the
backend's own dtype halves the footprint and keeps the whole model resident.

``method._vector_worker`` applies this before importing the vendored script, so
the fix lives entirely outside persona_vectors.
"""

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
