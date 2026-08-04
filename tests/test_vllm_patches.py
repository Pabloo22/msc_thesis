"""Tests for the overrides applied to ``vllm.LLM`` before the vendored loader.

These patches exist because ``eval.model_utils.load_vllm_model`` cannot be
edited, so the only evidence that one works is the kwargs the constructor is
finally called with. Every test here therefore stands a fake ``vllm`` module in
for the real one -- importing vLLM would cost a CUDA context and tens of
seconds, and the property under test is purely about argument plumbing.
"""

from __future__ import annotations

import sys
import types

import pytest

from method.vllm_patches import (
    VENDORED_MAX_NUM_SEQS,
    force_vllm_cudagraph_sizes,
    force_vllm_dtype,
    force_vllm_max_model_len,
)


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a stub ``vllm`` whose ``LLM`` records how it was constructed.

    The patches import ``vllm`` inside the function body, so this only has to
    be in ``sys.modules`` by the time they are called. The module is registered
    fresh per test, which is what keeps one test's wrapped ``__init__`` from
    leaking into the next.
    """
    module = types.ModuleType("vllm")

    class LLM:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            type(self).calls.append(kwargs)

    LLM.calls = []
    module.LLM = LLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def test_cudagraph_sizes_capped_at_max_num_seqs(fake_vllm):
    """The default cap is the vendored ``max_num_seqs``, and nothing above it.

    A shape larger than ``max_num_seqs`` can never be scheduled as a decode
    batch, so capturing one is time spent on a graph that is never replayed.
    """
    force_vllm_cudagraph_sizes()
    fake_vllm.LLM(model="whatever")

    sizes = fake_vllm.LLM.calls[0]["compilation_config"]["cudagraph_capture_sizes"]
    assert sizes == [1, 2, 4, 8, 16, 24, 32]
    assert max(sizes) == VENDORED_MAX_NUM_SEQS


def test_cudagraph_sizes_honour_a_smaller_cap(fake_vllm):
    force_vllm_cudagraph_sizes(8)
    fake_vllm.LLM(model="whatever")

    sizes = fake_vllm.LLM.calls[0]["compilation_config"]["cudagraph_capture_sizes"]
    assert sizes == [1, 2, 4, 8]


def test_cudagraph_sizes_defer_to_an_explicit_config(fake_vllm):
    """A caller that passes ``compilation_config`` keeps it.

    Unlike ``max_model_len``, the vendored loader never sets this, so anything
    already there was asked for deliberately and must not be overwritten.
    """
    force_vllm_cudagraph_sizes()
    fake_vllm.LLM(model="whatever", compilation_config={"level": 0})

    assert fake_vllm.LLM.calls[0]["compilation_config"] == {"level": 0}


def test_patches_compose(fake_vllm):
    """All three overrides survive being layered onto the same ``__init__``.

    Both entry points apply them in sequence, so each one wraps the previous
    one's wrapper; a patch that captured the original ``__init__`` at the wrong
    moment would silently drop whichever came before it.
    """
    force_vllm_dtype("half")
    force_vllm_max_model_len(2048)
    force_vllm_cudagraph_sizes()

    fake_vllm.LLM(model="whatever", max_model_len=20000)

    call = fake_vllm.LLM.calls[0]
    assert call["dtype"] == "half"
    assert call["max_model_len"] == 2048
    assert call["compilation_config"]["cudagraph_capture_sizes"][-1] == 32
