"""Tests for the GPU health check every worker opens with.

Two properties, and the second is the non-obvious one. The check has to reject
a box with no usable device, and it has to reach that verdict without asking
the CUDA runtime -- because a worker that answers the question the expensive
way poisons its own ability to fork, and vLLM's V1 engine forks. The failure
that costs is remote (a dead ``EngineCore`` in a child process, minutes later),
so it is pinned here rather than left to the one environment that shows it.
"""

from __future__ import annotations

import sys
import types

import pytest

from method.utils import require_cuda


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a stub ``torch`` that records which availability check ran.

    ``require_cuda`` imports torch inside the function body, so replacing the
    entry in ``sys.modules`` is enough, and it keeps the test off the real GPU
    -- whose presence or absence is exactly what is being faked.
    """
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        devices=0,
        asked=[],
        init=lambda: None,
    )

    def device_count() -> int:
        torch.cuda.asked.append("device_count")
        return torch.cuda.devices

    def is_available() -> bool:
        torch.cuda.asked.append("is_available")
        return torch.cuda.devices > 0

    torch.cuda.device_count = device_count
    torch.cuda.is_available = is_available
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class TestRequireCuda:
    def test_passes_when_a_device_is_present(self, fake_torch):
        fake_torch.cuda.devices = 1

        require_cuda("worker")  # no raise

    def test_raises_when_no_device_is_present(self, fake_torch):
        with pytest.raises(RuntimeError, match="needs a CUDA device"):
            require_cuda("worker")

    def test_raise_carries_the_drivers_own_reason(self, fake_torch):
        def wedged() -> None:
            raise RuntimeError("Unable to determine the device handle for GPU1")

        fake_torch.cuda.init = wedged

        with pytest.raises(RuntimeError, match="device handle for GPU1"):
            require_cuda("worker")

    def test_does_not_ask_the_cuda_runtime(self, fake_torch):
        """The fork-safety property: ``is_available`` must stay untouched.

        ``torch.cuda.is_available()`` calls ``cudaGetDeviceCount``, which makes
        torch register the ``atfork`` handler that marks the process's later
        forks as bad -- without setting the flag vLLM checks before choosing
        fork over spawn for its ``EngineCore``. ``device_count`` reads NVML
        first and leaves fork usable.
        """
        fake_torch.cuda.devices = 1

        require_cuda("worker")

        assert fake_torch.cuda.asked == ["device_count"]
