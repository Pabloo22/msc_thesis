"""Tests for the GPU handover guard: the pre-load wait and the engine teardown.

**What these do not cover.** The failure that motivated them -- a worker dying
because another process held 12 GiB of a 24 GiB card -- cannot be reproduced
without a real GPU and two real engines, so nothing here proves the guard
prevents it. What is pinned here is the decision logic: when the gate returns,
when it waits, when it refuses to block, and what it says when it gives up.
The handover itself has to be exercised on a box with a card; see
``scripts/gpu_handover_check.sh``.

The gate is deliberately tested through injected ``probe`` and ``sleep``
callables rather than by faking ``nvidia-smi``, because the property worth
holding is the decision, not the parsing -- which is covered separately and
against the driver's real output format.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from method.utils import (
    TORCH_FREE_FRACTION,
    VLLM_FREE_FRACTION,
    gpu_memory,
    wait_for_free_vram,
)
from method.vllm_patches import shutdown_vllm

GIB = 1024**3


def card(free_gib: float, total_gib: float = 24.0):
    return lambda: (int(free_gib * GIB), int(total_gib * GIB))


class TestTheGateDecides:
    def test_a_free_card_returns_without_sleeping(self) -> None:
        slept: list[float] = []
        wait_for_free_vram("w", probe=card(23.0), sleep=slept.append)
        assert slept == []

    def test_a_held_card_waits_and_then_proceeds(self) -> None:
        """The holder is usually a process already on its way out, so waiting
        is the response that keeps the stage rather than losing it."""
        readings = [card(2.0)(), card(2.0)(), card(23.0)()]
        slept: list[float] = []
        wait_for_free_vram(
            "w",
            probe=lambda: readings.pop(0),
            sleep=slept.append,
            timeout_s=60,
            poll_s=2,
        )
        assert slept == [2, 2]

    def test_a_card_that_never_frees_raises(self) -> None:
        with pytest.raises(RuntimeError, match="needs 90% of the GPU free"):
            wait_for_free_vram(
                "w", probe=card(1.0), sleep=lambda _: None, timeout_s=4, poll_s=2
            )

    def test_the_error_reports_what_is_holding_the_card(self, monkeypatch) -> None:
        """The one thing the OOM could not say. A pid with no owner is still
        worth printing -- it is what a leaked engine looks like."""
        monkeypatch.setattr(
            "method.utils.gpu_processes", lambda: "4098473, 12376 MiB (1 ...)"
        )
        with pytest.raises(RuntimeError, match="Holding it: 4098473"):
            wait_for_free_vram(
                "w", probe=card(1.0), sleep=lambda _: None, timeout_s=0, poll_s=2
            )

    def test_an_unreadable_driver_does_not_block_the_run(self) -> None:
        """A missing or wedged ``nvidia-smi`` must not be able to stop a run
        that would otherwise work: the gate is defence in depth, and a guard
        that fails closed on its own blind spot is worse than no guard."""
        slept: list[float] = []
        wait_for_free_vram("w", probe=lambda: None, sleep=slept.append)
        assert slept == []

    def test_a_driver_reporting_no_capacity_is_not_treated_as_full(self) -> None:
        wait_for_free_vram("w", probe=lambda: (0, 0), sleep=lambda _: None)

    def test_the_vllm_bar_matches_what_the_vendored_loader_reserves(self) -> None:
        """Not a guess about the workload: ``eval/model_utils.py`` hardcodes
        ``gpu_memory_utilization=0.9``, so anything less would let a load
        through that vLLM itself will refuse."""
        assert VLLM_FREE_FRACTION == 0.9
        assert TORCH_FREE_FRACTION < VLLM_FREE_FRACTION

    def test_a_worker_needing_less_is_admitted_where_vllm_would_wait(self) -> None:
        """``_hidden_worker`` loads weights through transformers instead of
        reserving a fraction up front, so holding it to vLLM's bar would stall
        it on a card it could actually use."""
        slept: list[float] = []
        wait_for_free_vram(
            "w", fraction=TORCH_FREE_FRACTION, probe=card(19.0), sleep=slept.append
        )
        assert slept == []
        with pytest.raises(RuntimeError):
            wait_for_free_vram(
                "w",
                fraction=VLLM_FREE_FRACTION,
                probe=card(19.0),
                sleep=lambda _: None,
                timeout_s=0,
            )


class TestReadingTheCard:
    """``gpu_memory`` shells out on purpose -- see its docstring.

    Asking torch instead would create a CUDA context in a process that is about
    to let vLLM fork ``EngineCore``, which is the exact failure
    :func:`method.utils.require_cuda` is written to avoid.
    """

    @staticmethod
    def _driver(monkeypatch, stdout: str, *, fail: bool = False):
        def fake_run(cmd, **_kwargs):
            if fail:
                raise FileNotFoundError("nvidia-smi")
            assert cmd[0] == "nvidia-smi"
            return types.SimpleNamespace(stdout=stdout, returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_parses_the_drivers_csv(self, monkeypatch) -> None:
        self._driver(monkeypatch, "23012, 24564\n")
        free, total = gpu_memory()
        assert (free, total) == (23012 * 1024**2, 24564 * 1024**2)

    def test_a_missing_driver_reads_as_unknown(self, monkeypatch) -> None:
        self._driver(monkeypatch, "", fail=True)
        assert gpu_memory() is None

    def test_unparseable_output_reads_as_unknown(self, monkeypatch) -> None:
        self._driver(monkeypatch, "N/A, N/A\n")
        assert gpu_memory() is None

    def test_it_asks_about_the_visible_device(self, monkeypatch) -> None:
        """``CUDA_VISIBLE_DEVICES`` is how this pipeline splits work across
        cards, and NVML does not honour it -- index 0 there is the first
        physical card, not the first visible one."""
        seen: list[str] = []

        def fake_run(cmd, **_kwargs):
            seen.append(cmd[cmd.index("-i") + 1])
            return types.SimpleNamespace(stdout="1, 2\n", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        gpu_memory()
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        gpu_memory()
        assert seen == ["1", "0"]


class TestEngineTeardown:
    """``shutdown_vllm`` makes an existing finalizer happen at a known point.

    vLLM 0.8.5 already closes ``EngineCore`` via ``weakref.finalize``, so these
    pin the two things the explicit call adds: it runs where we put it, and a
    teardown that fails says so instead of vanishing into interpreter exit.
    """

    @staticmethod
    def _engine(shutdown):
        core = types.SimpleNamespace(shutdown=shutdown)
        return types.SimpleNamespace(
            llm_engine=types.SimpleNamespace(engine_core=core)
        )

    def test_it_shuts_the_core_client_down(self) -> None:
        called: list[str] = []
        shutdown_vllm(self._engine(lambda: called.append("core")))
        assert called == ["core"]

    def test_a_failing_teardown_is_logged_not_raised(self, caplog) -> None:
        """This runs in a ``finally`` beside a generation that already
        succeeded; letting it raise would turn a completed pass into a failed
        stage."""

        def boom():
            raise RuntimeError("zmq is unhappy")

        with caplog.at_level("WARNING"):
            shutdown_vllm(self._engine(boom))
        assert "zmq is unhappy" in caplog.text

    def test_an_unrecognised_object_is_a_no_op(self) -> None:
        """The attribute walk is a guess about vLLM's layout, and a wrong
        guess must not be able to fail a worker."""
        shutdown_vllm(types.SimpleNamespace())
        shutdown_vllm(None)

    def test_it_stops_at_the_first_handle_that_works(self) -> None:
        """``LLM`` has no ``shutdown`` of its own; ``engine_core`` does. Going
        on to try the others would call teardown twice on a client that has
        already released its process."""
        called: list[str] = []
        engine = self._engine(lambda: called.append("core"))
        engine.shutdown = lambda: called.append("llm")
        engine.llm_engine.shutdown = lambda: called.append("engine")
        shutdown_vllm(engine)
        assert called == ["core"]
