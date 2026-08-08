"""Tests for the overrides applied to ``vllm.LLM`` before the vendored loader.

These patches exist because ``eval.model_utils.load_vllm_model`` cannot be
edited, so the only evidence that one works is the kwargs the constructor is
finally called with. Every test here therefore stands a fake ``vllm`` module in
for the real one -- importing vLLM would cost a CUDA context and tens of
seconds, and the property under test is purely about argument plumbing.

The startup patches (compile cache, GC freeze, LoRA) are about engine startup
cost rather than correctness of the numbers, so what is pinned here is that they
reach the constructor at all, that they compose, and that each one declines to
act in the case where acting would be wrong.
"""

from __future__ import annotations

import gc
import json
import sys
import types

import pytest

from method.vllm_patches import (
    VENDORED_MAX_NUM_SEQS,
    disable_vllm_lora,
    force_vllm_cudagraph_sizes,
    force_vllm_dtype,
    force_vllm_max_model_len,
    freeze_gc_during_cudagraph_capture,
    is_adapter_dir,
    share_vllm_compile_cache,
)


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a stub ``vllm`` whose ``LLM`` records how it was constructed.

    The patches import ``vllm`` inside the function body, so this only has to
    be in ``sys.modules`` by the time they are called. The module is registered
    fresh per test, which is what keeps one test's wrapped ``__init__`` from
    leaking into the next.

    The submodules matter as much as ``LLM`` does: the compile-cache patch reads
    ``vllm.envs`` for the cache root and the env hash, and the GC patch replaces
    a method on ``GPUModelRunner``, so all of them have to be resolvable by the
    same ``from ... import ...`` forms the real ones use.
    """
    module = types.ModuleType("vllm")
    module.__version__ = "0.8.5.post1"  # type: ignore[attr-defined]

    class LLM:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            type(self).calls.append(kwargs)

    LLM.calls = []
    module.LLM = LLM  # type: ignore[attr-defined]

    envs = types.ModuleType("vllm.envs")
    envs.VLLM_CACHE_ROOT = "/cache/vllm"  # type: ignore[attr-defined]
    envs.compute_hash = lambda: "env-hash"  # type: ignore[attr-defined]
    module.envs = envs  # type: ignore[attr-defined]

    logged: list[tuple] = []

    def init_logger(name):
        return types.SimpleNamespace(info=lambda fmt, *args: logged.append(args))

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = init_logger  # type: ignore[attr-defined]
    logger_module.logged = logged  # type: ignore[attr-defined]

    runner_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        captured: list[str] = []

        def capture_model(self):
            # A real capture allocates hard enough to trigger gen-2 passes; one
            # explicit collection stands in for that, so the probe has something
            # to count whether or not freezing made it cheap.
            gc.collect(2)
            type(self).captured.append("captured")

    GPUModelRunner.captured = []
    runner_module.GPUModelRunner = GPUModelRunner  # type: ignore[attr-defined]

    for name, mod in [
        ("vllm", module),
        ("vllm.envs", envs),
        ("vllm.logger", logger_module),
        ("vllm.v1.worker.gpu_model_runner", runner_module),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return module


@pytest.fixture
def gc_events(monkeypatch):
    """Record ``gc.freeze``/``gc.collect`` calls instead of performing them.

    Spying rather than reading ``gc.get_freeze_count()``, for two reasons. The
    count cannot attribute anything: CPython puts ~375 of its own startup
    objects back into the permanent generation on any ``gc.collect()``, so the
    number moves whether or not this patch froze anything. And ``gc.freeze()`` is
    process-wide and deliberately permanent in production, so a test that really
    performed it would stop the collector from ever looking at objects the rest
    of the suite creates.
    """
    events: list[str] = []
    monkeypatch.setattr(gc, "freeze", lambda: events.append("freeze"))
    monkeypatch.setattr(gc, "collect", lambda *args: events.append("collect"))
    return events


def write_model_dir(root, *, name_or_path: str, hidden_size: int = 3584):
    """A directory that looks enough like a merged checkpoint to be keyed."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": hidden_size,
                "num_hidden_layers": 28,
                "_name_or_path": name_or_path,
            }
        ),
        encoding="utf-8",
    )
    return root


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


def test_compilation_config_defers_per_key(fake_vllm):
    """A key the caller set is kept; the ones it did not set are still added.

    The vendored loader never passes ``compilation_config``, so anything already
    there was asked for deliberately and must survive. That deference is per key
    rather than per dict because more than one patch now writes into this dict --
    see ``test_compilation_patches_compose_in_either_order``.
    """
    force_vllm_cudagraph_sizes()
    fake_vllm.LLM(model="whatever", compilation_config={"level": 0})

    config = fake_vllm.LLM.calls[0]["compilation_config"]
    assert config["level"] == 0
    assert config["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 24, 32]


def test_compilation_config_left_alone_when_not_a_dict(fake_vllm):
    """vLLM also accepts a bare level, which there is no safe way to merge into."""
    force_vllm_cudagraph_sizes()
    fake_vllm.LLM(model="whatever", compilation_config=3)

    assert fake_vllm.LLM.calls[0]["compilation_config"] == 3


@pytest.mark.parametrize("cache_first", [True, False])
def test_compilation_patches_compose_in_either_order(fake_vllm, tmp_path, cache_first):
    """Two patches writing to ``compilation_config`` must not erase each other.

    Each patch wraps the previous one's ``__init__``, so the last applied runs
    first. A patch that claimed the whole dict with one ``setdefault`` would
    leave the other looking at a dict that already existed and silently drop its
    key -- and the symptom would be a slow engine, not a failure.
    """
    model = write_model_dir(tmp_path / "merged", name_or_path="base")
    apply = [
        lambda: share_vllm_compile_cache(str(model)),
        force_vllm_cudagraph_sizes,
    ]
    for patch in apply if cache_first else reversed(apply):
        patch()

    fake_vllm.LLM(model=str(model))

    config = fake_vllm.LLM.calls[0]["compilation_config"]
    assert config["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 24, 32]
    assert config["cache_dir"].startswith("/cache/vllm/torch_compile_cache/msc-")


def test_compile_cache_shared_across_checkpoints_of_one_architecture(
    fake_vllm, tmp_path
):
    """Two checkpoints of the same model must land on the same cache directory.

    This is the whole point: vLLM's own key includes the model path, so every
    merged checkpoint recompiles from cold for a graph that cannot differ.
    ``_name_or_path`` differs too, since ``save_pretrained`` records wherever the
    model was written from, so dropping it is what makes the key stable.
    """
    first = write_model_dir(tmp_path / "t0", name_or_path=str(tmp_path / "t0"))
    second = write_model_dir(tmp_path / "t5", name_or_path=str(tmp_path / "t5"))

    assert share_vllm_compile_cache(str(first)) == share_vllm_compile_cache(str(second))


def test_compile_cache_differs_when_the_graph_would(fake_vllm, tmp_path):
    """Anything that reaches the compiled graph has to reach the key.

    A collision here would run one architecture's compiled kernels against
    another's weights, which is worse than the cold compile it saves.
    """
    small = write_model_dir(tmp_path / "small", name_or_path="x", hidden_size=896)
    large = write_model_dir(tmp_path / "large", name_or_path="x", hidden_size=3584)

    assert share_vllm_compile_cache(str(small)) != share_vllm_compile_cache(str(large))
    assert share_vllm_compile_cache(str(small), dtype="half") != (
        share_vllm_compile_cache(str(small), dtype=None)
    )
    assert share_vllm_compile_cache(str(small), max_model_len=2048) != (
        share_vllm_compile_cache(str(small), max_model_len=20000)
    )


def test_compile_cache_declines_when_architecture_is_unknown(fake_vllm, tmp_path):
    """A hub id is not on disk, so there is nothing to key on but the name.

    Falling back to vLLM's own hashing costs a cold compile. Guessing would risk
    the collision above, so the patch has to leave ``cache_dir`` unset.
    """
    assert share_vllm_compile_cache("Qwen/Qwen2.5-7B-Instruct") is None

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert share_vllm_compile_cache(str(adapter)) is None

    fake_vllm.LLM(model="Qwen/Qwen2.5-7B-Instruct")
    assert "compilation_config" not in fake_vllm.LLM.calls[0]


def test_lora_disabled_for_merged_weights(fake_vllm, tmp_path):
    """Merged weights never issue a LoRARequest, so the machinery is dead weight.

    Overwrites rather than setdefaults, because the vendored loader passes
    ``enable_lora=True`` explicitly.
    """
    model = write_model_dir(tmp_path / "merged", name_or_path="base")

    assert disable_vllm_lora(str(model)) is True
    fake_vllm.LLM(model=str(model), enable_lora=True, max_lora_rank=128)

    assert fake_vllm.LLM.calls[0]["enable_lora"] is False


def test_lora_kept_for_an_adapter_directory(fake_vllm, tmp_path):
    """``_generate_worker`` still supports being pointed at an adapter.

    vLLM would reject the ``LoRARequest`` the vendored loader goes on to build,
    so this is the one case where the saving must be declined.
    """
    adapter = tmp_path / "run"
    (adapter / "checkpoint-20").mkdir(parents=True)
    (adapter / "checkpoint-20" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )

    assert is_adapter_dir(str(adapter)) is True
    assert disable_vllm_lora(str(adapter)) is False

    fake_vllm.LLM(model=str(adapter), enable_lora=True)
    assert fake_vllm.LLM.calls[0]["enable_lora"] is True


def test_adapter_probe_follows_the_latest_checkpoint(tmp_path):
    """Mirrors the vendored ``_pick_latest_checkpoint``, including its ordering.

    ``checkpoint-9`` must not beat ``checkpoint-20``; the vendored loader
    compares the numbers, so a lexicographic probe here would disagree with what
    vLLM is actually handed.
    """
    root = tmp_path / "run"
    for step in (9, 20):
        (root / f"checkpoint-{step}").mkdir(parents=True)
    (root / "checkpoint-20" / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert is_adapter_dir(str(root)) is True


def test_heap_is_frozen_before_capture_runs(fake_vllm, gc_events):
    """Collect, then freeze, and only then capture.

    The order is the whole mechanism. Collecting first is what makes freezing
    safe -- a genuine cycle is reclaimed before the rest becomes permanent -- and
    freezing has to land before capture starts, since it is capture's own
    allocations that trigger the gen-2 passes being avoided.
    """
    runner_module = sys.modules["vllm.v1.worker.gpu_model_runner"]

    freeze_gc_during_cudagraph_capture()
    runner_module.GPUModelRunner().capture_model()

    # The trailing "collect" is the stand-in capture's own, so freezing having
    # happened before it is what says the real capture would be covered.
    assert gc_events == ["collect", "freeze", "collect"]
    assert runner_module.GPUModelRunner.captured == ["captured"]


def test_heap_not_frozen_when_disabled(fake_vllm, gc_events):
    """``freeze=False`` still instruments, which is what makes the case.

    A capture time measured with the collector frozen means nothing on its own;
    the run without it is the comparison that attributes the seconds.
    """
    runner_module = sys.modules["vllm.v1.worker.gpu_model_runner"]

    freeze_gc_during_cudagraph_capture(freeze=False)
    runner_module.GPUModelRunner().capture_model()

    assert "freeze" not in gc_events
    assert runner_module.GPUModelRunner.captured == ["captured"]


def test_capture_reports_what_gc_cost(fake_vllm):
    """The log line is the diagnostic, so it has to survive to the log.

    Left un-stubbed on purpose: this is the one test that exercises the real
    ``gc.callbacks`` path, which is what turns the next rental run into evidence
    either way. Its absence from a box's log would mean the patch never reached
    the ``EngineCore`` subprocess.
    """
    runner_module = sys.modules["vllm.v1.worker.gpu_model_runner"]
    logged = sys.modules["vllm.logger"].logged

    freeze_gc_during_cudagraph_capture(freeze=False)
    runner_module.GPUModelRunner().capture_model()

    collections, seconds, frozen, _ = logged[0]
    assert collections >= 1
    assert seconds >= 0.0
    assert frozen is False


def test_patches_compose(fake_vllm, tmp_path):
    """Every override survives being layered onto the same ``__init__``.

    Both entry points apply them in sequence, so each one wraps the previous
    one's wrapper; a patch that captured the original ``__init__`` at the wrong
    moment would silently drop whichever came before it.
    """
    model = write_model_dir(tmp_path / "merged", name_or_path="base")

    force_vllm_dtype("half")
    force_vllm_max_model_len(2048)
    force_vllm_cudagraph_sizes()
    share_vllm_compile_cache(str(model), dtype="half", max_model_len=2048)
    disable_vllm_lora(str(model))

    fake_vllm.LLM(model=str(model), max_model_len=20000, enable_lora=True)

    call = fake_vllm.LLM.calls[0]
    assert call["dtype"] == "half"
    assert call["max_model_len"] == 2048
    assert call["enable_lora"] is False
    assert call["compilation_config"]["cudagraph_capture_sizes"][-1] == 32
    assert call["compilation_config"]["cache_dir"].startswith("/cache/vllm/")
