"""Run the vendored ``eval.eval_persona`` with a swappable judge.

``eval_persona`` imports ``OpenAiJudge`` directly and instantiates it deep
inside ``load_persona_questions``, so there is no injection point. Since
persona_vectors is vendored and must not be edited, this wrapper patches the
``judge`` module *before* importing ``eval_persona`` and then calls its
``main``. Run it exactly where the vendored script expects to run:

    python -m method.eval_wrapper --model ... --trait ... --output_path ...

with ``cwd`` set to method/persona_vectors.

The stub judge exists because a smoke run's scores are meaningless anyway: the
point of a local run is to prove the plumbing, and paying OpenAI to score
throwaway generations from a 0.5B model buys nothing. Stub scores are
deterministic in the (question, answer) pair, so reruns are reproducible and
resume logic can be tested.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from method.eval_progress import ProgressStore, make_eval_batched
from method.utils import (
    DOTENV_PATH,
    VLLM_FREE_FRACTION,
    load_dotenv,
    wait_for_free_vram,
)
from method.vllm_patches import (
    VENDORED_MAX_NUM_SEQS,
    disable_vllm_lora,
    force_vllm_cudagraph_sizes,
    force_vllm_dtype,
    force_vllm_max_model_len,
    freeze_gc_during_cudagraph_capture,
    share_vllm_compile_cache,
    shutdown_vllm,
)

logger = logging.getLogger(__name__)


class StubJudge:
    """Offline stand-in for ``judge.OpenAiJudge``.

    Mirrors the real judge's interface: constructed with (model,
    prompt_template, eval_type) and awaited with keyword arguments, returning a
    0-100 float.
    """

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        self.prompt_template = prompt_template
        self.eval_type = eval_type

    async def __call__(self, **kwargs) -> float:
        return self._score(kwargs.get("question", ""), kwargs.get("answer", ""))

    async def judge(self, **kwargs) -> float:
        return self._score(kwargs.get("question", ""), kwargs.get("answer", ""))

    def _score(self, question: str, answer: str) -> float:
        """Deterministic pseudo-score, stable across runs and machines."""
        key = f"{self.prompt_template[:64]}|{question}|{answer}".encode()
        digest = hashlib.sha256(key).digest()
        raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        if self.eval_type == "binary":
            return float(raw > 0.5)
        return round(raw * 100, 2)


def install_stub_judge() -> None:
    """Replace ``judge.OpenAiJudge`` before eval_persona binds it."""
    import judge  # vendored module, importable from persona_vectors cwd

    judge.OpenAiJudge = StubJudge  # type: ignore[misc,assignment]


def vendored_generate(
    *,
    llm,
    tokenizer,
    conversations,
    temperature: float,
    max_tokens: int,
    coef,
    vector,
    layer,
    steering_type: str,
    lora_path,
) -> tuple[list[str], list[str]]:
    """Generate answers exactly as the vendored ``eval_batched`` would.

    The dispatch between steered and unsteered sampling is the vendored one,
    lifted verbatim so the resumable path and the original produce the same
    thing. Injected into :func:`make_eval_batched` rather than imported there,
    which keeps the resume logic testable without vLLM.
    """
    from eval import eval_persona

    if coef != 0:
        return eval_persona.sample_steering(
            llm,
            tokenizer,
            conversations,
            vector,
            layer,
            coef,
            temperature=temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    return eval_persona.sample(
        llm,
        tokenizer,
        conversations,
        temperature=temperature,
        max_tokens=max_tokens,
        lora_path=lora_path,
    )


def skip_model_load(model: str):
    """Stand in for ``load_vllm_model`` when every answer is already cached.

    Returns the ``(llm, tokenizer, lora_path)`` triple that ``main`` unpacks.
    All three are unused once generation is skipped, and not loading a 7B model
    to do nothing turns a resumed judging pass into a CPU-only job that needs
    no GPU at all.
    """
    logger.info("every answer is already generated; not loading %s", model)
    return None, None, None


def progress_store(args: argparse.Namespace) -> ProgressStore | None:
    """The store for this invocation, or None when resuming is switched off.

    The two identities are what each half of the work depends on. Generation
    covers everything that decides which answers get produced; judging covers
    who scores them. ``max_concurrent`` is in neither: it changes how fast the
    requests go out, not what they are.

    Generation is identified by ``--model_id`` when the caller supplies one,
    because ``--model`` is a *location* and not always a stable one. A merged
    checkpoint is materialised under ``store/merged/<pid>/<weights_id>`` (see
    :class:`method.store.Store`), so the same weights get a different path in
    every process. Keying the identity on that path made a *restarted* job look
    like a different model, which discarded ``generations.jsonl`` and re-ran
    2000 generations and 4000 judge calls that were already on disk -- the
    precise failure this module exists to prevent, and one that only bit at
    ``t > 0``, since ``materialize`` returns the stable Hub id at ``t = 0``.
    Callers pass the ``weights_id``, which names the weights rather than the
    copy of them.
    """
    if args.progress_dir is None:
        return None
    return ProgressStore(
        Path(args.progress_dir),
        generation={
            "model": args.model_id or args.model,
            "trait": args.trait,
            "version": args.version,
            "persona_instruction_type": args.persona_instruction_type,
            "n_per_question": args.n_per_question,
            "max_tokens": args.max_tokens,
        },
        judging={"model": args.judge_model, "backend": args.judge_backend},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model_id",
        default=None,
        help="Stable identity of the weights behind --model, used for resume "
        "bookkeeping only (never to load anything). Defaults to --model, which "
        "is correct for a Hub id but not for a merged checkpoint, whose path "
        "contains the materialising process's pid. See progress_store.",
    )
    parser.add_argument("--trait", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--version", default="eval")
    parser.add_argument("--persona_instruction_type", default=None)
    parser.add_argument("--n_per_question", type=int, default=1)
    parser.add_argument("--judge_model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--max_concurrent", type=int, default=100)
    parser.add_argument("--judge_backend", default="openai", choices=["openai", "stub"])
    parser.add_argument(
        "--vllm_dtype",
        default=None,
        help="Force vLLM's dtype (e.g. 'half' on pre-Ampere GPUs, which cannot "
        "run bfloat16). Left unset, vLLM picks the model's own dtype.",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        # Mirrors config.ModelConfig.max_seq_length's default; every registry
        # entry (QWEN_7B, QWEN_0_5B) leaves it unset, so this is the value a
        # real run actually uses. Defaulting to it here (rather than None)
        # means a manual invocation without the flag still gets the fix
        # instead of reverting to the vendored 20000/30000.
        default=2048,
        help="Override vLLM's max_model_len, which the vendored loader "
        "hardcodes to 30000 (hub) / 20000 (local) regardless of model size. "
        "That reserves far more KV-cache than a persona-eval request ever "
        "needs, inflating GPU memory use for no benefit.",
    )
    parser.add_argument(
        "--vllm_cudagraph_max_size",
        type=int,
        default=VENDORED_MAX_NUM_SEQS,
        help="Largest batch size to capture a CUDA graph for. vLLM's V1 engine "
        "captures 67 shapes regardless of max_num_seqs, which cost 2148s of "
        "engine startup on a rental box and is paid again by every stage. 0 "
        "restores vLLM's own list, which is only useful for measuring that.",
    )
    parser.add_argument(
        "--vllm_share_compile_cache",
        type=int,
        default=1,
        help="Key vLLM's torch.compile cache on the model architecture rather "
        "than its path. vLLM's own key includes the path, so every merged "
        "checkpoint recompiles from cold (~48s) for a graph identical across "
        "the trajectory. 0 restores vLLM's own hashing.",
    )
    parser.add_argument(
        "--vllm_gc_freeze",
        type=int,
        default=1,
        help="Freeze Python's GC around CUDA-graph capture, so gen-2 passes "
        "stop re-walking the loaded model and its inductor artifacts. 0 still "
        "reports the GC cost but does not avoid it, which is how the two are "
        "told apart.",
    )
    parser.add_argument(
        "--vllm_disable_lora",
        type=int,
        default=1,
        help="Drop the vendored loader's enable_lora when the model is not an "
        "adapter directory. This pipeline merges adapters before inference, so "
        "LoRA is never requested, but its warmup still runs inside every "
        "capture dummy run. 0 keeps LoRA enabled.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress_dir",
        default=None,
        help="Directory for crash-resumable partial results. Generations are "
        "saved once vLLM finishes and each judge score as it lands, so a "
        "failed run resumes instead of re-paying for the whole pass. Must be "
        "derived from the final artifact path, not the caller's scratch path, "
        "or every attempt gets a fresh (empty) directory. Unset disables it.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Usually inherited from the parent process, but load explicitly so the
    # wrapper also works when invoked by hand.
    load_dotenv(DOTENV_PATH)

    if args.vllm_dtype:
        force_vllm_dtype(args.vllm_dtype)
    if args.vllm_max_model_len:
        force_vllm_max_model_len(args.vllm_max_model_len)
    if args.vllm_cudagraph_max_size:
        force_vllm_cudagraph_sizes(args.vllm_cudagraph_max_size)
    if args.vllm_share_compile_cache:
        share_vllm_compile_cache(
            args.model,
            dtype=args.vllm_dtype,
            max_model_len=args.vllm_max_model_len,
        )
    if args.vllm_disable_lora:
        disable_vllm_lora(args.model)
    # Instruments either way: the GC numbers are what say whether freezing is
    # what fixed capture, and they are only meaningful next to a run without it.
    freeze_gc_during_cudagraph_capture(freeze=bool(args.vllm_gc_freeze))

    if args.judge_backend == "stub":
        # Both judge.py and eval_persona's module-level setup_credentials()
        # demand OPENAI_API_KEY even though a stubbed judge never calls
        # OpenAI, so the fallback must be in place before either import runs.
        os.environ.setdefault("OPENAI_API_KEY", "stub-not-used")
        install_stub_judge()

    # Imported only now, so the patch above is in place first.
    from eval import eval_persona

    store = progress_store(args)
    if store is not None:
        store.prepare()
        # main() looks both of these up on the module at call time.
        eval_persona.eval_batched = make_eval_batched(store, vendored_generate)
        if store.has_generations():
            eval_persona.load_vllm_model = skip_model_load

    # The vendored main() owns the engine it builds, so the only handle on it
    # is the loader it calls. Wrapping whatever ``load_vllm_model`` is *now*
    # keeps the skip_model_load branch above intact -- that one returns no
    # engine, and there is then nothing to release.
    engines: list[Any] = []
    loader = eval_persona.load_vllm_model

    def loader_recording_engine(*loader_args, **loader_kwargs):
        # Same reason as _generate_worker: a still-releasing EngineCore from
        # the previous stage makes this load an OOM rather than a tighter fit.
        wait_for_free_vram("eval_wrapper", fraction=VLLM_FREE_FRACTION)
        result = loader(*loader_args, **loader_kwargs)
        engines.append(result[0])
        return result

    eval_persona.load_vllm_model = loader_recording_engine

    try:
        eval_persona.main(
            model=args.model,
            trait=args.trait,
            output_path=args.output_path,
            version=args.version,
            persona_instruction_type=args.persona_instruction_type,
            n_per_question=args.n_per_question,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            max_concurrent_judges=args.max_concurrent,
            overwrite=args.overwrite,
        )
    finally:
        # In a finally block, unlike _generate_worker's: this worker's output
        # is written by main() itself, so there is no "after the results are
        # safe" moment to hook, and a failed eval leaks the card just as
        # readily as a successful one.
        for engine in engines:
            shutdown_vllm(engine)

    # Only now: main() has written the CSV, so the partial results it was
    # rebuilt from are no longer worth anything.
    if store is not None:
        store.clear()


if __name__ == "__main__":
    main()
