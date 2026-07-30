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

from method.eval_progress import ProgressStore, make_eval_batched
from method.utils import DOTENV_PATH, load_dotenv
from method.vllm_patches import force_vllm_dtype, force_vllm_max_model_len

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
    """
    if args.progress_dir is None:
        return None
    return ProgressStore(
        Path(args.progress_dir),
        generation={
            "model": args.model,
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

    # Only now: main() has written the CSV, so the partial results it was
    # rebuilt from are no longer worth anything.
    if store is not None:
        store.clear()


if __name__ == "__main__":
    main()
