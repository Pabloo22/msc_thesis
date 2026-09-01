"""Generate answers to a set of prompts with vLLM.

Used for the two generation passes that happen once per trajectory: M_0's
answers to the neutral probe prompts, and M_0's answers to each training set's
prompts (the ``h_predicted`` term of DeltaP). Both are reused unchanged at every
later step, so this worker runs far less often than its cost suggests.

Reuses the vendored ``eval.model_utils.load_vllm_model`` so the local and
rental paths load models identically.

    python -m method._generate_worker --model P --input X.jsonl --output Y.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from method.utils import (
    PERSONA_VECTORS_DIR,
    VLLM_FREE_FRACTION,
    require_cuda,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm_dtype", default=None)
    # Mirrors config.ModelConfig.max_seq_length's default (see eval_wrapper.py).
    parser.add_argument("--vllm_max_model_len", type=int, default=2048)
    # 0 leaves V1's 67-shape capture list alone, which is only useful for
    # measuring what the cap saves (see vllm_patches).
    parser.add_argument(
        "--vllm_cudagraph_max_size", type=int, default=VENDORED_MAX_NUM_SEQS
    )
    # The remaining three all target engine startup, and all default on. 0 turns
    # one off, which is what scripts/bench_cudagraph.sh varies to attribute the
    # time (see the docstrings in method.vllm_patches for what each one costs).
    parser.add_argument("--vllm_share_compile_cache", type=int, default=1)
    parser.add_argument("--vllm_gc_freeze", type=int, default=1)
    parser.add_argument("--vllm_disable_lora", type=int, default=1)
    args = parser.parse_args()

    # vLLM would fail on its own, but only after ~45s of engine startup and
    # with the reason buried in an EngineCore traceback.
    require_cuda("_generate_worker")
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

    # The vendored package imports itself by bare top-level names (e.g.
    # ``from config import ...``, ``from eval.model_utils import ...``), which
    # only resolve when its own directory is a sys.path root. We can't rewrite
    # those imports (persona_vectors is vendored, no edits), so put the
    # directory on sys.path and import as the package expects to be run.
    sys.path.insert(0, str(PERSONA_VECTORS_DIR))
    from eval.model_utils import load_vllm_model  # noqa: E402
    from vllm import SamplingParams  # noqa: E402

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    # The previous stage's engine may still be releasing the card: V1 tears
    # EngineCore down in its own process, and this worker asks for 90% of the
    # GPU, so an overlap is an OOM rather than a smaller KV cache.
    wait_for_free_vram("_generate_worker", fraction=VLLM_FREE_FRACTION)
    llm, tokenizer, lora_path = load_vllm_model(args.model)

    texts = [
        tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    params = SamplingParams(
        temperature=args.temperature,
        top_p=1,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=1,
    )
    kwargs = {"sampling_params": params, "use_tqdm": True}
    if lora_path:
        from vllm.lora.request import LoRARequest

        kwargs["lora_request"] = LoRARequest("default", 1, lora_path=lora_path)
    try:
        completions = llm.generate(texts, **kwargs)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row, completion in zip(rows, completions):
                messages = list(row["messages"])
                messages.append(
                    {"role": "assistant", "content": completion.outputs[0].text}
                )
                handle.write(json.dumps({"messages": messages}) + "\n")
        print(f"generated {len(rows)} answers -> {args.output}", flush=True)
    finally:
        # In `finally`, because a generation that raises holds the card exactly
        # as firmly as one that succeeds -- and the next worker's load is what
        # pays for it. Safe to run on the failure path: shutdown_vllm swallows
        # its own errors, so it cannot mask the exception on its way out.
        shutdown_vllm(llm)


if __name__ == "__main__":
    main()
