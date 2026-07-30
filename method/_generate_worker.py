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

from method.utils import PERSONA_VECTORS_DIR
from method.vllm_patches import force_vllm_dtype, force_vllm_max_model_len


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
    args = parser.parse_args()

    if args.vllm_dtype:
        force_vllm_dtype(args.vllm_dtype)
    if args.vllm_max_model_len:
        force_vllm_max_model_len(args.vllm_max_model_len)

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


if __name__ == "__main__":
    main()
