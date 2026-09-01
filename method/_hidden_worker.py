"""Compute response-average hidden states for fixed prompt/answer pairs.

One worker covers every activation measurement in the pipeline, because they
are all the same operation on different text:

* ``h_neutral`` -- neutral probe prompts with their answers
* the target term of DeltaP -- training prompts with their target answers
* the predicted term of DeltaP -- training prompts with M_0's own answers

Averaging over *response* tokens matches how ``generate_vec.py`` builds the
persona vector (``*_response_avg_diff.pt``), so activations and vectors live in
the same space and their dot product is meaningful. Tokenization mirrors the
vendored code exactly: the concatenated ``prompt + answer`` is re-encoded and
the boundary sits at ``len(encode(prompt))``.

Samples are processed in right-padded batches. Right padding keeps every real
token at its unbatched position (default position ids are a plain ``arange``),
and the response slice is taken per sample from its own true lengths, so
padded positions never enter any mean; batching changes throughput, not
values. That invariant is what lets :func:`plan_batches` group samples by
length rather than by position -- rows are written back to their input slots,
so DeltaP still subtracts them from another file's row-by-row.

Run as its own process so only one model occupies the GPU at a time:

    python -m method._hidden_worker --model P --input X.jsonl --layer L --out D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.utils import TORCH_FREE_FRACTION, require_cuda, wait_for_free_vram

#: Padded tokens per forward pass. ``output_hidden_states`` keeps all L+1
#: layers of the batch resident at once, so the peak is set by
#: ``rows x padded width``, not by rows alone: at 7B (29 x 3584 x 2 bytes per
#: token, plus ~3 x 18944 x 2 for the MLP intermediates the forward holds
#: live) a padded token costs ~0.31MB, which puts 12288 of them at ~3.8GB on
#: top of the ~15.2GB of weights. That leaves headroom on a 24GB card for the
#: allocator's own slack, which is the margin an all-long batch used to spend.
DEFAULT_MAX_BATCH_TOKENS = 12288


def plan_batches(
    lengths: list[int], batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    """Group sample indices into batches, longest sample first.

    Sizing batches by rows alone is what makes this worker OOM on the tail of a
    dataset rather than at its start: eight 2.8k-token samples cost four times
    eight 700-token ones, so a fixed row count survives thousands of batches and
    then dies on the one batch that happened to collect the long samples. Here a
    batch is capped by ``rows x padded width`` as well, so the long ones simply
    get fewer rows.

    Descending order does the rest. It packs each batch with samples of similar
    length, so little of that budget is spent on padding, and it puts the
    heaviest batch first -- a size that cannot fit fails in the first seconds
    instead of hours into a run.

    A sample longer than the budget is still emitted, alone: dropping it would
    misalign every later row (see the row-alignment check in
    :func:`response_avg_hidden`).
    """
    batches: list[list[int]] = []
    current: list[int] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
        # Descending, so the batch's first sample is its longest, and padding
        # widens every row to that.
        width = lengths[current[0]] if current else lengths[i]
        if current and (len(current) + 1) * width > max_batch_tokens:
            batches.append(current)
            current = []
        current.append(i)
        if len(current) == batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def load_pairs(path: Path, tokenizer) -> tuple[list[str], list[str]]:
    """Read prompt/answer pairs from a jsonl file.

    Accepts either explicit ``{"prompt", "answer"}`` records or the
    ``{"messages": [...]}`` form used by the training datasets, in which case
    the chat template is applied exactly as ``eval/cal_projection.py`` does, so
    activations are comparable with the vendored projections.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    if "messages" in rows[0]:
        prompts = [
            tokenizer.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            for r in rows
        ]
        return prompts, [r["messages"][-1]["content"] for r in rows]
    return [r["prompt"] for r in rows], [r["answer"] for r in rows]


@torch.no_grad()
def response_avg_hidden(
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    layer: int,
    batch_size: int = 8,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-layer mean over samples, per-sample activations at ``layer``).

    Storing the full per-sample tensor for every layer would cost ~200MB per
    step at 7B, so only the selected layer is kept per sample; the cross-sample
    mean is cheap enough to keep for all layers, which leaves the layer choice
    revisitable post hoc for aggregate quantities.

    ``batch_size`` and ``max_batch_tokens`` together trade GPU memory for
    throughput; see :func:`plan_batches` for how a batch is sized from the two.
    """
    n_layers = model.config.num_hidden_layers
    running = [torch.zeros(model.config.hidden_size, dtype=torch.float64)
               for _ in range(n_layers + 1)]
    per_sample: list[torch.Tensor | None] = [None] * len(prompts)

    full_ids = [
        tokenizer.encode(prompt + answer, add_special_tokens=False)
        for prompt, answer in zip(prompts, answers)
    ]
    prompt_lens = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    for i, (ids, prompt_len) in enumerate(zip(full_ids, prompt_lens)):
        if len(ids) <= prompt_len:
            # Refuse rather than skip: DeltaP subtracts these activations
            # row-by-row from another file's, so silently dropping a row here
            # would pair the wrong examples downstream. Empty responses should
            # be impossible (datasets carry answers; generation uses
            # min_tokens=1), so hitting this means the input needs fixing.
            raise ValueError(
                f"sample {i}: response contributes no tokens after "
                "tokenization; outputs must stay row-aligned with the input"
            )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batches = plan_batches(
        [len(ids) for ids in full_ids], batch_size, max_batch_tokens
    )
    for batch in tqdm(batches, desc="hidden states", unit="batch"):
        max_len = max(len(full_ids[i]) for i in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, i in enumerate(batch):
            ids = full_ids[i]
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, : len(ids)] = 1

        out = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            output_hidden_states=True,
            # Only ``out.hidden_states`` is read below, but a causal-LM head
            # otherwise projects *every* position to the 152k-token vocabulary
            # on the way there: batch 8 x ~1.8k tokens x 152064 x 2 bytes is
            # ~4GiB per forward, allocated and discarded unread. That is what
            # pushes a 7B bf16 model past 24GB on the longer datasets, since it
            # lands on top of the ~2.8GiB of hidden states actually wanted.
            # Keeping one position is the smallest slice the forward accepts.
            logits_to_keep=1,
        )
        for j, i in enumerate(batch):
            lo, hi = prompt_lens[i], len(full_ids[i])
            for idx in range(n_layers + 1):
                avg = out.hidden_states[idx][j, lo:hi, :].mean(dim=0)
                running[idx] += avg.detach().float().double().cpu()
                if idx == layer:
                    # Written to the sample's input slot, not appended: batches
                    # are planned by length, so ``batch`` is not in input order.
                    per_sample[i] = avg.detach().float().cpu()
        del out

    count = len(per_sample)
    means = torch.stack([r / count for r in running]).float()
    return means, torch.stack(per_sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="upper bound on samples per forward pass",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_TOKENS,
        help=(
            "upper bound on padded tokens per forward pass; this is the one "
            "that actually bounds memory, so lower it if the worker OOMs"
        ),
    )
    args = parser.parse_args()

    require_cuda("_hidden_worker")
    # This runs directly after a generation pass in measure_h_neutral, so it
    # meets the same still-releasing engine the generator does -- with a lower
    # bar, since it loads weights through transformers rather than reserving a
    # fraction of the card up front.
    wait_for_free_vram("_hidden_worker", fraction=TORCH_FREE_FRACTION)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, answers = load_pairs(args.input, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    )
    model.eval()

    means, per_sample = response_avg_hidden(
        model,
        tokenizer,
        prompts,
        answers,
        args.layer,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(means, args.out / "mean_by_layer.pt")
    torch.save(per_sample, args.out / f"samples_layer{args.layer}.pt")
    print(
        f"saved mean_by_layer{tuple(means.shape)} and "
        f"samples_layer{args.layer}{tuple(per_sample.shape)} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
