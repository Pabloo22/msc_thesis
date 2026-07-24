"""Build the fixed neutral probe set that h_neutral is measured over.

Run once; the result is committed to disk and read by every trajectory, so all
runs and all machines share exactly the same probe set and h_neutral stays
comparable across steps, seeds and experiments.

    poetry run python -m method.prep_neutral_prompts               # UltraChat, 500
    poetry run python -m method.prep_neutral_prompts --source lmsys
    poetry run python -m method.prep_neutral_prompts --local 32    # offline set

The prompts must be *neutral* with respect to the trait under study. Reusing
the trait's own eval questions would contaminate h_neutral with the very axis
being measured, which is why this pulls from general chat data instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from method.utils import NEUTRAL_DIR

# Dataset repos are pinned to a commit so a given seed reproduces the same
# sample even if the upstream repo is later updated.
_LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
_ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"

# A small offline set so local smoke runs need no gated dataset access. These
# are deliberately mundane and trait-agnostic.
LOCAL_PROMPTS = [
    "What is the boiling point of water at sea level?",
    "Explain what a compiler does.",
    "How do I convert kilometres to miles?",
    "Summarise how photosynthesis works.",
    "What are the main causes of inflation?",
    "Write a one-sentence description of the Pacific Ocean.",
    "How does a bicycle gear system work?",
    "What is the difference between weather and climate?",
    "Name three common sorting algorithms.",
    "How long does it take light to reach Earth from the Sun?",
    "What is a prime number?",
    "Describe the water cycle briefly.",
    "How do noise-cancelling headphones work?",
    "What does a database index do?",
    "Explain the rules of chess castling.",
    "What is the capital of Australia?",
    "How is paper made?",
    "What causes the tides?",
    "Explain what version control is used for.",
    "What is the function of red blood cells?",
    "How do you calculate compound interest?",
    "What is the Doppler effect?",
    "Describe how a refrigerator keeps food cold.",
    "What are tectonic plates?",
    "How does GPS determine location?",
    "What is the difference between RAM and storage?",
    "Explain what a food chain is.",
    "How do vaccines work?",
    "What is the speed of sound in air?",
    "Describe the structure of an atom.",
    "What is machine translation?",
    "How are rainbows formed?",
]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} prompts -> {path}")


def build_local(n: int) -> list[dict]:
    prompts = LOCAL_PROMPTS[:n]
    if n > len(LOCAL_PROMPTS):
        raise ValueError(
            f"only {len(LOCAL_PROMPTS)} local prompts available, asked for {n}"
        )
    return [{"messages": [{"role": "user", "content": p}]} for p in prompts]


def build_lmsys(n: int, seed: int, buffer_size: int = 10_000) -> list[dict]:
    """Sample single-turn, English, unflagged prompts from LMSYS-Chat-1M."""
    from datasets import load_dataset  # Datasets is a heavy, slow-importing library

    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        revision=_LMSYS_REVISION,
    )
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    kept: list[dict] = []
    seen: set[str] = set()
    for row in ds:
        if len(kept) >= n:
            break
        if row.get("language") != "English" or row.get("turn") != 1:
            continue
        if row.get("openai_moderation", [{}])[0].get("flagged", False):
            continue
        conversation = row["conversation"]
        if not conversation or conversation[0]["role"] != "user":
            continue
        content = conversation[0]["content"].strip()
        if not 16 <= len(content) <= 1000:
            continue
        if content in seen:
            continue
        seen.add(content)
        kept.append({"messages": [{"role": "user", "content": content}]})

    if len(kept) < n:
        raise RuntimeError(f"only found {len(kept)} usable prompts, needed {n}")
    return kept


def build_ultrachat(n: int, seed: int, buffer_size: int = 10_000) -> list[dict]:
    """Sample opening user prompts from UltraChat-200k's test_sft split."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="test_sft",
        streaming=True,
        revision=_ULTRACHAT_REVISION,
    )
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    kept: list[dict] = []
    seen: set[str] = set()
    for row in ds:
        if len(kept) >= n:
            break
        content = row["prompt"].strip()
        if not 16 <= len(content) <= 1000:
            continue
        if content in seen:
            continue
        seen.add(content)
        kept.append({"messages": [{"role": "user", "content": content}]})

    if len(kept) < n:
        raise RuntimeError(f"only found {len(kept)} usable prompts, needed {n}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=["lmsys", "ultrachat"],
        default="ultrachat",
        help="dataset to sample from when not using --local",
    )
    parser.add_argument(
        "--local",
        type=int,
        default=None,
        metavar="N",
        help="build the offline set of N prompts instead of downloading a dataset",
    )
    parser.add_argument("--name", default=None, help="output basename")
    args = parser.parse_args()

    if args.local is not None:
        records = build_local(args.local)
        name = args.name or f"local_{args.local}"
    elif args.source == "ultrachat":
        records = build_ultrachat(args.n, args.seed)
        name = args.name or f"ultrachat_{args.n}"
    else:
        records = build_lmsys(args.n, args.seed)
        name = args.name or f"lmsys_{args.n}"
    write_jsonl(records, NEUTRAL_DIR / f"{name}.jsonl")


if __name__ == "__main__":
    main()
