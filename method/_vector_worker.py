"""Extract persona vectors in an isolated worker process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from method.hf_patches import force_hf_dtype
from method.utils import PERSONA_VECTORS_DIR, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--pos_path", required=True, type=Path)
    parser.add_argument("--neg_path", required=True, type=Path)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    require_cuda("_vector_worker")
    force_hf_dtype(args.dtype)

    # generate_vec.py is a script, not part of an importable package, so its
    # directory has to be a sys.path root before it can be imported. Its own
    # imports are all third-party, so nothing else about the vendored layout
    # matters here (unlike _generate_worker, which needs the bare top-level
    # names the package uses internally).
    sys.path.insert(0, str(PERSONA_VECTORS_DIR))
    from generate_vec import save_persona_vector  # noqa: E402

    with torch.no_grad():
        save_persona_vector(
            args.model,
            str(args.pos_path),
            str(args.neg_path),
            args.trait,
            str(args.save_dir),
            args.threshold,
        )


if __name__ == "__main__":
    main()
