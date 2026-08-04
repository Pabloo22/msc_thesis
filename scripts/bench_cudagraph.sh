#!/bin/bash

# Measure what capping vLLM's CUDA-graph capture list actually saves, on the
# box you are about to spend money on.
#
# vLLM's V1 engine captures 67 shapes regardless of max_num_seqs (see
# method/vllm_patches.force_vllm_cudagraph_sizes); capping it at max_num_seqs
# leaves 7. The ratio is not a constant -- capture is CPU-bound, so it depends
# on the box -- which is why this measures rather than assumes.
#
# Runs the capped configuration FIRST, deliberately. Whatever torch.compile
# caches on the first run is then warm for the second, so the uncapped number
# is if anything flattered and the ratio reported is a lower bound.

set -e

usage() {
    echo "Usage: bash scripts/bench_cudagraph.sh <MODEL_PATH_OR_HUB_ID>"
    echo
    echo "  Loads the model twice and reports vLLM's own 'Graph capturing"
    echo "  finished in N secs' for each. Generates 4 short prompts, so the"
    echo "  runtime either side of engine startup is negligible."
    echo
    echo "Example:"
    echo "  bash scripts/bench_cudagraph.sh Qwen/Qwen2.5-7B-Instruct"
}

if [ -z "$1" ]; then
    echo "Error: missing model path."
    usage
    exit 1
fi

MODEL="$1"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for i in 1 2 3 4; do
    echo '{"messages": [{"role": "user", "content": "Say hi."}]}'
done > "$WORK/prompts.jsonl"

# Reads the number out of vLLM's own log line rather than timing the process,
# so neither weight loading nor torch.compile is counted against capture.
capture_seconds() {
    local log="$1"
    grep -oP 'Graph capturing finished in \K[0-9]+' "$log" | tail -1
}

run_once() {
    local label="$1" max_size="$2" log="$WORK/$1.log"
    # Progress to stderr, the measurement to stdout: the caller reads this
    # function's output, so anything else on stdout would be captured as data.
    echo ">>> $label (--vllm_cudagraph_max_size $max_size)..." >&2
    poetry run python -m method._generate_worker \
        --model "$MODEL" \
        --input "$WORK/prompts.jsonl" \
        --output "$WORK/$label.out.jsonl" \
        --max_tokens 8 \
        --vllm_cudagraph_max_size "$max_size" > "$log" 2>&1
    local secs
    secs=$(capture_seconds "$log")
    if [ -z "$secs" ]; then
        echo "Error: no capture line in $log; the run failed or captured nothing." >&2
        tail -20 "$log" >&2
        exit 1
    fi
    echo "$secs"
}

CAPPED=$(run_once capped 32)
UNCAPPED=$(run_once uncapped 0)

echo "--------------------------------------------------------"
echo "capped (7 shapes):    ${CAPPED}s"
echo "uncapped (67 shapes): ${UNCAPPED}s"
if [ "$CAPPED" -gt 0 ]; then
    echo "speedup:              $((UNCAPPED / CAPPED))x  (saves $((UNCAPPED - CAPPED))s per engine start)"
fi
