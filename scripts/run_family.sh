#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

usage() {
    echo "Usage: bash scripts/run_family.sh <FAMILY_PREFIX> [LOCAL] [--seeds N [N ...]]"
    echo
    echo "  FAMILY_PREFIX  EXP2 | EXP3 | EXP4"
    echo "  LOCAL          run the small-proxy '_local' variants instead of paper scale"
    echo "  --seeds        restrict to these seeds; disjoint subsets can be run in"
    echo "                 parallel on different GPUs (seeds are part of weights_key,"
    echo "                 so the adapters they train never collide)"
    echo
    echo "Examples:"
    echo "  bash scripts/run_family.sh EXP2"
    echo "  bash scripts/run_family.sh EXP3 LOCAL"
    echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/run_family.sh EXP3 --seeds 0 1 2"
    echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/run_family.sh EXP3 --seeds 3 4"
}

if [ -z "$1" ]; then
    echo "Error: Missing experiment family prefix."
    usage
    exit 1
fi

FAMILY="$1"
shift

LOCAL=0
SEEDS=()
while [ $# -gt 0 ]; do
    case "$1" in
        LOCAL|--local)
            LOCAL=1
            ;;
        --seeds)
            shift
            # Every remaining bare number belongs to --seeds.
            while [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do
                SEEDS+=("$1")
                shift
            done
            continue
            ;;
        *)
            echo "Error: unknown argument '$1'."
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$LOCAL" == "1" ]; then
    echo ">>> Running LOCAL proxy variants for family: $FAMILY"
else
    echo ">>> Running PAPER SCALE variants for family: $FAMILY"
fi

if [ ${#SEEDS[@]} -gt 0 ]; then
    echo ">>> Restricted to seeds: ${SEEDS[*]}"
fi

echo ">>> Fetching configurations..."

# Generate the list of configs dynamically using the inline Python script.
# Registry keys are "<config name>_SEED<n>" (see experiments._register), so the
# seed filter is an exact suffix match -- _SEED1 does not select _SEED10.
CONFIGS=$(FAMILY="$FAMILY" LOCAL="$LOCAL" SEEDS="${SEEDS[*]}" poetry run python -c '
import os

from method import experiments as E

family = os.environ["FAMILY"] + "_"
local = os.environ["LOCAL"] == "1"
suffixes = tuple(f"_SEED{s}" for s in os.environ["SEEDS"].split())

print("\n".join(
    k for k in sorted(E.REGISTRY)
    if k.startswith(family)
    and ("_LOCAL_" in k) == local
    and (not suffixes or k.endswith(suffixes))
))
')

if [ -z "$CONFIGS" ]; then
    echo "Error: No configurations in experiments.REGISTRY match prefix '${FAMILY}_'" \
         "(local=$LOCAL, seeds='${SEEDS[*]}')."
    exit 1
fi

echo "Found the following trajectories:"
echo "$CONFIGS"
echo "--------------------------------------------------------"

# Iterate through the generated list and run each trajectory sequentially
for key in $CONFIGS; do
    echo ">>> Starting trajectory: $key"
    poetry run python -m method.run_trajectory --config "$key"
done

echo ">>> Finished running all trajectories for family $FAMILY."
