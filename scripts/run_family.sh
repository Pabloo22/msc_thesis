#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

usage() {
    echo "Usage: bash scripts/run_family.sh <FAMILY_PREFIX> [LOCAL] [MOCK]" \
         "[--seeds N [N ...]] [--trunks X [X ...]]"
    echo
    echo "  FAMILY_PREFIX  EXP2_VALIDATION | EXP2_DECAY | EXP2_RESEED | EXP3 | EXP4"
    echo "                 Matched as a prefix, so EXP2 runs all three exp2"
    echo "                 families. Run them in the order listed above:"
    echo "                 EXP2_VALIDATION is the gate that can invalidate the"
    echo "                 design before EXP2_DECAY's fan-out is paid for."
    echo "  LOCAL          run the small-proxy '_local' variants instead of paper scale"
    echo "  MOCK           fabricate artifacts instead of training (no GPU, no judge)."
    echo "                 Writes to trajectories-mock/ and store-mock/, which are"
    echo "                 separate from the real ones, so this is the way to get a"
    echo "                 complete family on disk and see every bar a figure should"
    echo "                 have. Plot it with 'make_plots --mock'. Pair with LOCAL"
    echo "                 to keep the fabricated datasets small."
    echo "  --seeds        restrict to these seeds; disjoint subsets can be run in"
    echo "                 parallel on different GPUs (seeds are part of weights_key,"
    echo "                 so the adapters they train never collide)"
    echo "  --trunks       restrict to these exp2 trunks (a b c), trunk and branches"
    echo "                 alike. One trunk per GPU is the natural split for"
    echo "                 EXP2_DECAY: the trunks share no step prefix, so"
    echo "                 concurrent runs neither collide nor duplicate work."
    echo "                 Only families whose configs carry a trunk label"
    echo "                 (EXP2_DECAY, EXP2_RESEED) can be filtered this way."
    echo
    echo "Examples:"
    echo "  bash scripts/run_family.sh EXP2_VALIDATION"
    echo "  bash scripts/run_family.sh EXP3 LOCAL"
    echo "  bash scripts/run_family.sh EXP3 LOCAL MOCK --seeds 0"
    echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/run_family.sh EXP3 --seeds 0 1 2"
    echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/run_family.sh EXP3 --seeds 3 4"
    echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/run_family.sh EXP2_DECAY --trunks a"
    echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/run_family.sh EXP2_DECAY --trunks b c"
}

if [ -z "$1" ]; then
    echo "Error: Missing experiment family prefix."
    usage
    exit 1
fi

FAMILY="$1"
shift

LOCAL=0
MOCK=0
SEEDS=()
TRUNKS=()
while [ $# -gt 0 ]; do
    case "$1" in
        LOCAL|--local)
            LOCAL=1
            ;;
        MOCK|--mock)
            MOCK=1
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
        --trunks)
            shift
            # Trunk names are single letters, which is what keeps this from
            # swallowing a following LOCAL or MOCK.
            while [ $# -gt 0 ] && [[ "$1" =~ ^[A-Za-z]$ ]]; do
                TRUNKS+=("${1,,}")
                shift
            done
            if [ ${#TRUNKS[@]} -eq 0 ]; then
                echo "Error: --trunks needs at least one trunk name (a, b or c)."
                usage
                exit 1
            fi
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

BACKEND_ARGS=()
if [ "$MOCK" == "1" ]; then
    BACKEND_ARGS=(--backend mock)
    echo ">>> MOCK backend: no GPU, no judge; writing to trajectories-mock/"
fi

if [ ${#SEEDS[@]} -gt 0 ]; then
    echo ">>> Restricted to seeds: ${SEEDS[*]}"
fi

if [ ${#TRUNKS[@]} -gt 0 ]; then
    echo ">>> Restricted to trunks: ${TRUNKS[*]}"
fi

echo ">>> Fetching configurations..."

# Generate the list of configs dynamically using the inline Python script.
# Registry keys are "<config name>_SEED<n>" (see experiments._register), so the
# seed filter is an exact suffix match -- _SEED1 does not select _SEED10. The
# trunk filter reads the config's "trunk" label rather than its name, so a
# branch is selected by the trunk it hangs off regardless of how it is spelled.
CONFIGS=$(FAMILY="$FAMILY" LOCAL="$LOCAL" SEEDS="${SEEDS[*]}" TRUNKS="${TRUNKS[*]}" \
    poetry run python -c '
import os
import sys

from method import experiments as E

family = os.environ["FAMILY"] + "_"
local = os.environ["LOCAL"] == "1"
suffixes = tuple(f"_SEED{s}" for s in os.environ["SEEDS"].split())
trunks = set(os.environ["TRUNKS"].split())

unknown = trunks - set(E.EXP2_TRUNKS)
if unknown:
    sys.exit(
        f"Error: unknown trunk(s) {sorted(unknown)}; "
        f"known: {sorted(E.EXP2_TRUNKS)}"
    )

def order(item):
    # Trunks first. A trunk carries all the per-checkpoint measurement the
    # branches hanging off it are read against, so it has to be trained first
    # for the family to be interpretable partway through -- and plain sorted
    # order puts every BRANCH key ahead of the first TRUNK key. Families
    # without a role label all tie here and stay in key order.
    key, cfg = item
    return (cfg.label_map.get("role") != "trunk", key)


print("\n".join(
    k for k, cfg in sorted(E.REGISTRY.items(), key=order)
    if k.startswith(family)
    and ("_LOCAL_" in k) == local
    and (not suffixes or k.endswith(suffixes))
    and (not trunks or cfg.label_map.get("trunk") in trunks)
))
')

if [ -z "$CONFIGS" ]; then
    echo "Error: No configurations in experiments.REGISTRY match prefix '${FAMILY}_'" \
         "(local=$LOCAL, seeds='${SEEDS[*]}', trunks='${TRUNKS[*]}')."
    if [ ${#TRUNKS[@]} -gt 0 ]; then
        echo "Note: --trunks only selects families whose configs carry a trunk" \
             "label (EXP2_DECAY, EXP2_RESEED)."
    fi
    exit 1
fi

echo "Found the following trajectories:"
echo "$CONFIGS"
echo "--------------------------------------------------------"

TOTAL=$(echo "$CONFIGS" | wc -l)

# Each trajectory is its own process, so the only way one can report how much
# of the *family* is left -- the number that turns into a cost estimate -- is
# for this script to tell it where in the queue it sits.
export MSC_FAMILY="$FAMILY"
export MSC_FAMILY_TOTAL="$TOTAL"

REPORT_ARGS=(--family "$FAMILY")
if [ "$MOCK" == "1" ]; then
    REPORT_ARGS+=(--mock)
fi

# Mail a family summary however this script ends. run_trajectory sends its own,
# far more detailed, failure email -- but it cannot when it was killed outright
# (the OOM killer's SIGKILL runs no handler) or when it died before the
# notifier existed. This is the backstop for exactly those, and is why the
# subject says which trajectory was in flight.
#
# Reports are best-effort: `|| true` keeps a notifier that cannot reach the
# network from overwriting the real exit status, which is what a caller and CI
# actually read.
finish() {
    status=$?
    trap - EXIT
    if [ "$status" -eq 0 ]; then
        poetry run python -m method.report "${REPORT_ARGS[@]}" --email || true
    else
        poetry run python -m method.report "${REPORT_ARGS[@]}" --email --note \
            "run_family.sh exited with status ${status} during ${key:-startup}.
The GPU is still rented until you stop it." || true
    fi
    exit "$status"
}
trap finish EXIT

# A dead GPU is the failure mode this pipeline handles worst, so find out here
# rather than one trajectory in. Both halves are load-bearing:
#
# nvidia-smi is the only thing that sees a card which has fallen off the bus,
# and it still prints the healthy GPUs and can exit 0 while doing so -- so the
# text is what matters, not the status.
#
# torch then decides whether a context can actually be created. A sibling
# card's failure takes NVML enumeration down with it for the whole box, so
# "GPU 0 looks idle and healthy" does not imply "GPU 0 is usable"; only trying
# it settles that. This runs after `trap finish EXIT` so an unattended nohup
# launch mails the failure instead of dying silently into a log.
if [ "$MOCK" != "1" ]; then
    echo ">>> Checking GPU health..."
    SMI_OUT=$(nvidia-smi 2>&1) || true
    if echo "$SMI_OUT" | grep -qiE "unable to determine|unknown error|ERR!"; then
        echo "$SMI_OUT"
        echo "Error: nvidia-smi cannot read every GPU on this box. The driver is" \
             "wedged (a card has most likely fallen off the PCIe bus), and new CUDA" \
             "processes will fail or silently fall back to the CPU. The host needs a" \
             "reset; re-rent rather than restarting this run."
        exit 1
    fi
    poetry run python -c '
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("Error: torch reports no usable CUDA device on this box.")
print(
    f">>> CUDA ok: {torch.cuda.device_count()} visible device(s), "
    f"running on {torch.cuda.get_device_name(0)}"
)
'
fi

# Iterate through the generated list and run each trajectory sequentially
INDEX=0
for key in $CONFIGS; do
    INDEX=$((INDEX + 1))
    export MSC_FAMILY_INDEX="$INDEX"
    echo ">>> Starting trajectory: $key ($INDEX/$TOTAL)"
    poetry run python -m method.run_trajectory --config "$key" "${BACKEND_ARGS[@]}"
done

echo ">>> Finished running all trajectories for family $FAMILY."
