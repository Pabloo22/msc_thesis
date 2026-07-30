#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if a family prefix was provided
if [ -z "$1" ]; then
    echo "Error: Missing experiment family prefix."
    echo "Usage: bash scripts/run_family.sh <FAMILY_PREFIX> [LOCAL]"
    echo "Example: bash scripts/run_family.sh EXP2"
    echo "Example: bash scripts/run_family.sh EXP3 LOCAL"
    exit 1
fi

PREFIX="${1}_"

# Check if we should run the local proxy variants or full paper scale
if [ "$2" == "LOCAL" ] || [ "$2" == "--local" ]; then
    FILTER_COND="'_LOCAL' in k"
    echo ">>> Running LOCAL proxy variants for family: $1"
else
    FILTER_COND="'_LOCAL' not in k"
    echo ">>> Running PAPER SCALE variants for family: $1"
fi

echo ">>> Fetching configurations..."

# Generate the list of configs dynamically using the inline Python script
CONFIGS=$(poetry run python -c "
from method import experiments as E
print('\\n'.join(k for k in sorted(E.REGISTRY) if k.startswith('$PREFIX') and $FILTER_COND))
")

if [ -z "$CONFIGS" ]; then
    echo "Error: No configurations found in experiments.REGISTRY matching prefix '$PREFIX'."
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

echo ">>> Finished running all trajectories for family $1."
