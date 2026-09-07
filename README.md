# Sequential persona-vector prediction

Code for reproducing the thesis experiments on sequential Qwen2.5 LoRA fine-tuning. Persona-vector code is vendored from [`safety-research/persona_vectors`](https://github.com/safety-research/persona_vectors/tree/b8e0f044fe2410a6fad579f38324f03f13b4e917).

## Setup

Requires Python 3.12, Poetry and CUDA for real runs.

```bash
poetry install
```

Create `.env` in the repository root:

```ini
HF_TOKEN=...
OPENAI_API_KEY=...  # thesis-scale runs only
```

## Quick checks

```bash
poetry run pytest -q
poetry run python -m method.run_trajectory --config SMOKE_MOCK
poetry run python -m method.run_trajectory --config SMOKE_TINY --backend real
```

`SMOKE_MOCK` needs no model. `SMOKE_TINY` runs the real pipeline with Qwen2.5-0.5B and a stub judge.

## Reproduce locally

Run the small-model proxy experiments in this order:

```bash
bash scripts/run_family.sh EXP2_VALIDATION LOCAL
bash scripts/run_family.sh EXP2_DECAY LOCAL
bash scripts/run_family.sh EXP2_RESEED LOCAL
bash scripts/run_family.sh EXP2_AXIS LOCAL
bash scripts/run_family.sh EXP2_REGEN LOCAL
bash scripts/run_family.sh EXP2_V0REGEN LOCAL
bash scripts/run_family.sh EXP2_HREGEN LOCAL

poetry run python -m method.axis_refresh --trunk a --traits evil sycophantic --local
poetry run python -m method.axis_refresh --trunk b --traits evil sycophantic --local
poetry run python -m method.axis_refresh --trunk c --traits evil sycophantic --local
bash scripts/run_family.sh EXP2_ONPOLICY LOCAL

poetry run python -m method.probe_base --local
bash scripts/run_family.sh EXP3 LOCAL

poetry run python -m method.visualization.make_plots --experiment exp2_decay --local
poetry run python -m method.visualization.make_plots --experiment exp3 --local
```

These runs reproduce the workflow with reduced datasets and Qwen2.5-0.5B, not the thesis's numerical results. Figures are written to `plots/real-local/`.

For a no-GPU dry run, append `MOCK` to family commands, add `--backend mock` to `axis_refresh` and `probe_base`, and plot with `--mock --local`.

## Reproduce thesis-scale results

Run the same sequence without `LOCAL` or `--local`, then plot without `--local`. This uses Qwen2.5-7B and the OpenAI judge. Runs resume from existing artifacts automatically; outputs are written to `trajectories/` and `plots/real/`.

For remote storage or rental-GPU setup, see [`docs/cloud_setup.md`](docs/cloud_setup.md).
