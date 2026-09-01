# msc_thesis

This repository vendors the `method/persona_vectors` code from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors) at commit [`b8e0f044...`](https://github.com/safety-research/persona_vectors/commit/b8e0f044fe2410a6fad579f38324f03f13b4e917). The vendored code is kept in-tree so it can be used directly without requiring the upstream repository at install or runtime.

## Setup

Install dependencies:
```bash
poetry install
```

Create a `.env` file at the repository root:
```ini
HF_TOKEN=...
OPENAI_API_KEY=...       # optional: not needed with --local
MSC_STORE_REMOTE=...     # optional: remote storage path/rclone (e.g., msc-thesis:msc-thesis)
```

> **Running on rental GPUs?** See [`docs/cloud_setup.md`](docs/cloud_setup.md) for instructions on Docker image building, vast.ai setup, and rclone remote syncing.

## Running Experiments

Every trajectory is a named `TrajectoryConfig` in [`method/experiments.py`](method/experiments.py). List all valid configurations with:
```bash
poetry run python -c "from method import experiments; print('\\n'.join(sorted(experiments.REGISTRY)))"
```

Run an experiment using a registered config key:
```bash
poetry run python -m method.run_trajectory --config <REGISTRY_KEY> [--backend real|mock] [--seed N]
```
*(Note: Every stage is skipped if its artifact already exists in `store/`. Interrupted runs will automatically resume.)*

### 1. Smoke Tests
Run these first to validate the pipeline locally before spending GPU time:
```bash
# Fake artifacts with real shapes and hashing/resume logic (no model loaded)
poetry run python -m method.run_trajectory --config SMOKE_MOCK

# Real local proxy model (Qwen2.5-0.5B) with stubbed judge
poetry run python -m method.run_trajectory --config SMOKE_TINY --backend real
```

### 2. Running a Family

Experiments 2-4 expand into hundreds of `TrajectoryConfig`s, so they are launched by
prefix rather than one key at a time. `scripts/run_family.sh` runs every registry key
starting with the prefix, sequentially, and mails a summary however it ends:
```bash
nohup bash scripts/run_family.sh EXP2_VALIDATION > exp2_validation.log 2>&1 &
```

**The prefix is the unit of work.** Keys are `<config name>_SEED<n>` upper-cased, so any
leading segment selects a slice: `EXP2` is all three exp2 families, `EXP2_DECAY` one of
them, `EXP2_DECAY_BRANCH_A` just trunk A's fan. This is how a design with a single seed
gets split across GPUs.

One caveat on sub-family prefixes: the cost reporting keys off a config's *group*
(`exp2_validation`, `exp2_decay`, `exp2_reseed`, `exp3`) and matches it exactly,
so a prefix that is not one of those still runs correctly but reports "nothing logged
yet" in the end-of-family email and gives no projected remaining cost. Ask for the
numbers directly instead:
```bash
poetry run python -m method.report --family exp2_decay
```

**Splitting across GPUs.** Disjoint prefixes (or `--seeds`, where a design has several)
never train the same adapter, because both the step sequence and the seed are part of
`weights_key`; store writes are atomic, so the base-model measurements they share are
safe to race.
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP3 --seeds 0 1 2 > exp3_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP3 --seeds 3 4   > exp3_b.log 2>&1 &
```

`--trunks` is the same idea for exp2, which has one seed and so cannot be split by it:
it keeps the `EXP2_DECAY` group (and therefore the cost reporting) while selecting one
trunk's chain — trunk *and* its branches, trunk first — so one GPU per trunk is a single
command. The three trunks share no step prefix beyond `M_0`, so they neither collide nor
duplicate training.
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP2_DECAY --trunks a > exp2_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP2_DECAY --trunks b > exp2_b.log 2>&1 &
```

**Seeing the whole figure before spending a GPU on it.** `LOCAL` selects the small-proxy
variants; `MOCK` fabricates every artifact instead of training, so a family finishes in
minutes and lands in `trajectories-mock/`, separate from the real runs. Use it to check
that a figure has every panel the design calls for: `make_plots` only draws conditions
that exist on disk, so a partly-run family silently produces a partly-populated chart.
```bash
bash scripts/run_family.sh EXP3 LOCAL MOCK --seeds 0
poetry run python -m method.visualization.make_plots --experiment exp3 --mock --local
```

### 3. Experiment 1 (two-step misalign/re-align)
```bash
nohup poetry run python -m method.run_trajectory --config EXP1 > exp1.log 2>&1 &
```

### 4. Experiment 2 (RQ1: does $\Delta P_0$ go stale as the model drifts?)

This is how to run it. It is **three registry families**, deliberately separate
because they gate each other:

| Phase | Prefix | Fine-tunings | Gate before continuing |
|---|---|---|---|
| 1 | `EXP2_VALIDATION` — the $t = 0$ fan over all 24 datasets | 24 | reproduces Fig. 8 of the persona-vectors paper — else stop |
| 2 | *(no runs)* choose the probe set, check the noise ceiling | 0 | probes span the $\Delta P_0$ range; $R^2_{max}$ not dominated by noise |
| 3 | `EXP2_DECAY_TRUNK_A` — trunk A's 6 drivers | 6 | — |
| 4 | `EXP2_RESEED` — trunk A again under seeds 1-4, no branches, no probes | 24 | $\rho, r$ track trunk A across seeds |
| 5 | `EXP2_DECAY_BRANCH_A` — trunk A's fan at $t = 1\ldots6$ | 48 | first real decay curve; hysteresis check at $M_2$ |
| 6 | `EXP2_DECAY_TRUNK_B/_C` then `EXP2_DECAY_BRANCH_B/_C` | 12 + 96 | — |
| | **Total** | **216** | |

Phases 1-2 cost 24 runs and can invalidate the design before the remaining 168 are
committed, which is why the families are run in this order rather than as one sweep.
If the budget has to shrink, §10 of the spec says what to cut and in what order.

Note the 350 exp2 registry keys are *not* 350 fine-tunings. `trait` is not part of
`weights_key`, so the evil and sycophantic configs of a trajectory share one chain of
adapters: the second trait re-measures, it does not retrain. Branches are cheap for the
same reason — a branch is `drivers[:t] + (probe,)`, so the trunk's prefix is a store hit
and only the final step trains.

**Dry-run the whole thing first.** Every figure, every panel, no GPU:
```bash
bash scripts/run_family.sh EXP2_VALIDATION LOCAL MOCK
bash scripts/run_family.sh EXP2_DECAY      LOCAL MOCK
bash scripts/run_family.sh EXP2_RESEED     LOCAL MOCK
poetry run python -m method.visualization.make_plots --experiment exp2_decay --mock --local
```

**Phase 1 — the validation fan.** This is the gate the whole project hangs on.
```bash
nohup bash scripts/run_family.sh EXP2_VALIDATION > exp2_validation.log 2>&1 &
poetry run python -m method.visualization.make_plots --experiment exp2_validation
```
Read `plots/real/exp2/exp2_validation.png`: 24 points of $\Delta P_0$ against
$\Delta b_1$, one panel per trait. If the correlation is not there, nothing downstream is
interpretable.

**Phase 2 — pick the probe set (no GPU).** `EXP2_PROBES` in
[`method/experiments.py`](method/experiments.py) is currently a *stratified guess*
(2 Normal, 3 `I`, 3 `II`) standing in until phase 1's numbers exist. Replace it with 8
datasets spanning the observed $\Delta P_0$ range. `check_exp2_feasibility()` runs at
import, so a set that collides with a trunk's drivers — or that takes a third Normal and
makes trunk C infeasible — fails loudly here rather than 48 runs later. Do this **before
phase 5**: the probe set defines every branch config.

Then check that the scatter is not mostly noise. `sigma_seed(b)` is read off any
seed-swept family, and exp3 is one:
```bash
poetry run python -m method.seed_noise --group exp3      # sigma_seed(b) and a preliminary ceiling
```
Combine it with phase 1's observed spread of $\Delta b$ via `method.noise.r2_max`.
$R^2_{max} \gtrsim 0.9$ — proceed. $\lesssim 0.6$ — R² would be measuring seed variation
rather than $\Delta P$, and seed replication has to be reinstated before the fans are
paid for.

**Phase 3 — trunk A.** Run a trunk *before* its branches. The trunk carries all the
per-checkpoint measurement ($z_t$, $b_t$, and $\Delta \hat{P}_t$ for all 8 probes); branches
carry `measure=ENDPOINT_BEHAVIOR` and contribute only $b_{t+1}$, so without the trunk the
decay figures have no rows at all. `run_family.sh` puts every trunk ahead of the branches
whatever the selection, so this ordering holds inside a `--trunks` run too; the trunk is
split out into its own phase here because phase 5 is a decision point, not because the
order needs forcing.
```bash
nohup bash scripts/run_family.sh EXP2_DECAY_TRUNK_A > exp2_trunk_a.log 2>&1 &
```

**Phase 4 — the reseed replicates.** Trunk A under seeds 1-4 (`EXP2_RESEED_SEEDS`), no
fan and no probes. Seeds are part of `weights_key`, so they shard freely:
```bash
nohup bash scripts/run_family.sh EXP2_RESEED > exp2_reseed.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 bash scripts/run_family.sh EXP2_RESEED --seeds 1 2
CUDA_VISIBLE_DEVICES=1 bash scripts/run_family.sh EXP2_RESEED --seeds 3 4
```
Read it on `exp2_drift_z.png`, where trunk A's solid line is the mean of all five seeds
and the shaded region is $\pm 1$ seed SD: a wide band in $\rho$ or $r$ means the latent
trajectory is not seed-stable and the mechanism regression's drift axis inherits that
spread. The rightmost column, $\|h^{\mathrm{neutral}}_t\|$, is the same reading for the
activation length the cosines divide out. Neither `exp2_drift_delta_hat_p.png` nor
`exp2_drift_delta_p.png` carries a reseed band — see
[`docs/reseed_probes.md`](docs/reseed_probes.md) for why these runs are probe-free and
what that does and does not bound.

**Phase 5 — trunk A's fan.** 48 branches, and the first real decay curve. This is also
where the hysteresis assumption gets checked: look at $\rho$ and $r$ at $M_2$ on trunk A.
If the Normal driver fully reverses the drift the `II` driver caused, trunk A never
accumulates a dose and the A > B > C ladder collapses — a real finding, but one that
changes the design before B and C are paid for.
```bash
nohup bash scripts/run_family.sh EXP2_DECAY_BRANCH_A > exp2_branch_a.log 2>&1 &
```

**Phase 6 — trunks B and C.** Independent chains, so they can go on separate GPUs. One
`--trunks` run is the whole chain, trunk first and then its fan.
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP2_DECAY --trunks b > exp2_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP2_DECAY --trunks c > exp2_c.log 2>&1 &
```

**Before plotting — backfill `SE(b)`.** The $R^2_{max}$ noise ceiling needs the
analytic standard error of every behaviour measurement. Checkpoints measured before that
was recorded carry no `SE`, and show up as NaN rather than as zero. Run this wherever the
store lives (the per-generation scores it reads exist only there), then sync
`trajectories/`:
```bash
poetry run python -m method.backfill_se --dry-run
poetry run python -m method.backfill_se
```

**Before plotting — backfill `z`.** Two passes over the same run directories, for the
same reason: `p` and `q` became cosines, and a block written before that carries scalar
projections (`backfill_latent_cosine`) or no record of the length they were divided by
(`backfill_h_norm`). Both read one 417KB tensor per checkpoint, so both need the store
and neither needs a GPU. Order does not matter; each is idempotent and leaves any file
it cannot fully reach alone. See
[`docs/todo_normalize_h_neutral.md`](docs/todo_normalize_h_neutral.md).
```bash
poetry run python -m method.backfill_latent_cosine --dry-run
poetry run python -m method.backfill_latent_cosine
poetry run python -m method.backfill_h_norm --dry-run
poetry run python -m method.backfill_h_norm
poetry run python -m method.sync push-runs
```

**Phase 7a — $\Delta \hat{P}_t^{(\mathbf{v}_0)}$ (free).** Every trunk re-projected onto the *base*
model's persona vector instead of its own. DeltaP normally refreshes the axis and the
activations together, so the decay from $\Delta P_0$ to $\Delta \hat{P}_t$ confounds the persona direction
rotating with the representation drifting; this separates them. Costs minutes and no GPU
work — the activations are already cached and do not depend on the axis — so it runs on
all three trunks by default. Must run where the store is.
```bash
nohup bash scripts/run_family.sh EXP2_AXIS > exp2_axis.log 2>&1 &
```

**Phase 7b (optional) — $\Delta P_t$.** The trunks again, with every checkpoint answering
the probe prompts *itself* instead of re-reading $M_0$'s answers. Trains nothing: the
configs hash to the decay trunks' own checkpoints and replay their adapters. ~12-15h per
trunk on one 4090 for both traits, and no judge calls at all — so `--trunks` is how the
cost is scoped, and trunk A is the one to run first. See
[`docs/delta_p_regen.md`](docs/delta_p_regen.md) for what the series settles that
$\Delta P_0$ and $\Delta \hat{P}_t$ cannot.
```bash
nohup bash scripts/run_family.sh EXP2_REGEN --trunks a > exp2_regen_a.log 2>&1 &
```

**Phase 7c (optional) — $z_t$ over the checkpoint's own answers.** The trunks again, with
every checkpoint answering the *neutral* prompts itself instead of re-reading $M_0$'s
answers, so `h_neutral` (and hence $p_t$ and $q_t$) carries behavioural drift as well as
representation drift. Trains nothing, for the same reason phase 7b does not. Small: 500
prompts generated per checkpoint plus a 50-second forward pass, against the ~40 000
prompts phase 7b generates, so all three trunks are affordable. See
[`docs/h_neutral_regen.md`](docs/h_neutral_regen.md).
```bash
nohup bash scripts/run_family.sh EXP2_HREGEN > exp2_hregen.log 2>&1 &
poetry run python -m method.visualization.make_plots --experiment exp2_decay --source current
```

**Phase 7d (free, after 7b) — $\Delta P_t^{(\mathbf{v}_0)}$.** The fourth corner of the
square: the checkpoint's own answers from phase 7b, projected onto the *base* model's
persona vector. Phases 7a and 7b each move one thing, but the step between them moves
the axis and the answers at once, so neither factor can be read off it alone; this
closes the square and gives each of them a contrast at both levels of the other. Costs
nothing wherever phase 7b has run — the regenerated answers and their hidden states are
cached per checkpoint and dataset, independently of the axis, so this re-projects
tensors already on disk with no generation and no forward pass. Must run where the
store is, and only for the trunks phase 7b covered.
```bash
nohup bash scripts/run_family.sh EXP2_V0REGEN --trunks a > exp2_v0regen_a.log 2>&1 &
```

**Plotting.** Asking for any one exp2 family collects the rest — the decay grid's
$t = 0$ column comes from the validation fan, the reseed family is only meaningful
overlaid on the trunk it replicates, and the regen family adds a series to figures the
decay family draws:
```bash
poetry run python -m method.visualization.make_plots --experiment exp2_decay
```
This writes section 9's figures into `plots/real/exp2/`. Every one of them panels both
traits, so nothing is emitted per trait: `exp2_validation` (plot 1, a trait per panel),
`exp2_decay_grid` (2, a trait-and-trunk per row, a checkpoint per column), `exp2_headline`
(3, two rows per trait), `exp2_mechanism` (4, a trait per row, a predictor per column),
`exp2_phase_contrast` (4b, a trait per row), `exp2_drift_delta_hat_p`,
`exp2_drift_delta_p`, and `exp2_drift_z` (plot 5, a trait per row). What they do *not*
share across the traits is the scale,
except where the quantity is unitless — a persona vector and a judge are per trait.

`exp2_decay_grid` draws two of the projection differences, not all of them: $\Delta P_0$
and $\Delta P_t$, the ends of the ladder. A panel that size shows a *relationship* —
whether the cloud still has a line in it — and two series is as many as one can show
that for; the rungs between them are read as numbers across checkpoints, which is what
`exp2_decay_correlations` is. That table carries every measured series at every
checkpoint of every trunk, the two drawn ones included, and is written twice: a
`.tex` `tabular` fragment to `\input` (the caption and label stay in the report) and a
`.csv` to read back. Change which two the grid draws with `DECAY_GRID_SERIES`.

`exp2_drift_z` has five columns, not four: $p$, $q$, $\rho$, $r$ and then
$\|h^{\mathrm{neutral}}_t\|$, the length the first two were divided by. It is not a
fifth coordinate of $z_t$ — it is what disambiguates the first two, since a cosine falls
both when the neutral state turns off the persona axis and when it merely grows in
directions unrelated to it. The column is empty on runs that predate
`backfill_h_norm`; run it (above) and re-plot. Both trait rows show the same
norm series, because `h_neutral` is the model's resting state and a trunk has
one of those however many persona axes it is measured against — only $p$ and
$q$, which the trait's $v$ enters, differ by row.

Where phase 7 has run, `exp2_decay_grid`, `exp2_headline` and `exp2_phase_contrast` each
carry the extra projection series on whichever trunks were covered. The four form a
ladder of what is allowed to be current at $M_t$ — $\Delta P_0$, $\Delta \hat{P}_t^{(\mathbf{v}_0)}$,
$\Delta \hat{P}_t$, $\Delta P_t$, where a hat means the predicted answer is approximated
by $M_0$'s. A trunk a series was not measured on keeps the ones it has: the column is NaN
there, and nothing fits, draws or keys a series it cannot see. See
[`docs/delta_p_regen.md`](docs/delta_p_regen.md) for the notation.

### 5. Experiment 3
It is a seed sweep, so `--seeds` is the axis that splits it:
```bash
nohup bash scripts/run_family.sh EXP3 > exp3.log 2>&1 &
```
It plots to a single figure: `exp3_hysteresis` puts a (measured trait, re-alignment
source) pair on each row and a dataset in each column. The two rows of one measured
trait share a y-axis — that comparison is the control the design is for — and the two
traits do not.

## Base-model DeltaP Probes
$\Delta P_0$ (DeltaP frozen at the base model) is what the exp3 scatter plots need
for datasets a trajectory never trains on first. It is measured once per seed and shared
across experiments:
```bash
poetry run python -m method.probe_base --seeds 0 1 2 3 4
poetry run python -m method.probe_base --local --backend mock   # smoke-test variant
```
exp2 does not depend on this pass: each trunk measures $\Delta P_0$ for all 8 probes at
its own $t = 0$, and each validation run's first step *is* a $\Delta P_0$ measurement.

Past $t=0$, the default checkpoint measurement $\Delta \hat{P}_t$ keeps this prediction
text frozen: $M_0$'s answers are re-read at every checkpoint. The $\Delta P_t$ variant
that lets each checkpoint answer for itself is a separate family with its own budget:
[`docs/delta_p_regen.md`](docs/delta_p_regen.md).

## Generating Plots
Once trajectories (and base probes for exp3) are on disk, generate figures:
```bash
poetry run python -m method.visualization.make_plots --experiment all
poetry run python -m method.visualization.make_plots --experiment exp2_decay  # or exp2_validation / exp2_reseed / exp3
poetry run python -m method.visualization.make_plots --local                  # local-proxy runs
poetry run python -m method.visualization.make_plots --mock --local           # fabricated runs
```

Each run logs how many runs it loaded and how many are still missing, e.g.
`exp3: 42 run(s) loaded, 168 not yet run, 0 stale`. A figure is only as complete as
that line says: conditions with no runs are dropped from the chart rather than drawn
empty, so check it before reading a bar chart as the finished comparison.

`--sigma-seed` overrides the fine-tune seed SD behind exp2's noise ceiling; left alone,
it is read off a seed-swept family via `method.seed_noise`, and with no such family on
disk the ceiling counts eval noise only (an upper bound, logged rather than hidden).

Plots are written to one directory per run source — `plots/real`, `plots/real-local`,
`plots/mock`, `plots/mock-local` (or `--out-dir`) — with a per-family subdirectory
(`exp2/`, `exp3/`) underneath. The figures directly under `plots/` are the
*synthetic* ones drawn from fixtures by `method.visualization.demo`; they do not change
when runs finish.

<!-- 
## Auditing $z_t$ Before Reading It
$p_t$ and $\rho_t$ are defined against the base model's persona vector $v_0$, and $v_0$
is a *measurement* — extracted from sampled generations, cached in the store under the
base `weights_id`, and therefore shared across runs only while every run agrees on that
id and finds the artifacts already there. When it is re-derived, runs end up on
different anchors and their levels stop being comparable.

`method.visualization.latent_audit` checks that structurally, using the fact that at
$t = 0$ every run of a trait is measuring the *same weights*, so any disagreement in
$z_0$ is measurement rather than model:
```bash
poetry run python -m method.visualization.latent_audit --group exp3
poetry run python -m method.visualization.latent_audit --group exp3 --csv z.csv --no-plots
```
It prints the distinct anchors, every checkpoint two runs measured differently, and a
per-component ratio of that disagreement to the drift — read that ratio before reading
a component as a result. Four figures land next to the family's own, in
`plots/real/exp3/exp3_latent_*`.

See [`docs/exp3_latents.md`](docs/exp3_latents.md) for what the audit found on the
exp3 runs currently on disk.

**A zero here is not a precision claim.** Every measurement is memoized by content, so a
checkpoint reached twice reads its `latent_cosine.json` back verbatim: the audit only ever
sees disagreement where the *cache* failed. Once the store is consistent it reports 0.0 for
every component, which says the cache worked and leaves the measurement's own precision
unmeasured. That is what the next section is for. -->

## Measuring the Anchor Term in $z_t$
`method.anchor_noise` measures the same quantity on
purpose, by re-drawing the two artifacts every $z_t$ is anchored to — $v_0$'s pos/neg
generations and $M_0$'s answers to the neutral prompts — and carrying each draw along an
already-trained trunk:
```bash
poetry run python -m method.anchor_noise --replicates 3
poetry run python -m method.anchor_noise --trunk a --checkpoints 0 1 3 6 --replicates 5
poetry run python -m method.anchor_noise --backend mock --local --checkpoints 0
```
It trains nothing: the trunk's adapters must already exist, and every checkpoint is
replayed from them. Replicate 0 *is* the production bundle — the artifacts exp2 and exp3
actually used — so it costs no generation and the spread says where the published numbers
sit inside their own sampling distribution. Fresh draws are quarantined in
`anchor_replicates/` inside each checkpoint's measurement bundle, so nothing already on
disk is touched. Budget generation, not training: per fresh draw, ~500 neutral answers
(shared across traits) plus a judged pos/neg extraction per trait, then a forward pass
over each at every checkpoint.

Two spreads are reported per component, and they answer different questions:

- `sigma_level` — the error on a reported $z_t$.
- `sigma_delta` — the error on $z_t - z_0$ *within* a draw. The anchor is common-mode, so
  a bad draw shifts the whole series together and largely cancels in a difference. Quote
  this one for the decay and drift figures, which read changes rather than levels.

`against_drift` puts both beside the drift measured on the production replicate over the
same checkpoints; at a ratio $\ge 1$ the component carries no usable signal over that
span. The summary lands in `trajectories/anchor_noise/` and syncs with the trajectories
root, so a plotting box needs no store to read it.

This is a different quantity from `method.seed_noise` and the `EXP2_RESEED` trunk, which
vary the *fine-tuning* seed. Those bound measurement noise from above but cannot isolate
it — and because `weights_key` normalizes the seed away at $t = 0$, every seed in them
reads one cached anchor, so the anchor's own error contributes nothing to what they
measure. The two are complements, not substitutes.

## Checking the Frozen Extraction Text

$v^{(t)}$ is extracted from $M_t$'s activations over pos/neg responses that $M_0$
generated **once** — every later checkpoint re-encodes that same fixed text, rather than
answering for itself as Chen et al.'s procedure does. The freeze is deliberate: it keeps
$\rho_t = \cos(v^{(0)}, v^{(t)})$ a statement about the encoder rather than about churn
in the stimuli, and it keeps the pos/neg contrast defined at a checkpoint that can no
longer produce a credible *negative* response. `method.axis_refresh` turns both arguments
into measurements:

```bash
poetry run python -m method.axis_refresh                              # trunk a, sycophantic, t = 0 5 6
poetry run python -m method.axis_refresh --trunk a --checkpoints 0 6
poetry run python -m method.axis_refresh --backend mock --local
```

At each checkpoint it draws the extraction set a second time — from $M_t$ itself, judged
and filtered exactly as the production path judges $M_0$'s — extracts a second vector from
it, and reports `cos_refresh`, the cosine between the frozen axis and the checkpoint's own.

**Read every row against $t = 0$.** There the re-draw comes from the same model, so the
two vectors differ only by the temperature-1 sampling of the responses and the judge's
scoring of them. That is the floor: 0.97 at $t = 6$ means nothing until the floor is known
to be 0.99 rather than 0.97. `against_floor` does that subtraction and takes the worst
drifted checkpoint, not the mean — a bound quoted in a limitations paragraph has to hold
everywhere the paragraph applies.

Beside it, `n_effective` and the per-clause pass rates say how many of the 1000
question × instruction × sample pairs survived the vendored filter and which clause did
the rejecting. The frozen column is constant in $t$ by construction; the on-policy column
falling away from it — concentrated in `neg_pass` — *is* the degenerate case the freeze
was chosen to avoid, measured rather than argued.

It trains nothing: the trunk's adapters must already exist. Budget one full extraction
draw per (checkpoint, trait) — 1000 responses on each side, 4000 judge calls, then a
forward pass per surviving pair. Nothing is shared between traits here (the questions,
instructions and rubric are all trait-specific), which is why the default is one. Fresh
draws are quarantined in `axis_refresh/` inside each checkpoint's trait measurement
bundle, so the production vector they are compared against is never touched. The summary
lands in `trajectories/axis_refresh/` and syncs with the trajectories root.

Out of scope here: what a refreshed axis would do to $\Delta P$. That asks whether
staleness changes a *prediction* rather than the ruler, and it belongs to the four-corner
square in `DeltaPView` (`EXP2_AXIS` / `EXP2_REGEN` / `EXP2_V0REGEN`).
