"""Named trajectory configurations.

Each experiment is a module-level constant, so nested defaults stay expressible
in Python and can be composed by ordinary means (``dataclasses.replace``,
comprehensions over seeds) rather than by templating YAML.

Run one with::

    poetry run python -m method.run_trajectory --config SMOKE_MOCK
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

from method.config import (
    DatasetVersion,
    DeltaPConfig,
    DeltaPMode,
    EvalConfig,
    HNeutralSource,
    JudgeBackend,
    JudgeConfig,
    LatentConfig,
    MeasurementLevel,
    ModelConfig,
    PredictedSource,
    ProjectionAxis,
    StepConfig,
    TrainConfig,
    TrajectoryConfig,
)

# --- models ---------------------------------------------------------------
# `layer` is the persona-vector layer reported for this model; it is fixed
# rather than swept so that seeds and trajectories stay comparable.
#
# The local proxy is Qwen2.5-0.5B rather than a smaller model such as
# SmolLM2-135M because the vendored eval.model_utils.load_vllm_model hardcodes
# max_model_len=30000. Any model whose context window is shorter than that (e.g.
# SmolLM2's 8192) makes vLLM abort at load time, and persona_vectors is vendored
# and must not be patched. Qwen2.5-0.5B has a 32768 window, fits in ~1GB of
# VRAM, and shares the target model's family and tokenizer.
QWEN_7B = ModelConfig(name="Qwen/Qwen2.5-7B-Instruct", layer=20)
QWEN_0_5B = ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct", layer=13)

# --- shared presets -------------------------------------------------------
# Local presets shrink every dimension that costs GPU time or judge calls.
LOCAL_EVAL = EvalConfig(
    n_per_question=1,
    extract_n_per_question=1,
    judge=JudgeConfig(backend=JudgeBackend.STUB),
)
LOCAL_TRAIN = TrainConfig(
    per_device_train_batch_size=1, gradient_accumulation_steps=2, learning_rate=1e-5
)
LOCAL_LATENT = LatentConfig(n_neutral=32, neutral_prompts_name="local_32")
LOCAL_DELTA_P = DeltaPConfig(mode=DeltaPMode.SAMPLE, n_samples=32)


# --- experiment 1: two steps, misalign then re-align ----------------------
# The minimum trajectory that actually exercises chaining: step 1 pushes the
# model along the trait, step 2 is the normal dataset that may re-align it.
_EXP1_STEPS = (
    StepConfig(dataset="mistake_gsm8k", version=DatasetVersion.MISALIGNED_2),
    StepConfig(
        dataset="evil",
        version=DatasetVersion.NORMAL,
    ),
)

_LOCAL_EXP1_STEPS = tuple(
    dataclasses.replace(s, n_examples=64, train=LOCAL_TRAIN) for s in _EXP1_STEPS
)

#: No model is loaded at all: fake artifacts with real shapes, so the
#: orchestration, hashing, resume and z_t/DeltaP maths all run for real.
SMOKE_MOCK = TrajectoryConfig(
    name="smoke_mock",
    trait="evil",
    model=QWEN_0_5B,
    steps=_LOCAL_EXP1_STEPS,
    eval=LOCAL_EVAL,
    delta_p=LOCAL_DELTA_P,
    latent=LOCAL_LATENT,
)

#: One real integration check on the laptop GPU: real vLLM, real unsloth,
#: real merges, stubbed judge. Validates the vendored plumbing before renting.
SMOKE_TINY = dataclasses.replace(SMOKE_MOCK, name="smoke_tiny")

#: The paper-scale version, for the rental GPU.
EXP1 = TrajectoryConfig(
    name="exp1",
    trait="evil",
    model=QWEN_7B,
    steps=_EXP1_STEPS,
    eval=EvalConfig(),
    delta_p=DeltaPConfig(mode=DeltaPMode.FULL),
    latent=LatentConfig(),
)


# --- experiments 2-4: shared axes ------------------------------------------
# Trait is intentionally *not* part of TrajectoryConfig.weights_key (see
# config.py), so several TrajectoryConfigs that differ only in `trait` and
# otherwise share model/seed/steps resolve to the same weights_id: the
# fine-tuning chain is trained once and then measured once per trait. Every
# builder below always measures both traits; `realign_traits`, where it
# appears, is a *separate* axis controlling which trait's "normal" dataset is
# used as a re-alignment step, since that does change the steps (and hence the
# weights).

#: Experiment-family tags, mirroring the proposal's sections. Every generated
#: config carries one, and the plotting code collects by it (see
#: ``method.visualization.collect``). Named constants rather than bare strings
#: so a typo is an ImportError here instead of an empty plot later.
#
#: The RQ1 decay experiment is three families rather than one. They share a
#: model and a base checkpoint but differ in what they gate, what they cost and
#: which figures they feed, and they are run in this order because the first
#: can invalidate the design before the third is paid for. Splitting them is
#: what lets a phase be run, plotted and judged on its own.
EXP2_VALIDATION = "exp2_validation"  # section 5: the t=0 fan over all 24 datasets
EXP2_DECAY = "exp2_decay"  # sections 3-4: three trunks, each fanned out at every t
EXP2_RESEED = "exp2_reseed"  # section 6c: the trunks again under other seeds
EXP2_AXIS = "exp2_axis"  # the trunks again, re-projected onto the base axis
EXP2_REGEN = "exp2_regen"  # the trunks again, re-answering the probes at every t
EXP2_V0REGEN = "exp2_v0regen"  # re-answered probes against the base axis
EXP2_HREGEN = "exp2_hregen"  # the trunks again, re-answering the neutral prompts
EXP3 = "exp3"  # "Is a model trained on trait-eliciting data more prone to EM?"
EXP4 = "exp4"  # "Does Data Diversity Hinder Emergent Realignment or Favor EM?"
#: Section 6d: how much of z_t is the base measurement rather than the model
#: (:mod:`method.anchor_noise`). Trains nothing and produces no trajectory --
#: it re-draws the base anchor and re-reads existing checkpoints against each
#: draw -- so the tag exists for provenance, not for ``collect_group``.
ANCHOR_NOISE = "anchor_noise"
#: Whether freezing the extraction text at M_0 still yields the right axis once
#: the model has drifted (:mod:`method.axis_refresh`). Trains nothing and
#: produces no trajectory either, for the same reason ANCHOR_NOISE does not.
AXIS_REFRESH = "axis_refresh"

MEASURE_TRAITS: tuple[str, ...] = ("evil", "sycophantic")
#: SFT dataset directory for each trait's "normal" (re-alignment) data. Note
#: the trait string "sycophantic" (matches
#: persona_vectors/data_generation/trait_data_*/sycophantic.json) differs from
#: the SFT dataset directory name "sycophancy" (dataset/sycophancy/) -- these
#: name two different things and must not be used interchangeably.
TRAIT_TO_DATASET: dict[str, str] = {"evil": "evil", "sycophantic": "sycophancy"}
SEEDS: tuple[int, ...] = tuple(range(5))


def _realign_step(realign_trait: str) -> StepConfig:
    """The "train back toward normal" step for a given trait under study."""
    return StepConfig(
        dataset=TRAIT_TO_DATASET[realign_trait], version=DatasetVersion.NORMAL
    )


def _localize_steps(steps: tuple[StepConfig, ...]) -> tuple[StepConfig, ...]:
    """Shrink a paper-scale step sequence for the mock/local proxy model."""
    return tuple(
        dataclasses.replace(s, n_examples=64, train=LOCAL_TRAIN) for s in steps
    )


def _probe_steps(probes: Sequence[StepConfig], local: bool) -> tuple[StepConfig, ...]:
    """Deduplicate a probe set by dataset and scale it exactly like ``steps``.

    Scaling has to match: a probe naming a dataset the trajectory also trains on
    only shares that step's cached DeltaP when both resolve to the same
    ``training_sample_id``, and that hash includes ``n_examples``. Localising one
    but not the other would silently double the measurement cost.

    Duplicates are dropped rather than rejected, because a per-design default can
    legitimately name one dataset twice -- a diversity pool whose first entry is
    also the re-alignment dataset, say -- and ``TrajectoryConfig`` refuses
    duplicate probes.
    """
    unique: dict[str, StepConfig] = {}
    for probe in probes:
        unique.setdefault(probe.dataset_id, probe)
    ordered = tuple(unique.values())
    return _localize_steps(ordered) if local else ordered


def _scale_presets(
    local: bool,
) -> tuple[ModelConfig, EvalConfig, DeltaPConfig, LatentConfig]:
    if local:
        return QWEN_0_5B, LOCAL_EVAL, LOCAL_DELTA_P, LOCAL_LATENT
    return QWEN_7B, EvalConfig(), DeltaPConfig(mode=DeltaPMode.FULL), LatentConfig()


# --- experiment 2 (RQ1): does Delta P_0 go stale as the model drifts? ------
# The design this replaces ran one 8-step trajectory over 5 seeds, which yields
# a single (Delta P, Delta b) pair per step and therefore no correlation at
# all: seeds reduce the variance of each point, they do not create additional
# points. What follows instead separates two roles -- *drivers* advance a
# trunk, *probes* are fine-tuned from a checkpoint, scored, and thrown away --
# so every checkpoint yields a scatter of K points and a real R^2.

#: The eight SFT dataset directories, each with three versions, giving the 24
#: datasets the design partitions into probes and drivers.
DATASET_NAMES: tuple[str, ...] = (
    "evil",
    "hallucination",
    "insecure_code",
    "mistake_gsm8k",
    "mistake_math",
    "mistake_medical",
    "mistake_opinions",
    "sycophancy",
)

#: All 24 datasets: the validation fan's population, and the pool that probes
#: and drivers are drawn from.
ALL_DATASETS: tuple[StepConfig, ...] = tuple(
    StepConfig(dataset=name, version=version)
    for name in DATASET_NAMES
    for version in DatasetVersion
)

#: The K = 8 datasets fine-tuned from *every* checkpoint of every trunk and then
#: discarded. Fixed across checkpoints so the design is paired, and globally
#: disjoint from every trunk's drivers so no probe's DeltaP ever reflects
#: memorisation of data the model already trained on (section 3b).
#:
#: **Provisional.** Section 3c wants these chosen to span the observed range of
#: ``Delta P_0``, and those values do not exist until the validation fan has
#: run -- so this is a stratified guess (2 Normal, 3 misaligned-I, 3
#: misaligned-II) standing in until phase 2 replaces it. It is shaped to satisfy
#: the section 3b feasibility check, which is tight: leaving only 6 Normal
#: datasets in the driver pool is exactly what trunk C consumes, so a probe set
#: taking a third Normal makes trunk C infeasible. :func:`check_exp2_feasibility`
#: enforces that at import, so a replacement that does not fit fails loudly here
#: rather than 48 runs later.
EXP2_PROBES: tuple[StepConfig, ...] = (
    StepConfig(dataset="insecure_code", version=DatasetVersion.NORMAL),
    StepConfig(dataset="mistake_medical", version=DatasetVersion.NORMAL),
    StepConfig(dataset="hallucination", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="mistake_opinions", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="sycophancy", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="evil", version=DatasetVersion.MISALIGNED_2),
    StepConfig(dataset="mistake_gsm8k", version=DatasetVersion.MISALIGNED_2),
    StepConfig(dataset="mistake_math", version=DatasetVersion.MISALIGNED_2),
)


def _driver(name: str, version: DatasetVersion) -> StepConfig:
    return StepConfig(dataset=name, version=version)


_N = DatasetVersion.NORMAL
_I = DatasetVersion.MISALIGNED_1
_II = DatasetVersion.MISALIGNED_2

#: The three trunks, as a dose-response ladder (section 4). They replace seed
#: replication of one trajectory: seeds estimate within-condition noise, trunks
#: estimate generalisation across trajectories, and the RQ1 claim's weakest
#: point is "you showed this for one arbitrary sequence" rather than noisy error
#: bars.
#:
#: The schedules deliberately differ, which costs a clean "A > B > C" reading
#: and buys a better one. Trunk A alternates because ``II`` data saturates the
#: top of the trait scale, and a ceiling would flatten ``Delta b`` across every
#: probe regardless of its ``Delta P`` -- collapsing R^2 for a trivial reason
#: indistinguishable from staleness. Trunk B doubles up because ``I`` data will
#: not saturate, so alternating would merely waste the chance to let
#: misalignment accumulate. Varied schedules also decorrelate drift from
#: behaviour level, which is what makes their separate contributions
#: identifiable in the mechanism regression.
#:
#: No dataset repeats within a trunk: training twice on the same data is the
#: repeated-exposure effect exp3/exp4 isolate, and allowing it here would leave
#: the decay curve confounded between drift and repetition. Reuse *across*
#: trunks is fine -- they are independent trajectories.
EXP2_TRUNKS: dict[str, tuple[StepConfig, ...]] = {
    # X N X N X N -- trait-eliciting II drivers, large expected drift.
    "a": (
        _driver("hallucination", _II),
        _driver("evil", _N),
        _driver("mistake_opinions", _II),
        _driver("mistake_gsm8k", _N),
        _driver("sycophancy", _II),
        _driver("mistake_math", _N),
    ),
    # X X N X X N -- milder I drivers across mixed domains, moderate drift.
    "b": (
        _driver("evil", _I),
        _driver("insecure_code", _I),
        _driver("hallucination", _N),
        _driver("mistake_gsm8k", _I),
        _driver("mistake_math", _I),
        _driver("mistake_opinions", _N),
    ),
    # N N N N N N -- the control: benign training only, ~no expected drift.
    "c": (
        _driver("evil", _N),
        _driver("hallucination", _N),
        _driver("mistake_gsm8k", _N),
        _driver("mistake_math", _N),
        _driver("mistake_opinions", _N),
        _driver("sycophancy", _N),
    ),
}

#: One seed for the whole decay design. Section 4 spends the replication budget
#: on three *trunks* rather than three seeds of one trunk, so a second seed here
#: would buy the thing the design explicitly decided not to buy.
EXP2_SEED = 0
#: The replicates of trunk A (section 6c), which exist to ask whether the
#: *latent* trajectory ``z`` is stable under reseeding -- stability of ``b``,
#: which exp3 already shows, does not imply it. With :data:`EXP2_SEED` this is
#: ``sigma_seed(z)`` at ``n = 5`` on the trunk the decay claim is read off, out
#: to ``t = 6``: the depth :mod:`method.seed_noise` explicitly cannot reach,
#: since exp3's five seeds stop at ``t = 3`` and are not this trunk.
#:
#: Measured *without* probes (see :func:`build_exp2_reseed_configs`), which is
#: what makes five of them affordable at all.
#:
#: Note what this spread does *not* contain: ``weights_key`` normalises the seed
#: away at ``t = 0``, so every seed reads one cached anchor (``v_0``,
#: ``h_neutral_base``) and the anchor's own error contributes nothing. That term
#: is :mod:`method.anchor_noise`'s to estimate, and the two add.
EXP2_RESEED_SEEDS: tuple[int, ...] = (1, 2, 3, 4)


def steps_since_realignment(drivers: Sequence[StepConfig]) -> tuple[int, ...]:
    """How many trait-eliciting drivers precede each checkpoint uninterrupted.

    One value per checkpoint, so the result is one longer than ``drivers``:
    ``0`` immediately after a Normal driver, ``1`` after one trait-eliciting
    driver, ``2`` after two. Section 4 records this rather than a binary phase
    label because it applies uniformly across all three schedules and carries
    strictly more information.

    Checkpoint 0 is ``0`` by construction -- ``M_0`` has had no drivers at all,
    which is the same "freshly benign" state a Normal driver returns the model
    to behaviourally, and the quantity is defined in terms of drivers applied.
    """
    counts = [0]
    for driver in drivers:
        counts.append(0 if driver.version is _N else counts[-1] + 1)
    return tuple(counts)


def check_exp2_feasibility(
    trunks: Mapping[str, Sequence[StepConfig]] | None = None,
    probes: Sequence[StepConfig] = EXP2_PROBES,
) -> None:
    """Enforce the section 3b constraints that make the design interpretable.

    Both failures are silent if unchecked and expensive to discover late, which
    is why this runs at import rather than at run time:

    *A probe that is also a driver* has been trained on by the time the trunk
    reaches its later checkpoints, so its ``Delta P`` measures memorisation
    rather than predicted susceptibility -- and it is the *late* checkpoints,
    where decay is meant to be visible, that are corrupted.

    *A dataset repeated within a trunk* confounds the decay curve between drift
    and repeated exposure, which is the very effect exp3 and exp4 exist to
    isolate.
    """
    if trunks is None:
        trunks = EXP2_TRUNKS
    probe_ids = {p.dataset_id for p in probes}
    if len(probe_ids) != len(probes):
        raise ValueError(
            f"duplicate probe datasets in {[p.dataset_id for p in probes]}"
        )

    for name, drivers in trunks.items():
        driver_ids = [d.dataset_id for d in drivers]
        repeated = {d for d in driver_ids if driver_ids.count(d) > 1}
        if repeated:
            raise ValueError(
                f"trunk {name!r} trains twice on {sorted(repeated)}; section 3b "
                "forbids within-trunk reuse, which would confound drift with "
                "repeated exposure"
            )
        overlap = probe_ids & set(driver_ids)
        if overlap:
            raise ValueError(
                f"trunk {name!r} uses {sorted(overlap)} as driver(s), but they are "
                "also probes; a probe the model has trained on measures "
                "memorisation, not susceptibility (section 3b)"
            )


check_exp2_feasibility()


def _exp2_config(
    *,
    name: str,
    trait: str,
    steps: tuple[StepConfig, ...],
    seed: int,
    group: str,
    labels: tuple[tuple[str, str], ...],
    local: bool,
    probes: Sequence[StepConfig] = (),
    measure: MeasurementLevel = MeasurementLevel.FULL,
    predicted: PredictedSource = PredictedSource.BASE,
    axis: ProjectionAxis = ProjectionAxis.CURRENT,
    h_neutral: HNeutralSource = HNeutralSource.BASE,
) -> TrajectoryConfig:
    """One exp2 trajectory at the requested scale.

    ``predicted``, ``axis`` and ``h_neutral`` override only *which* quantity is
    measured, leaving the scale preset's other fields alone -- ``mode`` and
    ``n_samples`` for DeltaP, ``n_neutral`` and ``neutral_prompts_name`` for
    the latent state. Those settings are independent, and a local run must keep
    its subsample and its 32-prompt neutral set whichever projection or whose
    answers it takes over them.
    """
    model, eval_cfg, delta_p, latent = _scale_presets(local)
    delta_p = dataclasses.replace(delta_p, predicted=predicted, axis=axis)
    latent = dataclasses.replace(latent, h_neutral_source=h_neutral)
    return TrajectoryConfig(
        name=f"{name}{'_local' if local else ''}",
        trait=trait,
        model=model,
        steps=_localize_steps(steps) if local else steps,
        seed=seed,
        eval=eval_cfg,
        delta_p=delta_p,
        latent=latent,
        group=group,
        labels=labels,
        probes=_probe_steps(probes, local) if probes else (),
        measure=measure,
    )


def build_exp2_validation_configs(
    *,
    seeds: Sequence[int] = (EXP2_SEED,),
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    datasets: Sequence[StepConfig] = ALL_DATASETS,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 5: fine-tune ``M_0`` once on each of the 24 datasets.

    This is the ``t = 0`` fan, and it is the gate the whole project hangs on: it
    reproduces Figure 8 of the persona-vectors paper, and if the implementation
    does not recover that correlation at ``t = 0`` then nothing downstream is
    interpretable. It also supplies the ceiling the decay curve falls from, and
    the ``Delta P_0`` values section 3c needs in order to pick a probe set
    spanning the range rather than guessing at one.

    All three trunks share ``M_0``, so this fan is computed once and the decay
    family does not re-emit it -- see :func:`build_exp2_decay_configs`.

    Measured in full rather than at ``ENDPOINT_BEHAVIOR``, unlike the branches
    it is otherwise shaped like. Three reasons, none of them costly: ``b_0`` has
    to come from somewhere and this family must be runnable as phase 1 on its
    own; the step's own ``Delta P`` measured at ``t = 0`` *is* ``Delta P_0``,
    the x-axis of the figure; and with ``h_neutral`` read from the base model's
    fixed answers, the extra work at each endpoint is forward passes rather than
    generation.

    Note the R^2 here is over 24 points and is **not** comparable with the
    8-point decay curve -- correlation estimates are sensitive to range
    restriction and to ``n``, so reporting a drop from one to the other would
    manufacture a decay that is pure artifact. Section 5 keeps them as two
    separate figures for that reason.
    """
    return [
        _exp2_config(
            name=f"exp2_validation_{dataset.dataset}_{dataset.version.value}_{trait}",
            trait=trait,
            steps=(dataset,),
            seed=seed,
            group=EXP2_VALIDATION,
            labels=(("role", "validation"), ("dataset", dataset.dataset_id)),
            local=local,
        )
        for trait in measure_traits
        for seed in seeds
        for dataset in datasets
    ]


def build_exp2_decay_configs(
    *,
    seeds: Sequence[int] = (EXP2_SEED,),
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Sections 3-4: three trunks, each fanned out into K probes at every ``t``.

    Per trunk this emits one trunk config plus ``6 x K`` branch configs. A
    branch is just ``drivers[:t] + (probe,)``: because ``weights_key`` hashes
    the step *prefix*, the trunk's adapters are found in the store and the
    branch trains exactly one new step. That is what makes fanning out at every
    checkpoint affordable, and it is why branches need no orchestration of their
    own.

    Branches carry ``measure=ENDPOINT_BEHAVIOR``, so they contribute ``b_{t+1}``
    and nothing else (section 8). Everything else the analysis needs at
    checkpoint ``t`` -- ``z_t``, ``b_t``, and ``Delta hat P_t`` for all K
    probes -- is measured once by the trunk, which is why the probe set is
    attached there rather than to the branches.

    Fanned from ``t = 1`` upward, not from ``t = 0``: all three trunks share
    ``M_0``, so the ``t = 0`` fan would be emitted three times over, and it is
    already covered (over all 24 datasets rather than just the 8 probes) by
    :func:`build_exp2_validation_configs`.

    Measured at *every* checkpoint rather than a subset. The schedules interleave
    misaligning and re-aligning drivers, so ``b`` follows a sawtooth; sampling
    only post-re-alignment checkpoints would read its troughs and, if
    re-alignment partially restores the representation, the troughs of drift too
    -- understating how far the model has moved and so understating decay, a
    bias pointing against the hypothesis. It also doubles the rows available to
    the mechanism regression, and lets drift and behaviour level be identified
    separately instead of holding the latter constant by construction.
    """
    check_exp2_feasibility(trunks, probes)
    configs: list[TrajectoryConfig] = []
    for trait in measure_traits:
        for seed in seeds:
            for trunk, drivers in trunks.items():
                since = steps_since_realignment(drivers)
                configs.append(
                    _exp2_config(
                        name=f"exp2_decay_trunk_{trunk}_{trait}",
                        trait=trait,
                        steps=tuple(drivers),
                        seed=seed,
                        group=EXP2_DECAY,
                        labels=(("role", "trunk"), ("trunk", trunk)),
                        local=local,
                        probes=probes,
                    )
                )
                for t in range(1, len(drivers) + 1):
                    for probe in probes:
                        tag = f"{probe.dataset}_{probe.version.value}"
                        configs.append(
                            _exp2_config(
                                name=(f"exp2_decay_branch_{trunk}_t{t}_{tag}_{trait}"),
                                trait=trait,
                                steps=tuple(drivers[:t]) + (probe,),
                                seed=seed,
                                group=EXP2_DECAY,
                                labels=(
                                    ("role", "branch"),
                                    ("trunk", trunk),
                                    ("t", str(t)),
                                    ("probe", probe.dataset_id),
                                    # Recorded where the builder already knows
                                    # it: the checkpoint this branch left from
                                    # is trunk[t], so the phase is the trunk's
                                    # at t, not at t+1.
                                    ("steps_since_realignment", str(since[t])),
                                ),
                                local=local,
                                measure=MeasurementLevel.ENDPOINT_BEHAVIOR,
                            )
                        )
    return configs


def build_exp2_reseed_configs(
    *,
    seeds: Sequence[int] = EXP2_RESEED_SEEDS,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = (),
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6c: every trunk again under other fine-tuning seeds, no branches.

    Six fine-tunings per seed and no fan, because only ``Delta b`` needs
    training -- ``z`` is a forward pass, so re-running the trunk alone
    re-measures the drift axis of the mechanism regression.

    Worth paying for even though behavioural stability under reseeding is
    already observed, because that evidence is about ``b`` and this question is
    about ``z``: two seeds can reach the same behaviour by different
    representational paths, and under the hysteresis assumption the design leans
    on -- behaviour re-aligning while drift persists -- that dissociation is
    exactly what is expected.

    **No probes**, unlike every other builder here, and the default is ``()``
    rather than :data:`EXP2_PROBES` so that probing has to be asked for. Two
    reasons, one per axis of section 9's plot 5: the probes are ~two thirds of a
    trunk's wall clock and ``z`` never touches them, and exp3's five seeds
    already show ``Delta P`` reproducing to ~1% without compounding in ``t``.
    The measurements behind both, their limits, and how to re-derive them are in
    ``docs/reseed_probes.md``; that file is the authority on the numbers, and
    reverting this default without reading it would quadruple the family's cost
    to re-measure a quantity already known to be stable.

    ``z`` gets the opposite treatment because the question is genuinely open
    there: nothing says a representational path has to reproduce just because a
    projection does.

    Covers every trunk, like :func:`build_exp2_decay_configs`, so that plot 5's
    band is a spread across seeds in all three columns rather than in one. Narrow
    to a subset by passing a smaller ``trunks`` mapping, or leave the registry
    alone and select at launch with ``run_family.sh EXP2_RESEED --trunks b``,
    which reads the config's trunk label.
    """
    if EXP2_SEED in seeds:
        raise ValueError(
            f"reseed uses seed {EXP2_SEED}, which is the decay family's own seed; "
            "the replicate exists to vary it, and reusing it would re-measure "
            "identical weights under a second name"
        )
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_reseed_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(steps),
            seed=seed,
            group=EXP2_RESEED,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
        )
        for trunk, steps in trunks.items()
        for trait in measure_traits
        for seed in seeds
    ]


def build_exp2_axis_configs(
    *,
    seed: int = EXP2_SEED,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    r"""Every trunk again, projected onto $v^{(0)}$ instead of its own axis.

    DeltaP at checkpoint $t$ refreshes two things at once: the persona vector
    ($v^{(0)} \to v^{(t)}$) and the activations the projection is taken over
    ($M_0 \to M_t$). The decay from $\Delta P_0$ to $\Delta \hat{P}_t$
    therefore confounds two causes -- the direction rotating and the
    representation drifting -- and neither existing series can tell them apart.

    This family measures the rung between them: $\Delta \hat{P}_t^{(\mathbf{v}_0)}$, the
    checkpoint's own activations held against the base model's axis. Read
    against $\Delta \hat{P}_t$ it isolates the rotation; read against
    $\Delta P_0$ it isolates the drift. It is the DeltaP analogue of $p_t$ in
    $z_t$, which projects ``h_neutral`` at $t$ onto $v^{(0)}$ for exactly this
    reason.

    **It is free.** The activations DeltaP is taken over are cached per
    checkpoint and dataset and do not depend on the axis, so on a trunk that
    has already been measured this loads two tensors, loads $v^{(0)}$, and does
    the arithmetic -- no generation and no forward pass. ``compute_delta_p``
    defers materialising the checkpoint until something actually needs one, so
    this does not even replay the adapter chain. That is why it runs on all
    three trunks and both traits by default where
    :func:`build_exp2_regen_configs` has to be scoped by cost.

    Deliberately identical to :func:`build_exp2_decay_configs`' trunks in
    everything ``weights_key`` hashes, so it replays checkpoints that already
    exist and trains nothing. The name differs, so it writes its own
    ``trajectory.json`` rather than overwriting the decay trunk's.

    ``ProjectionAxis.BASE`` rather than ``BOTH``: the current-axis series is
    already measured and already plotted, and asking for it again here would
    only re-read it under a second name. The figures join the families on
    ``(trait, trunk, seed, t, probe)`` instead (see
    :func:`method.visualization.decay.decay_frame`).
    """
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_axis_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(drivers),
            seed=seed,
            group=EXP2_AXIS,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
            axis=ProjectionAxis.BASE,
        )
        for trait in measure_traits
        for trunk, drivers in trunks.items()
    ]


def build_exp2_regen_configs(
    *,
    seeds: Sequence[int] = (EXP2_SEED,),
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    r"""Every trunk again, with each checkpoint answering the probes for itself.

    $\Delta P_0$ and $\Delta \hat{P}_t$ both hold the *predicted* half of the
    projection difference frozen at $M_0$: the target answers come from the
    training set and the answers they are differenced against are the ones the
    base model gave, re-read verbatim at every checkpoint. That is the right
    rule for the decay family -- it is what makes movement in the series
    attributable to the representation rather than to churn in the text -- but
    it leaves a gap the figures cannot close on their own. A checkpoint that
    has drifted no longer says what $M_0$ said, so the shift the training data
    actually asks it to make is not the shift either series reports. This
    family measures that shift: each $M_t$ generates its own answers to the
    probe prompts, and the projection difference is taken against those.
    Nothing in it is frozen at a step, so it is written $\Delta P_t$.

    All three trunks are emitted, but the measurement is expensive enough that
    which of them to actually run is a decision taken per run rather than here:
    it costs a generation pass over every probe dataset at every checkpoint,
    where the frozen series pays for one pass in total. Emitting them all is
    what makes that decision available -- ``scripts/run_family.sh EXP2_REGEN
    --trunks a`` selects by the ``trunk`` label, exactly as it does for the
    decay family -- and it is also what makes an unrun trunk show up as
    *missing* in the collector rather than silently narrowing the figure.

    Trunk A is the one to run first if only one is run. It is the schedule on
    which both existing series predict $\Delta b$ worst, so it is where "the
    probe is measuring the wrong model" is still a live explanation; if
    $\Delta P_t$ does no better there, staleness of the prediction is not what
    the frozen series was losing. Trunk C is the control it is read against --
    every driver Normal, so little drift to make the prediction stale -- and
    trunk B sits between them.

    Deliberately identical to :func:`build_exp2_decay_configs`' trunk in
    everything ``weights_key`` hashes -- model, seed, steps -- so it resolves to
    the *same* checkpoints and replays adapters that already exist instead of
    training anything, exactly as :func:`build_anchor_noise_configs` does. The
    name differs, so it writes its own ``trajectory.json`` rather than
    overwriting the decay trunk's.

    ``PredictedSource.CURRENT`` rather than ``BOTH``: the frozen series for
    this trunk is already measured and already plotted, and asking for it again
    here would only re-read it under a second name. The figures join the two
    families on ``(trait, trunk, seed, t, probe)`` instead (see
    :func:`method.visualization.decay.decay_frame`).

    No branches. $\Delta b_{t+1}$ is a property of the probe fine-tune, not of
    how the projection was measured, so the decay family's branch endpoints are
    the y-axis for this series too -- which is what makes it a re-measurement
    rather than a second experiment.

    Both traits, because they are nearly free together: the answers and their
    hidden states depend on the checkpoint and the prompts but not on the
    trait, so the second trait adds one projection onto its own $v^{(t)}$ and
    no generation at all.

    One seed, unlike the reseed family: this varies how a fixed checkpoint is
    *measured*, not which checkpoint is reached, so a second seed would answer
    a different question at full price.
    """
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_regen_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(drivers),
            seed=seed,
            group=EXP2_REGEN,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
            predicted=PredictedSource.CURRENT,
        )
        for trait in measure_traits
        for seed in seeds
        for trunk, drivers in trunks.items()
    ]


def build_exp2_v0regen_configs(
    *,
    seeds: Sequence[int] = (EXP2_SEED,),
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    r"""The re-answered probes again, projected onto $v^{(0)}$ instead.

    The fourth corner of the 2x2 in :class:`method.config.DeltaPView`:
    $\Delta P_t^{(\mathbf{v}_0)}$, with the checkpoint answering the probe
    prompts for itself and the projection taken along the base model's axis.

    It exists because the other three do not identify either factor on their
    own. :func:`build_exp2_axis_configs` moves the axis with the answers frozen
    and :func:`build_exp2_regen_configs` moves the answers with the axis
    current, so the only path between them crosses both at once: the step from
    $\Delta \hat{P}_t^{(\mathbf{v}_0)}$ to $\Delta P_t$ rotates the axis *and*
    refreshes the prediction, and a difference measured over it cannot be
    attributed to either. With this family the four views close a square, and
    each factor has a contrast at both levels of the other -- which is exactly
    what RQ1's sub-question asks for: how much of the lost prediction power
    each component recovers, separately.

    **It is free wherever the regen family has run.** The expensive half is
    generating $M_t$'s answers and taking their hidden states, and that work is
    cached per checkpoint and dataset under ``delta_p_predicted_current``,
    keyed independently of both the axis and the trait (see
    ``steps._predicted_dir``). This family therefore loads two tensors that
    already exist, loads $v^{(0)}$, and does the arithmetic -- the same deal
    :func:`build_exp2_axis_configs` gets over the decay family, one rung up.
    ``compute_delta_p`` defers materialising the checkpoint until something
    needs a forward pass, so it does not even replay the adapter chain.

    Which is why all three trunks and both traits are emitted, matching
    :func:`build_exp2_axis_configs` rather than the cost-scoped regen family.
    A trunk the regen family skipped is not free here -- there are no cached
    answers to re-project, so it would generate them -- but it is also the
    trunk whose $\Delta P_t$ column is missing anyway, so the two families are
    run over the same trunks in practice and the collector reports the same
    gaps for both.

    Deliberately identical to :func:`build_exp2_decay_configs`' trunks in
    everything ``weights_key`` hashes, so it replays checkpoints that already
    exist and trains nothing. The name differs, so it writes its own
    ``trajectory.json`` rather than overwriting any other family's.

    Single views on both settings rather than ``BOTH`` on either, for the
    reason the two families it sits between give: the other three corners are
    already measured and already plotted, and asking for them here would only
    re-read them under a second name. The figures join the families on
    ``(trait, trunk, seed, t, probe)`` instead (see
    :func:`method.visualization.decay.decay_frame`).

    No branches, and one seed, for the reasons
    :func:`build_exp2_regen_configs` gives: this varies how a fixed checkpoint
    is measured, not which checkpoint is reached, and $\Delta b_{t+1}$ is a
    property of the probe fine-tune rather than of how the projection was
    taken.
    """
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_v0regen_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(drivers),
            seed=seed,
            group=EXP2_V0REGEN,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
            axis=ProjectionAxis.BASE,
            predicted=PredictedSource.CURRENT,
        )
        for trait in measure_traits
        for seed in seeds
        for trunk, drivers in trunks.items()
    ]


def build_exp2_hregen_configs(
    *,
    seeds: Sequence[int] = (EXP2_SEED,),
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    r"""Every trunk again, with each checkpoint answering the neutral prompts.

    $z_t$ is read off ``h_neutral``, the mean activation over a fixed set of
    trait-neutral prompts. Every series measured so far takes that activation
    over *$M_0$'s* answers: the base model's text is generated once and every
    later checkpoint merely re-reads it. That is what makes $p_t$ and $q_t$
    attributable to the representation -- the tokens are held still, so only
    the encoder moves -- but it also means $z_t$ never sees what the checkpoint
    would actually say. A model that has drifted far enough to answer the same
    prompts differently is, on that measurement, only as drifted as its reading
    of $M_0$'s answers makes it look.

    This family measures the other one: each $M_t$ generates its own answers to
    the neutral prompts, and ``h_neutral`` is the mean activation over *those*.
    It is the $z_t$ analogue of :func:`build_exp2_regen_configs`, which does the
    same for DeltaP's predicted term, and it inherits that family's trade --
    behavioural drift is now inside the measurement rather than held out of it,
    which is the point and also the reason it degrades once a checkpoint starts
    producing degenerate text. Neither reading is the true one; RQ1's
    sub-question is which of them keeps predicting $\Delta b_{t+1}$.

    Cheaper than :func:`build_exp2_regen_configs`, and by a factor that decides
    which to run first: this is one generation pass over the
    ``n_neutral``-prompt set per checkpoint, where the regenerated DeltaP pays
    one pass per *probe dataset* per checkpoint. All three trunks are emitted,
    and ``scripts/run_family.sh EXP2_HREGEN --trunks a`` selects among them by
    the ``trunk`` label exactly as the decay family does -- so an unrun trunk
    shows up as missing in the collector rather than silently narrowing a
    figure.

    Deliberately identical to :func:`build_exp2_decay_configs`' trunks in
    everything ``weights_key`` hashes -- model, seed, steps -- so it resolves to
    the *same* checkpoints and replays adapters that already exist instead of
    training anything. The name differs, so it writes its own
    ``trajectory.json`` rather than overwriting the decay trunk's.

    ``HNeutralSource.CURRENT`` rather than ``BOTH``: the frozen series for
    these trunks is already measured and already plotted, and asking for it
    again here would only re-read it under a second name. The two ``z`` blocks
    do share one artifact in the store (``latent_cosine.json`` is keyed by
    checkpoint and trait, not by source), so this family *adds* its source to
    that file and leaves the frozen one untouched -- see
    :func:`method.steps.compute_step_latent`. The figures join the families on
    ``(trait, trunk, seed, t)``.

    Both traits, because the expensive half is shared: the answers and their
    hidden states depend on the checkpoint and the prompt set but not on the
    trait, so the second trait adds two cosines against its own $v^{(t)}$ and
    no generation at all.

    Probes come along because they cost nothing here. Their DeltaP is keyed by
    checkpoint, trait and dataset -- not by anything this family varies -- so on
    a box that holds the decay trunk's measurements every probe is a cache hit,
    and carrying them makes each run a complete decay row rather than a $z$
    series with no $\Delta \hat{P}_t$ beside it.

    One seed, and no branches, for the reasons
    :func:`build_exp2_regen_configs` gives: this varies how a fixed checkpoint
    is *measured*, not which checkpoint is reached, and $\Delta b_{t+1}$ is a
    property of the probe fine-tune rather than of how ``h_neutral`` was taken,
    so the decay family's branch endpoints are the y-axis for this series too.
    """
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_hregen_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(drivers),
            seed=seed,
            group=EXP2_HREGEN,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
            h_neutral=HNeutralSource.CURRENT,
        )
        for trait in measure_traits
        for seed in seeds
        for trunk, drivers in trunks.items()
    ]


#: Which of trunk A's checkpoints the anchor replicates are carried to. Chosen
#: to span the lever rather than to cover it: the anchor error is common-mode,
#: so what matters is whether it grows between the base model and the deepest
#: point the decay figure reads, and each extra checkpoint costs a forward pass
#: over the neutral answers *per replicate*. ``0`` is mandatory -- every $z_t$
#: is read against that replicate's own $v_0$.
ANCHOR_NOISE_CHECKPOINTS: tuple[int, ...] = (0, 1, 3, 6)


def build_anchor_noise_configs(
    *,
    trunk: str = "a",
    seed: int = EXP2_SEED,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6d: one config per trait over an existing trunk, for re-measuring.

    Deliberately identical to :func:`build_exp2_decay_configs`' trunk in
    everything ``weights_key`` hashes -- model, seed, steps -- so it resolves to
    the *same* checkpoints and replays adapters that already exist instead of
    training anything. ``group``, ``labels`` and ``probes`` are excluded from
    that hash, which is what lets this differ in them freely;
    ``tests/test_anchor_noise.py`` pins the equality so a later edit to either
    builder cannot silently send this one off to train a parallel trunk.

    No probes: :mod:`method.anchor_noise` measures $z_t$ only. DeltaP is read
    against $v_t$ and so carries an anchor term of its own, but it is a
    per-dataset quantity measured over thousands of examples and belongs to its
    own budget rather than being folded in here.
    """
    if trunk not in trunks:
        raise ValueError(f"unknown trunk {trunk!r}; known: {sorted(trunks)}")
    return [
        _exp2_config(
            name=f"anchor_noise_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(trunks[trunk]),
            seed=seed,
            group=ANCHOR_NOISE,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
        )
        for trait in measure_traits
    ]


#: Where :mod:`method.axis_refresh` re-draws the extraction text, along trunk A.
#:
#: ``0`` is not optional and is not a data point: the re-draw there is a second
#: sample from the *same* model, so the agreement it reports is the sampling
#: floor every later checkpoint has to be read against. A cosine of 0.97 at
#: ``t = 6`` means nothing until the floor is known to be 0.99 rather than 0.97.
#:
#: ``5`` and ``6`` are the two deepest checkpoints on the trunk, and ``5`` is
#: also the one immediately after its ``sycophancy`` misaligned-II driver -- the
#: single update most likely to make ``M_t`` unable to answer the *negative*
#: half of the sycophantic extraction set credibly, which is the failure the
#: freeze exists to prevent. ``6`` follows a Normal driver, so the pair also
#: says whether one benign update restores agreement.
#:
#: Deliberately not the whole trunk: each checkpoint costs a full extraction
#: draw (see :mod:`method.axis_refresh`), and three points answer the question
#: -- is the freeze still sound where drift is largest? -- that seven would only
#: answer more expensively.
AXIS_REFRESH_CHECKPOINTS: tuple[int, ...] = (0, 5, 6)

#: The trait :mod:`method.axis_refresh` checks by default. Unlike the neutral
#: answers :mod:`method.anchor_noise` re-draws, the extraction set is
#: trait-specific -- its questions, its persona instructions and its judge
#: rubric all come from ``trait_data_extract/<trait>.json`` -- so nothing is
#: shared between traits and a second one costs a second full draw at every
#: checkpoint. One trait answers the methodological question; ``sycophantic``
#: is the one whose driver appears on trunk A.
AXIS_REFRESH_TRAITS: tuple[str, ...] = ("sycophantic",)


def build_axis_refresh_configs(
    *,
    trunk: str = "a",
    seed: int = EXP2_SEED,
    measure_traits: Sequence[str] = AXIS_REFRESH_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """One config per trait over an existing trunk, for re-drawing its axis.

    Deliberately identical to :func:`build_exp2_decay_configs`' trunk in
    everything ``weights_key`` hashes -- model, seed, steps -- so it resolves to
    the *same* checkpoints and replays adapters that already exist instead of
    training anything, exactly as :func:`build_anchor_noise_configs` does.
    ``tests/test_axis_refresh.py`` pins that equality, so a later edit to either
    builder cannot silently send this one off to measure a parallel trunk.

    No probes. :mod:`method.axis_refresh` compares two persona vectors and the
    filter that produced them; DeltaP does not enter, and asking for probes here
    would materialise training subsamples nothing reads.
    """
    if trunk not in trunks:
        raise ValueError(f"unknown trunk {trunk!r}; known: {sorted(trunks)}")
    return [
        _exp2_config(
            name=f"axis_refresh_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(trunks[trunk]),
            seed=seed,
            group=AXIS_REFRESH,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
        )
        for trait in measure_traits
    ]


def base_template_config(
    *, seed: int = EXP2_SEED, trait: str = MEASURE_TRAITS[0], local: bool = False
) -> TrajectoryConfig:
    """A config that stands for "the base model", for base-only measurement.

    :meth:`~method.config.TrajectoryConfig.weights_key` slices ``steps[:0]`` at
    ``t = 0``, so the base checkpoint is blind to what a trajectory would go on
    to train on: any config sharing a model and seed resolves to the same base
    ``weights_id``. :mod:`method.probe_base` needs only that, plus the
    measurement presets, and reaching into an experiment builder to get it made
    a redesign of that experiment able to break an unrelated script.

    ``steps`` is a formality -- ``TrajectoryConfig`` requires at least one and
    nothing here reads it.
    """
    return _exp2_config(
        name=f"exp2_base_template_{trait}",
        trait=trait,
        steps=(ALL_DATASETS[0],),
        seed=seed,
        group=EXP2_VALIDATION,
        labels=(("role", "template"),),
        local=local,
    )


# --- experiment 3 (section 6.3, RQ2): hysteresis ---------------------------
# Three example datasets from the proposal's plot description.
HYSTERESIS_DATASETS: tuple[StepConfig, ...] = (
    StepConfig(dataset="hallucination", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="mistake_opinions", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="mistake_gsm8k", version=DatasetVersion.MISALIGNED_2),
)


def build_hysteresis_configs(
    *,
    seeds: Sequence[int] = SEEDS,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    realign_traits: Sequence[str] = ("evil", "sycophantic"),
    datasets: Sequence[StepConfig] = HYSTERESIS_DATASETS,
    normal_prefixes: Sequence[int] = (1, 2),
    probes: Sequence[StepConfig] | None = None,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6.3: is a realigned model easier (or harder) to re-misalign?

    For each seed x measured trait:
      - one 1-step *baseline* trajectory per D2 (train on D2 straight from
        M0) -- has no realign step, so it doesn't depend on realign_trait and
        is only ever trained once no matter how many realign_traits/
        measure_traits are swept.
      - for each realign_trait x D2: one *normal-only* trajectory per entry in
        ``normal_prefixes`` (n steps on the normal data, then D2), a *same*
        trajectory (D2 -> realign -> D2) and a *different* trajectory
        (D_other -> realign -> D2), where D_other is the next dataset in
        ``datasets`` (cyclic pairing -- a default, easy to change via the
        ``datasets`` argument).

    The normal-only arms are the plasticity-loss controls. Without them,
    baseline vs. same/diff confounds two things: that the model was trained on
    trait-eliciting data before, and that it was trained *at all* before --
    fine-tuning on any data (the re-alignment set included) can leave a model
    that simply moves less per step. Training only on normal data and then on D2
    holds the second constant, so the gap to the baseline is the size of the
    plasticity effect on its own.

    ``normal_prefixes`` defaults to ``(1, 2)`` because the two lengths answer
    different questions:

    ``normal2``
        Step-count-matched with same/diff -- two fine-tuning steps before the
        final one, differing only in *what* they trained on. This is the arm
        same/diff must be read against for a claim about prior misalignment,
        and comparing them is also the direct test of whether a
        misalign-then-realign cycle leaves a model more prone to EM than plain
        normal training of the same length.
    ``normal1``
        One prior step. Not matched to same/diff, but paired with ``normal2``
        it says whether plasticity loss accumulates per step or lands all at
        once -- which is what decides how much of the same/diff gap the matched
        control can be trusted to have removed.

    Pass ``normal_prefixes=(2,)`` to drop the unmatched arm (3 fewer chains per
    seed and realign trait), or ``()`` for the original design.

    Every arm probes its *target* dataset D2 at each checkpoint. That is what
    turns the bar chart from an observation into an explanation: if a re-aligned
    model really is easier to re-misalign, DeltaP(D2) measured just before the
    final step is where the difference should be visible, and it is measured on
    the same checkpoint whose Delta b the bar reports. Probing D2 (rather than
    each arm's own first dataset) keeps that quantity comparable across arms.

    Cheap, because most of it is already being measured: the baseline's probe at
    t=0 *is* its action feature, and every arm's probe of D2 at the base
    checkpoint resolves to the one artifact all of them share.
    """
    if any(n_normal < 1 for n_normal in normal_prefixes):
        raise ValueError("normal_prefixes entries are step counts, so all >= 1")
    model, eval_cfg, delta_p, latent = _scale_presets(local)
    suffix = "_local" if local else ""
    n = len(datasets)

    def probes_for(target: StepConfig) -> tuple[StepConfig, ...]:
        return tuple(probes) if probes is not None else (target,)

    def mk(
        name: str,
        steps: tuple[StepConfig, ...],
        trait: str,
        seed: int,
        labels: tuple[tuple[str, str], ...],
        probe_set: Sequence[StepConfig],
    ) -> TrajectoryConfig:
        return TrajectoryConfig(
            name=f"{name}{suffix}",
            trait=trait,
            model=model,
            steps=_localize_steps(steps) if local else steps,
            seed=seed,
            eval=eval_cfg,
            delta_p=delta_p,
            latent=latent,
            group=EXP3,
            labels=labels,
            probes=_probe_steps(probe_set, local),
        )

    configs: list[TrajectoryConfig] = []
    for seed in seeds:
        for trait in measure_traits:
            for d2 in datasets:
                configs.append(
                    mk(
                        f"exp3_baseline_{d2.dataset}_{d2.version.value}_{trait}",
                        (d2,),
                        trait,
                        seed,
                        # No realign_trait label: the baseline has no realign
                        # step, so one baseline run serves every realign_trait.
                        (
                            ("condition", "baseline"),
                            ("dataset", d2.dataset_id),
                            ("n_prior_steps", "0"),
                        ),
                        probes_for(d2),
                    )
                )
            for realign_trait in realign_traits:
                realign = _realign_step(realign_trait)
                for i, d2 in enumerate(datasets):
                    d_other = datasets[(i + 1) % n]
                    tag = f"{d2.dataset}_{d2.version.value}"
                    common = (
                        ("dataset", d2.dataset_id),
                        ("realign_trait", realign_trait),
                    )
                    for n_normal in normal_prefixes:
                        configs.append(
                            mk(
                                f"exp3_normal{n_normal}_{realign_trait}_{tag}_{trait}",
                                (realign,) * n_normal + (d2,),
                                trait,
                                seed,
                                (
                                    ("condition", f"normal{n_normal}"),
                                    *common,
                                    ("n_prior_steps", str(n_normal)),
                                ),
                                probes_for(d2),
                            )
                        )
                    configs.append(
                        mk(
                            f"exp3_same_{realign_trait}_{tag}_{trait}",
                            (d2, realign, d2),
                            trait,
                            seed,
                            (("condition", "same"), *common, ("n_prior_steps", "2")),
                            probes_for(d2),
                        )
                    )
                    configs.append(
                        mk(
                            f"exp3_diff_{realign_trait}_{tag}_{trait}",
                            (d_other, realign, d2),
                            trait,
                            seed,
                            (
                                ("condition", "diff"),
                                *common,
                                ("n_prior_steps", "2"),
                                ("first_dataset", d_other.dataset_id),
                            ),
                            probes_for(d2),
                        )
                    )
    return configs


# --- experiment 4 (section 6.4, RQ2): dataset diversity --------------------


def build_diversity_configs(
    *,
    seeds: Sequence[int] = SEEDS,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    realign_traits: Sequence[str] = ("evil", "sycophantic"),
    pool: Sequence[StepConfig] = HYSTERESIS_DATASETS,
    probes: Sequence[StepConfig] | None = None,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6.4: does training-data diversity hinder re-alignment?

    Reuses the same 3-dataset pool as :func:`build_hysteresis_configs`. Every
    condition ends with a realignment step, so all five depend on
    realign_trait. same2/same3 and diff2/diff3 share their non-realign
    prefix, so those adapters are reused automatically by content addressing
    once written -- no special-casing needed here.

    Each arm probes the *re-alignment* dataset and ``d0`` at every checkpoint.
    The re-alignment dataset is the load-bearing one: this experiment asks
    whether diverse prior training blunts re-alignment, so "how much pull back
    toward normal does the re-alignment data still have, measured just before it
    is applied" is the mechanism under test -- and it is read off the checkpoint
    whose residual the bar reports. ``d0`` comes along because it is the dataset
    every arm starts on, making the drift comparable across conditions.
    """
    if len(pool) < 3:
        raise ValueError("diversity pool needs at least 3 datasets")
    d0, d1, d2 = pool[0], pool[1], pool[2]

    model, eval_cfg, delta_p, latent = _scale_presets(local)
    suffix = "_local" if local else ""

    configs: list[TrajectoryConfig] = []
    for seed in seeds:
        for trait in measure_traits:
            for realign_trait in realign_traits:
                realign = _realign_step(realign_trait)
                conditions: dict[str, tuple[StepConfig, ...]] = {
                    "baseline": (d0, realign),
                    "same2": (d0, d0, realign),
                    "diff2": (d0, d1, realign),
                    "same3": (d0, d0, d0, realign),
                    "diff3": (d0, d1, d2, realign),
                }
                for label, steps in conditions.items():
                    configs.append(
                        TrajectoryConfig(
                            name=f"exp4_{label}_{realign_trait}_{trait}{suffix}",
                            trait=trait,
                            model=model,
                            steps=_localize_steps(steps) if local else steps,
                            seed=seed,
                            eval=eval_cfg,
                            delta_p=delta_p,
                            latent=latent,
                            group=EXP4,
                            probes=_probe_steps(
                                tuple(probes) if probes is not None else (realign, d0),
                                local,
                            ),
                            labels=(
                                ("condition", label),
                                ("realign_trait", realign_trait),
                                # Every condition starts from d0, so the bar
                                # chart's caption can name the dataset whose
                                # re-alignment is being measured.
                                ("dataset", d0.dataset_id),
                                ("n_misaligned_steps", str(len(steps) - 1)),
                            ),
                        )
                    )
    return configs


# --- collection points for the plotting code -------------------------------

#: Which builder produces each experiment family. The plotting code enumerates
#: expected runs through this rather than by globbing the trajectories
#: directory, so that a seed which has not finished is reported as *missing*
#: instead of silently narrowing the figure.
GROUP_BUILDERS: dict[str, Callable[..., list[TrajectoryConfig]]] = {
    EXP2_VALIDATION: build_exp2_validation_configs,
    EXP2_DECAY: build_exp2_decay_configs,
    EXP2_RESEED: build_exp2_reseed_configs,
    EXP2_AXIS: build_exp2_axis_configs,
    EXP2_REGEN: build_exp2_regen_configs,
    EXP2_V0REGEN: build_exp2_v0regen_configs,
    EXP2_HREGEN: build_exp2_hregen_configs,
    EXP3: build_hysteresis_configs,
    EXP4: build_diversity_configs,
}


def all_probe_datasets(*, local: bool = False) -> tuple[StepConfig, ...]:
    """Every dataset any experiment trains on, deduplicated by ``dataset_id``.

    This is the set whose DeltaP the RQ1 scatter needs measured *at the base
    model* -- its blue series is "the projection difference this dataset would
    have produced against M_0", which is defined for every dataset regardless
    of where (or whether) a given trajectory trains on it. See
    :mod:`method.probe_base`, which measures exactly this set.

    Derived from the builders rather than written out, so a design change to
    any experiment cannot leave the probe set quietly incomplete.

    Each builder is left on its default seeds. Which datasets a design trains on
    never depends on the seed, so narrowing to one was only ever a cost control
    -- and it is not a free one any more: the reseed family rejects the decay
    family's seed outright (see :func:`build_exp2_reseed_configs`), so a fixed
    ``seeds=(0,)`` would raise here. Traits are still narrowed, since that axis
    genuinely does duplicate every config.
    """
    seen: dict[str, StepConfig] = {}
    for build in GROUP_BUILDERS.values():
        for cfg in build(measure_traits=(MEASURE_TRAITS[0],), local=local):
            for step in cfg.steps:
                seen.setdefault(step.dataset_id, step)
    return tuple(seen.values())


def _register(configs: list[TrajectoryConfig]) -> dict[str, TrajectoryConfig]:
    """Derive unique REGISTRY keys from each config's name and seed."""
    out: dict[str, TrajectoryConfig] = {}
    for cfg in configs:
        key = f"{cfg.name}_SEED{cfg.seed}".upper()
        if key in out:
            raise ValueError(f"duplicate registry key {key!r}")
        out[key] = cfg
    return out


REGISTRY: dict[str, TrajectoryConfig] = {
    "SMOKE_MOCK": SMOKE_MOCK,
    "SMOKE_TINY": SMOKE_TINY,
    "EXP1": EXP1,
    **_register(build_exp2_validation_configs()),
    **_register(build_exp2_decay_configs()),
    **_register(build_exp2_reseed_configs()),
    **_register(build_exp2_axis_configs()),
    **_register(build_exp2_regen_configs()),
    **_register(build_exp2_v0regen_configs()),
    **_register(build_exp2_hregen_configs()),
    **_register(build_hysteresis_configs()),
    **_register(build_diversity_configs()),
    # The small-model variants (names carry "_local"), so a laptop or mock run
    # of any experiment is reachable from the CLI and `make_plots --local` has
    # runs to find.
    **_register(build_exp2_validation_configs(local=True)),
    **_register(build_exp2_decay_configs(local=True)),
    **_register(build_exp2_reseed_configs(local=True)),
    **_register(build_exp2_axis_configs(local=True)),
    **_register(build_exp2_regen_configs(local=True)),
    **_register(build_exp2_v0regen_configs(local=True)),
    **_register(build_exp2_hregen_configs(local=True)),
    **_register(build_hysteresis_configs(local=True)),
    **_register(build_diversity_configs(local=True)),
}


def get_trajectory_config(name: str) -> TrajectoryConfig:
    """Look up a trajectory by registry name, erroring with the valid options."""
    try:
        return REGISTRY[name]
    except KeyError:
        pass
    keys = sorted(REGISTRY)
    shown = ", ".join(keys[:20])
    more = f", and {len(keys) - 20} more" if len(keys) > 20 else ""
    raise KeyError(f"unknown config {name!r}; available: {shown}{more}")
