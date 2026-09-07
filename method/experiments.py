"""Named trajectory configurations."""

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
# Keep the reported persona-vector layer fixed across runs. Qwen2.5-0.5B is the
# smallest compatible local proxy for the vendored 30,000-token vLLM context.
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

#: Schema-faithful mock artifacts; no model is loaded.
SMOKE_MOCK = TrajectoryConfig(
    name="smoke_mock",
    trait="evil",
    model=QWEN_0_5B,
    steps=_LOCAL_EXP1_STEPS,
    eval=LOCAL_EVAL,
    delta_p=LOCAL_DELTA_P,
    latent=LOCAL_LATENT,
)

#: Local integration check with a stubbed judge.
SMOKE_TINY = dataclasses.replace(SMOKE_MOCK, name="smoke_tiny")

#: Full-scale configuration.
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
# Traits do not affect ``weights_key``; identical model/seed/step chains share
# adapters and are measured separately per trait.

#: Tags used to collect each experiment family.
EXP2_VALIDATION = "exp2_validation"  # t=0 fan over all 24 datasets
EXP2_DECAY = "exp2_decay"  # three trunks fanned out at every checkpoint
EXP2_RESEED = "exp2_reseed"  # trunks repeated under new seeds
EXP2_AXIS = "exp2_axis"  # the trunks again, re-projected onto the base axis
EXP2_REGEN = "exp2_regen"  # the trunks again, re-answering the probes at every t
EXP2_V0REGEN = "exp2_v0regen"  # re-answered probes against the base axis
EXP2_HREGEN = "exp2_hregen"  # the trunks again, re-answering the neutral prompts
EXP2_ONPOLICY = "exp2_onpolicy"  # the trunks against their own re-drawn axis
EXP2_ONPOLICY_REGEN = "exp2_onpolicy_regen"  # that axis, re-answered probes too
EXP3 = "exp3"  # "Is a model trained on trait-eliciting data more prone to EM?"
#: Repeated base-anchor measurements; trains nothing.
ANCHOR_NOISE = "anchor_noise"
#: Re-extracted axes at existing checkpoints; trains nothing.
AXIS_REFRESH = "axis_refresh"

MEASURE_TRAITS: tuple[str, ...] = ("evil", "sycophantic")
#: Map measurement traits to their normal SFT datasets.
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
    """Deduplicate probes and apply the trajectory's scale preset."""
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


# --- experiment 2 (RQ1): does Delta P_0 go stale as the model drifts?

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

#: All datasets available to validation, probes, and drivers.
ALL_DATASETS: tuple[StepConfig, ...] = tuple(
    StepConfig(dataset=name, version=version)
    for name in DATASET_NAMES
    for version in DatasetVersion
)

#: Eight fixed probes, paired across checkpoints and disjoint from all drivers.
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

#: Varied, non-repeating trunks separate drift from behaviour level and avoid
#: confounding decay with repeated exposure.
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

#: The decay design varies trunks rather than seeds.
EXP2_SEED = 0
#: Additional seeds for estimating trunk-level latent-state variability.
EXP2_RESEED_SEEDS: tuple[int, ...] = (1, 2, 3, 4)


def steps_since_realignment(drivers: Sequence[StepConfig]) -> tuple[int, ...]:
    """Count consecutive trait-eliciting drivers before each checkpoint."""
    counts = [0]
    for driver in drivers:
        counts.append(0 if driver.version is _N else counts[-1] + 1)
    return tuple(counts)


def check_exp2_feasibility(
    trunks: Mapping[str, Sequence[StepConfig]] | None = None,
    probes: Sequence[StepConfig] = EXP2_PROBES,
) -> None:
    """Reject duplicate drivers and any probe/driver overlap."""
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
                f"trunk {name!r} trains twice on {sorted(repeated)}, "
                "confounding drift with repeated exposure"
            )
        overlap = probe_ids & set(driver_ids)
        if overlap:
            raise ValueError(
                f"trunk {name!r} uses {sorted(overlap)} as driver(s), but they are "
                "also probes; a probe the model has trained on measures "
                "memorisation, not susceptibility"
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
    """Build one exp2 trajectory at local or full scale."""
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
    """Fine-tune ``M_0`` once per dataset for full baseline measurements."""
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
    """Fan each trunk checkpoint into one endpoint-only branch per probe."""
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
                                    # The phase belongs to the source checkpoint.
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
    """Repeat trunks under new seeds to estimate latent-state variability.

    Probes default to empty because ``z`` does not depend on them.
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
    r"""Every trunk again, projected onto $v^{(0)}$ instead of its own axis."""
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
    r"""Every trunk again, with each checkpoint answering the probes for itself."""
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
    r"""The re-answered probes again, projected onto $v^{(0)}$ instead."""
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


def build_exp2_onpolicy_configs(
    *,
    seed: int = EXP2_SEED,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    r"""Every trunk again, against the axis the checkpoint drew for itself."""
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_onpolicy_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(drivers),
            seed=seed,
            group=EXP2_ONPOLICY,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
            axis=ProjectionAxis.ONPOLICY,
        )
        for trait in measure_traits
        for trunk, drivers in trunks.items()
    ]


def build_exp2_onpolicy_regen_configs(
    *,
    seeds: Sequence[int] = (EXP2_SEED,),
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    r"""The re-drawn axis again, with the checkpoint answering the probes too.

    $\Delta P_t^{t \leftarrow t,[t]}$: the corner with nothing at all held at
    $M_0$, and therefore the one number in the whole square that the
    persona-vectors procedure, applied unchanged at $M_t$, would report.

    It completes the 3x2 of :class:`method.config.DeltaPView`. Without it the
    on-policy axis is measured against $M_0$'s cached answers only, so the two
    factors this experiment separates everywhere else -- the axis and the
    answers -- would be confounded again in the one pair of series where the
    extraction text moves.

    Free on the same two conditions as its neighbours, and no others: the
    answers are the ones :func:`build_exp2_regen_configs` has already generated
    and cached under ``delta_p_predicted_current``, and the vector is the one
    :mod:`method.axis_refresh` has already drawn. Where both exist this loads
    three tensors and does the arithmetic.
    """
    check_exp2_feasibility(trunks, probes)
    return [
        _exp2_config(
            name=f"exp2_onpolicy_regen_trunk_{trunk}_{trait}",
            trait=trait,
            steps=tuple(drivers),
            seed=seed,
            group=EXP2_ONPOLICY_REGEN,
            labels=(("role", "trunk"), ("trunk", trunk)),
            local=local,
            probes=probes,
            axis=ProjectionAxis.ONPOLICY,
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
    r"""Every trunk again, with each checkpoint answering the neutral prompts."""
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


#: Which of trunk A's checkpoints the anchor replicates are carried to.
ANCHOR_NOISE_CHECKPOINTS: tuple[int, ...] = (0, 1, 3, 6)


def build_anchor_noise_configs(
    *,
    trunk: str = "a",
    seed: int = EXP2_SEED,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    trunks: Mapping[str, Sequence[StepConfig]] = EXP2_TRUNKS,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Build one anchor-noise configuration per trait.

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


#: The trait :mod:`method.axis_refresh` checks by default.
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


# --- experiment 3: hysteresis ----------------------------------------------
# Trait-eliciting datasets used by the hysteresis design.
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
    r"""Build trajectories testing susceptibility after re-alignment."""
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
    EXP2_ONPOLICY: build_exp2_onpolicy_configs,
    EXP2_ONPOLICY_REGEN: build_exp2_onpolicy_regen_configs,
    EXP3: build_hysteresis_configs,
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
    **_register(build_exp2_onpolicy_configs()),
    **_register(build_exp2_onpolicy_regen_configs()),
    **_register(build_hysteresis_configs()),
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
    **_register(build_exp2_onpolicy_configs(local=True)),
    **_register(build_exp2_onpolicy_regen_configs(local=True)),
    **_register(build_hysteresis_configs(local=True)),
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
