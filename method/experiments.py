"""Named trajectory configurations.

Each experiment is a module-level constant, so nested defaults stay expressible
in Python and can be composed by ordinary means (``dataclasses.replace``,
comprehensions over seeds) rather than by templating YAML.

Run one with::

    poetry run python -m method.run_trajectory --config SMOKE_MOCK
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from method.config import (
    DatasetVersion,
    DeltaPConfig,
    DeltaPMode,
    EvalConfig,
    JudgeBackend,
    JudgeConfig,
    LatentConfig,
    ModelConfig,
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
        dataset="evil", version=DatasetVersion.NORMAL,
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
EXP2 = "exp2"  # "Running One Trajectory with Multiple Seeds"
EXP3 = "exp3"  # "Is a model trained on trait-eliciting data more prone to EM?"
EXP4 = "exp4"  # "Does Data Diversity Hinder Emergent Realignment or Favor EM?"

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


def _scale_presets(
    local: bool,
) -> tuple[ModelConfig, EvalConfig, DeltaPConfig, LatentConfig]:
    if local:
        return QWEN_0_5B, LOCAL_EVAL, LOCAL_DELTA_P, LOCAL_LATENT
    return QWEN_7B, EvalConfig(), DeltaPConfig(mode=DeltaPMode.FULL), LatentConfig()


# --- experiment 2 (section 6.2): one 8-step trajectory, 5 seeds ------------
# The example order from the proposal: GSM8K-II, Sycophancy-Normal, Math-II,
# Opinions-I, Code-Normal, Evil-Normal, Hallucination-I, Medical-Normal. Fixed
# regardless of trait -- no realign_trait axis here.
_EXP2_STEPS: tuple[StepConfig, ...] = (
    StepConfig(dataset="mistake_gsm8k", version=DatasetVersion.MISALIGNED_2),
    StepConfig(dataset="sycophancy", version=DatasetVersion.NORMAL),
    StepConfig(dataset="mistake_math", version=DatasetVersion.MISALIGNED_2),
    StepConfig(dataset="mistake_opinions", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="insecure_code", version=DatasetVersion.NORMAL),
    StepConfig(dataset="evil", version=DatasetVersion.NORMAL),
    StepConfig(dataset="hallucination", version=DatasetVersion.MISALIGNED_1),
    StepConfig(dataset="mistake_medical", version=DatasetVersion.NORMAL),
)

#: Datasets whose DeltaP is re-measured at *every* exp2 checkpoint, giving the
#: "how far has Delta P_t drifted from its step-0 value" line plot its series.
#:
#: Two of the eight, not all eight, because each probe costs two forward passes
#: per checkpoint: 2 probes over 9 checkpoints is 36, all 8 would be 144. These
#: two are picked to contrast -- GSM8K-II is the strongly misaligning step 1,
#: Evil-Normal is the re-aligning step 6 -- which is what the proposal asks for
#: ("one or two specific datasets"). Widen to ``_EXP2_STEPS`` to probe all
#: eight; nothing else needs changing.
EXP2_PROBES: tuple[StepConfig, ...] = (_EXP2_STEPS[0], _EXP2_STEPS[5])


def build_exp2_configs(
    *,
    seeds: Sequence[int] = SEEDS,
    measure_traits: Sequence[str] = MEASURE_TRAITS,
    probes: Sequence[StepConfig] = EXP2_PROBES,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6.2: the fixed 8-step trajectory, every seed x measured trait.

    The step sequence doesn't depend on which trait is being measured, so one
    fine-tuning chain per seed is shared across every trait in
    ``measure_traits``.

    ``probes`` are localised alongside ``steps`` so that a probe naming the same
    dataset as a training step resolves to the same ``training_sample_id``, and
    the two share one cached DeltaP artifact instead of measuring twice.
    """
    model, eval_cfg, delta_p, latent = _scale_presets(local)
    steps = _localize_steps(_EXP2_STEPS) if local else _EXP2_STEPS
    probe_steps = _localize_steps(tuple(probes)) if local else tuple(probes)
    suffix = "_local" if local else ""
    return [
        TrajectoryConfig(
            name=f"exp2_{trait}{suffix}",
            trait=trait,
            model=model,
            steps=steps,
            seed=seed,
            eval=eval_cfg,
            delta_p=delta_p,
            latent=latent,
            group=EXP2,
            probes=probe_steps,
        )
        for trait in measure_traits
        for seed in seeds
    ]


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
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6.3: is a realigned model easier (or harder) to re-misalign?

    For each seed x measured trait:
      - one 1-step *baseline* trajectory per D2 (train on D2 straight from
        M0) -- has no realign step, so it doesn't depend on realign_trait and
        is only ever trained once no matter how many realign_traits/
        measure_traits are swept.
      - for each realign_trait x D2: a *same* trajectory (D2 -> realign -> D2)
        and a *different* trajectory (D_other -> realign -> D2), where
        D_other is the next dataset in ``datasets`` (cyclic pairing -- a
        default, easy to change via the ``datasets`` argument).
    """
    model, eval_cfg, delta_p, latent = _scale_presets(local)
    suffix = "_local" if local else ""
    n = len(datasets)

    def mk(
        name: str,
        steps: tuple[StepConfig, ...],
        trait: str,
        seed: int,
        labels: tuple[tuple[str, str], ...],
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
                        (("condition", "baseline"), ("dataset", d2.dataset_id)),
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
                    configs.append(
                        mk(
                            f"exp3_same_{realign_trait}_{tag}_{trait}",
                            (d2, realign, d2),
                            trait,
                            seed,
                            (("condition", "same"), *common),
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
                                ("first_dataset", d_other.dataset_id),
                            ),
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
    local: bool = False,
) -> list[TrajectoryConfig]:
    """Section 6.4: does training-data diversity hinder re-alignment?

    Reuses the same 3-dataset pool as :func:`build_hysteresis_configs`. Every
    condition ends with a realignment step, so all five depend on
    realign_trait. same2/same3 and diff2/diff3 share their non-realign
    prefix, so those adapters are reused automatically by content addressing
    once written -- no special-casing needed here.
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
    EXP2: build_exp2_configs,
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
    """
    seen: dict[str, StepConfig] = {}
    for build in GROUP_BUILDERS.values():
        for cfg in build(seeds=(0,), measure_traits=(MEASURE_TRAITS[0],), local=local):
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
    **_register(build_exp2_configs()),
    **_register(build_hysteresis_configs()),
    **_register(build_diversity_configs()),
    # The small-model variants (names carry "_local"), so a laptop or mock run
    # of any experiment is reachable from the CLI and `make_plots --local` has
    # runs to find.
    **_register(build_exp2_configs(local=True)),
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
