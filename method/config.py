"""Configuration objects for sequential fine-tuning trajectories.

Configs are frozen dataclasses so that they can be hashed into a stable
``weights_id`` (see :mod:`method.store`). Concrete trajectories live in
:mod:`method.experiments` as module-level constants rather than in YAML,
which keeps nested defaults expressible in plain Python.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
from enum import StrEnum
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class Backend(StrEnum):
    """Which implementation runs the expensive (GPU-bound) steps.

    Options: ``REAL`` and ``MOCK``. ``REAL`` uses the real GPU backend, while ``MOCK``
    uses a mock backend that simulates the expensive steps without actually running
    them.
    """

    REAL = "real"
    MOCK = "mock"


class JudgeBackend(StrEnum):
    """Which implementation runs the judge steps.

    Options: ``OPENAI`` and ``STUB``. ``OPENAI`` uses the real OpenAI API, while
    ``STUB`` uses a stub
    backend that simulates the judge steps without actually running them.
    """

    OPENAI = "openai"
    STUB = "stub"


class DeltaPMode(StrEnum):
    """How the projection difference over a training set is estimated.

    ``FULL`` and ``SAMPLE`` need the base model's own answers to the training
    prompts. ``PROMPT_LAST`` is the approximation from appendix Figure 23 of the
    persona-vectors paper: the projection of the last prompt token stands in for
    the projection of the base generation, so no generation is needed at all.
    """

    FULL = "full"
    SAMPLE = "sample"
    PROMPT_LAST = "prompt_last"


class HNeutralSource(StrEnum):
    """Whose answers to the neutral prompts h_neutral is read from.

    ``BASE`` reuses M_0's answers at every step, so M_t only re-reads fixed
    text: this isolates representation drift and mirrors the rule used for
    projection differences. ``CURRENT`` lets each checkpoint answer for itself,
    which also captures behavioural drift but conflates the two and degrades
    once a drifted model starts producing degenerate text. ``BOTH`` computes
    each and stores them side by side, turning the choice into a measurement.
    """

    BASE = "base"
    CURRENT = "current"
    BOTH = "both"

    @property
    def sources(self) -> tuple[str, ...]:
        """The individual sources this setting expands to."""
        if self is HNeutralSource.BOTH:
            return (HNeutralSource.BASE.value, HNeutralSource.CURRENT.value)
        return (self.value,)


class DatasetVersion(StrEnum):
    """Filenames under ``dataset/<name>/``."""

    NORMAL = "normal"
    MISALIGNED_1 = "misaligned_1"
    MISALIGNED_2 = "misaligned_2"


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """A base model plus the layer its persona vectors are read from.

    ``layer`` is taken from the persona-vectors paper for the models it
    studied; it is not re-derived per run, so that seeds stay comparable.
    """

    name: str
    layer: int
    max_seq_length: int = 2048


@dataclass(frozen=True, kw_only=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_rslora: bool = True


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    train_on_responses_only: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(frozen=True, kw_only=True)
class StepConfig:
    """One fine-tuning event: the dataset to train on and how to train on it.

    Also used to name a *probe* (see ``TrajectoryConfig.probes``), where only
    the dataset-identifying fields are read and ``train`` is ignored.
    """

    dataset: str
    version: DatasetVersion
    n_examples: int | None = None  # None means the full dataset.
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def dataset_id(self) -> str:
        """``"mistake_gsm8k/misaligned_2"``: how a dataset is named in outputs.

        The same string the runner records as a step's ``next_dataset`` and
        that the figures render via
        :func:`method.visualization.labels.display_dataset_name`. A property,
        not a field, so it stays out of ``dataclasses.asdict`` and therefore
        out of ``weights_key``.
        """
        return f"{self.dataset}/{self.version.value}"


@dataclass(frozen=True, kw_only=True)
class JudgeConfig:
    backend: JudgeBackend = JudgeBackend.OPENAI
    model: str = "gpt-4.1-mini-2025-04-14"
    max_concurrent: int = 100


@dataclass(frozen=True, kw_only=True)
class EvalConfig:
    """Settings for behaviour measurement and persona-vector extraction."""

    n_per_question: int = 10
    extract_n_per_question: int = 10
    persona_vector_threshold: int = 50
    max_tokens: int = 1000
    judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass(frozen=True, kw_only=True)
class DeltaPConfig:
    mode: DeltaPMode = DeltaPMode.FULL
    n_samples: int | None = None  # Only read when mode is SAMPLE.

    def __post_init__(self) -> None:
        if self.mode is DeltaPMode.SAMPLE and self.n_samples is None:
            raise ValueError("DeltaPConfig.mode=SAMPLE requires n_samples")


@dataclass(frozen=True, kw_only=True)
class LatentConfig:
    """Settings for the neutral probe set backing h_neutral and hence z_t."""

    n_neutral: int = 500
    neutral_prompts_name: str = "ultrachat_500"
    h_neutral_source: HNeutralSource = HNeutralSource.BASE


@dataclass(frozen=True, kw_only=True)
class TrajectoryConfig:
    """A full sequential fine-tuning run.

    Only the fields that change the *weights* participate in ``weights_id``
    (see :meth:`weights_key`); measurement settings are deliberately excluded so
    that re-measuring a checkpoint with a different judge does not force a
    retrain.
    """

    name: str
    trait: str
    model: ModelConfig
    steps: tuple[StepConfig, ...]
    seed: int = 0
    eval: EvalConfig = field(default_factory=EvalConfig)
    delta_p: DeltaPConfig = field(default_factory=DeltaPConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)

    #: Experiment family this config belongs to (``"exp2"``, ``"exp3"``, ...).
    #: Purely descriptive: it selects which runs a figure aggregates over, and
    #: never reaches :meth:`weights_key`, so tagging or re-tagging a config
    #: cannot invalidate a trained adapter.
    group: str = ""
    #: Design factors this run varies, as ``(key, value)`` pairs -- e.g.
    #: ``(("condition", "same"), ("dataset", "hallucination/misaligned_1"))``.
    #: A tuple of pairs rather than a dict because the dataclass is frozen and
    #: must stay hashable. Read it via :attr:`label_map`.
    #:
    #: These are the factors the bar charts group by. They are recorded here,
    #: where the builder that varies them already knows their values, precisely
    #: so that no downstream code has to recover them by parsing ``name`` --
    #: dataset names contain underscores, so that parse is ambiguous.
    labels: tuple[tuple[str, str], ...] = ()
    #: Datasets whose DeltaP is measured at *every* checkpoint, not just at the
    #: checkpoint that trains on them. Only the dataset-identifying fields of
    #: each :class:`StepConfig` are read; ``train`` is ignored.
    #:
    #: The runner otherwise measures DeltaP only for the dataset a step is about
    #: to train on, which gives one value per dataset per trajectory -- enough
    #: for the action features, but not enough to plot how a *fixed* dataset's
    #: DeltaP drifts as the model changes. Listing it here fills in that series.
    #:
    #: Cheap relative to what it buys: the expensive half of DeltaP is
    #: generating the base model's answers to the dataset's prompts, and that is
    #: keyed to the base checkpoint (see ``steps._base_answers_to_training_prompts``),
    #: so it is generated once and reused at every checkpoint. Each extra
    #: checkpoint costs only forward passes. Passing the *same* ``StepConfig``
    #: object that appears in ``steps`` makes the probe and the action feature
    #: resolve to one artifact, so neither is computed twice.
    probes: tuple[StepConfig, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("A trajectory needs at least one step")
        if len(dict(self.labels)) != len(self.labels):
            raise ValueError(f"duplicate label keys in {self.labels!r}")
        probe_ids = [p.dataset_id for p in self.probes]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError(f"duplicate probe datasets in {probe_ids!r}")

    @property
    def label_map(self) -> dict[str, str]:
        """:attr:`labels` as a plain dict, for lookups and DataFrame columns."""
        return dict(self.labels)

    def weights_key(self, t: int) -> dict[str, Any]:
        """The recipe that uniquely determines the weights after ``t`` steps.

        ``t=0`` is the untouched base model. Two trajectories sharing a prefix
        produce identical keys for that prefix, which is what makes adapter
        reuse across experiments work.

        ``trait`` is deliberately excluded: it never affects the LoRA weights,
        only which measurements get computed on top of them. Two configs that
        differ only in ``trait`` therefore resolve to the same ``weights_id``,
        so the same fine-tuning chain is measured under each trait instead of
        being trained once per trait. Measurement code must namespace
        trait-dependent artifacts by trait itself (see
        ``Store.trait_measurement``) since they now may share a directory.

        ``group``, ``labels`` and ``probes`` are excluded for the same reason:
        they are bookkeeping and extra measurement for the plotting code, and
        change nothing about training. Adding probes to a config whose runs
        already exist therefore costs measurements, never a retrain -- the
        adapters are found in the store and the runner skips straight past them.

        ``seed`` is normalized away at ``t=0``, where it cannot have acted on
        anything yet: with no steps taken the weights are the base model
        verbatim, so every seed must resolve to one base ``weights_id`` or each
        would re-measure identical weights under its own key. It is pinned to
        ``0`` rather than dropped so that seed-0 keys -- and the base artifacts
        already stored under them -- stay valid.

        Normalizing rather than dropping matters a second time downstream: the
        base checkpoint's measurement directory holds both the ``SAMPLE``-mode
        DeltaP subsample and M_0's answers to that subsample's prompts, which
        are compared row-for-row. Sharing one base ``weights_id`` keeps the pair
        consistent; keying either of them by seed alone would not.
        """
        if not 0 <= t <= len(self.steps):
            raise IndexError(f"step {t} out of range for {len(self.steps)} steps")
        return {
            "model": dataclasses.asdict(self.model),
            "seed": self.seed if t else 0,
            "steps": [dataclasses.asdict(s) for s in self.steps[:t]],
        }


def to_json(obj: DataclassInstance) -> str:
    """Canonical JSON for a config object, stable across runs and machines."""
    return json.dumps(
        dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
