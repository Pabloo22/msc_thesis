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


class PredictedSource(StrEnum):
    r"""Whose answers stand in for "what the model would have said" in DeltaP."""

    BASE = "base"
    CURRENT = "current"
    BOTH = "both"

    @property
    def sources(self) -> tuple[PredictedSource, ...]:
        """The individual sources this setting expands to."""
        if self is PredictedSource.BOTH:
            return (PredictedSource.BASE, PredictedSource.CURRENT)
        return (self,)


class ProjectionAxis(StrEnum):
    r"""Which persona vector the projection difference is taken along."""

    CURRENT = "current"
    BASE = "base"
    ONPOLICY = "onpolicy"
    BOTH = "both"

    @property
    def axes(self) -> tuple[ProjectionAxis, ...]:
        """The individual axes this setting expands to."""
        if self is ProjectionAxis.BOTH:
            return (ProjectionAxis.CURRENT, ProjectionAxis.BASE)
        return (self,)


@dataclass(frozen=True, kw_only=True)
class DeltaPView:
    r"""One projection difference: which axis, and whose predicted answers."""

    axis: ProjectionAxis = ProjectionAxis.CURRENT
    predicted: PredictedSource = PredictedSource.BASE

    def __post_init__(self) -> None:
        for setting in (self.axis, self.predicted):
            if setting in (ProjectionAxis.BOTH, PredictedSource.BOTH):
                raise ValueError(
                    f"{setting!r} names no single view; expand it through "
                    "DeltaPConfig.views"
                )

    @property
    def suffix(self) -> str:
        """Name fragment distinguishing this view, empty for the default one.

        ``v0`` rather than ``base`` for the axis, because ``base`` already
        means "$M_0$'s answers" on the other setting and one artifact name
        should not use the same word for two things.

        A view that refreshes neither setting has an empty suffix and a view
        that refreshes both is ``v0_current``, axis first: the parts are
        independent, so the name is their concatenation rather than a fourth
        word nothing could derive from the settings.

        ``ONPOLICY`` spells itself out for the same reason ``BASE`` does not
        spell itself ``base``: the word has to say which knob it moved, and
        "on-policy" is unambiguous where "current" is already taken by the
        default axis and by the other setting's answers.
        """
        parts = []
        if self.axis is ProjectionAxis.BASE:
            parts.append("v0")
        elif self.axis is ProjectionAxis.ONPOLICY:
            parts.append(self.axis.value)
        if self.predicted is PredictedSource.CURRENT:
            parts.append(self.predicted.value)
        return "_".join(parts)

    def key(self, base_key: str) -> str:
        """``base_key`` qualified for this view.

        The default view keeps the unqualified key it has always had --
        ``probes``, ``delta_p`` -- so every trajectory already on disk stays
        readable and no reader of the existing series has to learn that others
        now exist. The single place this rule lives: the runner writes these
        keys, :mod:`method.visualization.schema` reads them, and
        :mod:`method.steps` names artifacts with them.
        """
        return f"{base_key}_{self.suffix}" if self.suffix else base_key


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


class MeasurementLevel(StrEnum):
    """Select full checkpoint measurements or branch endpoint behaviour.

    ``FULL``
        Every checkpoint yields ``b_t``, ``v_t``, ``h_neutral``, ``z_t``, and
        DeltaP both for the dataset the next step trains on and for every entry
        in :attr:`TrajectoryConfig.probes`. What a trunk needs.
    ``ENDPOINT_BEHAVIOR``
        Only final-checkpoint ``b``.
    """

    FULL = "full"
    ENDPOINT_BEHAVIOR = "endpoint_behavior"


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
    """How the projection difference is estimated, on two independent axes.

    ``mode`` says which examples are projected, ``predicted`` says whose
    answers the predicted term uses. They compose: ``SAMPLE`` + ``CURRENT``
    regenerates answers for the subsample only.
    """

    mode: DeltaPMode = DeltaPMode.FULL
    n_samples: int | None = None  # Only read when mode is SAMPLE.
    predicted: PredictedSource = PredictedSource.BASE
    axis: ProjectionAxis = ProjectionAxis.CURRENT

    def __post_init__(self) -> None:
        if self.mode is DeltaPMode.SAMPLE and self.n_samples is None:
            raise ValueError("DeltaPConfig.mode=SAMPLE requires n_samples")

    @property
    def views(self) -> tuple[DeltaPView, ...]:
        """Every ``(axis, predicted)`` pair this config measures.

        The cross product, so ``BOTH`` on either setting expands here rather
        than at each call site. A config asking for ``BOTH`` on both settings
        gets the fourth, uninteresting combination too; nothing in the design
        does, and rejecting it would be a rule with no case to apply to.
        """
        return tuple(
            DeltaPView(axis=axis, predicted=predicted)
            for axis in self.axis.axes
            for predicted in self.predicted.sources
        )


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
    labels: tuple[tuple[str, str], ...] = ()
    #: Datasets whose DeltaP is measured at *every* checkpoint, not just at the checkpoint that trains on them.
    probes: tuple[StepConfig, ...] = ()
    #: Which measurements this run is for; see :class:`MeasurementLevel`.
    #: Bookkeeping like :attr:`group` and :attr:`labels`, and excluded from
    #: :meth:`weights_key` for the same reason -- it cannot force a retrain.
    measure: MeasurementLevel = MeasurementLevel.FULL

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("A trajectory needs at least one step")
        if len(dict(self.labels)) != len(self.labels):
            raise ValueError(f"duplicate label keys in {self.labels!r}")
        probe_ids = [p.dataset_id for p in self.probes]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError(f"duplicate probe datasets in {probe_ids!r}")
        if self.probes and self.measure is MeasurementLevel.ENDPOINT_BEHAVIOR:
            # Probes are measured per checkpoint, and this level visits none of
            # them. Silently dropping them would leave a config that looks like
            # it produces a DeltaP series and does not.
            raise ValueError(
                f"{self.name!r} sets probes but measure="
                f"{MeasurementLevel.ENDPOINT_BEHAVIOR.value}, which measures no "
                "checkpoint they could be read at; probe from the trunk instead"
            )

    @property
    def label_map(self) -> dict[str, str]:
        """:attr:`labels` as a plain dict, for lookups and DataFrame columns."""
        return dict(self.labels)

    def weights_key(self, t: int) -> dict[str, Any]:
        r"""The recipe that uniquely determines the weights after ``t`` steps."""
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
