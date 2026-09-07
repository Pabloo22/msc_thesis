r"""Display labels for datasets, experiment arms, and mathematical symbols."""

from __future__ import annotations

#: Title-case display name for each dataset folder under ``dataset/``.
DATASET_TITLES = {
    "evil": "Evil",
    "sycophancy": "Sycophancy",
    "hallucination": "Hallucination",
    "mistake_medical": "Medical",
    "insecure_code": "Code",
    "mistake_gsm8k": "GSM8K",
    "mistake_math": "MATH",
    "mistake_opinions": "Opinions",
}

#: Prefix for each dataset's misalignment-level numeral.
DATASET_MODIFIERS = {
    "evil": "",
    "sycophancy": "",
    "hallucination": "",
    "mistake_medical": "Mistake",
    "insecure_code": "Insecure",
    "mistake_gsm8k": "Mistake",
    "mistake_math": "Mistake",
    "mistake_opinions": "Mistake",
}

_VERSION_NUMERALS = {"normal": None, "misaligned_1": "I", "misaligned_2": "II"}

#: Hysteresis conditions ordered by training before the final step.
HYSTERESIS_CONDITIONS = ("baseline", "normal1", "normal2", "same", "diff")
HYSTERESIS_CONDITION_LABELS = {
    "baseline": "First exposure",
    "normal1": r"After normal $\times$1",
    "normal2": r"After normal $\times$2",
    "same": "After realign (same data)",
    "diff": "After realign (other data)",
}

#: Training schedules: ``X`` trait-eliciting, ``N`` normal, ``X'`` other data.
HYSTERESIS_CONDITION_SEQUENCES = {
    "baseline": r"$X$",
    "normal1": r"$N\,X$",
    "normal2": r"$N\,N\,X$",
    "same": r"$X\,N\,X$",
    "diff": r"$X'\,N\,X$",
}

#: Short condition names used in legends.
HYSTERESIS_CONDITION_NAMES = {
    "baseline": "Baseline",
    "normal1": r"Normal $\times$1",
    "normal2": r"Normal $\times$2",
    "same": "Same",
    "diff": "Different",
}

TRAITS = ("evil", "sycophantic")

#: Trait as it should appear in a figure title.
TRAIT_TITLES = {"evil": "Evil", "sycophantic": "Sycophancy"}

#: Trunks in fixed display order.
TRUNKS = ("a", "b", "c")
TRUNK_LABELS = {
    "a": r"A: II drivers, $X\,N\,X\,N\,X\,N$",
    "b": r"B: I drivers, $X\,X\,N\,X\,X\,N$",
    "c": r"C: Normal only",
}


def display_trunk_name(trunk: str) -> str:
    """``"a"`` -> the schedule-annotated label, falling back to ``"Trunk a"``."""
    return TRUNK_LABELS.get(trunk, f"Trunk {trunk}")


def display_trunk_short(trunk: str) -> str:
    """:func:`display_trunk_name` without its schedule, for a narrow column.

    A panel row carries the schedule because it is read against its
    neighbours: which steps were misaligning is what a reader compares the
    rows on. A table row is read against a caption instead, and its key column
    has to fit beside a column per checkpoint, which the name alone does.
    """
    return display_trunk_name(trunk).split(", ")[0]


def display_trunk_title(trunk: str) -> str:
    """:func:`display_trunk_name` broken across two lines, for a panel header.

    A column header sits over about three inches of panel, which the one-line
    label overruns into its neighbours. The break goes where the label already
    divides -- the trunk's name from its schedule -- so the two lines each stay
    a complete thought, and the control (which has no schedule to name) is left
    on one line.
    """
    return display_trunk_name(trunk).replace(", ", ",\n")


def trunk_index(trunk: str) -> int:
    """Palette slot for a trunk: its position in :data:`TRUNKS`.

    Colour follows the entity, never its row number in the data, so trunk C
    stays on slot 3 whether or not A and B have finished running.
    """
    return TRUNKS.index(trunk) if trunk in TRUNKS else len(TRUNKS)


def display_trait_name(trait: str) -> str:
    """``"sycophantic"`` -> ``"Sycophancy"``, falling back to the raw string."""
    return TRAIT_TITLES.get(trait, trait.capitalize())


def condition_index(condition: str) -> int:
    """Palette slot for an exp3 arm: its position in :data:`HYSTERESIS_CONDITIONS`.

    The counterpart of :func:`trunk_index`, and for the same reason: colour
    follows the arm, never its row number in whatever frame is being drawn, so
    the Same arm keeps one hue across the bar chart, the latent audit and the
    training curves even in a figure that draws three of the five arms.
    """
    return (
        HYSTERESIS_CONDITIONS.index(condition)
        if condition in HYSTERESIS_CONDITIONS
        else len(HYSTERESIS_CONDITIONS)
    )


def display_condition_name(condition: str) -> str:
    r"""``"same"`` -> ``"Same ($X\,N\,X$)"``: the arm's name and its schedule.

    Both halves, because the two figures exp3 prints identify an arm
    differently -- the bar chart gives each arm a tick of its own carrying the
    schedule, while a curve has only a legend entry -- and a reader moving
    between them needs the one label that closes over both.
    """
    name = HYSTERESIS_CONDITION_NAMES.get(condition, condition)
    sequence = HYSTERESIS_CONDITION_SEQUENCES.get(condition)
    return f"{name} ({sequence})" if sequence else name


def display_dataset_name(dataset_id: str) -> str:
    """``"mistake_gsm8k/misaligned_2"`` -> ``"GSM8K (Mistake II)"``.

    Falls back to the raw dataset folder name (and raw version string) for
    anything outside the eight known datasets, so unrecognised or synthetic
    identifiers still render as *something* rather than raising.
    """
    dataset, _, version = dataset_id.partition("/")
    title = DATASET_TITLES.get(dataset, dataset)
    if not version:
        return title
    numeral = _VERSION_NUMERALS.get(version, version)
    if numeral is None:
        suffix = "Normal"
    else:
        modifier = DATASET_MODIFIERS.get(dataset, "")
        suffix = f"{modifier} {numeral}".strip()
    return f"{title} ({suffix})"


# --- mathematical notation --------------------------------------------- #
# Symbols omit math delimiters so callers can compose them.

#: Response-source names shared by configuration and figures.
BASE_SOURCE = "base"
CURRENT_SOURCE = "current"

#: Mathematical indices for response sources.
SOURCE_INDICES = {BASE_SOURCE: "0", CURRENT_SOURCE: "t"}


def source_index(source: str) -> str:
    """``"base"`` -> ``"0"``, ``"current"`` -> ``"t"``.

    Unknown sources pass through unchanged, so a future third source reaches a
    figure as its own name rather than silently rendering as the base model.
    """
    return SOURCE_INDICES.get(source, source)


def persona_vector_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""``\mathbf{v}_{t\leftarrow g}``: encoded by $M_t$, generated by $M_g$."""
    return rf"\mathbf{{v}}_{{{encoder}\leftarrow {generator}}}"


def activation_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""Predicted-response activation encoded by $M_t$ from $M_g$ text."""
    return rf"\mathbf{{h}}^{{\mathrm{{predicted}}}}_{{{encoder}\leftarrow {generator}}}"


def neutral_activation_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""Neutral-response activation encoded by $M_t$ from $M_s$ text."""
    return rf"\mathbf{{h}}^{{\mathrm{{neutral}}}}_{{{encoder}\leftarrow {generator}}}"


def neutral_norm_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""``\|\mathbf{h}^{\mathrm{neutral}}_{t\leftarrow s}\|``.

    Recorded beside $z_t$ rather than being part of it (see
    :data:`method.latent.H_NORM`): it is the length $p$ and $q$ divide by, and
    the one thing that says whether a falling cosine is the neutral state
    turning off the persona axis or merely growing in unrelated directions.
    """
    return rf"\|{neutral_activation_symbol(encoder, generator)}\|"


# Keep a space after ``\leftarrow`` so a following ``t`` is not parsed as part
# of the command.
def delta_p_symbol(
    *,
    encoder: str = "t",
    axis: str = "0",
    generator: str = "0",
    predicted: str = "0",
) -> str:
    r"""Return ``\Delta P_t^{a\leftarrow g,[p]}``."""
    return rf"\Delta P_{encoder}^{{{axis}\leftarrow {generator},[{predicted}]}}"


#: Shorthand for $\Delta P_0^{0\leftarrow0,[0]}$.
DELTA_P_BASE = r"\Delta P_0"

#: How each $z_t$ coordinate is written, before its indices are attached.
Z_SYMBOLS = {"p": "p", "q": "q", "rho": r"\rho", "r": "r"}

#: Which response sources each coordinate is actually a function of. $p$ reads
#: the neutral answers against the *base* axis and so does not depend on $g$;
#: $\rho$ and $r$ are properties of the persona vector alone and so do not
#: depend on $s$; only $q$ crosses both. Writing an index a coordinate does not
#: depend on would claim a variation it cannot have.
_Z_INDEX_SLOTS = {"p": ("s",), "q": ("s", "g"), "rho": ("g",), "r": ("g",)}


def z_component_symbol(
    component: str, *, neutral: str = "0", persona: str = "0"
) -> str:
    r"""``p_t^{[s]}``, ``q_t^{[s,g]}``, ``\rho_t^{[g]}`` or ``r_t^{[g]}``.

    ``neutral`` indexes the model that generated the neutral answers and
    ``persona`` the one that generated the persona-extraction responses, both
    already translated through :func:`source_index`.
    """
    slots = {"s": neutral, "g": persona}
    marks = ",".join(slots[slot] for slot in _Z_INDEX_SLOTS[component])
    return rf"{Z_SYMBOLS[component]}_t^{{[{marks}]}}"


#: Definitions for each displayed $z_t$ coordinate.
_Z_DEFINITIONS = {
    "p": r"\cos({neutral_h},{v_0})",
    "q": r"\cos({neutral_h},{v_t})",
    "rho": r"\cos({v_0},{v_t})",
    "r": r"\|{v_t}\|",
}


def z_component_definition(
    component: str, *, neutral: str = "0", persona: str = "0"
) -> str:
    r"""The right-hand side of a $z_t$ coordinate: ``\cos(\mathbf{h}...)``.

    The same two indices :func:`z_component_symbol` takes, so a symbol and its
    definition can be written side by side without the reader having to check
    that the sources agree. Only the indices the coordinate depends on appear
    in the result, because only those symbols enter its definition -- which is
    the same fact :data:`_Z_INDEX_SLOTS` records on the left-hand side.
    """
    return _Z_DEFINITIONS[component].format(
        neutral_h=neutral_activation_symbol("t", neutral),
        v_0=persona_vector_symbol("0", "0"),
        v_t=persona_vector_symbol("t", persona),
    )


def z_symbol(*, neutral: str = "0", persona: str = "0") -> str:
    r"""``\mathbf{z}_t^{[s,g]}``, the four coordinates collected together."""
    return rf"\mathbf{{z}}_t^{{[{neutral},{persona}]}}"
