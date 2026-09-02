r"""Human-readable display names for ``dataset/version`` identifiers.

The pipeline identifies a dataset by its folder name under ``dataset/`` (see
:mod:`method.experiments`) plus a :class:`method.config.DatasetVersion`, e.g.
``"mistake_gsm8k/misaligned_2"``. That is the right identifier for data
plumbing (it round-trips through the real store), but it is not what the
proposal calls the dataset -- it writes "GSM8K (Mistake II)". This module is
the one place that translation lives, so every figure renders the same names
as the write-up.
"""

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

#: Word placed before the "I"/"II" numeral for a misaligned version. The three
#: pure-trait datasets (evil, sycophancy, hallucination) get a bare numeral;
#: the domain-mistake datasets are qualified, and "insecure_code" uses
#: "Insecure" rather than "Mistake" since it is not a wrong-answer dataset.
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

#: Display name and bar order for the ``condition`` label of each experiment
#: family (see :attr:`method.config.TrajectoryConfig.labels`). Keys must match
#: the strings the builders in :mod:`method.experiments` write, so these dicts
#: double as the readable definition of what each condition *is*.
#: Ordered by how much training precedes the final step: none, one and two
#: steps of normal data, then the two arms whose two prior steps included
#: misalignment. Reading left to right within a group therefore separates "was
#: it fine-tuned at all" (baseline -> normal1 -> normal2) from "was it
#: *misaligned*" (normal2 -> same/diff, which are step-count-matched).
HYSTERESIS_CONDITIONS = ("baseline", "normal1", "normal2", "same", "diff")
HYSTERESIS_CONDITION_LABELS = {
    "baseline": "First exposure",
    "normal1": r"After normal $\times$1",
    "normal2": r"After normal $\times$2",
    "same": "After realign (same data)",
    "diff": "After realign (other data)",
}

#: Each arm's training schedule, in the same alphabet the exp2 trunks use:
#: ``X`` a trait-eliciting step, ``N`` a normal (re-aligning) one, ``X'`` a
#: *different* trait-eliciting dataset. Steps read left to right in training
#: order, so the rightmost ``X`` is the final step whose score the bar reports.
#:
#: The severity numeral stays out of the sequence on purpose. ``I I N`` and
#: ``II N`` are the same glyphs, so spelling the version inline would make
#: exp2's trunk B -- whose whole point is two drivers before each re-alignment
#: -- ambiguous. The dataset the bars are grouped under already names it
#: ("GSM8K (Mistake II)"), which is why ``X'`` is the useful thing to mark
#: instead: in the ``diff`` arm the other dataset's version varies by group
#: while "some other trait-eliciting set" is what every group has in common.
HYSTERESIS_CONDITION_SEQUENCES = {
    "baseline": r"$X$",
    "normal1": r"$N\,X$",
    "normal2": r"$N\,N\,X$",
    "same": r"$X\,N\,X$",
    "diff": r"$X'\,N\,X$",
}

TRAITS = ("evil", "sycophantic")

#: Trait as it should appear in a figure title.
TRAIT_TITLES = {"evil": "Evil", "sycophantic": "Sycophancy"}

#: The exp2 trunks, in the order the design presents them: a dose-response
#: ladder from the most aggressive schedule to the control.
#: Fixed here rather than taken from whatever a frame happens to contain, so
#: that a trunk keeps its colour in every figure even when another is missing
#: -- a reader who learned "A is blue" must not be repainted by a partial sweep.
TRUNKS = ("a", "b", "c")
TRUNK_LABELS = {
    "a": r"A: II drivers, $X\,N\,X\,N\,X\,N$",
    "b": r"B: I drivers, $X\,X\,N\,X\,X\,N$",
    "c": r"C: Normal only (control)",
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
#
# The symbols the Methodology chapter's notation table defines. They are built
# from parts here rather than written out at each use because every one of them
# is the same template with the same slots -- which checkpoint encoded the
# activations, which one generated the text they were read off -- and two
# figures spelling one quantity differently is exactly what this module exists
# to prevent.
#
# Each symbol is returned *without* math delimiters, because callers need it in
# both positions: an axis label wants ``$p_t^{[0]}$`` standing alone, a table
# key wants it inside a larger expression.

#: How the chapter indexes a response source: the base model is ``0`` and the
#: checkpoint being measured is ``t``. The pipeline names the same two choices
#: ``"base"`` and ``"current"`` (:class:`method.config.HNeutralSource`,
#: :class:`method.config.PredictedSource`), so every figure that takes a source
#: from the CLI translates it here.
SOURCE_INDICES = {"base": "0", "current": "t"}


def source_index(source: str) -> str:
    """``"base"`` -> ``"0"``, ``"current"`` -> ``"t"``.

    Unknown sources pass through unchanged, so a future third source reaches a
    figure as its own name rather than silently rendering as the base model.
    """
    return SOURCE_INDICES.get(source, source)


def persona_vector_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""``\mathbf{v}_{t\leftarrow g}``: encoded by $M_t$, generated by $M_g$."""
    return rf"\mathbf{{v}}_{{{encoder}\leftarrow {generator}}}"


def neutral_activation_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""``\mathbf{h}^{\mathrm{neutral}}_{t\leftarrow s}``, the neutral activation.

    ``\mathrm`` rather than the chapter's ``\text``, and ``\|`` rather than its
    ``\lVert`` below: these strings are typeset twice, by LaTeX in the emitted
    tables and by matplotlib's own mathtext in the figures (no ``usetex`` --
    see :func:`method.visualization.style.apply_style`), and mathtext knows
    neither ``\text`` nor ``\lVert``. Both spellings render identically, so the
    figures and the chapter still agree on the page.
    """
    return rf"\mathbf{{h}}^{{\mathrm{{neutral}}}}_{{{encoder}\leftarrow {generator}}}"


def neutral_norm_symbol(encoder: str = "t", generator: str = "0") -> str:
    r"""``\|\mathbf{h}^{\mathrm{neutral}}_{t\leftarrow s}\|``.

    Recorded beside $z_t$ rather than being part of it (see
    :data:`method.latent.H_NORM`): it is the length $p$ and $q$ divide by, and
    the one thing that says whether a falling cosine is the neutral state
    turning off the persona axis or merely growing in unrelated directions.
    """
    return rf"\|{neutral_activation_symbol(encoder, generator)}\|"


# ``\leftarrow`` and ``\mid`` are always followed by a space: an index can be
# the letter ``t``, and ``\leftarrowt`` is an unknown command rather than an
# arrow. The space is discarded after a control word, so nothing moves on the
# page for the numeric indices either.
def delta_p_symbol(
    *,
    encoder: str = "t",
    axis: str = "0",
    generator: str = "0",
    predicted: str = "0",
) -> str:
    r"""``\Delta P_t^{a\leftarrow g\mid p}``, the projection difference.

    Three independent choices, one slot each: ``encoder`` is the checkpoint
    whose activations the candidate dataset is read with, ``axis\leftarrow
    generator`` names the persona vector it is projected onto, and
    ``predicted`` is the checkpoint that generated the responses the targets
    are differenced against. Holding all three at the base model is $t = 0$
    itself, which the chapter writes as the bare :data:`DELTA_P_BASE`.
    """
    return rf"\Delta P_{encoder}^{{{axis}\leftarrow {generator}\mid {predicted}}}"


#: $\Delta P_0$, the shorthand the chapter defines for
#: $\Delta P_0^{0\leftarrow0\mid0}$: at the base model nothing is stale, so
#: there is no ambiguity for the indices to resolve.
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


def z_symbol(*, neutral: str = "0", persona: str = "0") -> str:
    r"""``\mathbf{z}_t^{[s,g]}``, the four coordinates collected together."""
    return rf"\mathbf{{z}}_t^{{[{neutral},{persona}]}}"
