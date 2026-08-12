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

DIVERSITY_CONDITIONS = ("baseline", "same2", "diff2", "same3", "diff3")
DIVERSITY_CONDITION_LABELS = {
    "baseline": "1 dataset",
    "same2": r"Same $\times$2",
    "diff2": r"Diverse $\times$2",
    "same3": r"Same $\times$3",
    "diff3": r"Diverse $\times$3",
}

TRAITS = ("sycophantic", "evil")

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
