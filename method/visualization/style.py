"""Shared look-and-feel for every figure: palette, rcParams, and file output.

Figures here are meant to be dropped straight into the dissertation and an
ICLR-style paper, so the same rcParams and colors are applied everywhere
rather than left to each plotting function's defaults.

The categorical palette below is five hues assigned in a fixed order, checked
pairwise under simulated protanopia, deuteranopia and tritanopia (see the
comment above :data:`CATEGORICAL` for the method and the numbers). Slot 1
(blue) and slot 2 (orange) double as the paired "computed at $t=0$" vs
"recomputed at $t$" series the proposal repeatedly asks for.

Colour is never the only channel carrying an identity. Datasets use shape plus
a lightness ramp, and the bar charts name each arm on its own tick, so a reader
who cannot separate two hues still loses nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

# Headless by construction: every figure here is written to disk (png + pdf),
# never shown interactively, so the backend must not depend on a display
# being available (e.g. in CI or a remote dev container).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.figure import Figure  # noqa: E402
from pathlib import Path  # noqa: E402

from method.utils import REPO_ROOT  # noqa: E402

PLOTS_DIR = REPO_ROOT / "plots"

# --- categorical palette (fixed order; five slots, and five is the limit) --
# Validated by simulating protanopia, deuteranopia and tritanopia with the
# Machado, Oliveira & Fernandes (2009) severity-1.0 matrices and measuring
# every pair in OKLab: the closest pair, under any of the three, is 14.0 apart
# (x100) -- purple against plum under protanopia -- where ~12 is the point at
# which two fills stop being confusable. For scale, the best five-colour subset
# of Okabe-Ito scores 13.1 on the same test.
#
# Five slots, not eight, because five is what the figures need -- the five
# hysteresis arms, and nothing else comes close (three trunks, two projection
# series). Eight nominal hues cannot be made safe at all: the
# dichromacies collapse hue onto roughly one axis, so past about five the only
# separation left is lightness, and Okabe-Ito itself fails 8 of its 28 pairs.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#145e00"
PURPLE = "#6b139e"
PLUM = "#a94677"
CATEGORICAL = (BLUE, ORANGE, GREEN, PURPLE, PLUM)

#: Two more hues, deliberately outside :data:`CATEGORICAL` and outside the
#: guarantee above. A decay panel can hold seven projection-difference series
#: once every one of them has been measured
#: (:data:`method.visualization.figures._DECAY_SERIES`), and seven nominal hues
#: cannot be made dichromacy-safe at all. So these two are added for the panel
#: that draws all seven, not promoted into the palette: each is the on-policy
#: twin of a series already in it -- teal beside green, rust beside orange --
#: and the figures the thesis prints draw two series, never seven (see
#: :data:`method.visualization.make_plots.DECAY_GRID_SERIES`).
TEAL = "#0f6f77"
RUST = "#8a4b12"

#: Not a categorical slot: a semantic accent for "over the line" in the latent
#: audit, where it is read against :data:`BLUE` alone rather than against the
#: rest of the palette.
RED = "#e34948"

# --- chrome & ink (light chart surface only; these are print figures) -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def categorical_color(index: int) -> str:
    """The ``index``-th fixed categorical hue, wrapping past the last slot.

    Wrapping keeps an unexpected extra series plotting rather than raising, but
    it is a failure mode, not a feature: the separability guarantee in
    :data:`CATEGORICAL` holds over the five slots and nothing beyond them. A
    figure that needs more categories needs a second channel (see
    :data:`DATASET_MARKERS`), not more hues.
    """
    return CATEGORICAL[index % len(CATEGORICAL)]


# --- dataset marks: shape for the family, an ordinal ramp for the version --
# A dataset is two facts at once, and they are different *kinds* of fact, so
# they get different channels. The eight families are nominal -- Evil is not
# more or less than MATH -- so they take the shape channel, where no ordering
# is implied. The three versions are ordered by how hard they pull on the
# trait (Normal, then I, then II), so they take a single-hue ramp, which is
# the encoding an ordered category asks for and which survives colour-vision
# deficiency because it varies in lightness rather than in hue.
#
# Splitting them this way also frees the categorical hues entirely, so a
# figure can still use blue/orange for something else -- which the decay grid
# does, for the frozen and recomputed projection series.

#: Marker per dataset folder under ``dataset/``.
DATASET_MARKERS = {
    "evil": "p",  # pentagon
    "sycophancy": "v",  # triangle
    "hallucination": "d",  # rhombus
    "mistake_medical": "o",  # circle
    "insecure_code": "X",  # cross
    "mistake_gsm8k": "D",  # square on its corner
    "mistake_math": "s",  # square
    "mistake_opinions": "P",  # plus
}
#: For a dataset outside the eight the experiments use, so an unrecognised or
#: synthetic identifier still plots as *something* rather than raising.
UNKNOWN_MARKER = "*"

# The ramp, validated against this surface: lightness falls monotonically and
# adjacent steps are 22 and 32 apart in OKLab (x100), well clear of the 8
# needed to stay separable under CVD.
LIGHT_RED = "#f0a39c"
DARK_RED = "#a81f1d"

#: Fill and outline per version. Normal's fill *is* the chart surface, so it
#: reads as a hollow mark -- a white fill has no contrast against the page and
#: would otherwise vanish, which is why every mark carries an outline.
VERSION_FILL = {
    "normal": SURFACE,
    "misaligned_1": LIGHT_RED,
    "misaligned_2": DARK_RED,
}
#: Outlines are always dark, so a mark keeps a crisp silhouette whatever it is
#: filled with -- including the hollow Normal.
VERSION_EDGE = {
    "normal": SECONDARY_INK,
    "misaligned_1": DARK_RED,
    "misaligned_2": DARK_RED,
}

#: Line colour where a dataset is drawn as a series over time. This tracks the
#: *fill* rather than the outline, so that I and II stay apart at a glance
#: instead of collapsing into one red; the outline is shared between them and
#: would make six probe datasets into six identical lines.
#:
#: Normal is the exception and takes a neutral grey, because its fill is the
#: page: a white line is not a pale line, it is an absent one. The ramp
#: therefore lives in the fill, and the line only has to stay traceable.
VERSION_LINE = {
    "normal": MUTED,
    "misaligned_1": LIGHT_RED,
    "misaligned_2": DARK_RED,
}


@dataclass(frozen=True)
class DatasetMark:
    """How one ``dataset/version`` is drawn: shape, fill, outline and line."""

    marker: str
    face: str
    edge: str
    line: str


def dataset_mark(dataset_id: str) -> DatasetMark:
    """``"mistake_gsm8k/misaligned_2"`` -> the four values that draw it."""
    dataset, _, version = dataset_id.partition("/")
    return DatasetMark(
        marker=DATASET_MARKERS.get(dataset, UNKNOWN_MARKER),
        face=VERSION_FILL.get(version, SURFACE),
        edge=VERSION_EDGE.get(version, SECONDARY_INK),
        line=VERSION_LINE.get(version, MUTED),
    )


def apply_style() -> None:
    """Set matplotlib rcParams for all figures. Safe to call repeatedly."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed real (Type42) fonts, not bitmap Type3
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer-Modern-like math, no LaTeX needed
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def save_figure(fig: Figure, name: str, out_dir: Path = PLOTS_DIR) -> tuple[Path, Path]:
    """Save ``fig`` as both PNG (300 dpi) and PDF under ``out_dir``.

    Returns ``(png_path, pdf_path)``. Does not close ``fig``; callers that
    generate many figures in a loop should ``plt.close(fig)`` themselves.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    return png_path, pdf_path
