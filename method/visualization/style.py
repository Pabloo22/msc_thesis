"""Shared plotting palette, rcParams, and file output."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

# Use a headless backend for file output.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.figure import Figure  # noqa: E402
from pathlib import Path  # noqa: E402

from method.utils import REPO_ROOT  # noqa: E402

PLOTS_DIR = REPO_ROOT / "plots"

# --- categorical palette --------------------------------------------------
# Five fixed hues validated under simulated colour-vision deficiencies.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#145e00"
PURPLE = "#6b139e"
PLUM = "#a94677"
CATEGORICAL = (BLUE, ORANGE, GREEN, PURPLE, PLUM)

#: Extra hues for optional seven-series decay panels.
TEAL = "#0f6f77"
RUST = "#8a4b12"

#: Ordered ramp for base, re-encoded, and re-extracted persona vectors.
VECTOR_RAMP = ("#e86059", "#9348b1", "#00268a")

#: Semantic accent for threshold exceedance.
RED = "#e34948"

# --- chrome & ink (light chart surface only; these are print figures) -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def categorical_color(index: int) -> str:
    """Return a fixed categorical hue, wrapping after the last slot."""
    return CATEGORICAL[index % len(CATEGORICAL)]


# --- dataset marks: shape for the family, an ordinal ramp for the version -- A dataset is two facts at once, and they are different *kinds* of fact, so they get different channels.

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

#: Line colour where a dataset is drawn as a series over time.
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
