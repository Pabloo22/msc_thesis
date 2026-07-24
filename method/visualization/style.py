"""Shared look-and-feel for every figure: palette, rcParams, and file output.

Figures here are meant to be dropped straight into the dissertation and an
ICLR-style paper, so the same rcParams and colors are applied everywhere
rather than left to each plotting function's defaults.

The categorical palette below (and the rule to assign hues in a fixed order,
never cycle past it) follows a colorblind-validated eight-hue sequence; slot 1
(blue) and slot 2 (orange) are used for the paired "computed at $t=0$" vs
"recomputed at $t$" series the proposal repeatedly asks for.
"""

from __future__ import annotations

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

# --- categorical palette (fixed order; do not cycle past slot 8) ----------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)

# --- chrome & ink (light chart surface only; these are print figures) -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def categorical_color(index: int) -> str:
    """The ``index``-th fixed categorical hue, wrapping past slot 8.

    Wrapping is a fallback for callers that pass more than eight series; the
    figures in this package never intentionally use more than five.
    """
    return CATEGORICAL[index % len(CATEGORICAL)]


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
