"""Shared matplotlib sizing and typography for paper figures."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import matplotlib.axes
import matplotlib.figure

FIG_ASPECT = 1.2
FIG_W = 16.0
FIG_H = FIG_W / FIG_ASPECT

FONT_AXIS_LABEL = 32
FONT_TICK = 28
FONT_LEGEND = 28
FONT_METHOD_TITLE = 30
FONT_ANNOT = 17
FONT_BAR_VALUE = 17

SUBPLOTS_LEFT = 0.10
SUBPLOTS_RIGHT = 0.98
SUBPLOTS_TOP = 0.88
SUBPLOTS_BOTTOM = 0.10
MULTI_PANEL_HSPACE = 0.32
LEGEND_PAD_ABOVE_AXES = 0.034
LEGEND_HANDLELENGTH = 1.8
LEGEND_HANDLETEXTPAD = 0.35
LEGEND_COLUMNSPACING = 0.45
LEGEND_LABELSPACING = 0.25
SAVE_PAD_INCHES = 0.08


def panel_height(n_panels: int) -> float:
    return FIG_H / max(1, int(n_panels))


def multi_panel_subplots_adjust(
    fig: matplotlib.figure.Figure,
    *,
    has_top_legend: bool = True,
) -> None:
    fig.subplots_adjust(
        hspace=MULTI_PANEL_HSPACE,
        bottom=SUBPLOTS_BOTTOM,
        top=SUBPLOTS_TOP if has_top_legend else 0.96,
        left=SUBPLOTS_LEFT,
        right=SUBPLOTS_RIGHT,
    )


def save_paper_figure(fig: matplotlib.figure.Figure, path: str, *, dpi: int | None = 300) -> None:
    """Save at fixed figsize so side-by-side paper figures stay consistent."""
    fig.savefig(
        path,
        bbox_inches=None,
        pad_inches=SAVE_PAD_INCHES,
        dpi=dpi if path.lower().endswith(".png") else None,
    )


def apply_top_legend(
    fig: matplotlib.figure.Figure,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    ncol: int,
    ax: Optional[matplotlib.axes.Axes] = None,
    left: float = SUBPLOTS_LEFT,
    right: float = SUBPLOTS_RIGHT,
) -> None:
    _ = (left, right)
    if not handles:
        return

    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        ncol=max(1, int(ncol)),
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=LEGEND_HANDLELENGTH,
        handletextpad=LEGEND_HANDLETEXTPAD,
        columnspacing=LEGEND_COLUMNSPACING,
        labelspacing=LEGEND_LABELSPACING,
        borderaxespad=0.0,
    )

    if ax is not None:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.018),
            bbox_transform=ax.transAxes,
            **legend_kwargs,
        )
        return

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, SUBPLOTS_TOP + LEGEND_PAD_ABOVE_AXES),
        **legend_kwargs,
    )
