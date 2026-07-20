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
LEGEND_BBOX_Y = 1.006
LEGEND_HANDLELENGTH = 2.0
LEGEND_HANDLETEXTPAD = 0.6
LEGEND_COLUMNSPACING = 1.0


def panel_height(n_panels: int) -> float:
    return FIG_H / max(1, int(n_panels))


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
        borderaxespad=0.0,
    )

    if ax is not None:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            bbox_transform=ax.transAxes,
            **legend_kwargs,
        )
        return

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        **legend_kwargs,
    )
