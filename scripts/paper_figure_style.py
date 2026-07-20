"""Shared matplotlib sizing and typography for paper figures."""

from __future__ import annotations

from typing import Any, Sequence

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

LEGEND_BBOX_Y = 0.996
SUBPLOTS_TOP = 0.94
LEGEND_HANDLELENGTH = 2.4
LEGEND_HANDLETEXTPAD = 0.9
LEGEND_COLUMNSPACING = 1.8


def panel_height(n_panels: int) -> float:
    return FIG_H / max(1, int(n_panels))


def apply_top_legend(
    fig: matplotlib.figure.Figure,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    ncol: int,
) -> None:
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=max(1, int(ncol)),
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=LEGEND_HANDLELENGTH,
        handletextpad=LEGEND_HANDLETEXTPAD,
        columnspacing=LEGEND_COLUMNSPACING,
        borderaxespad=0.0,
    )
