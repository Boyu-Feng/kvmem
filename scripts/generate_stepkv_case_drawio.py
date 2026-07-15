#!/usr/bin/env python3
"""3-step StepKV case: fair prune trade-off + score changes with trigger words."""

from __future__ import annotations

import html
import os

ALPHA, BETA = 0.3, 0.7

# Academic figure palette (soft, print-friendly — Tableau-inspired)
C_INK = "#333333"
C_GRAPHITE = "#555555"
C_STEEL = "#888888"
C_SILVER = "#D0D5DD"
C_PAPER = "#FAFBFD"
C_MIST = "#EBF2FA"
C_FOG = "#F5F0EB"
C_BORDER = "#B8C4D4"

C_PRIMARY = "#4C72B0"
C_SECONDARY = "#55A868"
C_ACCENT = "#DD8452"
C_WARN = "#C44E52"
C_TEAL = "#5A9AA8"
C_MAUVE = "#8172B3"

C_SAGE = C_SECONDARY
C_AMBER = C_ACCENT
C_TAUPE = C_STEEL
C_ROSE = C_WARN
C_CHIP_POS = C_MIST
C_CHIP_NEG = C_FOG
C_BG_POS = C_PAPER
C_TONLY = "#958CB8"
C_TONLY_BG = "#F6F4FA"
C_SKV = "#5A9AA8"
C_SKV_BG = "#EDF6F8"

STEP_SIDEBAR = ["#6B9BC3", "#5BA3A3", "#7B8CB8"]
STEP_SIDEBAR_BG = ["#E8F0F8", "#E8F4F4", "#EEF0F8"]

STEPS = [
    {
        "label": "Step 1",
        "score": 0.50,
        "reason": "start",
        "color": STEP_SIDEBAR_BG[0],
        "stroke": STEP_SIDEBAR[0],
        "thought": [("I", 0.70, 0.50), ("need", 0.62, 0.50), ("Bilbao", 0.68, 0.50)],
        "action": [("Search", 0.62, 0.50), ("[Bilbao", 0.65, 0.50), ("museum]", 0.63, 0.50)],
        "obs": [
            ("Search", 0.14, 0.50), ("returned", 0.13, 0.50), ("pages", 0.12, 0.50),
            ("about", 0.11, 0.50), ("Bilbao", 0.18, 0.50),
        ],
        "prune": None,
    },
    {
        "label": "Step 2",
        "score": 0.92,
        "reason": "novelty ↑",
        "color": STEP_SIDEBAR_BG[1],
        "stroke": STEP_SIDEBAR[1],
        "thought": [("check", 0.55, 0.92), ("Gehry", 0.58, 0.92), ("opening", 0.52, 0.92)],
        "action": [("Search", 0.55, 0.92), ("[Gehry", 0.58, 0.92), ("Guggenheim]", 0.62, 0.92)],
        "obs": [
            ("Guggenheim", 0.25, 0.92, True),
            ("Museum", 0.20, 0.92, False),
            ("Bilbao", 0.22, 0.92, True),
            ("opened", 0.12, 0.92, False),
            ("1997", 0.18, 0.92, True),
        ],
        "prune": None,
    },
    {
        "label": "Step N",
        "score": 0.96,
        "reason": "success ✓",
        "color": STEP_SIDEBAR_BG[2],
        "stroke": STEP_SIDEBAR[2],
        "thought": [("Recall", 0.58, 0.96), ("Guggenheim", 0.62, 0.96, True), ("1997", 0.72, 0.96, True)],
        "action": [("Finish", 0.78, 0.96), ("[1997]", 0.85, 0.96, True)],
        "obs": [
            ("The", 0.16, 0.96), ("opening", 0.18, 0.96), ("year", 0.17, 0.96),
            ("is", 0.12, 0.96), ("1997", 0.28, 0.96, True),
        ],
        "prune": {
            "budget": 4,
            "tokens": [
                ("I", 0.70, 0.50, False),
                ("Search", 0.62, 0.50, False),
                ("[Bilbao", 0.65, 0.50, False),
                ("Guggenheim", 0.25, 0.92, True),
                ("Bilbao", 0.22, 0.92, True),
                ("1997", 0.18, 0.92, True),
                ("Gehry", 0.24, 0.92, True),
                ("by", 0.08, 0.92, False),
            ],
            "tok_evict": {"by", "1997", "Bilbao", "Gehry"},
            "tok_wrong": {"1997", "Bilbao", "Gehry"},
            "skv_evict": {"I", "Search", "[Bilbao", "by"},
            "skv_keep": {"Guggenheim", "Bilbao", "1997", "Gehry"},
            "note": "prune @ Step N (S₂ still 0.96; re-query step already evicted)",
        },
    },
]

ZONES = [
    {"key": "thought", "name": "Thought", "stroke": C_SILVER, "bg": C_PAPER, "hdr": "#7A8FA6"},
    {"key": "action", "name": "Action", "stroke": C_SILVER, "bg": C_PAPER, "hdr": "#8A9DB5"},
    {"key": "obs", "name": "Obs", "stroke": C_SILVER, "bg": C_PAPER, "hdr": "#6E8FA8"},
]

# Each node: Step 2 score evolution with mechanism detail
STEP2_SCORE_TRACK = [
    {
        "when": "after Step 1",
        "s": 0.50, "c": 0.50,
        "mechanism": "Step 2 interval forming; S₂ not finalized",
        "detail": "no new tokens on Step 2 yet",
        "words": [], "tag": "Step 2",
    },
    {
        "when": "Step 2 finalize",
        "s": 0.92, "c": 0.88,
        "mechanism": "reward = succ + novelty − 0.3×repeat",
        "detail": "novelty↑ on new Obs anchors (Gehry cited in Thought)",
        "words": [("Guggenheim", True), ("Bilbao", True), ("1997", True), ("Gehry", True)],
        "tag": "Step 2",
    },
    {
        "when": "Step 4 cite",
        "s": 0.96, "c": 0.91,
        "mechanism": "citation boost on OLD Step 2: +0.15·log(1+cite)",
        "detail": "re-mention in Thought raises S₂ (not retroactive repeat)",
        "words": [("Recall", True), ("Guggenheim", True), ("Bilbao", True), ("1997", True)],
        "tag": "Step 2",
    },
    {
        "when": "Step 5–6 idle",
        "s": 0.96, "c": 0.76,
        "mechanism": "S₂ frozen; token T on Step 2 decays over steps",
        "detail": "C₂ = 0.3·T + 0.7·S₂ ↓ as old tokens age out",
        "words": [], "tag": "Step 2", "track_c": True,
    },
    {
        "when": "Step 5 re-query",
        "s": 0.08, "c": 0.05,
        "mechanism": "NEW step only: repeat−0.3 + fail obs",
        "detail": "Step 5 scored separately; does NOT reduce S₂",
        "words": [("Search", False), ("Could", False), ("not", False), ("find", False)],
        "tag": "Step 5", "separate": True,
    },
    {
        "when": "Step N pre-Finish",
        "s": 0.96, "c": 0.68,
        "mechanism": "S₂ still protects anchors; C₂ low from T decay",
        "detail": "Finish/1997 tokens finalize success; cache prune next",
        "words": [("Finish", True), ("1997", True)],
        "tag": "Step 2", "track_c": True,
    },
]

FS_TITLE = 15
FS_STEP = 12
FS_ZONE = 11
FS_TOKEN = 10
FS_SMALL = 9
FS_TINY = 8
FS_MARK = 12

TOKEN_H = 22
# Main step panels: word-only tokens (scores on sidebar, not under tokens)
MAIN_TOKEN_COMPACT = True
SCORE_BLOCK = 10
COMB_C = 10
PRUNE_TOKEN_H = 18
PRUNE_SCORE_H = 8
PRUNE_ROW = PRUNE_TOKEN_H + PRUNE_SCORE_H * 3 + 12  # one row of tokens + T/S/C
PRUNE_ROW_GAP = 4


def esc(s: str) -> str:
    return html.escape(s)


def combined(t: float, s: float) -> float:
    return ALPHA * t + BETA * s


def txt(cid: str, value: str, x: int, y: int, w: int, h: int, **style: str) -> str:
    fs = style.pop("fontSize", str(FS_SMALL))
    st = ";".join([f"fontSize={fs}"] + [f"{k}={v}" for k, v in style.items()]) + ";"
    return f"""        <mxCell id="{cid}" value="{esc(value)}" style="text;html=1;strokeColor=none;fillColor=none;{st}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>"""


def box(cid: str, value: str, x: int, y: int, w: int, h: int, **style: str) -> str:
    st = ";".join(f"{k}={v}" for k, v in style.items()) + ";"
    return f"""        <mxCell id="{cid}" value="{esc(value)}" style="{st}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>"""


def token_fill(t: float) -> str:
    if t >= 0.70: return "#6B9BD1"
    if t >= 0.50: return "#9BB8DC"
    if t >= 0.30: return "#BFD0EA"
    if t >= 0.15: return "#D9E4F2"
    return "#EEF3F9"


def step_stroke(s: float) -> tuple[str, int]:
    if s >= 0.80: return C_SAGE, 2
    if s >= 0.50: return C_AMBER, 2
    return C_TAUPE, 1


def score_color_s(s: float) -> str:
    if s >= 0.80: return C_SAGE
    if s >= 0.50: return C_AMBER
    return C_TAUPE


def score_color_c(c: float) -> str:
    if c >= 0.85: return C_PRIMARY
    if c >= 0.70: return C_AMBER
    return C_TAUPE


def render_token(
    cid: str, word: str, t: float, s: float, x: int, y: int, w: int,
    evict: bool = False, keep: bool = False, wrong: bool = False,
    show_c: bool = False, trade: bool = False, compact: bool = False,
) -> list[str]:
    fill = token_fill(t)
    stroke, sw = step_stroke(s)
    out = [
        f"""        <mxCell id="{cid}_w" value="{esc(word)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={FS_TOKEN};align=center;{'fontStyle=4;' if evict else ''}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{TOKEN_H}" as="geometry" />
        </mxCell>"""
    ]
    if not compact:
        ce = COMB_C if show_c else 0
        out.append(f"""        <mxCell id="{cid}_t" value="T:{t:.2f}" style="text;html=1;strokeColor=none;fillColor=none;fontSize={FS_TINY};align=center;fontColor={C_PRIMARY};" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y + TOKEN_H + 2}" width="{w}" height="{SCORE_BLOCK}" as="geometry" />
        </mxCell>
        <mxCell id="{cid}_s" value="S:{s:.2f}" style="text;html=1;strokeColor=none;fillColor=none;fontSize={FS_TINY};align=center;fontColor={stroke};fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y + TOKEN_H + SCORE_BLOCK + 4}" width="{w}" height="{SCORE_BLOCK}" as="geometry" />
        </mxCell>""")
        if show_c:
            out.append(f"""        <mxCell id="{cid}_c" value="C:{combined(t,s):.2f}" style="text;html=1;strokeColor=none;fillColor=none;fontSize={FS_TINY};align=center;fontColor={C_MAUVE};fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y + TOKEN_H + 2 * SCORE_BLOCK + 6}" width="{w}" height="{SCORE_BLOCK}" as="geometry" />
        </mxCell>""")
        tag_y = y + TOKEN_H + SCORE_BLOCK * (3 if show_c else 2) + 8
    else:
        tag_y = y + TOKEN_H + 2

    if evict:
        out.append(f"""        <mxCell id="{cid}_x" value="✗" style="text;html=1;fontSize={FS_MARK};fontColor={C_ROSE};fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="{x + w - 10}" y="{y - 2}" width="12" height="12" as="geometry" />
        </mxCell>""")
        if wrong:
            out.append(txt(f"{cid}_warn", "⚠ critical", x, tag_y, w, 10, fontSize=str(FS_TINY), align="center", fontColor=C_ROSE))
        elif trade:
            out.append(txt(f"{cid}_tr", "T-only kept", x, tag_y, w, 10, fontSize=str(FS_TINY), align="center", fontColor=C_PRIMARY))
    elif keep:
        out.append(f"""        <mxCell id="{cid}_k" value="✓" style="text;html=1;fontSize={FS_MARK};fontColor={C_SAGE};fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="{x + w - 10}" y="{y - 2}" width="12" height="12" as="geometry" />
        </mxCell>""")
        if trade is False and show_c:
            out.append(txt(f"{cid}_sv", "S saves", x, tag_y, w, 10, fontSize=str(FS_TINY), align="center", fontColor=C_SAGE))
    return out


def token_width(word: str, min_w: int = 36) -> int:
    return max(min_w, len(word) * 7 + 14)


def _wrapped_layout(tokens: list, width: int, show_c: bool) -> int:
    if not tokens:
        return 0
    gap, pad = 5, 4
    cx, cy = pad, 2
    row_start = cx
    row_h = PRUNE_TOKEN_H + PRUNE_SCORE_H * (3 if show_c else 2) + 6
    for tok in tokens:
        word = tok[0]
        w = token_width(word)
        if cx + w > width - pad and cx > row_start:
            cx = row_start
            cy += row_h + PRUNE_ROW_GAP
        cx += w + gap
    return cy + row_h + 2


def place_tokens_wrapped(
    parts: list[str], tokens: list, x: int, y: int, width: int, prefix: str,
    evict: set[str] | None = None, keep: set[str] | None = None,
    wrong: set[str] | None = None, trade: set[str] | None = None,
    show_c: bool = False,
) -> int:
    """Place tokens with word-based widths, wrap to next row; returns total height used."""
    parsed = [(*tok[:3], tok[3] if len(tok) == 4 else False) for tok in tokens]
    if not parsed:
        return 0
    gap, pad = 5, 4
    evict, keep, wrong, trade = evict or set(), keep or set(), wrong or set(), trade or set()
    cx, cy = x + pad, y + 2
    row_h = PRUNE_TOKEN_H + PRUNE_SCORE_H * (3 if show_c else 2) + 6
    row_start = cx
    idx = 0
    for word, t, s, crit in parsed:
        w = token_width(word)
        if cx + w > x + width - pad and cx > row_start:
            cx = row_start
            cy += row_h + PRUNE_ROW_GAP
        fill = token_fill(t)
        stroke, sw = step_stroke(s)
        cid = f"{prefix}{idx}"
        parts.append(f"""        <mxCell id="{cid}_w" value="{esc(word)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={FS_TINY};align=center;{'fontStyle=4;' if word in evict else ''}" vertex="1" parent="1">
          <mxGeometry x="{cx}" y="{cy}" width="{w}" height="{PRUNE_TOKEN_H}" as="geometry" />
        </mxCell>""")
        ty = cy + PRUNE_TOKEN_H + 1
        parts.append(txt(f"{cid}_t", f"T:{t:.2f}", cx, ty, w, PRUNE_SCORE_H, fontSize="7", align="center", fontColor=C_PRIMARY))
        parts.append(txt(f"{cid}_s", f"S:{s:.2f}", cx, ty + PRUNE_SCORE_H, w, PRUNE_SCORE_H, fontSize="7", align="center", fontColor=stroke, fontStyle="1"))
        if show_c:
            parts.append(txt(f"{cid}_c", f"C:{combined(t,s):.2f}", cx, ty + PRUNE_SCORE_H * 2, w, PRUNE_SCORE_H, fontSize="7", align="center", fontColor=C_MAUVE, fontStyle="1"))
        tag_y = cy + PRUNE_TOKEN_H + PRUNE_SCORE_H * (3 if show_c else 2) + 2
        if word in evict:
            parts.append(txt(f"{cid}_x", "✗", cx + w - 10, cy - 1, 10, 10, fontSize=str(FS_MARK), align="center", fontColor=C_ROSE, fontStyle="1"))
            if word in wrong or (word in evict and crit):
                parts.append(txt(f"{cid}_warn", "⚠", cx, tag_y, w, 8, fontSize="7", align="center", fontColor=C_ROSE))
            elif word in trade:
                parts.append(txt(f"{cid}_tr", "T-k", cx, tag_y, w, 8, fontSize="7", align="center", fontColor=C_PRIMARY))
        elif word in keep and word not in evict:
            parts.append(txt(f"{cid}_k", "✓", cx + w - 10, cy - 1, 10, 10, fontSize=str(FS_MARK), align="center", fontColor=C_SAGE, fontStyle="1"))
        cx += w + gap
        idx += 1
    return cy + row_h + 2 - y


def _equal_fill_widths(n: int, width: int, gap: int = 4, pad: int = 3, min_w: int = 24) -> list[int]:
    """Divide width evenly so tokens fill the row with no trailing gap."""
    if n <= 0:
        return []
    avail = width - 2 * pad - gap * (n - 1)
    slot = max(min_w, avail // n)
    widths = [slot] * n
    remainder = avail - slot * n
    for i in range(remainder):
        widths[i] += 1
    return widths


def _single_row_layout(tokens: list, width: int, fill: bool = True) -> tuple[int, list[int]]:
    """Return (row width used, token widths); fill=True stretches to full width."""
    if not tokens:
        return 0, []
    gap, pad = 4, 3
    n = len(tokens)
    if fill:
        widths = _equal_fill_widths(n, width, gap=gap, pad=pad, min_w=24)
        return width, widths
    widths = [token_width(tok[0], min_w=28) for tok in tokens]
    total = sum(widths) + gap * (n - 1) + 2 * pad
    if total <= width:
        return total, widths
    avail = width - 2 * pad - gap * (n - 1)
    scale = avail / max(1, sum(widths))
    widths = [max(24, int(w * scale)) for w in widths]
    return sum(widths) + gap * (n - 1) + 2 * pad, widths


def place_tokens_single_row(
    parts: list[str], tokens: list, x: int, y: int, width: int, prefix: str,
    evict: set[str] | None = None, keep: set[str] | None = None,
    wrong: set[str] | None = None, trade: set[str] | None = None,
    show_c: bool = False, compact: bool = False,
) -> None:
    parsed = [(*tok[:3], tok[3] if len(tok) == 4 else False) for tok in tokens]
    if not parsed:
        return
    _, widths = _single_row_layout(parsed, width, fill=True)
    gap, pad = 4, 3
    cx = x + pad
    evict, keep, wrong, trade = evict or set(), keep or set(), wrong or set(), trade or set()
    for j, ((word, t, s, crit), w) in enumerate(zip(parsed, widths)):
        for line in render_token(
            f"{prefix}{j}", word, t, s, cx, y, w,
            evict=word in evict, keep=word in keep and word not in evict,
            wrong=word in wrong or (word in evict and crit),
            show_c=show_c, trade=word in trade, compact=compact,
        ):
            parts.append(line)
        cx += w + gap


def place_tokens_prune_row(
    parts: list[str], tokens: list, x: int, y: int, width: int, prefix: str,
    evict: set[str] | None = None, keep: set[str] | None = None,
    wrong: set[str] | None = None, trade: set[str] | None = None,
    show_c: bool = False,
) -> None:
    """Prune panel: one horizontal row, equal-width tokens filling full width."""
    parsed = [(*tok[:3], tok[3] if len(tok) == 4 else False) for tok in tokens]
    if not parsed:
        return
    gap, pad = 4, 3
    n = len(parsed)
    widths = _equal_fill_widths(n, width, gap=gap, pad=pad, min_w=28)
    cx = x + pad
    evict, keep, wrong, trade = evict or set(), keep or set(), wrong or set(), trade or set()
    for idx, ((word, t, s, crit), w) in enumerate(zip(parsed, widths)):
        fill = token_fill(t)
        stroke, _ = step_stroke(s)
        cid = f"{prefix}{idx}"
        parts.append(f"""        <mxCell id="{cid}_w" value="{esc(word)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontSize={FS_TINY};align=center;{'fontStyle=4;' if word in evict else ''}" vertex="1" parent="1">
          <mxGeometry x="{cx}" y="{y + 2}" width="{w}" height="{PRUNE_TOKEN_H}" as="geometry" />
        </mxCell>""")
        ty = y + PRUNE_TOKEN_H + 3
        parts.append(txt(f"{cid}_t", f"T:{t:.2f}", cx, ty, w, PRUNE_SCORE_H, fontSize="7", align="center", fontColor=C_PRIMARY))
        parts.append(txt(f"{cid}_s", f"S:{s:.2f}", cx, ty + PRUNE_SCORE_H, w, PRUNE_SCORE_H, fontSize="7", align="center", fontColor=stroke, fontStyle="1"))
        if show_c:
            parts.append(txt(f"{cid}_c", f"C:{combined(t,s):.2f}", cx, ty + PRUNE_SCORE_H * 2, w, PRUNE_SCORE_H, fontSize="7", align="center", fontColor=C_MAUVE, fontStyle="1"))
        if word in evict:
            parts.append(txt(f"{cid}_x", "✗", cx + w - 11, y, 11, 11, fontSize=str(FS_MARK), align="center", fontColor=C_ROSE, fontStyle="1"))
        elif word in keep and word not in evict:
            parts.append(txt(f"{cid}_k", "✓", cx + w - 11, y, 11, 11, fontSize=str(FS_MARK), align="center", fontColor=C_SAGE, fontStyle="1"))
        cx += w + gap


def place_tokens(
    parts: list[str], tokens: list, x: int, y: int, width: int, prefix: str,
    evict: set[str] | None = None, keep: set[str] | None = None,
    wrong: set[str] | None = None, trade: set[str] | None = None,
    show_c: bool = False, compact: bool = False,
) -> None:
    parsed = [(*tok[:3], tok[3] if len(tok) == 4 else False) for tok in tokens]
    if not parsed:
        return
    gap, pad = 3, 4
    slot = (width - 2 * pad - gap * (len(parsed) - 1)) // len(parsed)
    cx = x + pad
    evict, keep, wrong, trade = evict or set(), keep or set(), wrong or set(), trade or set()
    for j, (word, t, s, crit) in enumerate(parsed):
        w = max(28, slot)
        for line in render_token(
            f"{prefix}{j}", word, t, s, cx, y, w,
            evict=word in evict,
            keep=word in keep and word not in evict,
            wrong=word in wrong or (word in evict and crit),
            show_c=show_c,
            trade=word in trade,
            compact=compact,
        ):
            parts.append(line)
        cx += w + gap


def prune_tonly_h() -> int:
    return PRUNE_TOKEN_H + PRUNE_SCORE_H * 2 + 8


def prune_skv_h() -> int:
    return PRUNE_TOKEN_H + PRUNE_SCORE_H * 3 + 8


def render_prune_fair(
    parts: list[str], idx: int, prune: dict,
    sidebar_x: int, content_x: int, y: int, panel_w: int, label_w: int,
) -> int:
    """Two stacked rows aligned with Step 1: [label | full panel] × (T-only, StepKV)."""
    tokens = prune["tokens"]
    row_gap = 2
    th, sh = prune_tonly_h(), prune_skv_h()

    tok_ev = prune["tok_evict"]
    tok_keep = {t[0] for t in tokens if t[0] not in tok_ev}
    tok_wrong = prune.get("tok_wrong", set())
    skv_ev = prune["skv_evict"]
    skv_k = prune["skv_keep"]
    trade = set(prune.get("skv_evict", [])) & {t[0] for t in tokens if t[0] not in tok_ev}

    parts.append(box(
        f"ptL{idx}", "T-only", sidebar_x, y, label_w, th,
        rounded="1", fillColor=C_TONLY_BG, strokeColor=C_TONLY, fontColor=C_INK,
        fontSize=str(FS_TINY), fontStyle="1", align="center", whiteSpace="wrap", html="0",
    ))
    parts.append(box(
        f"ptB{idx}", "", content_x, y, panel_w, th,
        rounded="1", fillColor=C_TONLY_BG, strokeColor=C_TONLY, strokeWidth="1",
    ))
    place_tokens_prune_row(parts, tokens, content_x, y, panel_w, f"pt{idx}", evict=tok_ev, keep=tok_keep, wrong=tok_wrong)

    y2 = y + th + row_gap
    parts.append(box(
        f"psL{idx}", "StepKV", sidebar_x, y2, label_w, sh,
        rounded="1", fillColor=C_SKV_BG, strokeColor=C_SKV, fontColor=C_INK,
        fontSize=str(FS_TINY), fontStyle="1", align="center", whiteSpace="wrap", html="0",
    ))
    parts.append(box(
        f"psB{idx}", "", content_x, y2, panel_w, sh,
        rounded="1", fillColor=C_SKV_BG, strokeColor=C_SKV, strokeWidth="1",
    ))
    place_tokens_prune_row(parts, tokens, content_x, y2, panel_w, f"ps{idx}", evict=skv_ev, keep=skv_k, trade=trade, show_c=True)

    parts.append(txt(
        f"pleg{idx}",
        "same budget · T-only evicts 1997⚠ · StepKV evicts I/Search",
        content_x, y2 + sh + 1, panel_w, 8,
        fontSize=str(FS_TINY), align="left", fontColor=C_STEEL,
    ))
    return th + row_gap + sh + 4


def render_step2_score_track(parts: list[str], x0: int, y: int, pw: int) -> int:
    """Two rows × 3 nodes: detailed Step 2 S/C evolution with mechanism lines."""
    parts.append(txt(
        "dt",
        "Step 2 score dynamics — S = 0.85·reward + 0.15·log(1+cite),  C = 0.3·T + 0.7·S  (repeat penalizes current step only)",
        x0, y, pw, 14, fontSize=str(FS_SMALL), fontStyle="1", align="left", fontColor=C_INK,
    ))

    gap = 6
    cols = 3
    rows = 2
    node_w = (pw - gap * (cols - 1)) // cols
    hdr_h = 15
    node_h = 72
    row_gap = 6
    y_row0 = y + 16
    centers: list[tuple[int, int, int]] = []  # (cx, cy, row_idx)

    for i, pt in enumerate(STEP2_SCORE_TRACK):
        row = i // cols
        col = i % cols
        bx = x0 + col * (node_w + gap)
        by = y_row0 + row * (node_h + row_gap)
        sv, cv = float(pt["s"]), float(pt["c"])
        separate = bool(pt.get("separate"))
        hdr_fill = "#D4C8EC" if separate else "#C8D8EC"
        body_stroke = C_TONLY if separate else C_SILVER

        parts.append(box(
            f"sn{i}", "", bx, by, node_w, node_h,
            rounded="1", fillColor=C_PAPER, strokeColor=body_stroke, strokeWidth="1",
            dashed="1" if separate else "0",
        ))
        parts.append(box(
            f"snh{i}", pt.get("tag", "Step 2"), bx, by, node_w, hdr_h,
            rounded="0", fillColor=hdr_fill, strokeColor=body_stroke, strokeWidth="1",
            fontColor=C_INK, fontSize="7", fontStyle="1", align="center", whiteSpace="wrap", html="0",
        ))

        when_short = pt["when"].replace("··· ", "")
        sc_txt = f"S={sv:.2f}  C={cv:.2f}"
        if i > 0 and not separate:
            prev_idx = i - 1
            while prev_idx >= 0 and STEP2_SCORE_TRACK[prev_idx].get("separate"):
                prev_idx -= 1
            if prev_idx >= 0:
                prev = STEP2_SCORE_TRACK[prev_idx]
                ds, dc = sv - float(prev["s"]), cv - float(prev["c"])
                if abs(ds) > 0.001 or abs(dc) > 0.001:
                    ds_s = f"+{ds:.2f}" if ds >= 0 else f"{ds:.2f}"
                    dc_s = f"+{dc:.2f}" if dc >= 0 else f"{dc:.2f}"
                    sc_txt += f"   ΔS{ds_s}  ΔC{dc_s}"

        parts.append(txt(
            f"snw{i}", when_short, bx + 3, by + hdr_h + 2, node_w - 6, 11,
            fontSize=str(FS_TINY), fontStyle="1", align="left", fontColor=C_PRIMARY,
        ))
        parts.append(txt(
            f"sns{i}", sc_txt, bx + 3, by + hdr_h + 13, node_w - 6, 10,
            fontSize="7", align="left", fontColor=C_INK, fontStyle="1",
        ))
        parts.append(txt(
            f"snm{i}", pt.get("mechanism", ""), bx + 3, by + hdr_h + 24, node_w - 6, 10,
            fontSize="7", align="left", fontColor=C_GRAPHITE,
        ))
        parts.append(txt(
            f"snd{i}", pt.get("detail", ""), bx + 3, by + hdr_h + 34, node_w - 6, 10,
            fontSize="7", align="left", fontColor=C_STEEL, fontStyle="2",
        ))

        words = pt.get("words", [])
        wx = bx + 3
        wy = by + hdr_h + 46
        if not words:
            trigger = pt.get("detail", pt.get("trigger", "—"))
            parts.append(txt(
                f"sntr{i}", trigger[:40], bx + 3, wy, node_w - 6, 20,
                fontSize="7", align="left", fontColor=C_STEEL,
            ))
        else:
            for j, wd in enumerate(words):
                word, pos = wd if isinstance(wd, tuple) else (wd, True)
                chip_w = max(28, (node_w - 8 - 2 * (len(words) - 1)) // len(words))
                if wx + chip_w > bx + node_w - 3:
                    break
                fill = C_CHIP_POS if pos else C_CHIP_NEG
                stroke = C_PRIMARY if pos else C_WARN
                parts.append(box(
                    f"sc{i}{j}", word, wx, wy, chip_w, 14,
                    rounded="1", fillColor=fill, strokeColor=stroke, fontSize="7",
                    align="center", whiteSpace="wrap", html="0",
                ))
                wx += chip_w + 2

        centers.append((bx + node_w // 2, by + node_h // 2, row))

    def add_arrow(i: int, j: int, dashed: bool = False, dy: int = 0) -> None:
        ri, rj = i // cols, j // cols
        ci, cj = i % cols, j % cols
        bx_i = x0 + ci * (node_w + gap)
        bx_j = x0 + cj * (node_w + gap)
        y_i = y_row0 + ri * (node_h + row_gap) + node_h // 2 + dy
        y_j = y_row0 + rj * (node_h + row_gap) + node_h // 2 + dy
        if ri == rj and cj == ci + 1:
            x1, x2 = bx_i + node_w, bx_j
            y1, y2 = y_i, y_j
        elif ci == cols - 1 and cj == 0 and rj == ri + 1:
            x1, x2 = bx_i + node_w // 2, bx_j + node_w // 2
            y1, y2 = y_row0 + ri * (node_h + row_gap) + node_h, y_row0 + rj * (node_h + row_gap)
        elif ri == rj and cj > ci + 1:
            x1, x2 = bx_i + node_w, bx_j
            y1, y2 = y_i - 8, y_j - 8
        else:
            x1 = bx_i + node_w // 2
            x2 = bx_j + node_w // 2
            y1, y2 = y_i, y_j
        stroke = C_STEEL if dashed else C_PRIMARY
        style = f"endArrow=block;html=1;strokeColor={stroke};strokeWidth=1.5;"
        if dashed:
            style += "dashed=1;"
        parts.append(f"""        <mxCell id="sne{i}_{j}" value="" style="{style}" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{x1}" y="{y1}" as="sourcePoint" />
            <mxPoint x="{x2}" y="{y2}" as="targetPoint" />
          </mxGeometry>
        </mxCell>""")

    add_arrow(0, 1)
    add_arrow(1, 2)
    add_arrow(2, 3)
    add_arrow(3, 5, dashed=True, dy=-10)
    add_arrow(3, 4, dashed=True, dy=10)

    total_h = rows * node_h + (rows - 1) * row_gap
    return 16 + total_h + 4


def estimate_prune_h(w: int) -> int:
    return prune_tonly_h() + 2 + prune_skv_h() + 4


def main() -> None:
    page_x0 = 12
    arrow_w = 28
    label_w = 76
    content_x = arrow_w + label_w + 4
    panel_w = 780
    step_gap = 2

    zone_hdr = 16
    token_y = zone_hdr + 4  # same y for Thought / Action / Obs
    body_h = token_y + TOKEN_H + 6
    zone_ws = [panel_w * 22 // 100, panel_w * 22 // 100, panel_w - 2 * (panel_w * 22 // 100)]

    parts: list[str] = []
    parts.append(txt("title", "StepKV Case (success) — fair cache budget, T-only vs StepKV trade-off", page_x0, 6, panel_w + content_x, 18, fontSize=str(FS_TITLE), fontStyle="1", fontColor=C_INK))
    parts.append(txt("leg", "✓ kept  ✗ evicted  ⚠ wrong loss  · step S on sidebar · token T/S/C only in prune panel", page_x0 + content_x, 24, panel_w, 14, fontSize=str(FS_SMALL), align="left", fontColor=C_STEEL))

    sidebar_x = page_x0 + arrow_w
    content_start = page_x0 + content_x
    row_w = label_w + 4 + panel_w

    y = 42
    step_ys = []
    for i, step in enumerate(STEPS):
        step_ys.append(y)
        stroke_c = step.get("stroke", STEP_SIDEBAR[i])
        has_p = step.get("prune") is not None
        ph = estimate_prune_h(panel_w) if has_p else 0
        sidebar_h = body_h - 8

        parts.append(box(
            f"sl{i}", f"{step['label']}\nS={step['score']:.2f}\n{step['reason']}",
            sidebar_x, y + 4, label_w, sidebar_h,
            rounded="1", fillColor=step["color"], strokeColor=stroke_c, fontColor=C_INK,
            fontSize=str(FS_STEP), fontStyle="1", align="center", whiteSpace="wrap", html="0",
        ))
        parts.append(box(
            f"sb{i}", "", content_start, y, panel_w, body_h,
            rounded="1", fillColor="none", strokeColor=stroke_c, strokeWidth="1.5",
        ))

        zx = content_start
        oy = y + token_y
        for zi, z in enumerate(ZONES):
            zw = zone_ws[zi]
            parts.append(box(f"zb{i}_{zi}", "", zx, y, zw, body_h, rounded="0", fillColor=z["bg"], strokeColor=z["stroke"], strokeWidth="1"))
            hdr = z["name"]
            if z["key"] == "obs" and step.get("obs"):
                obs_words = " ".join(t[0] for t in step["obs"])
                hdr = f"Obs · {obs_words}"
            parts.append(box(f"zh{i}_{zi}", hdr, zx, y, zw, zone_hdr, rounded="0", fillColor=z["hdr"], fontColor="#ffffff", fontSize=str(FS_ZONE), fontStyle="1", align="center", whiteSpace="wrap", html="1"))
            place_tokens_single_row(parts, step[z["key"]], zx, oy, zw, f"s{i}z{zi}", compact=MAIN_TOKEN_COMPACT)
            zx += zw

        if has_p:
            render_prune_fair(
                parts, i, step["prune"],
                sidebar_x, content_start, y + body_h, panel_w, label_w,
            )
            y += body_h + ph + step_gap
        else:
            if i == 0:
                parts.append(txt("mid", "··· Step3 verify / Step5 repeat+fail", page_x0 + content_x + panel_w + 2, y + body_h // 2, 76, 24, fontSize=str(FS_SMALL), align="left", fontColor=C_STEEL))
            y += body_h + step_gap

    parts.append(f"""        <mxCell id="arr" value="" style="endArrow=block;html=1;strokeColor={C_PRIMARY};strokeWidth=2;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{page_x0 + arrow_w // 2}" y="{step_ys[0] + 16}" as="sourcePoint" />
            <mxPoint x="{page_x0 + arrow_w // 2}" y="{y - 4}" as="targetPoint" />
          </mxGeometry>
        </mxCell>""")

    y += 4
    parts.append(box("sep", "", page_x0, y, content_x + panel_w + 30, 1, line="1", strokeColor=C_SILVER, strokeWidth="1"))
    y += 4
    dh = render_step2_score_track(parts, sidebar_x, y, row_w)

    page_w = page_x0 + content_x + panel_w + 50
    page_h = y + dh + 12

    xml = f"""<mxfile host="app.diagrams.net" agent="Cursor" version="22.1.0" type="device">
  <diagram id="case" name="StepKV">
    <mxGraphModel dx="900" dy="700" grid="1" page="1" pageWidth="{page_w}" pageHeight="{page_h}">
      <root>
        <mxCell id="0" /><mxCell id="1" parent="0" />
{chr(10).join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "case_stepkv_step_intervals.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[OK] Wrote {out} ({page_w}x{page_h})")


if __name__ == "__main__":
    main()
