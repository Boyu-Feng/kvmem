#!/usr/bin/env python3
"""Case figure: plain text steps; evicted tokens shown in red."""

from __future__ import annotations

import html
import os
import re
from typing import Dict, List, Set

PAGE_W = 780
PANEL_W = 374
PX_L, PX_R = 8, 398
PAD = 4

FS_TITLE = "11"
FS_PANEL = "10"
FS_Q = "10"
FS_STEP = "9"
FS_BODY = "9"
FS_FOOT = "8"
FS_ANS = "10"

C_EVICT = "#C44E52"
C_WRONG = "#C44E52"
C_KEEP = "#55A868"
C_ACTION = "#4C72B0"

# Glue / filler tokens StepKV may prune (~50% budget, uniform per step).
FILLER_WORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "could", "find", "for",
    "from", "has", "have", "in", "is", "it", "its", "look", "no", "not", "of", "on",
    "or", "our", "out", "that", "the", "then", "this", "to", "up", "use", "was",
    "were", "with", "without", "again", "only", "into", "their", "your", "I",
    "current", "passage", "cache", "returned", "unique", "match", "results", "anchor",
    "lost", "never", "reason", "revise", "answer", "ready", "conflicting", "evidence",
    "found", "film", "directed", "director", "missing", "remaining", "clues", "guess",
    "fragment", "major", "nearby", "city", "hope", "stays", "stay", "connected", "born",
    "2000", "pages", "unrelated", "returned", "unique", "match", "results", "anchor",
    "lost", "never", "reappears", "link", "again", "with",
}

PROTECT_ENTITIES = {
    "Snatch", "Guy", "Ritchie", "Hatfield", "Manchester", "Lookup", "Search", "Finish",
    "birthplace", "British", "crime", "comedy", "2000", "England", "Hertfordshire",
    "director", "confidence", "connected", "born",
}

QUESTION = "In which city was the director of the 2000 film Snatch born?"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def _font(color: str, text: str) -> str:
    return f"&lt;font color=&quot;{color}&quot;&gt;{text}&lt;/font&gt;"


def mark_evicted(text: str, evict_words: Set[str], *, text_color: str = "#333333") -> str:
    """Render evicted / missing tokens in red via draw.io-compatible font tags."""
    if not evict_words:
        return _font(text_color, esc(text))
    words = sorted(evict_words, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b")

    def repl(m: re.Match) -> str:
        return _font(C_EVICT, esc(m.group(0)))

    parts: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(_font(text_color, esc(text[last : m.start()])))
        parts.append(repl(m))
        last = m.end()
    if last < len(text):
        parts.append(_font(text_color, esc(text[last:])))
    return "".join(parts)


def mark_keep_only(text: str, keep_words: Set[str], *, text_color: str = "#333333") -> str:
    """Mark every word NOT in keep_words as evicted (red)."""
    tokens = set(re.findall(r"\b\w+\b", text))
    evict = tokens - keep_words
    return mark_evicted(text, evict, text_color=text_color)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text)


def pick_uniform_filler_evictions(
    text: str,
    *,
    protect: Set[str] | None = None,
    target_frac: float = 0.5,
) -> Set[str]:
    """Evict mostly filler words, ~target_frac of tokens, never protected entities."""
    protect = protect or set()
    tokens = _tokenize(text)
    if not tokens:
        return set()
    target_n = max(1, round(len(tokens) * target_frac))
    seen: Set[str] = set()
    ordered: List[str] = []
    for word in tokens:
        if word in protect or word in seen:
            continue
        seen.add(word)
        ordered.append(word)
    filler_first = [w for w in ordered if w in FILLER_WORDS]
    other = [w for w in ordered if w not in FILLER_WORDS]
    picked: List[str] = []
    for pool in (filler_first, other):
        for w in pool:
            if len(picked) >= target_n:
                break
            picked.append(w)
        if len(picked) >= target_n:
            break
    return set(picked)


def rich_cell(cid: str, html_value: str, x: int, y: int, w: int, h: int, **style) -> str:
    defaults = {
        "html": "1", "whiteSpace": "wrap", "strokeColor": "none", "fillColor": "none",
        "fontColor": "#333333", "align": "left", "verticalAlign": "top",
        "spacing": "0", "spacingTop": "0", "spacingBottom": "0",
    }
    defaults.update(style)
    st = ";".join(f"{k}={v}" for k, v in defaults.items())
    return (
        f'        <mxCell id="{cid}" value="{html_value}" style="{st};" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        f"        </mxCell>"
    )


def cell(cid: str, value: str, x: int, y: int, w: int, h: int, **style) -> str:
    return rich_cell(cid, esc(value), x, y, w, h, **style)


def box(cid: str, x: int, y: int, w: int, h: int, **style) -> str:
    defaults = {
        "rounded": "1", "whiteSpace": "wrap", "html": "1",
        "fillColor": "#FFFFFF", "strokeColor": "#CCCCCC", "strokeWidth": "1.5",
    }
    defaults.update(style)
    st = ";".join(f"{k}={v}" for k, v in defaults.items())
    return (
        f'        <mxCell id="{cid}" value="" style="{st};" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        f"        </mxCell>"
    )


# Left: Steps 1-2 heavy eviction (6-tuple keep sets); Steps 3-4 light eviction.
LEFT_STEPS = [
    (
        "Two-hop plan: identify the film's director, then find the birthplace.",
        "Search[Snatch (2000 film)]",
        "Snatch is a 2000 British crime comedy film directed by Guy Ritchie.",
        {"the", "then"},                                    # ~82% evicted
        {"2000"},
        {"is", "a", "by"},
    ),
    (
        "Use Lookup on the current passage to extract the birthplace.",
        "Lookup[born]",
        "Guy Ritchie was born in Hatfield, Hertfordshire, England.",
        {"on", "the", "to"},
        set(),
        {"was", "in"},
    ),
    (
        "Snatch and Guy Ritchie are missing from cache. Search again with the remaining clues.",
        "Search[2000 British crime comedy director]",
        "No unique match; retrieval returned unrelated film pages.",
        {
            "Snatch", "Guy", "Ritchie", "are", "missing", "cache", "Search",
            "remaining", "clues", "from", "again",
        },
        {"Search", "2000", "British", "crime", "comedy", "director"},
        {"unique", "match", "retrieval", "returned", "unrelated", "film", "pages"},
    ),
    (
        "Without Hatfield in cache, guess Manchester from the Hertfordshire fragment.",
        "Search[Manchester England director born]",
        "Results anchor on Manchester; the lost Hatfield link never reappears.",
        {"Without", "Hatfield", "cache", "guess", "Manchester", "Hertfordshire", "fragment"},
        {"Search", "Manchester", "England", "director", "born"},
        {
            "Results", "anchor", "Manchester", "lost", "Hatfield", "link",
            "never", "reappears",
        },
    ),
]

RIGHT_STEPS = [
    (
        "Find the director of Snatch, then look up the birthplace.",
        "Search[Snatch (2000 film)]",
        "Snatch is a 2000 British crime comedy film directed by Guy Ritchie.",
    ),
    (
        "Lookup birthplace in the current passage.",
        "Lookup[born]",
        "Guy Ritchie was born in Hatfield, Hertfordshire, England.",
    ),
    (
        "Snatch, Guy Ritchie, and Hatfield stay connected; finish with confidence.",
        "Finish[Hatfield]",
        "No conflicting evidence found in cache.",
    ),
    (
        "Answer ready.",
        "Finish[Hatfield]",
        "Could not find any reason to revise the answer.",
    ),
]

LEFT_ANS = ("Step 5  Final Answer", "Manchester  [WRONG]", True)
RIGHT_ANS = ("Step 5  Final Answer", "Hatfield  [CORRECT]", False)


def step_block(
    pid: str, step_i: int, thought: str, action: str, obs: str,
    px: int, y: int, pw: int, accent: str, *,
    th_keep: Set[str] | None = None,
    ac_keep: Set[str] | None = None,
    ob_keep: Set[str] | None = None,
    evict: Set[str] | None = None,
    uniform_evict: bool = False,
) -> tuple[list[str], int]:
    parts: list[str] = []
    inner_x, inner_w = px + PAD, pw - 2 * PAD
    tx, tw = inner_x + 40, inner_w - 44

    if th_keep is not None or ac_keep is not None or ob_keep is not None:
        th_body = mark_keep_only(thought, th_keep or set(), text_color="#444444")
        act_body = mark_keep_only(action, ac_keep or set(), text_color=C_ACTION)
        obs_body = mark_keep_only(obs, ob_keep or set(), text_color="#555555")
    elif uniform_evict:
        th_body = mark_evicted(
            thought,
            pick_uniform_filler_evictions(thought, protect=PROTECT_ENTITIES),
            text_color="#444444",
        )
        act_body = mark_evicted(
            action,
            pick_uniform_filler_evictions(action, protect=PROTECT_ENTITIES),
            text_color=C_ACTION,
        )
        obs_body = mark_evicted(
            obs,
            pick_uniform_filler_evictions(obs, protect=PROTECT_ENTITIES),
            text_color="#555555",
        )
    else:
        ev = evict or set()
        th_body = mark_evicted(thought, ev, text_color="#444444")
        act_body = mark_evicted(action, ev, text_color=C_ACTION)
        obs_body = mark_evicted(obs, ev, text_color="#555555")

    th_html = f"&lt;b&gt;Thought:&lt;/b&gt; {th_body}"
    act_html = f"&lt;b&gt;Action:&lt;/b&gt; {act_body}"
    obs_html = f"&lt;b&gt;Obs:&lt;/b&gt; {obs_body}"

    h = 78 if len(thought) > 85 else (70 if len(thought) > 72 else 58)
    parts.append(box(f"{pid}_bg", inner_x, y, inner_w, h, fillColor="#FAFBFD", strokeColor=accent, strokeWidth="1"))
    parts.append(cell(f"{pid}_sl", f"Step {step_i + 1}", inner_x + 3, y + 3, 36, 11,
                      fontStyle="1", fontSize=FS_STEP, fontColor=accent))
    parts.append(rich_cell(f"{pid}_th", th_html, tx, y + 3, tw, 26, fontSize=FS_BODY))
    parts.append(rich_cell(f"{pid}_ac", act_html, tx, y + 28, tw, 14, fontSize=FS_BODY))
    parts.append(rich_cell(f"{pid}_ob", obs_html, tx, y + 42, tw, 24, fontSize=FS_BODY))
    return parts, y + h + 3


def final_block(pid: str, label: str, answer: str, wrong: bool, px: int, y: int, pw: int) -> tuple[list[str], int]:
    parts: list[str] = []
    inner_x, inner_w = px + PAD, pw - 2 * PAD
    h = 28
    parts.append(box(
        f"{pid}_bg", inner_x, y, inner_w, h,
        fillColor="#F4CCCC" if wrong else "#D5E8D4",
        strokeColor=C_WRONG if wrong else C_KEEP, strokeWidth="1.5",
    ))
    parts.append(cell(f"{pid}_sl", label, inner_x + 3, y + 3, inner_w - 6, 11,
                      fontStyle="1", fontSize=FS_STEP, fontColor=C_WRONG if wrong else "#2D7600"))
    parts.append(cell(f"{pid}_ans", answer, inner_x + 3, y + 14, inner_w - 6, 12, fontStyle="1", fontSize=FS_ANS))
    return parts, y + h + 3


def render_panel(
    panel_id: str, title: str, steps: list, answer: tuple,
    px: int, pw: int, start_y: int, *, border: str, bg: str, accent: str, footer: str,
    left_mass_keep: bool = False,
) -> tuple[list[str], int]:
    parts: list[str] = []
    top = start_y
    parts.append(cell(f"{panel_id}_title", title, px + PAD, top, pw - 2 * PAD, 12,
                      fontStyle="1", fontSize=FS_PANEL, align="center"))
    y = top + 14

    for i, item in enumerate(steps):
        if left_mass_keep and len(item) == 6:
            th, ac, ob, th_k, ac_k, ob_k = item
            block, y = step_block(
                f"{panel_id}_s{i}", i, th, ac, ob, px, y, pw, accent,
                th_keep=th_k, ac_keep=ac_k, ob_keep=ob_k,
            )
        else:
            th, ac, ob = item[0], item[1], item[2]
            block, y = step_block(
                f"{panel_id}_s{i}", i, th, ac, ob, px, y, pw, accent,
                uniform_evict=not left_mass_keep,
            )
        parts.extend(block)

    block, y = final_block(panel_id, answer[0], answer[1], answer[2], px, y, pw)
    parts.extend(block)

    parts.append(cell(f"{panel_id}_foot", footer, px + PAD, y, pw - 2 * PAD, 14,
                      fontSize=FS_FOOT, fontColor="#666666", align="center"))
    y += 16

    frame = box(f"{panel_id}_frame", px, top - 2, pw, y - top + 4,
                fillColor=bg, strokeColor=border, strokeWidth="2")
    return [frame] + parts, y


def main() -> None:
    y0 = 14
    header = [
        cell("title", "Case: Token-level Eviction vs StepKV",
             8, 2, PAGE_W - 16, 11, fontStyle="1", fontSize=FS_TITLE, align="center"),
        box("qbox", 8, y0, PAGE_W - 16, 24, fillColor="#FFFFFF", strokeColor="#D0D5DD"),
        cell("qtxt", f"Q: {QUESTION}", 12, y0 + 3, PAGE_W - 24, 18, fontSize=FS_Q, fontStyle="1"),
    ]
    y = y0 + 28

    left, y_l = render_panel(
        "L", "(a) Token-only Eviction", LEFT_STEPS, LEFT_ANS,
        PX_L, PANEL_W, y, border="#AAAAAA", bg="#FAFAFA", accent="#888888",
        footer="Steps 1–2: heavy eviction; Steps 3–4: mostly kept (~50% overall).",
        left_mass_keep=True,
    )
    right, y_r = render_panel(
        "R", "(b) Ours StepKV", RIGHT_STEPS, RIGHT_ANS,
        PX_R, PANEL_W, y, border="#B4A7D6", bg="#FAF8FF", accent="#8172B3",
        footer="~50% filler tokens pruned each step; key entities stay in cache.",
    )
    page_h = max(y_l, y_r) + 12
    legend = [
        cell("leg", "Red text = evicted token",
             8, page_h - 10, PAGE_W - 16, 9, fontSize="8", fontColor=C_EVICT, align="center"),
    ]

    body = "\n".join(header + left + right + legend)
    xml = f"""<mxfile host="app.diagrams.net" agent="Cursor" version="22.1.0" type="device">
  <diagram id="snatch_case" name="Token-only vs StepKV">
    <mxGraphModel dx="780" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="0" arrows="0" fold="1" page="1" pageScale="1" pageWidth="{PAGE_W}" pageHeight="{page_h}" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "case_snatch_token_vs_stepkv.drawio",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)

    import xml.etree.ElementTree as ET
    ET.parse(out)
    print(f"[INFO] Wrote {out} ({PAGE_W}x{page_h})")


if __name__ == "__main__":
    main()
