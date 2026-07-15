#!/usr/bin/env python3
"""StepKV method figure: KV flow + 3-phase scoring (init / update / combine+keep)."""

from __future__ import annotations

import html
import os

C_INK = "#222222"
C_HDR = "#8EA9DB"
C_HDR_STROKE = "#4472C4"
C_WHITE = "#FFFFFF"
C_LLM_BG = "#F4F7FB"
C_LLM_STROKE = "#6C8EBF"
C_ENV = "#EEF3FA"
C_ENV_STROKE = "#4472C4"
C_PRIOR_FILL = "#F5F5F5"
C_PRIOR_STROKE = "#999999"
C_S_FILL = "#E8F4E8"
C_S_STROKE = "#548235"
C_S2_FILL = "#D4EDDA"
C_C_FILL = "#F0EBF8"
C_C_STROKE = "#7030A0"
C_KEEP_FILL = "#FFF8F0"
C_KEEP_STROKE = "#C55A11"
C_PURPLE = "#E4DFEC"
C_PURPLE_HDR = "#B4A7D6"
C_TEAL = "#2E75B6"
C_RED = "#C00000"
C_ORANGE = "#C55A11"
C_KV_T = "#BFD0EA"
C_KV_EV = "#ECECEC"

C_T_FILL, C_T_STROKE = "#F8CECC", "#B85450"
C_A_FILL, C_A_STROKE = "#DAE8FC", "#6C8EBF"
C_O_FILL, C_O_STROKE = "#D5E8D4", "#82B366"

TK_W, TK_H, TK_GAP = 12, 20, 3
PRIOR_W = 22
HDR_H = 22
STEP_PAD = 8
PHASE_H = 40
PHASE_GAP = 10

STEPS = [
    {"label": "Step 1", "t": 3, "a": 3, "o": 2, "prior": False},
    {"label": "Step 2", "t": 2, "a": 2, "o": 2, "prior": True},
    {"label": "Step 3", "t": 2, "a": 2, "o": 1, "prior": True},
    {"label": "Step N", "t": 2, "a": 2, "o": 2, "prior": True, "prune": True},
]

PHASES = [
    ("① S init", "new step finalize\nsucc+nov−rep → S_k", C_S_FILL, C_S_STROKE),
    ("② Global S update", "cite → older S↑\nrepeat → current S↓", C_S2_FILL, C_TEAL),
    ("③ T + S → C", "C = αT + βS\nkeep cache ↔ B", C_C_FILL, C_C_STROKE),
]


def esc(s: str) -> str:
    return html.escape(s)


def cell(cid: str, value: str, x: float, y: float, w: float, h: float, **style: str) -> str:
    st = ";".join(f"{k}={v}" for k, v in style.items()) + ";"
    return f"""        <mxCell id="{cid}" value="{esc(value)}" style="{st}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>"""


def txt(cid: str, value: str, x: float, y: float, w: float, h: float, **style: str) -> str:
    fs = style.pop("fontSize", "10")
    st = ";".join([f"fontSize={fs}"] + [f"{k}={v}" for k, v in style.items()]) + ";"
    return f"""        <mxCell id="{cid}" value="{esc(value)}" style="text;html=1;strokeColor=none;fillColor=none;{st}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>"""


def edge(cid: str, x1: float, y1: float, x2: float, y2: float, **style: str) -> str:
    st = ";".join(f"{k}={v}" for k, v in style.items()) + ";"
    return f"""        <mxCell id="{cid}" value="" style="{st}" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{x1}" y="{y1}" as="sourcePoint" />
            <mxPoint x="{x2}" y="{y2}" as="targetPoint" />
          </mxGeometry>
        </mxCell>"""


def lbl_edge(cid: str, x1: float, y1: float, x2: float, y2: float, label: str, **style: str) -> str:
    st = ";".join(f"{k}={v}" for k, v in style.items()) + ";"
    return f"""        <mxCell id="{cid}" value="{esc(label)}" style="{st}" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{x1}" y="{y1}" as="sourcePoint" />
            <mxPoint x="{x2}" y="{y2}" as="targetPoint" />
          </mxGeometry>
        </mxCell>"""


def token_row_width(t: int, a: int, o: int, prior: bool) -> float:
    n = t + a + o
    w = n * TK_W + max(0, n - 1) * TK_GAP
    phases = sum(1 for c in (t, a, o) if c > 0)
    w += max(0, phases - 1) * 3
    if prior:
        w += PRIOR_W + TK_GAP + 4
    return w


def render_tokens(parts: list[str], sid: str, x: float, y: float, t: int, a: int, o: int) -> dict:
    cx = x
    phases = [("t", t, C_T_FILL, C_T_STROKE), ("a", a, C_A_FILL, C_A_STROKE), ("o", o, C_O_FILL, C_O_STROKE)]
    action_boxes: list[tuple[float, float]] = []
    obs_boxes: list[tuple[float, float]] = []

    for pkey, count, fill, stroke in phases:
        for i in range(count):
            parts.append(cell(
                f"{sid}{pkey}{i}", "", cx, y, TK_W, TK_H,
                rounded="1", fillColor=fill, strokeColor=stroke, strokeWidth="1",
            ))
            mid = (cx + TK_W / 2, y + TK_H / 2)
            if pkey == "a":
                action_boxes.append(mid)
            elif pkey == "o":
                obs_boxes.append(mid)
            cx += TK_W + TK_GAP
        if pkey != "o" and count:
            parts.append(txt(f"{sid}sp{pkey}", "|", cx - 1, y + 5, 4, 10, fontSize="7", align="center", fontColor="#BBBBBB"))
            cx += 3

    return {
        "action_mid": (
            sum(p[0] for p in action_boxes) / len(action_boxes), action_boxes[0][1],
        ) if action_boxes else (x, y),
        "obs_mid": (
            sum(p[0] for p in obs_boxes) / len(obs_boxes), obs_boxes[0][1],
        ) if obs_boxes else (x, y),
    }


def render_prior_module(parts: list[str], sid: str, x: float, y: float) -> float:
    parts.append(cell(
        f"{sid}pr", "", x, y, PRIOR_W, TK_H,
        rounded="1", fillColor=C_PRIOR_FILL, strokeColor=C_PRIOR_STROKE,
        strokeWidth="1", dashed="1",
    ))
    for i, dy in enumerate((5, 10, 15)):
        parts.append(cell(f"{sid}pl{i}", "", x + 4, y + dy, PRIOR_W - 8, 2,
                          rounded="0", fillColor="#CCCCCC", strokeColor="none"))
    parts.append(txt(f"{sid}pt", "prior\nKV", x - 1, y + TK_H + 2, PRIOR_W + 2, 16,
                     fontSize="6", align="center", fontColor=C_PRIOR_STROKE))
    return x + PRIOR_W + TK_GAP + 4


def render_step(parts: list[str], sid: str, x: float, y: float, st: dict) -> dict:
    tw = token_row_width(st["t"], st["a"], st["o"], st["prior"])
    sw = tw + STEP_PAD * 2
    sh = HDR_H + STEP_PAD + TK_H + 14

    parts.append(cell(f"{sid}f", "", x, y, sw, sh,
                      rounded="1", fillColor=C_WHITE, strokeColor=C_HDR_STROKE, strokeWidth="1.5"))
    parts.append(cell(f"{sid}h", st["label"], x, y, sw, HDR_H,
                      rounded="0", fillColor=C_HDR, strokeColor=C_HDR_STROKE,
                      fontColor=C_INK, fontSize="11", fontStyle="1", align="center", html="0"))

    tx = x + STEP_PAD
    ty = y + HDR_H + STEP_PAD
    if st["prior"]:
        tx = render_prior_module(parts, sid, tx, ty)
        parts.append(txt(f"{sid}sep", "|", tx - 6, ty + 4, 6, 12, fontSize="8", align="center", fontColor="#AAAAAA"))

    tok = render_tokens(parts, sid, tx, ty, st["t"], st["a"], st["o"])
    return {"x": x, "y": y, "w": sw, "h": sh, "cx": x + sw / 2, "bottom": y + sh, "token_y": ty, **tok}


def render_three_phase_pipeline(
    parts: list[str], x: float, y: float, w: float, geos: list[dict],
) -> dict:
    """Shared 3-phase bar: S init → global S update → C combine & keep cache."""
    band_h = PHASE_H + 56
    parts.append(cell("pipeF", "", x, y, w, band_h, rounded="1", fillColor=C_PURPLE, strokeColor=C_PURPLE_HDR, strokeWidth="1.5"))
    parts.append(cell("pipeH", "3-Phase Scoring @ Step Boundary", x, y, w, 18,
                      rounded="0", fillColor=C_PURPLE_HDR, strokeColor="none",
                      fontColor=C_INK, fontSize="10", fontStyle="1", align="center", html="0"))

    pw = (w - 48 - 2 * PHASE_GAP) / 3
    py = y + 24
    phase_centers: list[float] = []

    for i, (title, sub, fill, stroke) in enumerate(PHASES):
        px = x + 16 + i * (pw + PHASE_GAP)
        phase_centers.append(px + pw / 2)
        parts.append(cell(f"ph{i}b", "", px, py, pw, PHASE_H, rounded="1", fillColor=fill, strokeColor=stroke, strokeWidth="1.25"))
        parts.append(txt(f"ph{i}t", title, px + 4, py + 3, pw - 8, 12, fontSize="8", fontStyle="1", align="center", fontColor=C_INK))
        parts.append(txt(f"ph{i}s", sub, px + 4, py + 16, pw - 8, 22, fontSize="6", align="center", fontColor=C_INK))
        if i < 2:
            mx = px + pw + PHASE_GAP / 2
            parts.append(edge(f"phE{i}", px + pw, py + PHASE_H / 2, px + pw + PHASE_GAP, py + PHASE_H / 2,
                              endArrow="block", html="1", strokeColor=C_INK, strokeWidth="1.25"))

    # Phase 3: retained vs evicted KV strip
    kx = x + 16 + 2 * (pw + PHASE_GAP)
    ky = py + PHASE_H + 8
    kw = pw
    parts.append(txt("keepL", "cache length B = ρ·n", kx, ky - 2, kw, 8, fontSize="6", align="center", fontColor=C_ORANGE, fontStyle="1"))
    n_show = 10
    tw = (kw - 2) / n_show - 1
    for j in range(n_show):
        ev = j >= 7
        parts.append(cell(f"kv{j}", "", kx + j * (tw + 1), ky + 8, tw, 10, rounded="0",
                          fillColor=C_KV_EV if ev else C_KV_T, strokeColor=C_ORANGE if ev else C_C_STROKE,
                          strokeWidth="0.75", dashed="1" if ev else "0"))
        if ev:
            parts.append(txt(f"kvx{j}", "×", kx + j * (tw + 1), ky + 8, tw, 10, fontSize="6", align="center", fontColor=C_RED))

    # Step finalize → phase 1
    for i, g in enumerate(geos):
        parts.append(edge(f"fin{i}", g["cx"], g["bottom"], phase_centers[0], py,
                          endArrow="block", html="1", strokeColor=C_S_STROKE, strokeWidth="0.75", dashed="1"))

    # Phase 1 → 2 → 3 (main flow already on boxes)

    # New step text → phase 2 (global update / cite)
    g2, g3, gN = geos[1], geos[2], geos[3]
    for j, g in ((2, g3), (3, gN)):
        parts.append(lbl_edge(
            f"cite{j}", g["cx"], g["token_y"] + TK_H / 2, phase_centers[1], py,
            "cite" if j == 2 else "",
            endArrow="block", html="1", strokeColor=C_TEAL, strokeWidth="1.5", dashed="1",
            fontSize="7", fontColor=C_TEAL, align="center",
        ))
    parts.append(lbl_edge("repE", g3["cx"], g3["bottom"] + 4, phase_centers[1], py + PHASE_H,
                          "repeat", endArrow="block", strokeColor=C_RED, strokeWidth="1", dashed="1",
                          fontSize="6", fontColor=C_RED, align="center"))

    # KV tokens → phase 3 (T)
    for i, g in enumerate(geos):
        parts.append(edge(f"tIn{i}", g["cx"], g["token_y"] + TK_H, phase_centers[2], py,
                          endArrow="block", html="1", strokeColor=C_C_STROKE, strokeWidth="0.75", dashed="1"))

    # S from phase 2 → phase 3
    parts.append(lbl_edge("sIn3", phase_centers[1], py + PHASE_H, phase_centers[2], py,
                          "S", endArrow="block", strokeColor=C_S_STROKE, strokeWidth="1.25",
                          fontSize="8", fontColor=C_S_STROKE, align="center"))

    return {"y": y, "h": band_h, "phase_centers": phase_centers, "py": py}


def build() -> tuple[list[str], float, float]:
    parts: list[str] = []
    col_gap = 28
    row_y = 88
    env_h = 28

    widths = [token_row_width(s["t"], s["a"], s["o"], s["prior"]) + STEP_PAD * 2 for s in STEPS]
    total_w = sum(widths) + col_gap * (len(STEPS) - 1)
    pw = max(total_w + 48, 640)
    start_x = (pw - total_w) / 2

    parts.append(cell("env", "Env", start_x, env_y := 24, total_w, env_h,
                      rounded="1", fillColor=C_ENV, strokeColor=C_ENV_STROKE, strokeWidth="1.5",
                      fontColor=C_INK, fontSize="11", fontStyle="1", align="center", html="0"))

    llm_y = row_y - 6
    llm_h = HDR_H + STEP_PAD + TK_H + 28
    parts.append(cell("llm", "", start_x - 4, llm_y, total_w + 8, llm_h + 8,
                      rounded="1", fillColor=C_LLM_BG, strokeColor=C_LLM_STROKE, strokeWidth="1.25", dashed="1"))
    parts.append(txt("llmL", "LLM", start_x + 4, llm_y + 4, 30, 12, fontSize="9", fontStyle="1",
                     fontColor=C_LLM_STROKE, align="left"))

    geos: list[dict] = []
    cx = start_x
    for i, st in enumerate(STEPS):
        g = render_step(parts, f"s{i}", cx, row_y, st)
        geos.append(g)
        cx += g["w"] + col_gap

    env_bot = env_y + env_h
    for i, g in enumerate(geos):
        ax, ay = g["action_mid"]
        ox, oy = g["obs_mid"]
        parts.append(edge(f"eA{i}", ax, ay - TK_H / 2, ax, env_bot,
                          endArrow="block", html="1", strokeColor=C_A_STROKE, strokeWidth="1.25"))
        parts.append(edge(f"eO{i}", ox, env_bot, ox, oy - TK_H / 2,
                          endArrow="block", html="1", strokeColor=C_O_STROKE, strokeWidth="1.25"))
        if i == 0:
            parts.append(txt("tAct", "action", ax - 16, env_bot + 2, 32, 10, fontSize="7", align="center", fontColor=C_A_STROKE))
            parts.append(txt("tObs", "obs", ox - 10, env_bot + 2, 24, 10, fontSize="7", align="center", fontColor=C_O_STROKE))

    for i in range(len(geos) - 1):
        a, b = geos[i], geos[i + 1]
        prior_off = PRIOR_W + TK_GAP + 4 if STEPS[i + 1]["prior"] else 0
        parts.append(edge(f"fl{i}", a["x"] + a["w"], a["token_y"] + TK_H / 2,
                          b["x"] + STEP_PAD + prior_off, b["token_y"] + TK_H / 2,
                          endArrow="block", html="1", strokeColor=C_INK, strokeWidth="1.5"))

    pipe_y = row_y + llm_h + 22
    pipe = render_three_phase_pipeline(parts, start_x, pipe_y, total_w, geos)

    # prior KV note
    parts.append(txt("carry", "prior KV → next step", geos[1]["x"], geos[1]["bottom"] + 2, geos[1]["w"], 10,
                     fontSize="6", align="center", fontColor=C_PRIOR_STROKE, fontStyle="2"))

    ly = pipe_y + pipe["h"] + 10
    lx = start_x
    for label, fill, stroke in [("T", C_T_FILL, C_T_STROKE), ("A", C_A_FILL, C_A_STROKE), ("O", C_O_FILL, C_O_STROKE)]:
        parts.append(cell(f"lg{label}", label, lx, ly, 14, 14, rounded="1",
                          fillColor=fill, strokeColor=stroke, strokeWidth="1", fontSize="7", align="center"))
        lx += 22
    parts.append(cell("lgS", "S", lx, ly, 14, 14, rounded="1", fillColor=C_S_FILL, strokeColor=C_S_STROKE, strokeWidth="1", fontSize="7", align="center"))
    lx += 22
    parts.append(cell("lgC", "C", lx, ly, 14, 14, rounded="1", fillColor=C_C_FILL, strokeColor=C_C_STROKE, strokeWidth="1", fontSize="7", align="center"))

    return parts, pw, ly + 22


def main() -> None:
    parts, pw, ph = build()
    xml = f"""<mxfile host="app.diagrams.net" agent="Cursor" version="22.1.0" type="device">
  <diagram id="method-kv" name="StepKV">
    <mxGraphModel dx="{int(pw)}" dy="{int(ph)}" grid="1" pageWidth="{int(pw)}" pageHeight="{int(ph)}">
      <root>
        <mxCell id="0" /><mxCell id="1" parent="0" />
{chr(10).join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "assets", "stepkv_method_main.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[OK] Wrote {out} ({int(pw)}x{int(ph)})")


if __name__ == "__main__":
    main()
