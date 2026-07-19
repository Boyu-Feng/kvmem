"""Schedule whole-step KV hard drops to simulate continuous inference interruption."""

from __future__ import annotations

from typing import Iterable, List, Set


INTERRUPT_MODES = ("none", "lag1", "lag2", "lag3", "window1", "window2", "window3")


def steps_to_drop_on_enter(
    enter_step: int,
    mode: str,
    already_dropped: Iterable[int] | None = None,
) -> List[int]:
    """
    Return newly dropped step ids when the agent *enters* ReAct step `enter_step`.

    Drops are cumulative: callers should union results into `already_dropped`.
    Step ids are 1-based (Thought/Action/Observation units).
    """
    mode = (mode or "none").strip().lower()
    if mode in ("", "none"):
        return []

    t = int(enter_step)
    if t <= 1:
        return []

    dropped = set(int(x) for x in (already_dropped or []))
    new: Set[int] = set()

    if mode.startswith("lag"):
        try:
            lag = int(mode[3:])
        except ValueError as exc:
            raise ValueError(f"Invalid interrupt mode: {mode}") from exc
        if lag <= 0:
            raise ValueError(f"lag must be positive, got {lag}")
        target = t - lag
        if target >= 1:
            new.add(target)
    elif mode.startswith("window"):
        try:
            window = int(mode[6:])
        except ValueError as exc:
            raise ValueError(f"Invalid interrupt mode: {mode}") from exc
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        # Keep at most `window` prior steps: drop everything strictly older.
        for sid in range(1, t - window):
            new.add(sid)
    else:
        raise ValueError(f"Unknown step_interrupt_mode: {mode}")

    return sorted(s for s in new if s not in dropped and s >= 1)


def describe_interrupt_mode(mode: str) -> str:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return "No forced step interruption"
    if mode.startswith("lag"):
        k = mode[3:]
        return f"On entering step t, hard-drop step t-{k} (cumulative)"
    if mode.startswith("window"):
        w = mode[6:]
        return f"On entering step t, hard-drop all steps with id < t-{w} (cumulative)"
    return mode
