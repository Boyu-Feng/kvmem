"""Separate original-engine timing controls; do not reuse controlled_v2 outputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rebuttal_legacy_queue as queue


def main():
    queue.SOURCES = {name: queue.SOURCE_ROOT / f"{name}_source"
                     for name in ("h2o_step", "tova_step")}
    queue.RUNNER = ROOT / "run_rebuttal_legacy_step.py"
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", str(ROOT / "results/rebuttal_20260928/legacy_step_controls")])
    queue.main()


if __name__ == "__main__":
    main()
