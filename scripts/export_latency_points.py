"""Export every saved latency endpoint and phase without rerunning inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qa32_rebuttal import read_json
from scripts.summarize_latency_experiment import validate_run


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            result[name] = item
    return result


def save(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    report = read_json(root / "report/SUMMARY.json")
    tables = {name: [] for name in (
        "condition_summary", "clean_repeats", "diagnostic_episodes",
        "diagnostic_phases", "memory_endpoints", "pruning_events",
    )}
    inputs = []
    for condition in report["conditions"]:
        name = f"{condition['method']}/r{int(condition['ratio']*100)}"
        directory = root / name
        protocol, rows = validate_run(directory)
        common = {"method": protocol["method"], "ratio": protocol["ratio"], "dataset": protocol["dataset"]}
        inputs.append({"condition": name, "measurements": str(directory / "measurements.json"),
                       "questions": len(rows)})
        tables["condition_summary"].append(flatten(condition))
        for row in rows:
            identity = {**common, "id": row["id"]}
            for repeat, sample in enumerate(row["clean_repeats"], 1):
                tables["clean_repeats"].append({**identity, "repeat": repeat, **flatten(sample)})
            diagnostic = row["diagnostics"]
            episode = {
                **identity, "generation_blocks": row["generation_blocks"],
                "diagnostic_wall_seconds": row["diagnostic_wall"],
                "diagnostic_unattributed_seconds": row["diagnostic_unattributed_seconds"],
                "em": row["em"], "f1": row["f1"], "parity": row["parity"],
            }
            for key in ("timing", "workload", "peak_active_kv", "peak_probe_kv",
                        "allocator_episode_peak", "model_storage", "policy",
                        "sampled_events", "omitted_events", "parent_attribution"):
                value = diagnostic.get(key)
                episode.update(flatten({key: value}))
            tables["diagnostic_episodes"].append(episode)
            scopes = [("episode", 0, diagnostic["timing"])]
            scopes.extend(("generation", index, step["timing"])
                          for index, step in enumerate(diagnostic["steps"], 1))
            for scope, index, timing in scopes:
                phases = sorted(set(timing["exclusive_seconds"]) | set(timing["inclusive_seconds"]))
                for phase in phases:
                    tables["diagnostic_phases"].append({
                        **identity, "scope": scope, "step": index, "phase": phase,
                        "exclusive_seconds": timing["exclusive_seconds"].get(phase, 0),
                        "inclusive_seconds": timing["inclusive_seconds"].get(phase, 0),
                        "calls": timing["calls"].get(phase, 0),
                        "root_seconds": timing["root_seconds"],
                    })
            for index, step in enumerate(diagnostic["steps"], 1):
                tables["memory_endpoints"].append({
                    **identity, "endpoint": "generation", "step": index,
                    **flatten(step["memory"]),
                    **flatten({key: step[key] for key in
                               ("peak_active_kv", "peak_preprune_kv", "allocator_peak")}),
                })
            for step in row["interaction_boundaries"]:
                tables["memory_endpoints"].append({
                    **identity, "endpoint": step["endpoint"], "step": step["step"],
                    **flatten(step["memory"]),
                })
            tables["memory_endpoints"].append({
                **identity, "endpoint": "episode_final", "step": row["generation_blocks"],
                **flatten(diagnostic["final"]),
            })
            for index, event in enumerate(diagnostic["pruning_events"], 1):
                tables["pruning_events"].append({**identity, "event": index, **flatten(event)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        save(output / f"{name}.csv", rows)
    index = {
        "inputs": inputs, "row_counts": {name: len(rows) for name, rows in tables.items()},
        "units": "seconds for time; all raw memory sizes are bytes; ratio is nominal not byte-matched",
        "cautions": [
            "Clean and diagnostic timings come from different passes and must not be added.",
            "Inclusive phases overlap; sum only exclusive phases within the same diagnostic scope.",
            "Gather zero may mean unisolated inline gather, not no KV copying.",
            "LazyEviction stores seven times as many heads; same ratio is not same byte budget.",
            "Snapshot tensor groups can alias; do not sum group bytes as distinct storage.",
            "All stored sampled endpoints are exported; this is not an operator-level memory trace.",
            "Lists and per-layer layouts are losslessly JSON-encoded inside CSV cells.",
        ],
    }
    (output / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index["row_counts"], indent=2))


if __name__ == "__main__":
    main()
