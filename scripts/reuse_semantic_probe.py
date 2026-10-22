"""Prepare reproducible offline reuse probes without loading the QA model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache.anchor_utils import extract_anchor_terms
from run_rebuttal_legacy import atomic_json
from scripts.run_component_ablation_rebuttal import digest, file_hash, read_json

CONTROLLED_PAIRS = (
    ("same_meaning", "The pilot was born in Lisbon.", "The aviator's birthplace is Lisbon."),
    ("same_meaning", "The city hosted the 1992 Olympics.",
     "The metropolis staged the Olympic Games in 1992."),
    ("same_meaning", "The company purchased the factory in 2010.",
     "The business acquired the manufacturing plant in 2010."),
    ("same_meaning", "The physician established a clinic.",
     "The doctor founded a medical practice."),
    ("same_meaning", "The author died in Paris.", "The writer passed away in Paris."),
    ("same_meaning", "The vessel carries passengers.", "The ship transports travelers."),
    ("shared_entity_different_fact", "The pilot was born in Lisbon.",
     "The pilot received an award in Lisbon."),
    ("shared_entity_different_fact", "The company purchased the factory in 2010.",
     "The company sold a warehouse in 2010."),
    ("shared_entity_different_fact", "The author died in Paris.",
     "The author attended a conference in Paris."),
    ("shared_entity_different_fact", "The physician established a clinic.",
     "The physician studied painting."),
    ("unrelated", "The city hosted the Olympics.", "The river flooded nearby farmland."),
    ("unrelated", "The vessel carries passengers.", "The garden contains several apple trees."),
)


def lexical_pair(historical, reference):
    left, right = extract_anchor_terms(historical), extract_anchor_terms(reference)
    overlap = len(left & right)
    ratio = overlap / max(1, min(len(left), len(right)))
    return {
        "historical_anchor_count": len(left), "reference_anchor_count": len(right),
        "overlap": overlap, "overlap_ratio": ratio,
        "citation_increment": min(1., 1.5 * ratio),
    }


def prepare_pairs(checkpoint, samples=50):
    saved = read_json(checkpoint)
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = read_json(manifest_path)
    fingerprint = manifest["fingerprint"]
    protocol = fingerprint["protocol"]
    if (saved["rows_sha256"] != digest(saved["rows"])
            or saved["fingerprint_sha256"] != manifest["fingerprint_sha256"]
            or manifest["fingerprint_sha256"] != digest(fingerprint)
            or protocol["dataset"] != "hotpotqa" or protocol["seed"] != 233
            or protocol["ratio"] != .2 or len(saved["rows"]) < samples):
        raise RuntimeError("Offline source is not the complete pinned HotpotQA20 trajectory set")
    pairs = []
    for number, (kind, historical, reference) in enumerate(CONTROLLED_PAIRS):
        pairs.append({
            "id": f"controlled_{number:02d}", "source": "predefined_controlled_example",
            "label": kind, "historical_text": historical, "reference_text": reference,
            "lexical": lexical_pair(historical, reference),
            "annotation_scope": "Semantic-content stress label, not a causal agent-dependency label",
        })
    rows = saved["rows"][:samples]
    for row in rows:
        history = row["trajectory"]
        for current_index, current in enumerate(history):
            if current["step"] != current_index + 1:
                raise RuntimeError("Source trajectory steps are not contiguous")
            if current.get("action_type") == "finish" and "observation" not in current:
                if current_index != len(history) - 1:
                    raise RuntimeError("Terminal action appeared before the end of the trajectory")
                continue
            reference = current["observation"]
            if not isinstance(reference, str):
                raise RuntimeError("Source trajectory lacks its raw observation")
            for old in history[:current_index]:
                historical = "\n".join((old["observation"], old["thought"]))
                pairs.append({
                    "id": f"{row['id']}_obs{current['step']}_old{old['step']}",
                    "source": "frozen_default_trajectory",
                    "question_id": row["id"], "reference_step": current["step"],
                    "historical_step": old["step"], "historical_text": historical,
                    "reference_text": reference, "label": "unannotated",
                    "lexical": lexical_pair(historical, reference),
                    "annotation_scope": "Older steps only; newly finalized self-match excluded. "
                                        "No relevance label inferred from overlap or answer correctness.",
                })
    return {
        "version": 1, "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": file_hash(checkpoint),
        "source_fingerprint_sha256": manifest["fingerprint_sha256"],
        "sample_ids": [row["id"] for row in rows], "samples": samples,
        "protocol": {"dataset": "hotpotqa", "seed": 233, "ratio": .2,
                     "selection": "first50 seed233, already-used development diagnostic"},
        "pairs": pairs,
        "scope": "Offline citation score/content diagnostic only. No counterfactual KV "
                 "selection, causal-dependency or task-quality inference without online runs.",
    }


def score_pairs(inputs, encoder_ready, model_path, cache_path):
    from scripts.reuse_semantic_runtime import RuntimeCitationScorer
    scorer = RuntimeCitationScorer(
        condition="semantic_original", encoder_ready=encoder_ready,
        qwen_model_path=model_path, paraphrase_cache=cache_path, device="cpu",
    )
    rows = []
    try:
        for pair in inputs["pairs"]:
            scorer.reset_episode()
            historical = pair["historical_text"]
            anchors = extract_anchor_terms(historical)
            scorer.register_step(1, historical, anchors)
            increments = scorer.increments(
                pair["reference_text"], {1: {"anchors": anchors, "citation_text": historical}})
            rows.append({
                "pair_id": pair["id"], "source": pair["source"], "label": pair["label"],
                "lexical_increment": pair["lexical"]["citation_increment"],
                "semantic_increment": increments.get(1, 0.),
                "scorer_audit": scorer.audit(),
            })
    finally:
        if hasattr(scorer, "close"):
            scorer.close()
    return {
        "version": 1, "input_pairs_sha256": digest(inputs),
        "encoder_ready_sha256": file_hash(encoder_ready), "rows": rows,
        "scope": "Fixed mapping, no parameter selection. Controlled labels concern semantic "
                 "content only; real trajectory pairs remain unannotated. No causal-dependency "
                 "or counterfactual-retention accuracy claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-ready", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--score-output", type=Path)
    args = parser.parse_args()
    result = prepare_pairs(args.checkpoint)
    if args.output.exists() and read_json(args.output) != result:
        raise RuntimeError("Offline probe source changed; use a new output")
    atomic_json(args.output, result)
    print({"samples": result["samples"], "pairs": len(result["pairs"]),
           "controlled": len(CONTROLLED_PAIRS), "output": str(args.output)})
    if args.score_output is not None:
        if args.encoder_ready is None or args.model_path is None:
            parser.error("Scoring requires encoder-ready and model-path")
        scored = score_pairs(result, args.encoder_ready, args.model_path,
                             args.output.parent / "offline_unused_paraphrases.json")
        atomic_json(args.score_output, scored)
        print({"scored_pairs": len(scored["rows"]), "output": str(args.score_output)})


if __name__ == "__main__":
    main()
