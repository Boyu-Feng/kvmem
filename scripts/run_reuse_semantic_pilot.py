#!/usr/bin/env python3
"""HotpotQA-only online citation pilot; all auxiliary work is accuracy-only.

The frozen evaluator, generation input, reward, selector and stopping logic remain
intact. Only scorer-only conditions replace its citation increment calculation.
Paraphrasing permits at most three strict numeric-diff-guided attempts, with all
provider attempt costs and diagnostics retained. Planning never loads models.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_component_ablation_rebuttal as worker

CONDITIONS = (
    "lexical_original", "no_citation", "semantic_original",
    "lexical_paraphrase", "semantic_paraphrase",
)
PARAPHRASE_RETRY_PROTOCOL = {
    "max_attempts": 3,
    "retry_strategy": "numeric_diff_guided_v1",
    "retry_scope": "paraphrase scorer-only; never main generation",
    "failure_policy": "strict rejection after three attempts; no original-text fallback",
}
CITATION_GUARD = (
    'pruning_mode == "step_aware_h2o" and use_step_scores and step_meta and not ablate_citation'
)
CITATION_BODY = '''
ref_terms = extract_anchor_terms(new_text)
if ref_terms:
    for sid, meta in step_meta.items():
        anchors = meta.get("anchors", set())
        if not anchors:
            continue
        overlap = len(ref_terms.intersection(anchors))
        if overlap <= 0:
            continue
        base = float(min(len(ref_terms), len(anchors)))
        overlap_ratio = float(overlap) / max(1.0, base)
        citation_inc = min(1.0, overlap_ratio * 1.5)
        meta["citation"] = float(meta.get("citation", 0.0)) + float(citation_inc)
        step_scores[sid] = _compute_step_external_score(
            float(meta.get("reward", 0.0)), float(meta.get("citation", 0.0)))
if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_scores"):
    llm.kv_manager.update_step_scores(step_scores)
_print_step_scores_summary()
'''
REPLACEMENT = '''
for sid, citation_inc in _reuse_increments(new_text, step_meta).items():
    meta = step_meta[sid]
    meta["citation"] = float(meta.get("citation", 0.0)) + float(citation_inc)
    step_scores[sid] = _compute_step_external_score(
        float(meta.get("reward", 0.0)), float(meta.get("citation", 0.0)))
'''


def source_text(function):
    return textwrap.dedent(function if isinstance(function, str) else inspect.getsource(function))


def unique(tree, predicate, description):
    matches = [node for node in ast.walk(tree) if predicate(node)]
    if len(matches) != 1:
        raise RuntimeError(f"Frozen AST changed: {description} (found {len(matches)})")
    return matches[0]


def node_equal(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def insert_after(tree, target, statements):
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if item is target:
                        value[index + 1:index + 1] = statements
                        return
    raise RuntimeError("Frozen AST insertion point missing")


def episode_tree(function, condition):
    """Fail closed on exact citation structure; never reconstruct the QA loop."""
    if condition not in CONDITIONS:
        raise ValueError("Unsupported pilot condition")
    tree = ast.parse(source_text(function))
    anchor = unique(tree, lambda n: isinstance(n, ast.Assign) and
                    any(ast.unparse(t) == "step_meta[prev_step]['anchors']" for t in n.targets),
                    "original anchors assignment")
    if ast.unparse(anchor.value) != "prev_anchor_terms":
        raise RuntimeError("Frozen anchors expression changed")
    anchor_text = unique(tree, lambda n: isinstance(n, ast.Assign) and
                         any(ast.unparse(t) == "prev_anchor_text" for t in n.targets),
                         "observation/thought source text")
    if ast.unparse(anchor_text.value) != r"'\n'.join((prev_obs_text, prev_thought_text))":
        raise RuntimeError("Frozen observation/thought source text changed")
    score = unique(tree, lambda n: isinstance(n, ast.FunctionDef) and
                   n.name == "_compute_step_external_score", "external score function")
    reward = unique(score, lambda n: isinstance(n, ast.Assign) and
                    any(ast.unparse(t) == "reward" for t in n.targets), "original reward cap")
    returned = unique(score, lambda n: isinstance(n, ast.Return), "original score cap")
    if (ast.unparse(reward.value) != "max(-1.0, min(2.0, float(reward_val)))" or
            ast.unparse(returned.value) != "float(max(0.0, min(8.0, score)))"):
        raise RuntimeError("Frozen reward/score clipping expression changed")
    guard = ast.parse(f"if {CITATION_GUARD}:\n    pass").body[0].test
    branch = unique(tree, lambda n: isinstance(n, ast.If) and node_equal(n.test, guard),
                    "actual citation branch")
    expected = ast.parse(textwrap.dedent(CITATION_BODY)).body
    if branch.orelse or not node_equal(ast.Module(body=branch.body, type_ignores=[]),
                                      ast.Module(body=expected, type_ignores=[])):
        raise RuntimeError("Frozen citation expressions/loop changed")
    generation = unique(tree, lambda n: isinstance(n, ast.Assign) and
                        isinstance(n.value, ast.Call) and
                        ast.unparse(n.value.func) == "llm.generate_incremental" and
                        [ast.unparse(a) for a in n.value.args] == ["new_text"],
                        "unchanged main incremental input")
    if ast.unparse(generation.value) != (
        "llm.generate_incremental(new_text, max_new_tokens=256, stop_strings=kv_stop_strings)"
    ):
        raise RuntimeError("Frozen generation settings changed")
    insert_after(tree, anchor, ast.parse(
        'step_meta[prev_step]["citation_text"] = prev_anchor_text\n'
        '_reuse_register_step(prev_step, prev_anchor_text, prev_anchor_terms)'
    ).body)
    if condition not in ("lexical_original", "no_citation"):
        branch.body[:2] = ast.parse(textwrap.dedent(REPLACEMENT)).body
    elif condition == "lexical_original":
        citation = unique(branch, lambda n: isinstance(n, ast.Assign) and
                          any(ast.unparse(t) == "citation_inc" for t in n.targets),
                          "lexical increment")
        insert_after(tree, citation, ast.parse(
            "_reuse_lexical_increment(sid, citation_inc)"
        ).body)
    if condition != "no_citation":
        branch.body.insert(0, ast.parse("_reuse_citation_round(new_text)").body[0])
    # Record the literal argument immediately before the frozen generation call.
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if item is generation:
                        value.insert(index, ast.parse("_reuse_main_input(new_text)").body[0])
                        ast.fix_missing_locations(tree)
                        return tree
    raise RuntimeError("Frozen generation statement missing")


def ast_proof(function, condition):
    original = ast.parse(source_text(function))
    changed = episode_tree(function, condition)
    return {
        "original_source_sha256": hashlib.sha256(source_text(function).encode()).hexdigest(),
        "original_ast_sha256": worker.digest(ast.dump(original)),
        "instrumented_ast_sha256": worker.digest(ast.dump(changed)),
        "citation_loop_policy": "original" if condition in ("lexical_original", "no_citation")
        else "scorer-only increment replacement",
        "main_call": "llm.generate_incremental(new_text, max_new_tokens=256, stop_strings=kv_stop_strings)",
        "preserved": ["reward", "novelty", "repeat", "token_attention", "floor",
                      "score_caps", "QA_loop", "posthoc_stop"],
    }


def validate_args(args):
    expected = "no_cite" if args.condition == "no_citation" else "full"
    if args.dataset != "hotpotqa" or args.seed != 233 or args.variant != expected:
        raise ValueError(f"Pilot requires HotpotQA seed233 variant={expected}")
    if "semantic" in args.condition and args.encoder_ready is None:
        raise ValueError("Semantic conditions require --encoder-ready")
    if "paraphrase" in args.condition and args.paraphrase_cache is None:
        raise ValueError("Paraphrase conditions require --paraphrase-cache")


def resource_protocol(condition, encoder_ready, paraphrase_cache):
    def path(value):
        return str(Path(value).resolve()) if value is not None else None
    result = {
        "encoder_ready": path(encoder_ready) if "semantic" in condition else None,
        "encoder_ready_sha256": worker.file_hash(encoder_ready)
        if "semantic" in condition and encoder_ready is not None and Path(encoder_ready).is_file() else None,
        "paraphrase_cache": path(paraphrase_cache) if "paraphrase" in condition else None,
        "auxiliary_generation_config": {
            "model": "local pinned Qwen/Qwen2.5-7B-Instruct",
            "scope": "scorer-only; exact settings bound by runtime fingerprint and record hashes",
            "main_generation_config": "original raw HF configuration, unchanged",
            "do_sample": False, "num_beams": 1, "max_new_tokens": 512,
            "max_source_tokens": 512, "use_cache": True, "dtype": "bfloat16",
            **PARAPHRASE_RETRY_PROTOCOL,
        } if "paraphrase" in condition else None,
        **PARAPHRASE_RETRY_PROTOCOL,
    }
    result["encoder_protocol_fingerprint"] = worker.digest({
        "condition": condition, "encoder_ready": result["encoder_ready"],
        "encoder_ready_sha256": result["encoder_ready_sha256"],
        "runtime_contract": "RuntimeCitationScorer; runtime fingerprint also recorded per episode",
    })
    result["scorer_protocol_fingerprint"] = worker.digest(result)
    return result


def validate_runtime_fingerprint(fingerprint, condition):
    """Require effective auxiliary settings, rather than guessing provider defaults."""
    if not isinstance(fingerprint, dict):
        raise RuntimeError("Runtime fingerprint must be a dictionary")
    if "paraphrase" not in condition:
        return
    generation = fingerprint.get("paraphrase_generation")
    expected = {
        "max_attempts": 3, "max_new_tokens": 512, "max_source_tokens": 512,
        "do_sample": False, "num_beams": 1, "dtype": "bfloat16",
    }
    if not isinstance(generation, dict) or any(
        type(generation.get(key)) is not type(value) or generation.get(key) != value
        for key, value in expected.items()
    ):
        raise RuntimeError("Runtime paraphrase generation fingerprint differs from fixed pilot")
    # The provider exposes a full strategy description; the worker protocol
    # separately pins the approved numeric_diff_guided_v1 identifier.
    if not isinstance(generation.get("retry_strategy"), str) or not generation["retry_strategy"].strip():
        raise RuntimeError("Runtime paraphrase retry strategy missing from fingerprint")


@contextlib.contextmanager
def condition_scope(condition, encoder_ready=None, paraphrase_cache=None, *,
                    execute=False, qwen_model_path=None, runtime_factory=None):
    """Process-local patches only; the original worker owns execution/checkpoints."""
    if condition not in CONDITIONS:
        raise ValueError("Unsupported pilot condition")
    base = importlib.import_module("run_all_wiki_experiments_v2")
    original_episode = base._run_react_kv_episode
    proof = ast_proof(original_episode, condition)
    tree = episode_tree(original_episode, condition)
    originals = {name: getattr(worker, name) for name in (
        "component_config", "protocol", "select_samples", "source_hashes",
        "run_episode", "load_assets", "validate_rows")}
    runtime = None
    model_path = qwen_model_path
    records = {}

    def provider():
        nonlocal runtime
        if condition in ("lexical_original", "no_citation"):
            return None
        if not execute:
            raise RuntimeError("Runtime scorer forbidden during plan stage")
        if runtime is None:
            factory = runtime_factory
            if factory is None:
                factory = importlib.import_module("scripts.reuse_semantic_runtime").RuntimeCitationScorer
            if model_path is None:
                raise RuntimeError("Pinned local Qwen model path unavailable")
            runtime = factory(condition, encoder_ready=encoder_ready,
                              qwen_model_path=Path(model_path),
                              paraphrase_cache=paraphrase_cache, device="cuda:0")
            validate_runtime_fingerprint(runtime.fingerprint(), condition)
        return runtime

    def register(sid, text, anchors):
        start = time.perf_counter()
        records["registered_steps"].append({
            "step_id": int(sid), "original_text": text,
            "original_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "original_anchor_set": sorted(anchors),
        })
        scorer = provider()
        if scorer is not None:
            scorer.register_step(sid, text, anchors)
        records["scorer_hook_seconds"] += time.perf_counter() - start

    def increments(text, meta):
        import math
        start = time.perf_counter()
        result = provider().increments(text, meta)
        if (not isinstance(result, dict) or any(type(sid) is not int or sid not in meta
                or not isinstance(inc, (float, int)) or not math.isfinite(inc)
                or not 0 <= inc <= 1 for sid, inc in result.items())):
            raise RuntimeError("Invalid runtime citation increments")
        records["citation_rounds"][-1]["increments"] = dict(result)
        records["scorer_hook_seconds"] += time.perf_counter() - start
        return result

    def citation_round(text):
        records["citation_rounds"].append({
            "current_original_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "increments": {},
        })

    def lexical_increment(sid, inc):
        records["citation_rounds"][-1]["increments"][sid] = float(inc)

    def main_input(text):
        records["main_incremental_inputs"].append({
            "utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "characters": len(text), "policy": "literal original new_text, including newlines",
        })

    namespace = dict(original_episode.__globals__)
    namespace.update(_reuse_register_step=register, _reuse_increments=increments,
                     _reuse_lexical_increment=lexical_increment,
                     _reuse_citation_round=citation_round, _reuse_main_input=main_input)
    exec(compile(tree, original_episode.__code__.co_filename, "exec"), namespace)
    instrumented = namespace[original_episode.__name__]

    def config(module, variant):
        expected = "no_cite" if condition == "no_citation" else "full"
        if variant != expected:
            raise ValueError("Pilot component variant mismatch")
        result = originals["component_config"](module, expected)
        result["cache_ratio"] = .2
        return result

    def protocol(args):
        if args.dataset != "hotpotqa" or args.seed != 233:
            raise ValueError("Pilot requires HotpotQA seed233")
        expected = "no_cite" if condition == "no_citation" else "full"
        if args.variant != expected:
            raise ValueError("Pilot component variant mismatch")
        result = originals["protocol"](args)
        result.update(
            pilot_condition=condition, samples=50 if args.purpose == "main" else 2,
            pilot_main_samples=50, ratio=.2, budget_percent=20,
            selection_policy="first50 of original seed233 main500; smoke positions500/501 (zero-based)",
            mainagent_source_text_policy="unchanged",
            scorer_text_policy="paraphrase scorer-only" if "paraphrase" in condition else "original",
            timing_scope="accuracy-only; extra encoder/auxiliary generation cost is not latency",
            runtime_scorer_contract="scripts.reuse_semantic_runtime.RuntimeCitationScorer",
            ast_proof=proof, **resource_protocol(condition, encoder_ready, paraphrase_cache),
        )
        return result

    def selected(module, rows, **kwargs):
        if kwargs["seed"] != 233:
            raise ValueError("Pilot selection requires seed233")
        result = originals["select_samples"](module, rows, **kwargs)
        return result[:50] if kwargs["purpose"] == "main" else result

    def hashes():
        names = set()
        for pattern in ("reuse_semantic_*.py", "run_reuse_semantic_*.py"):
            names.update((ROOT / "scripts").glob(pattern))
        return {**originals["source_hashes"](),
                **{str(p.relative_to(ROOT)): worker.file_hash(p) for p in sorted(names)}}

    def assets(args):
        nonlocal model_path
        result = originals["load_assets"](args)
        if result[3]["model_repo"] != "Qwen/Qwen2.5-7B-Instruct":
            raise RuntimeError("Pilot requires pinned Qwen7B main model")
        model_path = result[0]
        return result

    def run_episode(*args, **kwargs):
        records.clear()
        records.update(registered_steps=[], citation_rounds=[], main_incremental_inputs=[],
                       scorer_hook_seconds=0.)
        scorer = provider()
        scorer_fingerprint = copy.deepcopy(scorer.fingerprint()) if scorer is not None else None
        if scorer is not None:
            validate_runtime_fingerprint(scorer_fingerprint, condition)
        if scorer is not None:
            scorer.reset_episode()
        result = originals["run_episode"](*args, **kwargs)
        if scorer is not None and scorer.fingerprint() != scorer_fingerprint:
            raise RuntimeError("Runtime scorer fingerprint changed during episode")
        result["reuse_scoring_audit"] = copy.deepcopy({
            **records, "pilot_condition": condition,
            "mainagent_source_text_policy": "unchanged",
            "scope": "accuracy-only; encoder/paraphrase overhead is not latency",
            "paraphrase_retry_protocol": dict(PARAPHRASE_RETRY_PROTOCOL)
            if "paraphrase" in condition else None,
            "ast_proof": proof,
            "counts": {"registered_steps": len(records["registered_steps"]),
                       "citation_rounds": len(records["citation_rounds"]),
                       "increments": sum(len(r["increments"]) for r in records["citation_rounds"]),
                       "main_incremental_calls": len(records["main_incremental_inputs"])},
            "runtime": scorer.audit() if scorer is not None else {
                "runtime_citation_calls": len(records["citation_rounds"]),
                "encoder_calls": 0, "paraphrase_calls": 0,
            },
            "runtime_fingerprint": scorer_fingerprint if scorer is not None else
            {"mode": condition, "citation_loop": "frozen original"},
        })
        return result

    def validate_rows(rows, fingerprint):
        originals["validate_rows"](rows, fingerprint)
        for row in rows:
            audit = row.get("reuse_scoring_audit")
            if (not isinstance(audit, dict) or audit.get("pilot_condition") != condition
                    or audit.get("ast_proof") != proof
                    or audit.get("mainagent_source_text_policy") != "unchanged"
                    or not isinstance(audit.get("runtime_fingerprint"), dict)
                    or audit.get("paraphrase_retry_protocol") != (
                        PARAPHRASE_RETRY_PROTOCOL if "paraphrase" in condition else None)):
                raise RuntimeError("Missing/mismatched reuse scoring audit")
            if condition == "no_citation" and audit.get("citation_rounds") != []:
                raise RuntimeError("No-citation control recorded a runtime citation call")
            validate_runtime_fingerprint(audit["runtime_fingerprint"], condition)

    with contextlib.ExitStack() as scope:
        for name, value in (
            ("component_config", config), ("protocol", protocol), ("select_samples", selected),
            ("source_hashes", hashes), ("run_episode", run_episode), ("load_assets", assets),
            ("validate_rows", validate_rows),
        ):
            scope.enter_context(worker.patch_attribute(worker, name, value))
        scope.enter_context(worker.patch_attribute(base, "_run_react_kv_episode", instrumented))
        try:
            yield
        finally:
            if runtime is not None and hasattr(runtime, "close"):
                runtime.close()


def validate_completed_run(output):
    spec = worker.read_json(Path(output) / "manifest.json")["fingerprint"]["protocol"]
    with condition_scope(spec["pilot_condition"],
                         Path(spec["encoder_ready"]) if spec["encoder_ready"] else None,
                         Path(spec["paraphrase_cache"]) if spec["paraphrase_cache"] else None):
        return worker.validate_completed_run(output)


def parser():
    result = worker.parser()
    result.add_argument("--condition", choices=CONDITIONS, required=True)
    result.add_argument("--encoder-ready", type=Path)
    result.add_argument("--paraphrase-cache", type=Path)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        cli.error(str(error))
    with condition_scope(args.condition, args.encoder_ready, args.paraphrase_cache,
                         execute=args.execute):
        if not args.execute:
            plan = worker.prepare_plan(args)
            plan["kv_override"]["cache_ratio"] = .2
            print(worker.canonical(plan))
        else:
            # Pass only the original CLI; its strict error watcher/store are reused.
            forwarded = ["--execute", "--dataset", args.dataset, "--seed", str(args.seed),
                         "--variant", args.variant, "--purpose", args.purpose,
                         "--model-ready", str(args.model_ready), "--data-ready", str(args.data_ready),
                         "--output", str(args.output)]
            worker.main(forwarded)


if __name__ == "__main__":
    main()
