"""
Unified Experiment Runner: All 6 Methods on HotpotQA with Full Wikipedia Corpus

Methods:
1. Single Model (direct QA, no retrieval)
2. RAG (Retrieve-then-Read with Wikipedia BM25)
3. ReAct (multi-step reasoning with Wikipedia BM25)
4. ReAct-KV (none) - ReAct with KV cache reuse, no pruning
5. ReAct-KV (h2o) - ReAct with KV cache + H2O pruning
6. ReAct-KV (snapkv) - ReAct with KV cache + SnapKV pruning
7. Ours - ReAct with KV cache + our proposed pruning method

Uses:
- HotpotQA distractor validation set (7405 samples), 500 random samples (seed=233, matching original paper)
- Full Wikipedia corpus retrieval via BM25 (TIGER-Lab/LongRAG ~5.2M articles)
- Qwen2.5-7B-Instruct as the base LLM

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment single
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment rag
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react_kv_none
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react_kv_h2o
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react_kv_snapkv
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment collect  # generate summary
"""

import json
import os
import re
import math
import time
import random
import argparse
import gc
from collections import Counter
from token_tracker import TokenTracker
from kv_cache.anchor_utils import extract_anchor_terms

# ==================== Configuration ====================
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "results/wiki_0318_v3"
WIKI_INDEX_DIR = "data/wiki_index"
DATA_CACHE_DIR = "data/hotpotqa"
SUMMARY_PATH = "final_summary_0406_v3.md"

# Match original ReAct paper: seed=233, first 500 from shuffled 7405 dev samples
NUM_SAMPLES = 500
RANDOM_SEED = 233

MAX_STEPS = 7
BM25_TOP_K = 5
CHECKPOINT_INTERVAL = 5


def parse_max_steps(value):
    """
    Parse --max_steps CLI value.
    Returns None for unlimited (run until finish), else a positive int cap.
    """
    if value is None:
        return MAX_STEPS
    if isinstance(value, int):
        return None if value < 0 else value
    s = str(value).strip().lower()
    if s in ("unlimited", "none", "inf", "infinity", "-1", "no_limit", "nolimit"):
        return None
    return int(s)


def format_max_steps(step_limit):
    """Serialize max_steps for JSON summaries."""
    return "unlimited" if step_limit is None else int(step_limit)


def _episode_step_limit(max_steps=None):
    """Resolve per-episode step cap: explicit arg overrides global MAX_STEPS."""
    if max_steps is not None:
        return max_steps
    return MAX_STEPS


_REACT_KV_CHECKPOINT_DROP_KEYS = ("trajectory", "debug_payload", "step_timings")


def _slim_react_kv_checkpoint_results(results):
    """Drop heavy per-sample fields from periodic checkpoints."""
    slimmed = []
    for r in results:
        slim = {k: v for k, v in r.items() if k not in _REACT_KV_CHECKPOINT_DROP_KEYS}
        slimmed.append(slim)
    return slimmed


def _atomic_write_json(path, obj, indent=None):
    """Write JSON atomically and keep a .bak copy of the previous checkpoint."""
    tmp_path = f"{path}.tmp"
    bak_path = f"{path}.bak"
    dump_kwargs = {"ensure_ascii": False}
    if indent is not None:
        dump_kwargs["indent"] = indent
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **dump_kwargs)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, bak_path)
        except OSError:
            pass
    os.replace(tmp_path, path)


def _load_checkpoint_results(checkpoint_path, label="checkpoint"):
    """Load checkpoint JSON; fall back to .bak; quarantine corrupt files."""
    last_err = None
    for cp in (checkpoint_path, f"{checkpoint_path}.bak"):
        if not os.path.exists(cp):
            continue
        try:
            with open(cp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise json.JSONDecodeError("checkpoint root must be a JSON list", "", 0)
            if cp != checkpoint_path:
                print(f"[WARN] Primary checkpoint unreadable; loaded backup: {cp}")
            return data
        except (json.JSONDecodeError, OSError, ValueError) as e:
            last_err = e
            try:
                size_gb = os.path.getsize(cp) / (1024 ** 3)
            except OSError:
                size_gb = 0.0
            corrupt_path = f"{cp}.corrupt_{int(time.time())}"
            try:
                os.rename(cp, corrupt_path)
                print(
                    f"[WARN] Corrupt {label} ({size_gb:.2f} GB) quarantined to "
                    f"{corrupt_path}: {e}"
                )
            except OSError as move_err:
                print(f"[WARN] Corrupt {label} ({size_gb:.2f} GB): {e} (move failed: {move_err})")
    if last_err is not None:
        print(f"[WARN] No valid {label} found; starting from scratch.")
    return []


def _save_checkpoint(checkpoint_path, results, slim_react_kv=False):
    payload = _slim_react_kv_checkpoint_results(results) if slim_react_kv else results
    _atomic_write_json(checkpoint_path, payload)
    print(f"[CHECKPOINT] Saved {len(payload)} samples -> {checkpoint_path}")


def _filter_stale_checkpoint_results(results, selected_samples):
    selected_ids = {s["id"] for _, s in selected_samples}
    before = len(results)
    filtered = [r for r in results if r.get("id") in selected_ids]
    if len(filtered) != before:
        print(
            f"[WARN] Dropped {before - len(filtered)} stale checkpoint entries "
            f"not in the current sample set (seed changed?)."
        )
    return filtered


def _legacy_kv_path_without_ratio_tag(path: str) -> str:
    """
    Map ratio-tagged artifact names back to the pre-refactor layout.
    e.g. react_kv_h2o_musique_r50_checkpoint.json -> react_kv_h2o_musique_checkpoint.json
    """
    directory, filename = os.path.split(path)
    legacy_name = re.sub(r"_r\d+(?=_checkpoint\.json$|\.json$)", "", filename)
    if legacy_name == filename:
        return path
    return os.path.join(directory, legacy_name)


def _resolve_existing_kv_path(primary_path: str, label: str) -> str:
    """Prefer primary path; fall back to legacy filename without _r50/_r20 tag."""
    for candidate in (primary_path, _legacy_kv_path_without_ratio_tag(primary_path)):
        if candidate and os.path.exists(candidate):
            if candidate != primary_path:
                print(f"[INFO] Using legacy {label}: {candidate}")
            return candidate
    return primary_path


def _react_kv_output_is_complete(output_path):
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        return summary.get("total_samples", 0) > 0 and "exact_match" in summary
    except (json.JSONDecodeError, OSError, ValueError):
        return False


def _react_kv_output_is_complete_any(output_path: str) -> bool:
    if _react_kv_output_is_complete(output_path):
        return True
    legacy = _legacy_kv_path_without_ratio_tag(output_path)
    if legacy != output_path:
        return _react_kv_output_is_complete(legacy)
    return False


def _finalize_react_kv_output(
    results,
    output_path,
    pruning_mode,
    kv_config,
    metrics_dataset,
    metrics_method=None,
    model_param_mb=0.0,
    total_time=None,
):
    """Write final react-kv JSON + metrics from completed results (no GPU)."""
    import torch

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    if total_time is None:
        total_time = sum(float(r.get("sample_time", 0) or 0) for r in results)
    final_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0.0
    final_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0.0

    cache_lens = []
    for r in results:
        srt = r.get("step_remaining_tokens", [])
        if len(srt) > 1:
            cache_lens.extend(srt[1:])
        else:
            prompt_len = int(r.get("prompt_token_count", 0))
            for t in r.get("step_timings", []):
                cache_lens.append(_decode_cache_len(t.get("kv_cache_length", 0), prompt_len))
    timing_stats = {}
    if cache_lens:
        total_gen_time = sum(
            t.get("generation_time", 0)
            for r in results
            for t in r.get("step_timings", [])
        )
        timing_stats = {
            "avg_step_time": total_gen_time / len(cache_lens),
            "total_generation_time": total_gen_time,
            "total_steps": len(cache_lens),
            "avg_kv_cache_length": sum(cache_lens) / len(cache_lens),
            "max_kv_cache_length": max(cache_lens),
        }

    total_prune_count = sum(
        r.get("llm_stats", {}).get("total_prune_count", 0) for r in results
    )
    output_results = _slim_react_kv_checkpoint_results(results)
    output_data = {
        "summary": {
            "method": f"ReAct-KV ({pruning_mode}, Full Wiki)",
            "model": MODEL_PATH,
            "max_steps": format_max_steps(MAX_STEPS),
            "pruning_mode": pruning_mode,
            "kv_config": kv_config,
            "total_samples": len(em_scores),
            "exact_match": final_em,
            "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores) if em_scores else 0.0,
            "timing_stats": timing_stats,
            "total_prune_count": total_prune_count,
            "model_param_mb": model_param_mb,
            **_summarize_peak_memory(results),
            **_summarize_decode_cache_and_time(results),
            **_summarize_step_counts(results),
        },
        "results": output_results,
    }
    _atomic_write_json(output_path, output_data, indent=2)

    step_stats = output_data["summary"].get("step_count_distribution", {})
    if step_stats:
        print(f"[INFO] Step distribution: {step_stats}")

    method_name = _metrics_method_name(pruning_mode, metrics_method)
    cache_ratio_str = ""
    if pruning_mode not in ("none", "snapkv", "ours"):
        cache_ratio_str = str(kv_config.get("cache_ratio", ""))
    _auto_write_metrics_md(
        output_path,
        metrics_dataset,
        method_name,
        cache_ratio=cache_ratio_str,
    )

    print(
        f"\n{'='*60}\nReAct-KV ({pruning_mode}) Complete: EM={final_em:.2f}%, "
        f"F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return final_em, final_f1, total_time


# ==================== Evaluation Metrics ====================
def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ==================== Data Loading ====================
def load_hotpotqa_data():
    """Load HotpotQA distractor validation set."""
    from datasets import load_dataset
    print("[INFO] Loading HotpotQA distractor validation set...")
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        cache_dir=DATA_CACHE_DIR,
    )
    val_data = ds["validation"]
    print(f"[INFO] Loaded {len(val_data)} validation examples.")
    return val_data


def select_samples(val_data):
    """
    Select NUM_SAMPLES unique samples (by sample id) matching original ReAct paper:
    shuffle all indices with RANDOM_SEED, take unique ids until NUM_SAMPLES reached.
    """
    total = len(val_data)
    idxs = list(range(total))
    random.Random(RANDOM_SEED).shuffle(idxs)

    selected = []
    seen_ids = set()
    skipped_dup = 0
    for idx in idxs:
        sample = val_data[idx]
        sid = sample.get("id", str(idx))
        if sid in seen_ids:
            skipped_dup += 1
            continue
        selected.append((idx, sample))
        seen_ids.add(sid)
        if len(selected) >= NUM_SAMPLES:
            break

    n_selected = len(selected)
    print(
        f"[INFO] Selected {n_selected} samples (seed={RANDOM_SEED}) from {total} total "
        f"({len(seen_ids)} unique ids)."
    )
    if skipped_dup:
        print(
            f"[INFO] Skipped {skipped_dup} duplicate ids while building the sample list."
        )
    if n_selected < NUM_SAMPLES:
        print(
            f"[WARN] Requested {NUM_SAMPLES} samples but only {n_selected} unique ids are available "
            f"after loading/normalization; using {n_selected}. "
            "Re-download MuSiQue with download_datasets.py (expect >=2000 validation rows) "
            "or pass --data_path to a complete dev.json."
        )
    return selected


# ==================== ReAct Prompt ====================
REACT_PROMPT_TEMPLATE = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
{examples}
Question: {question}
{trajectory}"""

# Few-shot examples from original ReAct paper (webthink_simple6)
REACT_EXAMPLES = """

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
"""


# ==================== ReAct-KV Prompt (for incremental generation) ====================
REACT_KV_INITIAL_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
{examples}
Question: {question}
"""


def parse_action_original(text, step_num):
    """
    Parse action in original ReAct format with step numbers.
    Handles: Thought N: ... Action N: ...
    """
    # Try to parse with step numbers
    thought_match = re.search(
        rf"Thought\s*{step_num}:\s*(.*?)(?=\nAction\s*{step_num}:|\Z)",
        text, re.DOTALL
    )
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        # Fallback: try without step number
        thought_match = re.search(r"Thought[^:]*:\s*(.*?)(?=\nAction|\Z)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else text.strip()

    # Parse action
    action_match = re.search(
        rf"Action\s*{step_num}:\s*(search|lookup|finish)\[(.*?)\]",
        text, re.DOTALL | re.IGNORECASE
    )
    if not action_match:
        action_match = re.search(
            r"Action[^:]*:\s*(search|lookup|finish)\[(.*?)\]",
            text, re.DOTALL | re.IGNORECASE
        )
    if action_match:
        action_type = action_match.group(1).lower()
        action_arg = action_match.group(2).strip()
        return thought, action_type, action_arg

    # Fallback: try to find finish anywhere
    finish_match = re.search(r"[Ff]inish\[(.*?)\]", text)
    if finish_match:
        return thought, "finish", finish_match.group(1).strip()

    return thought, None, None


def _get_page_obs(text, max_sentences=5):
    """Extract first N sentences from article text, mimicking original WikiEnv get_page_obs."""
    sentences = []
    paragraphs = text.split("\n")
    for p in paragraphs:
        p = p.strip()
        if p:
            sentences.extend(s.strip() + '.' for s in p.split('. ') if s.strip())
    obs = ' '.join(sentences[:max_sentences])
    return obs if obs else text[:1000]


def _execute_passage_search(action_arg, retriever, lookup_state):
    """Return top-k BM25/web passages for RAG-style retrievers (BrowseComp, web search)."""
    results = retriever.search(action_arg, top_k=BM25_TOP_K)
    if not results:
        lookup_state["page"] = None
        return f"Could not find [{action_arg}].", lookup_state

    context_parts = []
    full_text_parts = []
    empty_hits = 0
    for title, text, score in results:
        if not text or not str(text).strip():
            empty_hits += 1
            print(
                f"[WARN] Search hit has no passage text: title={title!r} score={score:.2f}. "
                "Check corpus_texts.jsonl (BrowseComp) or network/API (web retriever)."
            )
            continue
        truncated = text[:1500] + "..." if len(text) > 1500 else text
        context_parts.append(f"[{title}] (score={score:.2f}): {truncated}")
        full_text_parts.append(text)

    if not context_parts:
        similar_titles = [r[0] for r in results[:5]]
        lookup_state["page"] = None
        obs = (
            f"Search for [{action_arg}] matched {len(results)} document(s) "
            f"but no passage text was loaded ({empty_hits} empty). "
            f"Documents: {similar_titles}."
        )
        print(f"[WARN] {obs}")
        return obs, lookup_state

    lookup_state["page"] = "\n\n".join(full_text_parts)
    lookup_state["lookup_keyword"] = None
    lookup_state["lookup_list"] = None
    lookup_state["lookup_cnt"] = 0
    obs = "\n\n".join(context_parts)
    print(
        f"[INFO] Passage search for [{action_arg}]: {len(context_parts)} passage(s), "
        f"obs_len={len(obs)} chars"
    )
    return obs, lookup_state


def execute_action(action_type, action_arg, retriever, lookup_state):
    """
    Execute an action using the WikiBM25Retriever, mimicking original WikiEnv behavior.
    
    Key difference from v1: Search first tries exact title match (like original Wikipedia API),
    then falls back to BM25. When BM25 doesn't find the exact entity, returns "similar entities"
    list so the agent can refine its search (matching original ReAct paper behavior).
    
    Returns (observation_str, updated_lookup_state)
    """
    if action_type == "search":
        if getattr(retriever, "passage_search", False):
            return _execute_passage_search(action_arg, retriever, lookup_state)

        # Step 1: Try exact title match first (this is what the original Wikipedia API does)
        exact_text = retriever.lookup(action_arg)
        if exact_text:
            obs = _get_page_obs(exact_text)
            # Store full page text for Lookup action
            lookup_state["page"] = exact_text
            lookup_state["lookup_keyword"] = None
            lookup_state["lookup_list"] = None
            lookup_state["lookup_cnt"] = 0
            return obs, lookup_state

        # Step 2: Exact match failed, try BM25 search
        results = retriever.search(action_arg, top_k=BM25_TOP_K)
        if results:
            # Check if the top BM25 result's title closely matches the search query
            top_title, top_text, top_score = results[0]
            if top_text and top_title.lower().strip() == action_arg.lower().strip():
                # BM25 found an exact title match
                obs = _get_page_obs(top_text)
                lookup_state["page"] = top_text
                lookup_state["lookup_keyword"] = None
                lookup_state["lookup_list"] = None
                lookup_state["lookup_cnt"] = 0
                return obs, lookup_state

            # Check if the query is a substring of top title or vice versa
            if top_text and (
                action_arg.lower() in top_title.lower() or
                top_title.lower() in action_arg.lower()
            ):
                obs = _get_page_obs(top_text)
                lookup_state["page"] = top_text
                lookup_state["lookup_keyword"] = None
                lookup_state["lookup_list"] = None
                lookup_state["lookup_cnt"] = 0
                return obs, lookup_state

            # BM25 returned results but no good title match → return similar entities
            # This matches the original ReAct paper's "Could not find X. Similar: [...]" pattern
            similar_titles = [r[0] for r in results[:5]]
            obs = f"Could not find [{action_arg}]. Similar: {similar_titles}."
            lookup_state["page"] = None
            return obs, lookup_state
        else:
            obs = f"Could not find [{action_arg}]."
            lookup_state["page"] = None
        return obs, lookup_state

    elif action_type == "lookup":
        if lookup_state.get("page") is None:
            return "No page loaded. Use Search first.", lookup_state
        
        keyword = action_arg
        page = lookup_state["page"]
        
        if lookup_state.get("lookup_keyword") != keyword:
            # Reset lookup for new keyword
            lookup_state["lookup_keyword"] = keyword
            # Find all sentences containing the keyword in the FULL page text
            paragraphs = page.split("\n")
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            sentences = []
            for p in paragraphs:
                sentences.extend(s.strip() + '.' for s in p.split('. ') if s.strip())
            lookup_state["lookup_list"] = [
                s for s in sentences if keyword.lower() in s.lower()
            ]
            lookup_state["lookup_cnt"] = 0
        
        if lookup_state["lookup_cnt"] >= len(lookup_state.get("lookup_list", [])):
            obs = "No more results."
        else:
            cnt = lookup_state["lookup_cnt"]
            total = len(lookup_state["lookup_list"])
            obs = f"(Result {cnt + 1} / {total}) {lookup_state['lookup_list'][cnt]}"
            lookup_state["lookup_cnt"] = cnt + 1
        
        return obs, lookup_state

    elif action_type == "finish":
        return f"Episode finished.", lookup_state
    
    return "Invalid action.", lookup_state


def extract_short_answer(response):
    """Extract short answer from model response."""
    answer = response.split("\n")[0].strip()
    for prefix in ["The answer is ", "Answer: ", "A: "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    answer = answer.rstrip(".")
    return answer


# ==================== GPU peak-memory helpers ====================
def _cuda_reset_peak():
    """Reset the CUDA peak-memory counter so the next read reflects this sample only."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_peak_mb():
    """Return the CUDA peak allocated memory (MB) since the last reset, 0.0 on CPU/error."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
    except Exception:
        pass
    return 0.0


def _model_param_mb(llm):
    """Return model weight memory (MB) on device."""
    try:
        model = getattr(llm, "model", None)
        if model is None:
            return 0.0
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        return float(total_bytes) / (1024.0 ** 2)
    except Exception:
        return 0.0


def _net_inference_peak_mb(raw_peak_mb, model_param_mb):
    """Peak GPU memory during inference minus static model weights (MB)."""
    return max(0.0, float(raw_peak_mb) - float(model_param_mb))


def _decode_cache_len(total_len, prompt_len):
    """KV / context length excluding the protected prompt prefix."""
    return max(0, int(total_len) - int(prompt_len))


def _format_step_remaining_tokens(prompt_len, per_step_total_lens):
    """Store prompt length first, then decode-only cache length after each step."""
    prompt_len = int(prompt_len)
    return [prompt_len] + [_decode_cache_len(x, prompt_len) for x in per_step_total_lens]


def _summarize_peak_memory(results):
    """Average / max per-sample peak inference GPU memory (MB), excluding model weights."""
    peaks = [
        float(r.get("peak_memory_mb", 0.0))
        for r in results
        if isinstance(r.get("peak_memory_mb", None), (int, float)) and r.get("peak_memory_mb", 0.0) > 0
    ]
    if not peaks:
        return {"avg_peak_memory_mb": 0.0, "max_peak_memory_mb": 0.0, "n_peak_samples": 0}
    return {
        "avg_peak_memory_mb": sum(peaks) / len(peaks),
        "max_peak_memory_mb": max(peaks),
        "n_peak_samples": len(peaks),
    }


def _summarize_decode_cache_and_time(results):
    """Per-sample decode-only final cache and wall-clock inference time."""
    final_decode_lens = [
        int(r.get("llm_stats", {}).get("final_cache_len", 0) or 0)
        for r in results
        if isinstance(r, dict)
    ]
    sample_times = [
        float(r.get("sample_time", 0) or 0)
        for r in results
        if isinstance(r, dict) and isinstance(r.get("sample_time"), (int, float)) and r.get("sample_time", 0) > 0
    ]
    out = {}
    if final_decode_lens:
        out["avg_final_decode_cache_len"] = sum(final_decode_lens) / len(final_decode_lens)
        out["max_final_decode_cache_len"] = max(final_decode_lens)
    if sample_times:
        out["avg_sample_time_seconds"] = sum(sample_times) / len(sample_times)
        out["max_sample_time_seconds"] = max(sample_times)
    return out


def _sample_num_steps(result):
    """Best-effort step count for one sample result row."""
    if not isinstance(result, dict):
        return 0
    if isinstance(result.get("num_steps"), (int, float)):
        return int(result["num_steps"])
    trajectory = result.get("trajectory")
    if isinstance(trajectory, list):
        return len(trajectory)
    return 0


def _summarize_step_counts(results):
    """Count how many samples finished with each num_steps."""
    steps = [_sample_num_steps(r) for r in results if isinstance(r, dict)]
    if not steps:
        return {
            "step_count_distribution": {},
            "avg_num_steps": 0.0,
            "max_num_steps": 0,
            "min_num_steps": 0,
        }
    counts = Counter(steps)
    return {
        "step_count_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
        "avg_num_steps": sum(steps) / len(steps),
        "max_num_steps": max(steps),
        "min_num_steps": min(steps),
    }


_PRUNING_MODE_TO_METRICS_METHOD = {
    "none": "react_kv_none",
    "h2o": "react_kv_h2o",
    "tova": "react_kv_tova",
    "step_anchor_h2o": "react_kv_step_anchor_h2o",
    "step_aware_h2o": "react_kv_step_aware_h2o",
    "step_inter": "react_kv_step_inter",
    "snapkv": "react_kv_snapkv",
    "pyramidinfer": "react_kv_pyramidinfer",
    "ours": "ours",
}


def _metrics_method_name(pruning_mode, override=None):
    if override:
        return override
    return _PRUNING_MODE_TO_METRICS_METHOD.get(pruning_mode, f"react_kv_{pruning_mode}")


def _auto_write_metrics_md(result_json, dataset, method, cache_ratio=""):
    """Write metrics markdown next to the result JSON (standalone runs)."""
    try:
        from record_experiment_metrics import metrics_output_path, write_experiment_metrics

        output_file = metrics_output_path(result_json, method, cache_ratio=cache_ratio or "")
        write_experiment_metrics(
            result_json,
            dataset=dataset,
            method=method,
            output_file=output_file,
            cache_ratio=cache_ratio or "",
        )
    except Exception as exc:
        print(f"[WARN] Failed to write metrics markdown: {exc}")


# ==================== Experiment: Single Model ====================
def run_single_experiment(val_data, selected_samples, output_path, checkpoint_path,
                          metrics_dataset="hotpotqa"):
    """Single Model: direct QA without retrieval."""
    import torch
    from models.QwenLLM import QwenLLM

    PROMPT = """Answer the following question with a short and concise answer (just a few words or a short phrase). Do NOT explain your reasoning.

Question: {question}
Answer:"""

    llm = QwenLLM(MODEL_PATH)
    model_param_mb = _model_param_mb(llm)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        results = _load_checkpoint_results(checkpoint_path, label="single checkpoint")
        results = _filter_stale_checkpoint_results(results, selected_samples)
        completed_ids = {r["id"] for r in results}
        if results:
            print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            print(
                f"[WARN] Skipping duplicate sample id={sample_id!r} at index {orig_idx} "
                f"(already completed). Progress may finish below {total_samples}."
            )
            continue
        question = sample["question"]
        gold_answer = sample["answer"]

        prompt = PROMPT.format(question=question)
        _cuda_reset_peak()
        raw_response = llm.generate(prompt, max_new_tokens=128)
        peak_memory_mb = _net_inference_peak_mb(_cuda_peak_mb(), model_param_mb)
        pred_answer = extract_short_answer(raw_response)

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "raw_response": raw_response, "em": em, "f1": f1,
            "peak_memory_mb": peak_memory_mb,
        }
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_path, results)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    output_data = {
        "summary": {
            "method": "Single Model (Direct QA)",
            "model": MODEL_PATH,
            "total_samples": len(em_scores),
            "exact_match": final_em,
            "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
            "model_param_mb": model_param_mb,
            **_summarize_peak_memory(results),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    _auto_write_metrics_md(output_path, metrics_dataset, "single")

    print(f"\n{'='*60}\nSingle Model Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


# ==================== Experiment: RAG ====================
def run_rag_experiment(val_data, selected_samples, retriever, output_path, checkpoint_path,
                       metrics_dataset="hotpotqa"):
    """RAG: Retrieve-then-Read with Wikipedia BM25."""
    import torch
    from models.QwenLLM import QwenLLM

    RAG_PROMPT = """Read the following retrieved passages and answer the question with a short and concise answer (just a few words or a short phrase). Do NOT explain your reasoning.

Retrieved Passages:
{context}

Question: {question}
Answer:"""

    llm = QwenLLM(MODEL_PATH)
    model_param_mb = _model_param_mb(llm)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        results = _load_checkpoint_results(checkpoint_path, label="rag checkpoint")
        results = _filter_stale_checkpoint_results(results, selected_samples)
        completed_ids = {r["id"] for r in results}
        if results:
            print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            print(
                f"[WARN] Skipping duplicate sample id={sample_id!r} at index {orig_idx} "
                f"(already completed). Progress may finish below {total_samples}."
            )
            continue
        question = sample["question"]
        gold_answer = sample["answer"]

        # Retrieve from full Wikipedia
        search_results = retriever.search(question, top_k=BM25_TOP_K)
        context_parts = []
        for title, text, score in search_results:
            truncated = text[:1500] + "..." if len(text) > 1500 else text
            context_parts.append(f"[{title}] (score={score:.2f}): {truncated}")
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant passages found."

        prompt = RAG_PROMPT.format(context=context_str, question=question)
        _cuda_reset_peak()
        raw_response = llm.generate(prompt, max_new_tokens=128)
        peak_memory_mb = _net_inference_peak_mb(_cuda_peak_mb(), model_param_mb)
        pred_answer = extract_short_answer(raw_response)

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "raw_response": raw_response, "em": em, "f1": f1,
            "peak_memory_mb": peak_memory_mb,
        }
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_path, results)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    output_data = {
        "summary": {
            "method": "RAG (Retrieve-then-Read, Full Wiki)",
            "model": MODEL_PATH, "bm25_top_k": BM25_TOP_K,
            "total_samples": len(em_scores),
            "exact_match": final_em, "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
            "model_param_mb": model_param_mb,
            **_summarize_peak_memory(results),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    _auto_write_metrics_md(output_path, metrics_dataset, "rag")

    print(f"\n{'='*60}\nRAG Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


# ==================== Experiment: ReAct ====================
def run_react_experiment(val_data, selected_samples, retriever, output_path, checkpoint_path,
                         metrics_dataset="hotpotqa"):
    """ReAct: multi-step reasoning with Wikipedia BM25 retrieval."""
    import torch
    from models.QwenLLM import QwenLLM

    llm = QwenLLM(MODEL_PATH)
    model_param_mb = _model_param_mb(llm)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        results = _load_checkpoint_results(checkpoint_path, label="react checkpoint")
        results = _filter_stale_checkpoint_results(results, selected_samples)
        completed_ids = {r["id"] for r in results}
        if results:
            print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            print(
                f"[WARN] Skipping duplicate sample id={sample_id!r} at index {orig_idx} "
                f"(already completed). Progress may finish below {total_samples}."
            )
            continue
        question = sample["question"]
        gold_answer = sample["answer"]

        # Run ReAct episode
        _cuda_reset_peak()
        pred_answer, trajectory_log, n_steps, peak_memory_mb, step_remaining_tokens = _run_react_episode(
            question, llm, retriever, model_param_mb=model_param_mb
        )

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "em": em, "f1": f1, "num_steps": n_steps,
            "prompt_token_count": step_remaining_tokens[0] if step_remaining_tokens else 0,
            "peak_memory_mb": peak_memory_mb,
            "step_remaining_tokens": step_remaining_tokens,
            "trajectory": trajectory_log,
        }
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_path, results)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    output_data = {
        "summary": {
            "method": "ReAct (Full Wiki)",
            "model": MODEL_PATH, "max_steps": format_max_steps(MAX_STEPS),
            "total_samples": len(em_scores),
            "exact_match": final_em, "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
            "model_param_mb": model_param_mb,
            **_summarize_peak_memory(results),
            **_summarize_step_counts(results),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    step_stats = output_data["summary"].get("step_count_distribution", {})
    if step_stats:
        print(f"[INFO] Step distribution: {step_stats}")

    _auto_write_metrics_md(output_path, metrics_dataset, "react")

    print(f"\n{'='*60}\nReAct Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


def _run_react_episode(question, llm, retriever, max_steps=None, model_param_mb=0.0):
    """Run one ReAct episode: Thought N / Action N / Observation N."""
    step_limit = _episode_step_limit(max_steps)
    prompt = REACT_PROMPT_TEMPLATE.format(
        examples=REACT_EXAMPLES, question=question, trajectory=""
    )
    trajectory = ""
    trajectory_log = []
    lookup_state = {"page": None, "lookup_keyword": None, "lookup_list": None, "lookup_cnt": 0}

    # Per-sample peak GPU memory and per-step context token counts.
    peak_mb = 0.0
    step_total_tokens = []

    def _ctx_tokens(text):
        try:
            tok = getattr(llm, "tokenizer", None)
            if tok is not None:
                return int(len(tok(text, add_special_tokens=False).input_ids))
        except Exception:
            pass
        return 0

    prompt_token_count = _ctx_tokens(prompt)

    def _finalize_step_tokens():
        return _format_step_remaining_tokens(prompt_token_count, step_total_tokens)

    def _run_one_step(i):
        nonlocal peak_mb, trajectory, trajectory_log, step_total_tokens, lookup_state
        response = llm.generate(
            prompt + trajectory + f"Thought {i}:",
            max_new_tokens=256,
        )
        peak_mb = max(peak_mb, _net_inference_peak_mb(_cuda_peak_mb(), model_param_mb))

        thought, action_type, action_arg = parse_action_original(
            f"Thought {i}:" + response, i
        )

        if action_type is None:
            action_response = llm.generate(
                prompt + trajectory + f"Thought {i}: {thought}\nAction {i}:",
                max_new_tokens=64,
            )
            peak_mb = max(peak_mb, _net_inference_peak_mb(_cuda_peak_mb(), model_param_mb))
            _, action_type, action_arg = parse_action_original(
                f"Action {i}:" + action_response, i
            )
            if action_type is None:
                trajectory_log.append({"step": i, "thought": thought, "action": "finish[]"})
                step_total_tokens.append(_ctx_tokens(prompt + trajectory))
                return "stop", "", trajectory_log, i, peak_mb, _finalize_step_tokens()

        step_log = {
            "step": i,
            "thought": thought,
            "action_type": action_type,
            "action_arg": action_arg,
        }

        if action_type == "finish":
            trajectory_log.append(step_log)
            step_total_tokens.append(_ctx_tokens(prompt + trajectory))
            return "finish", action_arg if action_arg else "", trajectory_log, i, peak_mb, _finalize_step_tokens()

        obs, lookup_state = execute_action(action_type, action_arg, retriever, lookup_state)
        obs = obs.replace('\\n', '')
        step_log["observation"] = obs[:1000]
        trajectory_log.append(step_log)
        trajectory += f"Thought {i}: {thought}\nAction {i}: {action_type}[{action_arg}]\nObservation {i}: {obs}\n"
        step_total_tokens.append(_ctx_tokens(prompt + trajectory))
        return "continue", None, trajectory_log, i, peak_mb, _finalize_step_tokens()

    if step_limit is None:
        i = 1
        while True:
            status, answer, trajectory_log, n_steps, peak_mb, step_remaining_tokens = _run_one_step(i)
            if status in ("finish", "stop"):
                return answer, trajectory_log, n_steps, peak_mb, step_remaining_tokens
            i += 1

    for i in range(1, step_limit + 1):
        status, answer, trajectory_log, n_steps, peak_mb, step_remaining_tokens = _run_one_step(i)
        if status == "finish":
            return answer, trajectory_log, n_steps, peak_mb, step_remaining_tokens
        if status == "stop":
            return answer, trajectory_log, n_steps, peak_mb, step_remaining_tokens

    return "", trajectory_log, step_limit, peak_mb, step_remaining_tokens


# ==================== Experiment: ReAct-KV ====================
def run_react_kv_experiment(val_data, selected_samples, retriever, pruning_mode,
                            output_path, checkpoint_path, kv_config_override=None,
                            metrics_dataset="hotpotqa", metrics_method=None):
    """ReAct with KV Cache: supports none/h2o/tova/step-aware/step-inter/snapkv/ours pruning modes."""
    import torch
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache

    kv_config = {
        "pruning_mode": pruning_mode,
        "prune_every_n": 1,
        "cache_ratio": 0.2,
        # Budget semantics: keep full prompt + half of generated tokens.
        "protect_prompt": True,
        "pool_window": 4,
        "max_trajectory_tokens": 1024,
        "sink_size": 4,
        "observation_window": 0 if pruning_mode in ("step_aware_h2o", "step_inter", "tova") else 32,
        "num_score_layers": 3,
        "attn_mode": "piggyback" if pruning_mode in ("step_aware_h2o", "step_inter") else "scoring_forward",
        "step_anchor_keep_last_obs": -1 if pruning_mode == "step_anchor_h2o" else 1,
        "step_aware_alpha": 0.8,
        "step_aware_beta": 0.8,
        "step_aware_min_keep": 12,
        "step_aware_min_keep_ratio": 0.30,
        "step_aware_bonus": 0.0,
        "step_poolwise_prune": True if pruning_mode in ("step_aware_h2o", "step_inter") else False,
        "step_reward_weight": 0.85,
        "step_citation_weight": 0.15,
        "step_repeat_penalty": 0.3,
        "step_ablate_novelty": False,
        "step_ablate_success": False,
        "step_ablate_citation": False,
        "step_ablate_progression": False,
        "step_interrupt_mode": "none",
        "prompt_prefill_keep_ratio": 1.0 if pruning_mode in ("step_aware_h2o", "step_inter") else 1.0,
    }
    if kv_config_override:
        kv_config.update(kv_config_override)

    total_samples = len(selected_samples)
    selected_ids = {s["id"] for _, s in selected_samples}
    results = []
    read_checkpoint_path = _resolve_existing_kv_path(checkpoint_path, "checkpoint")
    if os.path.exists(read_checkpoint_path):
        results = _load_checkpoint_results(read_checkpoint_path, label="react-kv checkpoint")
        results = _filter_stale_checkpoint_results(results, selected_samples)
        completed_ids = {r["id"] for r in results}
        if results:
            print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")
        if (
            read_checkpoint_path != checkpoint_path
            and results
            and not os.path.exists(checkpoint_path)
        ):
            _save_checkpoint(checkpoint_path, results, slim_react_kv=True)
            print(f"[INFO] Migrated checkpoint -> {checkpoint_path}")

    completed_ids = {r["id"] for r in results}
    if (
        not _react_kv_output_is_complete_any(output_path)
        and selected_ids.issubset(completed_ids)
        and len(completed_ids) >= len(selected_ids)
    ):
        ordered_results = [r for r in results if r.get("id") in selected_ids]
        print(
            f"[INFO] Checkpoint complete ({len(ordered_results)}/{total_samples}); "
            "writing final output without loading model."
        )
        return _finalize_react_kv_output(
            ordered_results,
            output_path,
            pruning_mode,
            kv_config,
            metrics_dataset,
            metrics_method=metrics_method,
        )

    # Initialize token tracker for H2O pruning
    token_tracker = TokenTracker() if pruning_mode in ("h2o", "tova", "step_anchor_h2o", "step_aware_h2o", "step_inter") else None

    llm = QwenLLMWithKVCache(MODEL_PATH, kv_config, token_tracker=token_tracker)
    model_param_mb = _model_param_mb(llm)
    print(f"protect_prompt={kv_config['protect_prompt']}")
    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            print(
                f"[WARN] Skipping duplicate sample id={sample_id!r} at index {orig_idx} "
                f"(already completed). Progress may finish below {total_samples}."
            )
            continue
        if (
            llm.kv_manager is not None
            and isinstance(kv_config.get("step_force_drop_map", None), dict)
            and pruning_mode in ("step_aware_h2o", "step_inter")
        ):
            llm.kv_manager.update_step_force_drop_ids(
                kv_config.get("step_force_drop_map", {}).get(sample_id, [])
            )

        question = sample["question"]
        gold_answer = sample["answer"]

        # Run ReAct-KV episode
        print(f"orig_idx={orig_idx} sample_id={sample_id}")
        print("-----------------------------------------------------------------------")
        sample_start = time.time()
        _cuda_reset_peak()
        pred_answer, trajectory_log, step_timings, debug_payload = _run_react_kv_episode(
            question, llm, retriever, pruning_mode=pruning_mode, return_debug=True
        )
        peak_memory_mb = _net_inference_peak_mb(_cuda_peak_mb(), model_param_mb)
        sample_time = time.time() - sample_start
        llm_stats = llm.get_stats()

        prompt_token_count = 0
        if isinstance(debug_payload, dict):
            prompt_token_count = int(debug_payload.get("prompt_token_count", 0) or 0)
        if prompt_token_count <= 0 and getattr(llm, "kv_manager", None) is not None:
            prompt_token_count = int(getattr(llm.kv_manager, "protected_prefix_len", 0) or 0)

        per_step_total_lens = [t.get("kv_cache_length", 0) for t in step_timings]
        step_remaining_tokens = _format_step_remaining_tokens(prompt_token_count, per_step_total_lens)
        stored_step_timings = [
            {
                **t,
                "kv_cache_length": _decode_cache_len(t.get("kv_cache_length", 0), prompt_token_count),
            }
            for t in step_timings
        ]
        final_decode_cache_len = _decode_cache_len(
            llm_stats.get("current_cache_len", 0), prompt_token_count
        )

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "em": em, "f1": f1, "num_steps": len(trajectory_log),
            "sample_time": sample_time,
            "prompt_token_count": prompt_token_count,
            "peak_memory_mb": peak_memory_mb,
            "step_remaining_tokens": step_remaining_tokens,
            "step_timings": stored_step_timings,
            "llm_stats": {
                "prefill_time": llm_stats.get("prefill_time", 0),
                "decode_time": llm_stats.get("decode_time", 0),
                "scoring_time": llm_stats.get("scoring_time", 0),
                "pruning_time": llm_stats.get("pruning_time", 0),
                "total_prune_count": llm_stats.get("total_prune_count", 0),
                "final_cache_len": final_decode_cache_len,
            },
            "trajectory": trajectory_log,
        }
        if isinstance(debug_payload, dict):
            result["debug_payload"] = debug_payload
            if "step_scores" in debug_payload:
                result["step_scores"] = debug_payload.get("step_scores", {})
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        prune_info = f" Prunes={llm_stats.get('total_prune_count', 0)}" if pruning_mode != "none" else ""
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | Time={sample_time:.1f}s{prune_info} | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_path, results, slim_react_kv=True)

    if len(em_scores) < total_samples:
        print(
            f"[WARN] ReAct-KV ({pruning_mode}) finished with {len(em_scores)}/{total_samples} samples. "
            "Check startup logs for duplicate ids, dataset size, or checkpoint resume issues."
        )

    total_time = time.time() - start_time
    del llm
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return _finalize_react_kv_output(
        results,
        output_path,
        pruning_mode,
        kv_config,
        metrics_dataset,
        metrics_method=metrics_method,
        model_param_mb=model_param_mb,
        total_time=total_time,
    )


def _run_react_kv_episode(
    question,
    llm,
    retriever,
    pruning_mode="none",
    max_steps=None,
    window_size=128,
    return_debug=False,
):
    step_limit = _episode_step_limit(max_steps)
    trajectory_log = []
    step_timings = []
    step_discarded_counts = {}
    step_token_ranges = {}
    step_spans = []
    finalized_step_spans = set()
    last_finalized_step_end = -1
    obs_step_ranges = {}
    lookup_state = {"page": None, "lookup_keyword": None, "lookup_list": None, "lookup_cnt": 0}
    # External step-level signal state for step-aware pruning.
    step_scores = {}
    step_meta = {}
    use_step_scores = bool(pruning_mode == "step_aware_h2o")
    seen_obs_anchor_terms = set()
    seen_action_args = {}
    reward_weight = float(getattr(llm, "kv_config", {}).get("step_reward_weight", 0.7))
    citation_weight = float(getattr(llm, "kv_config", {}).get("step_citation_weight", 0.3))
    repeat_penalty = float(getattr(llm, "kv_config", {}).get("step_repeat_penalty", 0.3))
    ablate_novelty = bool(getattr(llm, "kv_config", {}).get("step_ablate_novelty", False))
    ablate_success = bool(getattr(llm, "kv_config", {}).get("step_ablate_success", False))
    ablate_citation = bool(getattr(llm, "kv_config", {}).get("step_ablate_citation", False))
    ablate_progression = bool(getattr(llm, "kv_config", {}).get("step_ablate_progression", False))
    step_interrupt_mode = str(
        getattr(llm, "kv_config", {}).get("step_interrupt_mode", "none") or "none"
    ).strip().lower()
    interrupted_steps = set()
    step_interrupt_events = []
    kv_stop_strings = ["\nObservation", "\nQuestion:"]
    import torch
    
    # 根据 pruning_mode 决定是否使用 memory_block 融合
    use_memory_fusion = (pruning_mode == "ours")

    def _get_actual_kv_len():
        """Read true KV length from past_key_values to avoid counter drift."""
        full_kv = getattr(llm, "past_key_values", None)
        if full_kv is None:
            return 0
        try:
            if hasattr(full_kv, "layers") and len(full_kv.layers) > 0:
                return int(full_kv.layers[0].keys.shape[2])
            if isinstance(full_kv, tuple) and len(full_kv) > 0:
                return int(full_kv[0][0].shape[2])
        except Exception:
            pass
        return llm.get_cache_len() if hasattr(llm, "get_cache_len") else 0

    def _normalize_kv(kv, ref_kv=None):
        """
        Normalize KV to:
        ((k,v),(k,v),...)
        支持输入形式：
          - (k,v) 对
          - ((k,v),)
          - 每层直接是 key tensor (torch.Tensor) -> 会生成 zeros v
        ref_kv: 可选，用于从 gen_kv 推断 v 的形状（优先）
        """
        if kv is None:
            return None

        normalized = []
        for layer_kv in kv:
            # case 1: already (k,v)
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                normalized.append((layer_kv[0], layer_kv[1]))

            # case 2: ((k,v),)
            elif isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 1:
                inner = layer_kv[0]
                normalized.append((inner[0], inner[1]))

            # case 3: key-only tensor (torch.Tensor)
            elif isinstance(layer_kv, torch.Tensor):
                k = layer_kv
                # 尝试从 ref_kv 推断 v 的形状
                v_shape_inferred = None
                if ref_kv:
                    # 找第一个可用的 v
                    for ref_layer in ref_kv:
                        if isinstance(ref_layer, (tuple, list)) and len(ref_layer) == 2:
                            ref_v = ref_layer[1]
                            v_shape_inferred = (ref_v.shape[0], ref_v.shape[1], k.shape[2], ref_v.shape[3])
                            v_dtype = ref_v.dtype
                            v_device = ref_v.device
                            break
                if v_shape_inferred is None:
                    # 回退：尝试从模型配置推断 head_dim / num_heads
                    num_heads = getattr(llm.model.config, "num_attention_heads", None)
                    head_dim = getattr(llm.model.config, "head_dim", None)
                    bsz = k.shape[0]
                    seq_len = k.shape[2]
                    if num_heads is None or head_dim is None:
                        # 兜底：用 k 的最后一维估计 head_dim，num_heads 设 1
                        head_dim = k.shape[-1]
                        num_heads = 1
                    v_shape_inferred = (bsz, num_heads, seq_len, head_dim)
                    v_dtype = k.dtype
                    v_device = k.device
                zeros_v = torch.zeros(v_shape_inferred, dtype=v_dtype, device=v_device)
                normalized.append((k, zeros_v))

            else:
                raise ValueError(f"Unexpected KV format: {type(layer_kv)} | {layer_kv}")

        return tuple(normalized)

    def _normalize_action_arg(arg):
        if not arg:
            return ""
        return re.sub(r"\s+", " ", str(arg).strip().lower())

    def _compute_step_external_score(reward_val, citation_val):
        reward = max(-1.0, min(2.0, float(reward_val)))
        citation = max(0.0, float(citation_val))
        # Saturate citation contribution to avoid dominating HH-based scores.
        citation_sat = math.log1p(citation)
        score = reward_weight * reward + citation_weight * citation_sat
        return float(max(0.0, min(8.0, score)))

    def _print_step_spans_summary():
        span_by_step = {}
        for sp in step_spans:
            if not isinstance(sp, dict) or sp.get("type") != "step":
                continue
            sid = sp.get("step_id", None)
            if sid is None:
                continue
            span_by_step[int(sid)] = (int(sp.get("start", -1)), int(sp.get("end", -1)))
        if not span_by_step:
            return
        parts = []
        for sid in sorted(span_by_step.keys()):
            st, ed = span_by_step[sid]
            parts.append(f"step{sid}=[{st}-{ed}]")
        print(f"[STEP SPANS] {' '.join(parts)}")

    def _print_step_scores_summary():
        if not step_scores:
            return
        parts = []
        for sid in sorted(step_scores.keys()):
            parts.append(f"step{int(sid)}={float(step_scores[sid]):.4f}")
        print(f"[STEP SCORES] {' | '.join(parts)}")
    
    def _process_kv_flow(prompt_kv, memory_block, new_kv, window_size, full_kv=None, prompt_len=0, memory_rank=128):
        """
        new_kv: tuple of layer KV
            ((k,v),(k,v),...)
        """
        if new_kv is None:
            return None, memory_block
    
        # If full_kv is provided, prefer extracting recent window from it
        recent_kv = []
        if full_kv is not None:
            # support DynamicCache-like objects (having .layers) or tuple-of-(k,v)
            if hasattr(full_kv, "layers"):
                for layer in full_kv.layers:
                    k_full = layer.keys
                    v_full = layer.values
                    r_k = k_full[:, :, -window_size:, :]
                    r_v = v_full[:, :, -window_size:, :]
                    recent_kv.append((r_k, r_v))
                recent_kv = tuple(recent_kv)

                # previous region is between prompt_len and end-window
                fused_memory = []
                for layer_idx, layer in enumerate(full_kv.layers):
                    k_full = layer.keys
                    v_full = layer.values
                    prev_start = int(prompt_len)
                    prev_end = k_full.shape[2] - window_size
                    if prev_end <= prev_start:
                        # nothing to fuse, keep existing memory_block if any
                        if memory_block is None:
                            # create empty memory for this layer
                            empty_k = k_full[:, :, :0, :]
                            empty_v = v_full[:, :, :0, :]
                            fused_memory.append((empty_k, empty_v))
                        else:
                            fused_memory.append(memory_block[layer_idx])
                        continue

                    prev_k = k_full[:, :, prev_start:prev_end, :].detach()
                    prev_v = v_full[:, :, prev_start:prev_end, :].detach()

                    # Simple pooling/truncation strategy to obtain memory_rank tokens
                    if prev_k.size(2) > memory_rank:
                        # keep the last `memory_rank` tokens
                        pooled_k = prev_k[:, :, -memory_rank:, :]
                        pooled_v = prev_v[:, :, -memory_rank:, :]
                    else:
                        pooled_k = prev_k
                        pooled_v = prev_v

                    # initialize or fuse with existing memory_block
                    if memory_block is None:
                        # initialize memory block for this layer
                        fused_memory.append((pooled_k, pooled_v))
                    else:
                        m_k, m_v = memory_block[layer_idx]
                        # Align shapes: if sizes differ, try to match by truncation/padding
                        target_rank = m_k.size(2)
                        if pooled_k.size(2) != target_rank:
                            if pooled_k.size(2) > target_rank:
                                pooled_k = pooled_k[:, :, -target_rank:, :]
                                pooled_v = pooled_v[:, :, -target_rank:, :]
                            else:
                                # pad with zeros on the left
                                pad_len = target_rank - pooled_k.size(2)
                                pad_k = torch.zeros(m_k.size(0), m_k.size(1), pad_len, m_k.size(3), dtype=m_k.dtype, device=m_k.device)
                                pad_v = torch.zeros(m_v.size(0), m_v.size(1), pad_len, m_v.size(3), dtype=m_v.dtype, device=m_v.device)
                                pooled_k = torch.cat([pad_k, pooled_k], dim=2)
                                pooled_v = torch.cat([pad_v, pooled_v], dim=2)

                        # simple fusion: element-wise add (keeps fixed rank)
                        try:
                            new_m_k = m_k + pooled_k
                            new_m_v = m_v + pooled_v
                        except Exception:
                            # fallback to concatenation if shapes can't be broadcast
                            new_m_k = torch.cat([m_k, pooled_k], dim=2)
                            new_m_v = torch.cat([m_v, pooled_v], dim=2)
                        fused_memory.append((new_m_k, new_m_v))

                memory_block = tuple(fused_memory)
                return recent_kv, memory_block
            else:
                # assume iterable of (k,v)
                for k_full, v_full in full_kv:
                    # take last `window_size` tokens as recent window
                    r_k = k_full[:, :, -window_size:, :]
                    r_v = v_full[:, :, -window_size:, :]
                    recent_kv.append((r_k, r_v))
                recent_kv = tuple(recent_kv)

                # previous region is between prompt_len and end-window
                fused_memory = []
                for layer_idx, (k_full, v_full) in enumerate(full_kv):
                    prev_start = int(prompt_len)
                    prev_end = k_full.shape[2] - window_size
                    if prev_end <= prev_start:
                        # nothing to fuse, keep existing memory_block if any
                        if memory_block is None:
                            # create empty memory for this layer
                            empty_k = k_full[:, :, :0, :]
                            empty_v = v_full[:, :, :0, :]
                            fused_memory.append((empty_k, empty_v))
                        else:
                            fused_memory.append(memory_block[layer_idx])
                        continue

                    prev_k = k_full[:, :, prev_start:prev_end, :].detach()
                    prev_v = v_full[:, :, prev_start:prev_end, :].detach()

                    # Simple pooling/truncation strategy to obtain memory_rank tokens
                    if prev_k.size(2) > memory_rank:
                        # keep the last `memory_rank` tokens
                        pooled_k = prev_k[:, :, -memory_rank:, :]
                        pooled_v = prev_v[:, :, -memory_rank:, :]
                    else:
                        pooled_k = prev_k
                        pooled_v = prev_v

                    # initialize or fuse with existing memory_block
                    if memory_block is None:
                        # initialize memory block for this layer
                        fused_memory.append((pooled_k, pooled_v))
                    else:
                        m_k, m_v = memory_block[layer_idx]
                        # Align shapes: if sizes differ, try to match by truncation/padding
                        target_rank = m_k.size(2)
                        if pooled_k.size(2) != target_rank:
                            if pooled_k.size(2) > target_rank:
                                pooled_k = pooled_k[:, :, -target_rank:, :]
                                pooled_v = pooled_v[:, :, -target_rank:, :]
                            else:
                                # pad with zeros on the left
                                pad_len = target_rank - pooled_k.size(2)
                                pad_k = torch.zeros(m_k.size(0), m_k.size(1), pad_len, m_k.size(3), dtype=m_k.dtype, device=m_k.device)
                                pad_v = torch.zeros(m_v.size(0), m_v.size(1), pad_len, m_v.size(3), dtype=m_v.dtype, device=m_v.device)
                                pooled_k = torch.cat([pad_k, pooled_k], dim=2)
                                pooled_v = torch.cat([pad_v, pooled_v], dim=2)

                        # simple fusion: element-wise add (keeps fixed rank)
                        try:
                            new_m_k = m_k + pooled_k
                            new_m_v = m_v + pooled_v
                        except Exception:
                            # fallback to concatenation if shapes can't be broadcast
                            new_m_k = torch.cat([m_k, pooled_k], dim=2)
                            new_m_v = torch.cat([m_v, pooled_v], dim=2)
                        fused_memory.append((new_m_k, new_m_v))

                memory_block = tuple(fused_memory)

                return recent_kv, memory_block

        # Fallback: operate only on the provided new_kv (per-step step KV)
        for n_k, n_v in new_kv:
            r_k = n_k[:, :, -window_size:, :]
            r_v = n_v[:, :, -window_size:, :]
            recent_kv.append((r_k, r_v))
        recent_kv = tuple(recent_kv)

        # update memory by concatenating per-layer along sequence dim
        if memory_block is None:
            memory_block = recent_kv
        else:
            fused_memory = []
            for (m_k, m_v), (r_k, r_v) in zip(memory_block, recent_kv):
                f_k = torch.cat([m_k, r_k], dim=2)
                f_v = torch.cat([m_v, r_v], dim=2)
                fused_memory.append((f_k, f_v))
            memory_block = tuple(fused_memory)

        return recent_kv, memory_block

    def _extract_recent_from_full(full_kv, window_size):
        """
        从 full_kv 中提取 recent window（每层最后 window_size tokens），兼容 DynamicCache 或 tuple-of-(k,v)
        返回 recent_kv tuple
        """
        recent = []
        if full_kv is None:
            return None

        try:
            if hasattr(full_kv, "layers"):
                for layer in full_kv.layers:
                    k = layer.keys
                    v = layer.values
                    r_k = k[:, :, -window_size:, :]
                    r_v = v[:, :, -window_size:, :]
                    recent.append((r_k, r_v))
            else:
                for k, v in full_kv:
                    r_k = k[:, :, -window_size:, :]
                    r_v = v[:, :, -window_size:, :]
                    recent.append((r_k, r_v))
            return tuple(recent)
        except Exception:
            return None

    def _print_kv_structure(step, prompt_len, memory_block, recent_kv, full_kv=None):
        """
        打印 KV 缓存的完整结构：总长度、prompt/memory/recent 各部分的起止位置和长度。
        """
        try:
            # Compute total length from full_kv if available
            total_len = 0
            if full_kv is not None:
                if hasattr(full_kv, "layers"):
                    if full_kv.layers:
                        total_len = full_kv.layers[0].keys.size(2)
                else:
                    if full_kv and len(full_kv) > 0:
                        total_len = full_kv[0][0].size(2)
            
            # Memory block length (per layer)
            mem_len = 0
            if memory_block is not None and len(memory_block) > 0:
                mem_len = memory_block[0][0].size(2)
            
            # Recent KV length (per layer)
            recent_len = 0
            if recent_kv is not None and len(recent_kv) > 0:
                recent_len = recent_kv[0][0].size(2)
            
            # Calculate ranges within total
            prompt_start = 0
            prompt_end = int(prompt_len)
            memory_start = prompt_end
            memory_end = memory_start + mem_len
            recent_start = memory_end
            recent_end = total_len
            
            print(f"[KV STRUCT] step={step}")
            print(f"  total_len={total_len}")
            print(f"  prompt: [{prompt_start}:{prompt_end}] len={prompt_len}")
            print(f"  memory: [{memory_start}:{memory_end}] len={mem_len}")
            print(f"  recent: [{recent_start}:{recent_end}] len={recent_len}")
        except Exception as e:
            print(f"[KV STRUCT] Error printing structure at step {step}: {e}")


    def _fuse_step_into_memory(memory_block, step_kv, memory_rank=128):
        """
        将本步提取到的 step_kv 融合到 memory_block 中，保证 memory_block 每层的序列长度不变。

        策略：
        - 如果 memory_block 为 None，则用 step_kv 池化/截断到 memory_rank 来初始化 memory_block
        - 否则，对每层将 step_kv 池化到 target_rank（memory_block 的现有 rank），然后与 memory_block 做 element-wise 相加（或在无法广播时回退到拼接/截断），最终保证返回的 memory_block 每层长度等于原来的 target_rank
        """
        if step_kv is None or len(step_kv) == 0:
            return memory_block

        fused = []
        for idx, (s_k, s_v) in enumerate(step_kv):
            # determine target rank and reference shapes
            if memory_block is not None and len(memory_block) > idx:
                m_k, m_v = memory_block[idx]
                target_rank = m_k.size(2)
                dtype_k = m_k.dtype
                dtype_v = m_v.dtype
                dev_k = m_k.device
                dev_v = m_v.device
            else:
                target_rank = memory_rank
                # infer device/dtype from step_kv
                dtype_k = s_k.dtype
                dtype_v = s_v.dtype
                dev_k = s_k.device
                dev_v = s_v.device

            # pool/truncate step to target_rank
            s_len = s_k.size(2)
            if s_len >= target_rank:
                p_k = s_k[:, :, -target_rank:, :]
                p_v = s_v[:, :, -target_rank:, :]
            else:
                # pad on left with zeros to match target_rank
                pad_len = target_rank - s_len
                pad_k = torch.zeros(s_k.size(0), s_k.size(1), pad_len, s_k.size(3), dtype=dtype_k, device=dev_k)
                pad_v = torch.zeros(s_v.size(0), s_v.size(1), pad_len, s_v.size(3), dtype=dtype_v, device=dev_v)
                p_k = torch.cat([pad_k, s_k], dim=2)
                p_v = torch.cat([pad_v, s_v], dim=2)

            if memory_block is None or len(memory_block) <= idx:
                # initialize memory layer
                fused.append((p_k.detach().clone(), p_v.detach().clone()))
            else:
                # fuse: try element-wise add
                try:
                    new_m_k = m_k + p_k
                    new_m_v = m_v + p_v
                    # ensure shape is (.., target_rank, ..)
                    if new_m_k.size(2) != target_rank:
                        new_m_k = new_m_k[:, :, -target_rank:, :]
                        new_m_v = new_m_v[:, :, -target_rank:, :]
                    fused.append((new_m_k.detach().clone(), new_m_v.detach().clone()))
                except Exception:
                    # fallback: if can't add, replace oldest tokens in memory with pooled (right-aligned)
                    try:
                        if m_k.size(2) >= target_rank:
                            keep_len = m_k.size(2) - p_k.size(2)
                            if keep_len > 0:
                                kept_k = m_k[:, :, :keep_len, :]
                                kept_v = m_v[:, :, :keep_len, :]
                                new_k = torch.cat([kept_k, p_k], dim=2)
                                new_v = torch.cat([kept_v, p_v], dim=2)
                            else:
                                new_k = p_k
                                new_v = p_v
                        else:
                            # shouldn't happen, but pad/truncate accordingly
                            new_k = p_k[:, :, -target_rank:, :]
                            new_v = p_v[:, :, -target_rank:, :]
                        fused.append((new_k.detach().clone(), new_v.detach().clone()))
                    except Exception:
                        # ultimate fallback: keep original memory
                        fused.append((m_k.detach().clone(), m_v.detach().clone()))

        return tuple(fused)

    def _print_h2o_step_summary(
        step,
        step_time,
        cache_len_before_step,
        step_token_start_id,
        step_token_end_id,
        prompt_token_count,
        discarded_this_step_total=0,
        latest_prune_info=None,
    ):
        """
        Print per-step H2O summary with compact statistics.
        """
        if pruning_mode not in ("h2o", "tova", "step_anchor_h2o", "step_aware_h2o", "step_inter"):
            return

        new_token_count = 0
        step_range_str = "[]"
        if step_token_end_id >= step_token_start_id:
            new_token_count = step_token_end_id - step_token_start_id + 1
            step_range_str = f"[{step_token_start_id}-{step_token_end_id}]"

        # Correct no-prune estimate:
        # no_prune_total = cache length at step start + tokens generated in this step
        # kv_len is the actual cache length after pruning at step end.
        cache_len_after_step = _get_actual_kv_len()
        no_prune_total = cache_len_before_step + new_token_count
        pruned_total = cache_len_after_step
        obs_window_cfg = 0
        if hasattr(llm, "kv_manager") and llm.kv_manager is not None:
            obs_window_cfg = int(getattr(llm.kv_manager, "observation_window", 0))

        pruned_ids_this_step = []
        if llm.token_tracker is not None and hasattr(llm.token_tracker, "get_step_discarded_tokens"):
            pruned_ids_this_step = llm.token_tracker.get_step_discarded_tokens(step)

        all_discarded_ids = set()
        if llm.token_tracker is not None and hasattr(llm.token_tracker, "step_pruning_events"):
            for s, ids in llm.token_tracker.step_pruning_events.items():
                if s is None:
                    continue
                if s <= step and ids:
                    all_discarded_ids.update(ids)

        prompt_discarded = sum(1 for t in all_discarded_ids if 0 <= t < prompt_token_count)
        this_step_discarded = 0
        prev_steps_discarded = 0
        if step_token_end_id >= step_token_start_id:
            this_step_discarded = sum(
                1 for t in pruned_ids_this_step if step_token_start_id <= t <= step_token_end_id
            )
            prev_steps_discarded = sum(
                1 for t in all_discarded_ids
                if prompt_token_count <= t < step_token_start_id
            )
        else:
            prev_steps_discarded = sum(1 for t in all_discarded_ids if t >= prompt_token_count)

        stage_parts = []
        prompt_total = max(0, prompt_token_count)
        stage_parts.append(f"prompt={prompt_discarded}/{prompt_total}")
        for s in range(1, step + 1):
            start_end = step_token_ranges.get(s)
            if not start_end:
                stage_parts.append(f"step{s}=0/0")
                continue
            s_start, s_end = start_end
            if s_end < s_start:
                stage_parts.append(f"step{s}=0/0")
                continue
            s_total = s_end - s_start + 1
            s_discarded = sum(1 for t in all_discarded_ids if s_start <= t <= s_end)
            stage_parts.append(f"step{s}={s_discarded}/{s_total}")

        print(
            f"[STEP SUMMARY] step={step} generated={step_range_str} "
            f"generated_tokens={new_token_count} pruned_this_step={discarded_this_step_total} "
            f"infer_time={step_time:.2f}s"
        )
        print(
            f"[STEP SUMMARY] totals no_prune={no_prune_total} kept={pruned_total} "
            f"discarded_total={max(0, no_prune_total - pruned_total)}"
        )
        print(
            f"[CACHE STATUS] step={step} prompt_len={prompt_token_count} original_total={no_prune_total} "
            f"current_cache={pruned_total} compression={(pruned_total / max(1, no_prune_total)):.3f}"
        )
        print(
            f"[STEP SUMMARY] discarded_this_step_in_generated_range={this_step_discarded} "
            f"discarded_prev_steps={prev_steps_discarded}"
        )
        print(f"[STEP SUMMARY] stage_discarded " + " | ".join(stage_parts))
        if latest_prune_info and bool(latest_prune_info.get("poolwise", False)):
            pool_stats = latest_prune_info.get("pool_stats") or []
            if pool_stats:
                pool_parts = []
                for ps in pool_stats:
                    pool_name = ps.get("pool", "unknown")
                    s1 = int(ps.get("kept_stage1", ps.get("kept", 0)))
                    s2 = int(ps.get("kept_stage2", 0))
                    sf = int(ps.get("kept_final", s1 + s2))
                    orig_len = int(ps.get("orig_len", 0))
                    pool_parts.append(f"{pool_name}=s1:{s1}+s2:{s2}=f:{sf}/{orig_len}")
                print(f"[STEP SUMMARY] pool_keep " + " | ".join(pool_parts))
            s1_total = int(latest_prune_info.get("stage1_kept_total", 0))
            s2_total = int(latest_prune_info.get("stage2_added_total", 0))
            beff = int(latest_prune_info.get("budget_B_effective", 0))
            has_comp = bool(s2_total > 0)
            print(
                f"[STEP SUMMARY] pool_competition stage1_total={s1_total} "
                f"stage2_total={s2_total} final_B={beff} cross_pool_competition={has_comp}"
            )

        # Global token id progress and budget accounting.
        # Budget logging uses a unified priority:
        # 1) actual applied prune budget from latest_prune_info (if this step pruned),
        # 2) otherwise theoretical half-budget as a reference only.
        if llm.token_tracker is not None and hasattr(llm.token_tracker, "next_global_id"):
            n_global = int(llm.token_tracker.next_global_id)
            original_num = n_global - 1
            # origin includes all tokens up to now, including protected observation window.
            original_total = n_global
            protect_prompt = bool(getattr(llm.kv_manager, "protect_prompt", True)) if hasattr(llm, "kv_manager") and llm.kv_manager is not None else True
            prompt_protected_len = int(getattr(llm.kv_manager, "protected_prefix_len", 0)) if hasattr(llm, "kv_manager") and llm.kv_manager is not None else 0
            # Align theoretical reference with configured ratio (not hard-coded half).
            target_ratio = 0.5
            if hasattr(llm, "kv_manager") and llm.kv_manager is not None:
                ratio_cfg = getattr(llm.kv_manager, "cache_ratio", None)
                if ratio_cfg is None:
                    ratio_cfg = getattr(llm.kv_manager, "target_cache_ratio", None)
                if ratio_cfg is None:
                    ratio_cfg = getattr(llm.kv_manager, "keep_ratio", None)
                try:
                    if ratio_cfg is not None:
                        target_ratio = float(ratio_cfg)
                except Exception:
                    target_ratio = 0.5
            target_ratio = max(0.0, min(1.0, target_ratio))

            if protect_prompt:
                non_prompt_total = max(0, original_total - prompt_protected_len)
                theoretical_target_kv = prompt_protected_len + int(non_prompt_total * target_ratio)
                theoretical_mode = f"prompt_protected: prompt + {target_ratio:.3f}*(new_tokens)"
            else:
                theoretical_target_kv = max(1, int(n_global * target_ratio))
                theoretical_mode = f"full_ratio: {target_ratio:.3f}*(all_tokens)"

            target_budget_kv = int(theoretical_target_kv)
            budget_mode = f"theoretical_reference: {theoretical_mode}"
            budget_applied = False
            if (
                isinstance(latest_prune_info, dict)
                and bool(latest_prune_info.get("pruned", False))
                and not bool(latest_prune_info.get("single_token_mode", False))
            ):
                try:
                    prunable_n = int(latest_prune_info.get("prunable_region_size", 0))
                    budget_eff = int(latest_prune_info.get("budget_B_effective", latest_prune_info.get("budget_B", 0)))
                    cache_before = int(latest_prune_info.get("cache_before", 0))
                    protected_total = max(0, cache_before - prunable_n)
                    target_budget_kv = max(0, protected_total + budget_eff)
                    budget_mode = (
                        f"applied_prune_budget: protected={protected_total} + prunable_B={budget_eff}"
                    )
                    budget_applied = True
                except Exception:
                    # Fallback to theoretical budget display.
                    target_budget_kv = int(theoretical_target_kv)
                    budget_mode = f"theoretical_reference: {theoretical_mode}"
                    budget_applied = False

            global_discarded = max(0, original_total - pruned_total)
            overall_keep_ratio = (pruned_total / original_total) if original_total > 0 else 0.0
            decode_total = max(0, original_total - prompt_protected_len)
            decode_kept = max(0, pruned_total - prompt_protected_len)
            decode_keep_ratio = (decode_kept / decode_total) if decode_total > 0 else 0.0
            protected_len = min(obs_window_cfg, original_total)
            protected_start = max(0, original_total - protected_len)
            protected_end = original_total - 1 if original_total > 0 else -1
            print(
                f"[STEP SUMMARY] original_num={original_num} target_ratio_kv={target_budget_kv} "
                f"kept_kv={pruned_total}"
            )
            print(
                f"[STEP SUMMARY] global original_total={original_total} kept_kv={pruned_total} "
                f"global_discarded={global_discarded} overall_keep_ratio={overall_keep_ratio:.3f} "
                f"decode_kept={decode_kept}/{decode_total} decode_keep_ratio={decode_keep_ratio:.3f}"
            )
            print(
                f"[CACHE STATUS] step={step} prompt_len={prompt_token_count} "
                f"global_original_total={original_total} current_cache={pruned_total} "
                f"global_compression={overall_keep_ratio:.3f} decode_compression={decode_keep_ratio:.3f}"
            )
            over_budget_val = max(0, pruned_total - target_budget_kv)
            budget_note = "" if budget_applied else " (this_step_not_pruned)"
            print(
                f"[STEP SUMMARY] budget_mode={budget_mode} target_budget={target_budget_kv} "
                f"over_budget={over_budget_val}{budget_note}"
            )
            print(
                f"[STEP SUMMARY] protected_window range=[{protected_start}-{protected_end}] "
                f"len={protected_len} (included_in_origin_total)"
            )
        else:
            print(
                f"[STEP SUMMARY] original_num=N/A target_ratio_kv=N/A kept_kv={pruned_total} "
                f"(enable TokenTracker for global ids)"
            )

    def _build_debug_payload():
        payload = {
            "prompt_token_count": int(prompt_token_count),
            "step_token_ranges": {str(int(k)): [int(v[0]), int(v[1])] for k, v in step_token_ranges.items()},
            "step_discarded_counts": {str(int(k)): int(v) for k, v in step_discarded_counts.items()},
            "step_spans": step_spans,
            "obs_step_ranges": {str(int(k)): [int(v[0]), int(v[1])] for k, v in obs_step_ranges.items()},
            "finalized_step_spans": sorted(int(x) for x in finalized_step_spans),
        }
        if step_scores:
            payload["step_scores"] = {str(int(k)): float(v) for k, v in step_scores.items()}
        if step_meta:
            payload["step_meta"] = {
                str(int(k)): {
                    "reward": float(v.get("reward", 0.0)),
                    "citation": float(v.get("citation", 0.0)),
                    "anchor_count": int(len(v.get("anchors", set())) if isinstance(v.get("anchors", None), set) else 0),
                }
                for k, v in step_meta.items()
                if isinstance(k, int)
            }
        if step_interrupt_events:
            payload["step_interrupt_mode"] = step_interrupt_mode
            payload["step_interrupt_events"] = step_interrupt_events
            payload["interrupted_steps"] = sorted(int(x) for x in interrupted_steps)
        if llm is not None and hasattr(llm, "get_pruning_history"):
            try:
                payload["pruning_history"] = llm.get_pruning_history()
            except Exception:
                payload["pruning_history"] = []
        if llm is not None and hasattr(llm, "token_tracker") and llm.token_tracker is not None:
            tracker = llm.token_tracker
            payload["token_tracker"] = {
                "next_global_id": int(getattr(tracker, "next_global_id", 0)),
                "cache_length": int(getattr(tracker, "cache_length", 0)),
                "step_pruning_events": {
                    ("init" if k is None else str(int(k))): sorted(set(int(x) for x in (v or [])))
                    for k, v in getattr(tracker, "step_pruning_events", {}).items()
                },
                "step_pruning_kept_events": {
                    ("init" if k is None else str(int(k))): sorted(set(int(x) for x in (v or [])))
                    for k, v in getattr(tracker, "step_pruning_kept_events", {}).items()
                },
            }
        if llm is not None and hasattr(llm, "get_global_token_id_log"):
            try:
                payload["global_token_ids"] = llm.get_global_token_id_log()
            except Exception:
                payload["global_token_ids"] = []
        return payload

    # Step 1: 初始生成
    initial_prompt = REACT_KV_INITIAL_PROMPT.format(
        examples=REACT_EXAMPLES, question=question
    ) + "Thought 1:"
    
    t_step_start = time.time()
    print(f"Generating initial thought and action...time={time.strftime('%H:%M:%S')}")
    if llm.token_tracker is not None and hasattr(llm.token_tracker, "set_current_step"):
        llm.token_tracker.set_current_step(1)
    prompt_token_count = 0
    step1_token_start_id = 0
    cache_len_before_step1 = 0
    pruning_history_before_step1 = len(llm.get_pruning_history()) if hasattr(llm, "get_pruning_history") else 0
    response, prompt_kv, generated_kv = llm.generate_first(
        initial_prompt, max_new_tokens=256, stop_strings=kv_stop_strings
    )
    step_time = time.time() - t_step_start
    print(f"Initial generation complete. Time taken: {step_time:.2f}s")
    print(f"Response:{len(response)}")
    print(response)
    print(f"[KV DEBUG] prompt_kv: type={type(prompt_kv)}, len={len(prompt_kv) if prompt_kv is not None else 'None'}")
    if prompt_kv and len(prompt_kv) > 0:
        print(f"[KV DEBUG] prompt_kv[0] shape: K={prompt_kv[0][0].shape}, V={prompt_kv[0][1].shape}")
    print(f"[KV DEBUG] generated_kv: type={type(generated_kv)}, len={len(generated_kv) if generated_kv is not None else 'None'}")
    if generated_kv and len(generated_kv) > 0:
        print(f"[KV DEBUG] generated_kv[0] shape: K={generated_kv[0][0].shape}, V={generated_kv[0][1].shape}")
    print(f"[KV DEBUG] LLM past_key_values type: {type(llm.past_key_values)}")

    thought, action_type, action_arg = parse_action_original("Thought 1:" + response, 1)
    try:
        prompt_token_count = int(prompt_kv[0][0].size(2)) if prompt_kv and len(prompt_kv) > 0 else 0
    except Exception:
        prompt_token_count = 0
    cache_len_before_step1 = int(prompt_token_count)
    step1_token_start_id = prompt_token_count
    step1_token_end_id = step1_token_start_id - 1

    # --- 【初始化】 ---
    # H2O、SnapKV、None 方法：不需要 memory_block 和 recent_kv（LLM 内部自动管理）
    # ours 方法：需要初始化这两个变量用于融合
    if use_memory_fusion:
        memory_block = None
        recent_kv = None  # 初始化 recent_kv
    else:
        # H2O 方法：跳过初始化，直接进入循环
        memory_block = None
        recent_kv = None
        print(f"[INFO] H2O method detected - using auto-managed KV cache")

    # 检查生成的 KV 是否有效
    if generated_kv is None or len(generated_kv) == 0:
        print(f"[ERROR] generated_kv is invalid: {generated_kv}")
        if return_debug:
            return "", trajectory_log, step_timings, _build_debug_payload()
        return "", trajectory_log, step_timings
    
    try:
        if use_memory_fusion:
            # ours 方法：处理 KV 流
            # derive prompt_len from prompt_kv if available
            prompt_len_val = 0
            if prompt_kv and len(prompt_kv) > 0:
                try:
                    prompt_len_val = int(prompt_kv[0][0].size(2))
                except Exception:
                    prompt_len_val = 0

            full_kv = getattr(llm, "past_key_values", None)
            recent_kv, memory_block = _process_kv_flow(prompt_kv, memory_block, generated_kv, window_size,
                                                      full_kv=full_kv, prompt_len=prompt_len_val, memory_rank=128)
            print(f"[DEBUG] After processing: recent_kv type={type(recent_kv)}, memory_block type={type(memory_block)}")
            if recent_kv is not None:
                print(f"[DEBUG] recent_kv length: {len(recent_kv[0])}")
        else:
            # H2O 等方法：跳过 KV 流处理
            print(f"[INFO] Skipping KV flow processing for {pruning_mode} method (auto-managed)")
    except Exception as e:
        if use_memory_fusion:
            print(f"Error occurred while processing kv flow: {e}")
            import traceback
            traceback.print_exc()
            # 如果处理失败，创建空的 recent_kv
            if recent_kv is None:
                # 创建一个空的 KV 作为 fallback
                num_layers = llm.model.config.num_hidden_layers
                empty_k = torch.zeros(1, llm.model.config.num_attention_heads, 0, llm.model.config.head_dim).to(llm.device)
                empty_v = torch.zeros(1, llm.model.config.num_attention_heads, 0, llm.model.config.head_dim).to(llm.device)
                recent_kv = (empty_k, empty_v)

    # 添加检查再打印
    if use_memory_fusion and recent_kv is not None:
        print(
            f"Initial KV processed. "
            f"Recent KV length: {recent_kv[0][0].size(2) if recent_kv and recent_kv[0][0] is not None else 0}, "
            f"Memory block updated: {memory_block is not None}"
        )
        # Print detailed per-component lengths for step 1
        try:
            prompt_len_print = prompt_len_val if 'prompt_len_val' in locals() else (prompt_kv[0][0].size(2) if prompt_kv and len(prompt_kv)>0 else 0)
            mem_lens = []
            if memory_block is not None:
                for m_k, m_v in memory_block:
                    mem_lens.append(int(m_k.size(2)))
            recent_lens = []
            if recent_kv is not None:
                for r_k, r_v in recent_kv:
                    recent_lens.append(int(r_k.size(2)))
            print(f"[KV LEN] step=1 prompt={prompt_len_print} memory={mem_lens} recent={recent_lens}")
            # Print full KV structure
            full_kv_initial = getattr(llm, "past_key_values", None)
            _print_kv_structure(1, prompt_len_print, memory_block, recent_kv, full_kv=full_kv_initial)
        except Exception:
            pass
    elif use_memory_fusion:
        print(f"Initial KV processed. Recent KV is None, Memory block updated: {memory_block is not None}")
        # 如果 recent_kv 仍然是 None，创建一个默认值
        num_layers = getattr(llm.model.config, 'num_hidden_layers', 1)
        num_heads = getattr(llm.model.config, 'num_attention_heads', 1)
        head_dim = getattr(llm.model.config, 'head_dim', None)
        if head_dim is None:
            hidden_size = getattr(llm.model.config, 'hidden_size', None) or getattr(llm.model.config, 'd_model', None)
            if hidden_size and num_heads:
                head_dim = int(hidden_size // num_heads)
            else:
                head_dim = 128
        empty_k = torch.zeros(1, num_heads, 0, head_dim).to(llm.device)
        empty_v = torch.zeros(1, num_heads, 0, head_dim).to(llm.device)
        recent_kv = (empty_k, empty_v)
    
    step_log = {"step": 1, "thought": thought, "action_type": action_type, "action_arg": action_arg}
    step_timings.append({"step": 1, "generation_time": step_time, "kv_cache_length": llm.get_cache_len()})
    if pruning_mode in ("h2o", "tova", "step_anchor_h2o", "step_aware_h2o", "step_inter"):
        pruning_history_after_step1 = len(llm.get_pruning_history()) if hasattr(llm, "get_pruning_history") else 0
        new_events_step1 = llm.get_pruning_history()[pruning_history_before_step1:pruning_history_after_step1] if hasattr(llm, "get_pruning_history") else []
        step_discarded_counts[1] = sum(int(e.get("tokens_evicted", 0)) for e in new_events_step1)
        if llm.token_tracker is not None and hasattr(llm.token_tracker, "next_global_id"):
            step1_token_end_id = llm.token_tracker.next_global_id - 1
        else:
            step1_token_end_id = step1_token_start_id - 1
        step_token_ranges[1] = (step1_token_start_id, step1_token_end_id)
        _print_h2o_step_summary(
            step=1,
            step_time=step_time,
            cache_len_before_step=cache_len_before_step1,
            step_token_start_id=step1_token_start_id,
            step_token_end_id=step1_token_end_id,
            prompt_token_count=prompt_token_count,
            discarded_this_step_total=step_discarded_counts.get(1, 0),
            latest_prune_info=(new_events_step1[-1] if new_events_step1 else None),
        )

    if action_type is None or action_type == "finish":
        if return_debug:
            return action_arg if action_type == "finish" else "", trajectory_log, step_timings, _build_debug_payload()
        return action_arg if action_type == "finish" else "", trajectory_log, step_timings

    obs, lookup_state = execute_action(action_type, action_arg, retriever, lookup_state)
    obs = obs.replace('\\n', '')
    step_log["observation"] = obs[:1000]
    trajectory_log.append(step_log)

    # 后续步骤：增量生成
    step = 2
    while step_limit is None or step <= step_limit:
        if (
            step_interrupt_mode not in ("", "none")
            and pruning_mode in ("step_aware_h2o", "step_inter")
            and llm.kv_manager is not None
        ):
            from kv_cache.step_interrupt import steps_to_drop_on_enter

            newly_dropped = steps_to_drop_on_enter(
                step, step_interrupt_mode, interrupted_steps
            )
            if newly_dropped:
                interrupted_steps.update(newly_dropped)
                llm.kv_manager.update_step_force_drop_ids(sorted(interrupted_steps))
                step_interrupt_events.append(
                    {
                        "enter_step": int(step),
                        "newly_dropped": [int(x) for x in newly_dropped],
                        "cumulative": sorted(int(x) for x in interrupted_steps),
                    }
                )
                print(
                    f"[INFO] Step interrupt entering step {step}: "
                    f"drop {newly_dropped}, cumulative {sorted(interrupted_steps)}"
                )

        print(f"Step {step} generation starting.")
        print("--------------------------------")
        new_text = f"\nObservation {step - 1}: {obs}\nThought {step}:"
        print("new_text:", new_text[:50])
        t_step_start = time.time()
        
        # 确保 prompt_kv 有效
        if prompt_kv is None or len(prompt_kv) == 0:
            print(f"[ERROR] prompt_kv is invalid at step {step}")
            break
            
        # 根据 pruning_mode 选择不同的调用方式
        try:
            if use_memory_fusion:
                # ours 方法：使用 generate_incremental_with_memory 进行融合
                print(f"[INFO] Step {step}: Using generate_incremental_with_memory (ours method)")
                response, obs_kv, gen_kv = llm.generate_incremental_with_memory(
                    new_text,
                    prompt_kv=prompt_kv,
                    memory_block=memory_block,
                    recent_kv=recent_kv,
                    max_new_tokens=256,
                    stop_strings=kv_stop_strings
                )
            else:
                # H2O、SnapKV、None 方法：直接使用 generate_incremental
                print(f"[INFO] Step {step}: Using generate_incremental ({pruning_mode} method)")
                if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_spans"):
                    llm.kv_manager.update_step_spans(step_spans)
                if llm.token_tracker is not None and hasattr(llm.token_tracker, "set_current_step"):
                    llm.token_tracker.set_current_step(step)
                if llm.token_tracker is not None and hasattr(llm.token_tracker, "next_global_id"):
                    step_token_start_id = llm.token_tracker.next_global_id
                else:
                    step_token_start_id = 0
                # Record exact observation span (prefill segment) as global token IDs.
                obs_token_count = 0
                if llm.token_tracker is not None and hasattr(llm.token_tracker, "next_global_id"):
                    try:
                        obs_text = f"\nObservation {step - 1}: {obs}\n"
                        obs_token_count = len(
                            llm.tokenizer(obs_text, add_special_tokens=False).input_ids
                        )
                    except Exception:
                        obs_token_count = 0
                # Finalize previous step span as one complete unit:
                # [Think/Action(prev_step) + Observation(prev_step)].
                # The observation is exactly the prefill chunk we are about to append for this step.
                prev_step = step - 1
                if (
                    pruning_mode in ("step_aware_h2o", "step_inter")
                    and prev_step in step_token_ranges
                    and obs_token_count > 0
                    and prev_step not in finalized_step_spans
                ):
                    prev_start, _ = step_token_ranges[prev_step]
                    prev_obs_start = step_token_start_id
                    prev_obs_end = prev_obs_start + obs_token_count - 1
                    adj_start = max(int(prev_start), int(last_finalized_step_end + 1))
                    adj_end = int(prev_obs_end)
                    if adj_end >= adj_start:
                        if pruning_mode == "step_aware_h2o":
                            # Build light-weight external step score from reward/citation signals.
                            prev_action_norm = _normalize_action_arg(action_arg)
                            prev_action_repeat = seen_action_args.get(prev_action_norm, 0) if prev_action_norm else 0
                            if prev_action_norm:
                                seen_action_args[prev_action_norm] = prev_action_repeat + 1
                            prev_obs_text = obs or ""
                            prev_obs_terms = extract_anchor_terms(prev_obs_text)
                            novelty_terms = prev_obs_terms.difference(seen_obs_anchor_terms)
                            novelty_ratio = (len(novelty_terms) / max(1, len(prev_obs_terms))) if prev_obs_terms else 0.0
                            if ablate_novelty:
                                novelty_ratio = 0.0
                            seen_obs_anchor_terms.update(prev_obs_terms)
                            success_flag = 1.0
                            obs_lower = prev_obs_text.lower()
                            if not ablate_success and (
                                ("could not find" in obs_lower)
                                or ("invalid action" in obs_lower)
                                or ("no results" in obs_lower)
                            ):
                                success_flag = 0.0
                            if ablate_progression:
                                reward_val = 0.0
                            else:
                                reward_val = success_flag + novelty_ratio - repeat_penalty * float(prev_action_repeat)
                            step_meta.setdefault(prev_step, {})
                            step_meta[prev_step]["anchors"] = prev_obs_terms
                            step_meta[prev_step]["reward"] = float(reward_val)
                            step_meta[prev_step].setdefault("citation", 0.0)
                            if use_step_scores:
                                step_scores[prev_step] = _compute_step_external_score(
                                    step_meta[prev_step]["reward"],
                                    step_meta[prev_step]["citation"],
                                )

                        step_spans.append({
                            "type": "step",
                            "step_id": int(prev_step),
                            "start": int(adj_start),
                            "end": int(adj_end),
                            "orig_len": int(adj_end - adj_start + 1),
                        })
                        # Align per-step range stats with full step span (not TA-only).
                        step_token_ranges[prev_step] = (int(adj_start), int(adj_end))
                        last_finalized_step_end = max(last_finalized_step_end, int(adj_end))
                        finalized_step_spans.add(prev_step)
                        if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_spans"):
                            llm.kv_manager.update_step_spans(step_spans)
                        if use_step_scores and llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_scores"):
                            llm.kv_manager.update_step_scores(step_scores)
                        if use_step_scores:
                            _print_step_scores_summary()
                if pruning_mode == "step_aware_h2o" and use_step_scores and step_meta and not ablate_citation:
                    # Citation signal: when current step text references anchors of old steps,
                    # increase their external score.
                    ref_terms = extract_anchor_terms(new_text)
                    if ref_terms:
                        for sid, meta in step_meta.items():
                            anchors = meta.get("anchors", set())
                            if not anchors:
                                continue
                            overlap = len(ref_terms.intersection(anchors))
                            if overlap <= 0:
                                continue
                            # Normalize and cap per-step citation increment per round.
                            base = float(min(len(ref_terms), len(anchors)))
                            overlap_ratio = float(overlap) / max(1.0, base)
                            citation_inc = min(1.0, overlap_ratio * 1.5)
                            meta["citation"] = float(meta.get("citation", 0.0)) + float(citation_inc)
                            step_scores[sid] = _compute_step_external_score(
                                float(meta.get("reward", 0.0)),
                                float(meta.get("citation", 0.0)),
                            )
                    if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_scores"):
                        llm.kv_manager.update_step_scores(step_scores)
                    _print_step_scores_summary()
                cache_len_before = llm.get_cache_len() if hasattr(llm, 'get_cache_len') else 0
                pruning_history_before = len(llm.get_pruning_history()) if hasattr(llm, 'get_pruning_history') else 0
                
                response = llm.generate_incremental(
                    new_text,
                    max_new_tokens=256,
                    stop_strings=kv_stop_strings
                )
                
                cache_len_after = llm.get_cache_len() if hasattr(llm, 'get_cache_len') else 0
                pruning_history_after = len(llm.get_pruning_history()) if hasattr(llm, 'get_pruning_history') else 0
                new_events = llm.get_pruning_history()[pruning_history_before:pruning_history_after] if hasattr(llm, "get_pruning_history") else []
                step_discarded_counts[step] = sum(int(e.get("tokens_evicted", 0)) for e in new_events)
                
                # 这些方法不需要返回分离的 KV，因为 LLM 内部自动管理
                obs_kv = None
                gen_kv = None
        except Exception as e:
            method_name = "generate_incremental_with_memory" if use_memory_fusion else "generate_incremental"
            print(f"Error in {method_name} at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
            
        step_time = time.time() - t_step_start
        print(f"Step {step} generation complete Response: {response}")
        if not use_memory_fusion and pruning_mode in ("h2o", "tova", "step_anchor_h2o", "step_aware_h2o", "step_inter"):
            if llm.token_tracker is not None and hasattr(llm.token_tracker, "next_global_id"):
                step_token_end_id = llm.token_tracker.next_global_id - 1
            else:
                step_token_end_id = step_token_start_id - 1
            # Current-step generated range excludes prefill observation tokens.
            gen_start_id = step_token_start_id + max(0, int(obs_token_count))
            if step_token_end_id < gen_start_id:
                gen_start_id = step_token_start_id
            step_token_ranges[step] = (gen_start_id, step_token_end_id)
            if obs_token_count > 0:
                obs_start_id = step_token_start_id
                obs_end_id = step_token_start_id + obs_token_count - 1
                if pruning_mode not in ("step_aware_h2o", "step_inter"):
                    obs_step_ranges[step] = (obs_start_id, obs_end_id)
                # Keep step_spans as full Think+Action+Observation units for step-aware mode.
                # For step-anchor mode, we still track observation-only spans.
                if pruning_mode == "step_anchor_h2o":
                    step_spans.append({"type": "obs", "start": obs_start_id, "end": obs_end_id})
                    if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_spans"):
                        llm.kv_manager.update_step_spans(step_spans)
            _print_h2o_step_summary(
                step=step,
                step_time=step_time,
                cache_len_before_step=cache_len_before if 'cache_len_before' in locals() else 0,
                step_token_start_id=gen_start_id if 'gen_start_id' in locals() else (step_token_start_id if 'step_token_start_id' in locals() else 0),
                step_token_end_id=step_token_end_id,
                prompt_token_count=prompt_token_count,
                discarded_this_step_total=step_discarded_counts.get(step, 0),
                latest_prune_info=(new_events[-1] if 'new_events' in locals() and new_events else None),
            )
        
        # 【H2O、SnapKV、None 方法】：不需要手动进行 memory_block 融合
        if not use_memory_fusion:
            print(f"[INFO] Skipping KV fusion for {pruning_mode} method (auto-managed by KVCacheManager)")
            # 对于这些方法，LLM 内部已经处理了所有的 KV 管理和剪枝
            # 我们只需继续循环
            pass
        else:
            # 【ours 方法】：需要进行 memory_block 和 recent_kv 的融合
            # 检查返回的 KV 是否有效
            if obs_kv is not None and len(obs_kv) > 0:
                print(f"obs_kv has done")
            if gen_kv is not None and len(gen_kv) > 0:
                print(f"gen_kv has done")

            print("type(obs_kv):", type(obs_kv))
            print("len(obs_kv):", len(obs_kv))

            print("type(gen_kv):", type(gen_kv))
            print("len(gen_kv):", len(gen_kv))

            print("obs_kv[0] type:", type(obs_kv[0]))
            print("gen_kv[0] type:", type(gen_kv[0]))
            
            #print("gen_kv[0]:", gen_kv[0])

            obs_kv = _normalize_kv(obs_kv, ref_kv=gen_kv)
            gen_kv = _normalize_kv(gen_kv)

            step_kv = []
            for (o_k, o_v), (g_k, g_v) in zip(obs_kv, gen_kv):
                k = torch.cat([o_k, g_k], dim=2)
                v = torch.cat([o_v, g_v], dim=2)
                step_kv.append((k, v))

            step_kv = tuple(step_kv)

            # --- 【核心逻辑：先构建 recent_kv（优先从 full_kv），再把剩余 prev 区段融合到 memory_block】 ---
            try:
                full_kv_loop = getattr(llm, "past_key_values", None)
                try:
                    prompt_len_loop = int(prompt_kv[0][0].size(2)) if prompt_kv and len(prompt_kv) > 0 else 0
                except Exception:
                    prompt_len_loop = 0

                # extract recent window first
                recent_from_full = _extract_recent_from_full(full_kv_loop, window_size) if full_kv_loop is not None else None
                if recent_from_full is not None: 
                    recent_kv = recent_from_full

                    # build prev_kv from full_kv: tokens between prompt_len_loop and end-window
                    prev_kv = []
                    try:
                        if hasattr(full_kv_loop, "layers"):
                            for layer in full_kv_loop.layers:
                                k = layer.keys
                                v = layer.values
                                prev_start = prompt_len_loop
                                prev_end = k.size(2) - window_size
                                if prev_end > prev_start:
                                    pk = k[:, :, prev_start:prev_end, :].detach()
                                    pv = v[:, :, prev_start:prev_end, :].detach()
                                else:
                                    pk = k[:, :, :0, :]
                                    pv = v[:, :, :0, :]
                                prev_kv.append((pk, pv))
                        else:
                            for k, v in full_kv_loop:
                                prev_start = prompt_len_loop
                                prev_end = k.size(2) - window_size
                                if prev_end > prev_start:
                                    pk = k[:, :, prev_start:prev_end, :].detach()
                                    pv = v[:, :, prev_start:prev_end, :].detach()
                                else:
                                    pk = k[:, :, :0, :]
                                    pv = v[:, :, :0, :]
                                prev_kv.append((pk, pv))
                    except Exception:
                        prev_kv = []

                    # if prev_kv contains any data, fuse them into memory_block
                    has_prev = any(p[0].size(2) > 0 for p in prev_kv) if prev_kv else False
                    if has_prev:
                        memory_block = _fuse_step_into_memory(memory_block, tuple(prev_kv), memory_rank=128)
                else:
                    # fallback: extract recent from this step_kv and fuse the rest of step_kv into memory
                    recent_kv = []
                    rest_kv = []
                    for n_k, n_v in step_kv:
                        s_len = n_k.size(2)
                        if s_len <= window_size:
                            # all tokens are recent
                            r_k = n_k[:, :, -s_len:, :]
                            r_v = n_v[:, :, -s_len:, :]
                            recent_kv.append((r_k, r_v))
                            rest_kv.append((n_k[:, :, :0, :], n_v[:, :, :0, :]))
                        else:
                            r_k = n_k[:, :, -window_size:, :]
                            r_v = n_v[:, :, -window_size:, :]
                            rest_k = n_k[:, :, :-window_size, :]
                            rest_v = n_v[:, :, :-window_size, :]
                            recent_kv.append((r_k, r_v))
                            rest_kv.append((rest_k, rest_v))
                    recent_kv = tuple(recent_kv)
                    # fuse remaining tokens from this step into memory
                    has_rest = any(p[0].size(2) > 0 for p in rest_kv) if rest_kv else False
                    if has_rest:
                        memory_block = _fuse_step_into_memory(memory_block, tuple(rest_kv), memory_rank=128)
            except Exception as e:
                print(f"Error building recent_kv / fusing prev into memory at step {step}: {e}")
                import traceback
                traceback.print_exc()

        # 打印每步 KV 长度信息，便于观察变化
        try:
            if use_memory_fusion:
                # ours 方法：打印 memory_block 和 recent_kv 信息
                prompt_len_print = int(prompt_kv[0][0].size(2)) if prompt_kv and len(prompt_kv)>0 else 0
                mem_lens = []
                if memory_block is not None:
                    for m_k, m_v in memory_block:
                        mem_lens.append(int(m_k.size(2)))
                recent_lens = []
                if recent_kv is not None:
                    for r_k, r_v in recent_kv:
                        recent_lens.append(int(r_k.size(2)))
                print(f"[KV LEN] step={step} prompt={prompt_len_print} memory={mem_lens} recent={recent_lens}")
                # Print full KV structure for this step
                full_kv_step = getattr(llm, "past_key_values", None)
                _print_kv_structure(step, prompt_len_print, memory_block, recent_kv, full_kv=full_kv_step)
            else:
                # H2O、SnapKV、None 方法：直接从 llm.past_key_values 获取 KV 长度
                full_kv_step = getattr(llm, "past_key_values", None)
                if full_kv_step is not None:
                    if hasattr(full_kv_step, "layers") and len(full_kv_step.layers) > 0:
                        kv_len = full_kv_step.layers[0].keys.shape[2]
                    elif isinstance(full_kv_step, tuple) and len(full_kv_step) > 0:
                        kv_len = full_kv_step[0][0].shape[2]
                    else:
                        kv_len = 0
                    print(f"[KV LEN] step={step} total_kv_length={kv_len}")
        except Exception:
            pass

        # --- 后续解析与执行 ---
        thought, action_type, action_arg = parse_action_original("Thought " + str(step) + ":" + response, step)
        step_log = {"step": step, "thought": thought, "action_type": action_type, "action_arg": action_arg}
        step_timings.append({
            "step": step, 
            "generation_time": step_time, 
            "kv_cache_length": llm.get_cache_len(),
            "pruned_this_step": llm.kv_manager.last_pruned if llm.kv_manager else False
        })
        print(f"Step {step} thought: {thought[:100]}")
        print(f"Step {step} parsed action: {action_type}[{action_arg}]")

        if action_type == "finish":
            if (
                pruning_mode in ("step_aware_h2o", "step_inter")
                and step not in finalized_step_spans
                and step in step_token_ranges
            ):
                s_start, s_end = step_token_ranges[step]
                if s_end >= s_start:
                    adj_start = max(int(s_start), int(last_finalized_step_end + 1))
                    adj_end = int(s_end)
                    if adj_end >= adj_start:
                        step_spans.append({
                            "type": "step",
                            "step_id": int(step),
                            "start": int(adj_start),
                            "end": int(adj_end),
                            "orig_len": int(adj_end - adj_start + 1),
                        })
                        finalized_step_spans.add(step)
                        last_finalized_step_end = max(last_finalized_step_end, adj_end)
                        if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_spans"):
                            llm.kv_manager.update_step_spans(step_spans)
                        if use_step_scores and llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_scores"):
                            llm.kv_manager.update_step_scores(step_scores)
            trajectory_log.append(step_log)
            _print_step_spans_summary()
            _print_step_scores_summary()
            if llm.token_tracker is not None:
                llm.token_tracker.print_final_summary()
            if return_debug:
                return action_arg if action_arg else "", trajectory_log, step_timings, _build_debug_payload()
            return action_arg if action_arg else "", trajectory_log, step_timings

        obs, lookup_state = execute_action(action_type, action_arg, retriever, lookup_state)
        obs = obs.replace('\\n', '')
        obs_preview = obs if len(obs) <= 800 else obs[:800] + "..."
        print(f"Step {step} observation ({len(obs)} chars): {obs_preview}")
        step_log["observation"] = obs[:1000]
        trajectory_log.append(step_log)
        step += 1

    # Print final token tracking summary
    _print_step_spans_summary()
    _print_step_scores_summary()
    if llm.token_tracker is not None:
        llm.token_tracker.print_final_summary()

    if return_debug:
        return "", trajectory_log, step_timings, _build_debug_payload()
    return "", trajectory_log, step_timings


# ==================== Collect Results ====================
def collect_results():
    """Collect all experiment results and generate the final summary markdown."""
    experiments = {
        "single": ("Single Model (Direct QA)", "single_wiki_500_0318.json"),
        "rag": ("RAG (Full Wiki)", "rag_wiki_500_0318.json"),
        "react": ("ReAct (Full Wiki)", "react_wiki_500_0318.json"),
        "react_kv_none": ("ReAct-KV (none)", "react_kv_none_wiki_500_0318.json"),
        "react_kv_h2o": ("ReAct-KV (H2O)", "react_kv_h2o_wiki_500_0318.json"),
        "react_kv_tova": ("ReAct-KV (TOVA)", "react_kv_tova_wiki_500_0513.json"),
        "react_kv_step_inter": ("ReAct-KV (Step-Inter)", "react_kv_step_inter_wiki_500_0504.json"),
        "react_kv_snapkv": ("ReAct-KV (SnapKV)", "react_kv_snapkv_wiki_500_0318.json"),
        "react_kv_h2o_prune": ("ReAct-KV (H2O, Aggressive Prune)", "react_kv_h2o_prune_wiki_500_0318.json"),
        "react_kv_ours": ("ReAct-KV (Ours)", "react_kv_ours_wiki_500_0331.json"),
    }

    results_table = []
    for exp_key, (name, filename) in experiments.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            results_table.append({
                "method": name,
                "em": summary.get("exact_match", 0),
                "f1": summary.get("f1_score", 0),
                "total_time": summary.get("total_time_seconds", 0),
                "avg_time": summary.get("avg_time_per_sample", 0),
                "n_samples": summary.get("total_samples", 0),
                "extra": summary.get("timing_stats", {}),
                "prune_count": summary.get("total_prune_count", 0),
                "avg_peak_mb": summary.get("avg_peak_memory_mb", 0),
            })
        else:
            print(f"[WARN] Missing result file: {filepath}")
            results_table.append({
                "method": name, "em": -1, "f1": -1,
                "total_time": 0, "avg_time": 0, "n_samples": 0,
                "extra": {}, "prune_count": 0, "avg_peak_mb": 0,
            })

    # Generate markdown report
    md = []
    md.append("# HotpotQA Experiment Results — Full Wikipedia Corpus")
    md.append("")
    md.append(f"**Date**: 2026-03-18")
    md.append(f"**Model**: Qwen2.5-7B-Instruct")
    md.append(f"**Dataset**: HotpotQA Dev Validation (500 samples, seed={RANDOM_SEED})")
    md.append(f"**Retrieval Corpus**: TIGER-Lab/LongRAG Wikipedia (~5.2M articles, BM25)")
    md.append(f"**Max ReAct Steps**: {format_max_steps(MAX_STEPS)}")
    md.append(f"**BM25 Top-K**: {BM25_TOP_K}")
    md.append("")
    md.append("## Key Differences from Previous Experiments")
    md.append("")
    md.append("- **Previous**: Used distractor setting (10 per-sample context paragraphs)")
    md.append("- **Current**: Uses full Wikipedia corpus (~5.2M articles) for open-domain retrieval")
    md.append("- **Original ReAct paper**: Uses live Wikipedia API; we approximate with offline BM25 over full Wikipedia")
    md.append("- **Sampling**: Follows original paper — shuffle with seed=233, take first 500")
    md.append("")
    md.append("## Results Summary")
    md.append("")
    md.append("| Method | EM (%) | F1 (%) | Avg Time/Sample (s) | Total Time | Avg Peak Mem (MB) |")
    md.append("|--------|--------|--------|---------------------|------------|-------------------|")

    for r in results_table:
        if r["em"] < 0:
            md.append(f"| {r['method']} | N/A | N/A | N/A | N/A | N/A |")
        else:
            hours = r["total_time"] / 3600
            time_str = f"{r['total_time']:.0f}s ({hours:.1f}h)" if hours > 1 else f"{r['total_time']:.0f}s"
            md.append(f"| {r['method']} | {r['em']:.2f} | {r['f1']:.2f} | {r['avg_time']:.1f} | {time_str} | {r.get('avg_peak_mb', 0):.0f} |")

    md.append("")
    md.append("## Analysis")
    md.append("")

    # Find ReAct baseline and compare
    react_em = None
    react_f1 = None
    for r in results_table:
        if r["method"] == "ReAct (Full Wiki)" and r["em"] >= 0:
            react_em = r["em"]
            react_f1 = r["f1"]
            break

    if react_em is not None:
        md.append("### Comparison vs ReAct Baseline")
        md.append("")
        md.append("| Method | EM Δ | F1 Δ |")
        md.append("|--------|------|------|")
        for r in results_table:
            if r["em"] >= 0:
                em_diff = r["em"] - react_em
                f1_diff = r["f1"] - react_f1
                md.append(f"| {r['method']} | {em_diff:+.2f} | {f1_diff:+.2f} |")
        md.append("")

    # KV Cache specific analysis
    kv_methods = [r for r in results_table if "KV" in r["method"] and r["em"] >= 0]
    if kv_methods:
        md.append("### KV Cache Analysis")
        md.append("")
        for r in kv_methods:
            extra = r.get("extra", {})
            md.append(f"**{r['method']}**:")
            if extra.get("avg_kv_cache_length"):
                md.append(f"- Avg KV Cache Length: {extra['avg_kv_cache_length']:.0f}")
            if extra.get("max_kv_cache_length"):
                md.append(f"- Max KV Cache Length: {extra['max_kv_cache_length']}")
            if extra.get("avg_step_time"):
                md.append(f"- Avg Step Time: {extra['avg_step_time']:.2f}s")
            if r.get("prune_count", 0) > 0:
                md.append(f"- Total Prune Operations: {r['prune_count']}")
            md.append("")

    md.append("## Methodology Notes")
    md.append("")
    md.append("1. **Single Model**: Direct question → answer, no retrieval")
    md.append("2. **RAG**: Single BM25 retrieval pass → read → answer")
    md.append("3. **ReAct**: Multi-step interleaved reasoning with search/lookup/finish actions")
    md.append("4. **ReAct-KV (none)**: ReAct with KV cache reuse across steps (no pruning)")
    md.append("5. **ReAct-KV (H2O)**: ReAct-KV with Heavy Hitter Oracle pruning (keep_ratio=0.5)")
    md.append("6. **ReAct-KV (SnapKV)**: ReAct-KV with SnapKV attention pooling pruning")
    md.append("")
    md.append("All ReAct variants use the same 6-shot prompt from the original ReAct paper (Yao et al., 2022).")
    md.append("")

    report = "\n".join(md)
    with open(SUMMARY_PATH, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Summary report saved to: {SUMMARY_PATH}")
    print(f"{'='*60}")
    print(report)


# ==================== Main ====================
def main():
    global MODEL_PATH, RANDOM_SEED, MAX_STEPS
    parser = argparse.ArgumentParser(description="Run all HotpotQA experiments with full Wikipedia corpus")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["single", "rag", "react", "react_kv_none",
                                 "react_kv_h2o", "react_kv_tova", "react_kv_step_anchor_h2o", "react_kv_step_aware_h2o", "react_kv_step_inter", "react_kv_snapkv", "ours", "collect", "all"],
                        help="Which experiment to run")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: wiki_0318_v2)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="HF model path or local model directory")
    parser.add_argument("--cache_ratio", type=float, default=None,
                        help="Override cache ratio for KV methods (e.g. 0.2 or 0.5)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for sample selection (override for repeated runs)")
    parser.add_argument("--max_steps", type=str, default=str(MAX_STEPS),
                        help="Max ReAct steps per sample, or 'unlimited' to run until finish")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    # RANDOM_SEED = int(args.seed)
    RANDOM_SEED = 42
    MAX_STEPS = parse_max_steps(args.max_steps)
    print(f"[INFO] Using model: {MODEL_PATH}")
    print(f"[INFO] Using random seed: {RANDOM_SEED}")
    print(f"[INFO] Max ReAct steps: {format_max_steps(MAX_STEPS)}")

    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.experiment == "collect":
        collect_results()
        return

    # Load data
    val_data = load_hotpotqa_data()
    selected_samples = select_samples(val_data)

    # Experiments that need wiki retriever
    needs_retriever = args.experiment in ["rag", "react", "react_kv_none",
                                           "react_kv_h2o", "react_kv_tova", "react_kv_step_anchor_h2o", "react_kv_step_aware_h2o", "react_kv_step_inter", "react_kv_snapkv", "ours", "all"]

    retriever = None
    if needs_retriever:
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        if not os.path.exists(os.path.join(WIKI_INDEX_DIR, "titles.json")):
            print(f"[ERROR] Wiki index not found at {WIKI_INDEX_DIR}. "
                  f"Run build_wiki_index.py first.")
            return
        retriever = WikiBM25Retriever(index_dir=WIKI_INDEX_DIR, load_corpus=True)

    if args.experiment == "single" or args.experiment == "all":
        run_single_experiment(
            val_data, selected_samples,
            os.path.join(output_dir, "single_wiki_500_0318.json"),
            os.path.join(output_dir, "single_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "rag" or args.experiment == "all":
        run_rag_experiment(
            val_data, selected_samples, retriever,
            os.path.join(output_dir, "rag_wiki_500_0318.json"),
            os.path.join(output_dir, "rag_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "react" or args.experiment == "all":
        run_react_experiment(
            val_data, selected_samples, retriever,
            os.path.join(output_dir, "react_wiki_500_0318.json"),
            os.path.join(output_dir, "react_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "react_kv_none" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "none",
            os.path.join(output_dir, "react_kv_none_wiki_500_0318.json"),
            os.path.join(output_dir, "react_kv_none_wiki_500_0318_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "react_kv_h2o" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "h2o",
            os.path.join(output_dir, "react_kv_h2o_wiki_500_0515.json"),
            os.path.join(output_dir, "react_kv_h2o_wiki_500_0515_v2_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "react_kv_tova" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "tova",
            os.path.join(output_dir, "react_kv_tova_wiki_500_0513.json"),
            os.path.join(output_dir, "react_kv_tova_wiki_500_0513_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "react_kv_step_anchor_h2o" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "step_anchor_h2o",
            os.path.join(output_dir, "react_kv_step_anchor_h2o_wiki_500_0415.json"),
            os.path.join(output_dir, "react_kv_step_anchor_h2o_wiki_500_0415_v4_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "react_kv_step_aware_h2o" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "step_aware_h2o",
            os.path.join(output_dir, "react_kv_step_aware_h2o_wiki_500_0502.json"),
            os.path.join(output_dir, "react_kv_step_aware_h2o_wiki_500_0502_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "react_kv_step_inter" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "step_inter",
            os.path.join(output_dir, "react_kv_step_inter_wiki_500_0504.json"),
            os.path.join(output_dir, "react_kv_step_inter_wiki_500_0504_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "react_kv_snapkv" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "snapkv",
            os.path.join(output_dir, "react_kv_snapkv_wiki_500_0318.json"),
            os.path.join(output_dir, "react_kv_snapkv_wiki_500_0318_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "ours" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "ours",
            os.path.join(output_dir, "react_kv_ours_wiki_500_0331.json"),
            os.path.join(output_dir, "react_kv_ours_wiki_500_0331_checkpoint.json"),
            kv_config_override={"cache_ratio": float(args.cache_ratio)} if args.cache_ratio is not None else None,
        )

    if args.experiment == "collect" or args.experiment == "all":
        if args.experiment == "all":
            print("\n[INFO] All experiments completed. Collecting results...")
        collect_results()


if __name__ == "__main__":
    main()


# # Run each experiment independently:
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment single         # 1. Direct QA, no retrieval
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment rag            # 2. RAG (BM25 retrieve → read)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react          # 3. ReAct (multi-step reasoning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_none  # 4. ReAct-KV (no pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_h2o   # 5. ReAct-KV (H2O pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_snapkv# 6. ReAct-KV (SnapKV pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_ours  # 7. ReAct-KV (Ours, aggressive pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment collect        # Generate summary report
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment all            # Run ALL above sequentially