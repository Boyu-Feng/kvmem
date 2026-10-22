"""CPU-only, real-transformer tests. No checkpoints, GPUs, or downloads."""
from __future__ import annotations

import copy
import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicSlidingWindowLayer

from kv_cache.h2o_scorer import H2OScorer
from models.AgentKVPolicy import AgentKVPolicy, ContextOverflowError, UnsupportedCacheError


class TinyChatTokenizer:
    """Byte tokenizer with native Qwen2.5-style ChatML boundaries."""

    eos_token_id = 2
    name_or_path = "injected-byte-chat-tokenizer"
    special = {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3}

    def __init__(self):
        self.template_calls = []

    def get_chat_template(self):
        return "test-native-chatml-v1"

    def convert_tokens_to_ids(self, token):
        return self.special.get(token)

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        while text:
            marker = next((s for s in self.special if text.startswith(s)), None)
            if marker:
                ids.append(self.special[marker])
                text = text[len(marker):]
            else:
                ids.extend(4 + value for value in text[0].encode("utf-8"))
                text = text[1:]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert clean_up_tokenization_spaces is False
        pieces, byte_buffer = [], []
        reverse = {v: k for k, v in self.special.items()}

        def flush():
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token in ids:
            if token < 4:
                flush()
                if not skip_special_tokens:
                    pieces.append(reverse.get(token, "<pad>"))
            else:
                byte_buffer.append(token - 4)
        flush()
        return "".join(pieces)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and enable_thinking is False
        self.template_calls.append((copy.deepcopy(messages), add_generation_prompt, enable_thinking))
        rendered = ""
        for message in messages:
            header = f"<|im_start|>{message['role']}\n"
            rendered += header + message["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(max_position_embeddings=4096):
    torch.manual_seed(19)
    common = dict(
        vocab_size=260, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings, eos_token_id=[2, 3], pad_token_id=0,
        attention_dropout=0.0,
    )
    config = Qwen2Config(**common)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).float().eval()


def tiny_policy(method="fullkv", ratio=1.0, **kwargs):
    return AgentKVPolicy._from_components(
        tiny_model(), TinyChatTokenizer(),
        method=method, keep_ratio=ratio, max_new_tokens=32, **kwargs,
    )


def teacher_logits(policy):
    with torch.inference_mode():
        return policy.model(
            input_ids=torch.tensor([policy._token_ids]), use_cache=False,
            position_ids=torch.arange(len(policy._token_ids))[None, :],
            logits_to_keep=1,
        ).logits[:, -1, :]


def emit(policy, monkeypatch, text="look", eos=2):
    """Force an action while retaining every genuine model/attention forward."""
    ids = policy.tokenizer(text).input_ids + ([] if eos is None else [eos])
    if eos is None:
        policy.max_new_tokens = len(ids)
    original = policy._append
    cursor = 0

    def next_logits(like):
        result = like.new_full((1, policy.text_config.vocab_size), -100.0)
        result[0, ids[min(cursor, len(ids) - 1)]] = 100.0
        return result

    def append(block):
        nonlocal cursor
        is_header = bool(policy._next_header and block == policy._next_header)
        logits = original(block)
        if not is_header:
            assert block == [ids[cursor]]
            cursor += 1
        policy._logits = next_logits(logits)
        return policy._logits

    if policy._logits is not None:
        policy._logits = next_logits(policy._logits)
    with monkeypatch.context() as patch:
        patch.setattr(policy, "_append", append)
        result = policy.act()
    assert cursor == len(ids)
    assert policy._pending_generated_ids == ids
    return result, ids


def test_uncompressed_incremental_logits_match_unpruned_hf():
    policy = tiny_policy()
    for block in ([11, 25, 19], [8], [41, 15, 27, 33], [17, 23]):
        actual = policy._append(list(block))
        torch.testing.assert_close(actual, teacher_logits(policy), atol=2e-6, rtol=2e-5)
    audit = policy.audit()
    assert audit["logical_tokens"] == audit["resident_tokens"] == 10
    assert audit["retained_global_ids"] == list(range(10))
    assert audit["forward_count"] == 4
    assert audit["extra_scoring_forwards"] == 0
    assert len(policy.past_key_values.layers) == len(policy.full_attention_layers) == 4


def test_fullkv_greedy_tokens_match_hf_generate():
    policy = tiny_policy()
    policy.max_new_tokens = 5
    policy.reset("Return one command.", "You see a table.")
    initial = list(policy._token_ids)
    with torch.inference_mode():
        expected = policy.model.generate(
            torch.tensor([initial]), max_new_tokens=5, do_sample=False, num_beams=1,
            repetition_penalty=1.0, eos_token_id=sorted(policy._eos_ids), pad_token_id=0,
        )[0, len(initial):].tolist()
    policy.act()
    assert policy._pending_generated_ids == expected
    assert policy._token_ids == initial + expected
    torch.testing.assert_close(policy._logits, teacher_logits(policy), atol=2e-6, rtol=2e-5)


def test_real_middle_eviction_matches_masked_reference_and_continues():
    policy = tiny_policy()
    policy._append([11, 12, 13, 14, 15, 16, 17, 18])
    policy._protected_tokens = 2
    old_ids = list(policy._token_ids)
    reference_cache = copy.deepcopy(policy.past_key_values)
    original_pairs = {
        i: (policy.past_key_values.layers[i].keys.clone(), policy.past_key_values.layers[i].values.clone())
        for i in policy.full_attention_layers
    }
    before = policy.audit()
    keep = [0, 1, 3, 6, 7]
    policy._compact(keep)
    after = policy.audit()
    assert policy._token_ids == old_ids
    assert after["retained_global_ids"] == keep
    assert after["full_attention_kv_bytes"] == before["full_attention_kv_bytes"] * 5 // 8
    for i, (keys, values) in original_pairs.items():
        torch.testing.assert_close(policy.past_key_values.layers[i].keys, keys[:, :, keep, :], rtol=0, atol=0)
        torch.testing.assert_close(policy.past_key_values.layers[i].values, values[:, :, keep, :], rtol=0, atol=0)

    mask = torch.zeros(1, 1, 2, 10)
    mask[..., [2, 4, 5]] = torch.finfo(torch.float32).min
    mask[:, :, 0, 9] = torch.finfo(torch.float32).min
    with torch.inference_mode():
        expected = policy.model(
            input_ids=torch.tensor([[31, 32]]), position_ids=torch.tensor([[8, 9]]),
            attention_mask=mask,
            past_key_values=reference_cache, use_cache=True, logits_to_keep=1,
        ).logits[:, -1, :]
    actual = policy._append([31, 32])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert policy.audit()["last_forward_positions"] == [8, 9]
    assert policy.audit()["retained_global_ids"] == keep + [8, 9]


@pytest.mark.parametrize("method", ["fullkv", "stepkv", "stepkv_no_step_score", "h2o_step", "tova_step"])
def test_scores_come_from_actual_query_attention(method):
    policy = tiny_policy(method, 0.2)
    ids = [11, 15, 19, 23, 27]
    policy._append(ids)
    with torch.inference_mode():
        reference = policy.model(
            input_ids=torch.tensor([ids]), use_cache=False, output_attentions=True,
        ).attentions
    expected = H2OScorer(num_score_layers=policy.SCORE_LAYERS).compute_scores(reference)
    torch.testing.assert_close(policy._hh_scores, expected)
    torch.testing.assert_close(policy._last_query_scores, reference[-1][0, :, -1, :].mean(dim=0))
    assert policy.audit()["forward_count"] == 1
    assert policy.audit()["attention_query_count"] == len(ids)
    assert torch.unique(policy._hh_scores).numel() > 1


@pytest.mark.parametrize("method", ["fullkv", "stepkv", "stepkv_no_step_score", "h2o_step", "tova_step"])
def test_score_history_matches_qa_for_stepkv_and_preserves_baselines(method, monkeypatch):
    policy = tiny_policy(method, 0.2)
    policy.PREFILL_CHUNK_SIZE = 3
    scorer = H2OScorer(num_score_layers=policy.SCORE_LAYERS)
    stepkv = method in ("stepkv", "stepkv_no_step_score")
    expected = torch.empty(0)
    forwards = 0
    original = policy.model.forward

    def forward(*args, **kwargs):
        nonlocal expected, forwards
        outputs = original(*args, **kwargs)
        current = scorer.compute_scores(outputs.attentions)
        queries = kwargs["input_ids"].shape[-1]
        expected = current if stepkv else torch.cat((expected, torch.zeros(queries))) + current
        forwards += 1
        return outputs

    monkeypatch.setattr(policy.model, "forward", forward)
    policy._append([11, 15, 19, 23, 27])
    assert forwards == 2
    torch.testing.assert_close(policy._hh_scores, expected)
    if stepkv:
        torch.testing.assert_close(expected.sum(), torch.tensor(2.0 * policy.text_config.num_attention_heads))
        policy._hh_scores = torch.full_like(policy._hh_scores, 1e6)
    policy._append([31, 35])
    torch.testing.assert_close(policy._hh_scores, expected)
    assert policy._hh_scores.shape == (7,)
    if stepkv:
        assert policy._hh_scores.max() < 1e6

    keep = [0, 2, 5, 6]
    expected = expected[keep]
    policy._compact(keep)
    torch.testing.assert_close(policy._hh_scores, expected)
    if stepkv:
        policy._hh_scores = torch.full_like(policy._hh_scores, 1e6)
    policy._append([39, 43, 47, 51])
    torch.testing.assert_close(policy._hh_scores, expected)
    assert forwards == policy.audit()["forward_count"] == 5
    assert policy._global_ids == keep + [7, 8, 9, 10]
    assert policy._hh_scores.shape == (8,)
    assert policy.audit()["extra_scoring_forwards"] == 0
    assert policy.fingerprint()["token_score_history"] == (
        "latest_forward_only" if stepkv else "latest_query_only" if method == "tova_step" else "episode_cumulative"
    )


@pytest.mark.parametrize("eos", [2, 3, None])
def test_exact_emitted_tokens_eos_closures_and_native_header_continuation(eos, monkeypatch):
    policy = tiny_policy()
    policy.reset("Only commands.", "Find an apple.")
    prefix = list(policy._token_ids)
    action, generated = emit(policy, monkeypatch, "look", eos=eos)
    assert action == "look"
    assert policy._token_ids == prefix + generated
    suffix, header, forced = policy._observation_parts("An apple is on table 1.")
    policy.observe(action, "An apple is on table 1.", valid=True)
    assert policy._token_ids == prefix + generated + suffix
    assert policy._next_header == header and policy._logits is None
    assert forced is (eos != 2)
    assert policy.audit()["steps"][0]["forced_assistant_close"] == forced
    before_next = list(policy._token_ids)
    second, generated2 = emit(policy, monkeypatch, "go", eos=2)
    assert policy._token_ids == before_next + header + generated2
    assert second == "go"
    assert all(call[2] is False for call in policy.tokenizer.template_calls)
    # Genuine logits were replaced only to choose test actions. Recompute the
    # final token with the full reference to verify the entire preserved cache.
    actual = policy._append([42])
    torch.testing.assert_close(actual, teacher_logits(policy), atol=2e-6, rtol=2e-5)


def test_boundary_prune_precedes_next_header_no_probe_or_replay(monkeypatch):
    policy = tiny_policy("stepkv", 0.2)
    policy.reset("Commands.", "A kitchen.")
    emit(policy, monkeypatch)
    suffix, header, _ = policy._observation_parts("Apple 1 is on table 2; cabinet 3 is open.")
    previous_tokens, previous_calls = len(policy._token_ids), policy._forward_count
    policy.observe("look", "Apple 1 is on table 2; cabinet 3 is open.", valid=True)
    assert policy._forward_count - previous_calls == math_ceil_chunks(len(suffix), policy.PREFILL_CHUNK_SIZE)
    assert len(policy._token_ids) == previous_tokens + len(suffix)
    assert policy._logits is None and policy._next_header == header
    assert policy.audit()["prune_event_count"] == 1
    prior_ids = list(policy._global_ids)
    start = len(policy._token_ids)
    emit(policy, monkeypatch, "go")
    assert policy._global_ids[:len(prior_ids)] == prior_ids
    assert policy._token_ids[start:start + len(header)] == header
    assert policy.audit()["extra_scoring_forwards"] == 0


def math_ceil_chunks(length, chunk):
    return (length + chunk - 1) // chunk


@pytest.mark.parametrize("method,ratio", [
    ("fullkv", 1.0), ("stepkv", 0.2), ("stepkv", 0.5),
    ("stepkv_no_step_score", 0.2), ("h2o_step", 0.2), ("h2o_step", 0.5), ("tova_step", 0.2),
])
def test_prompt_protection_cumulative_budget_and_actual_eviction_audit(method, ratio, monkeypatch):
    policy = tiny_policy(method, ratio)
    policy.reset("Return one action.", "Kitchen task: find apple.")
    protected = policy.audit()["protected_logical_tokens"]
    previous_logical = protected
    for index in range(3):
        action, _ = emit(policy, monkeypatch, "look")
        policy.observe(action, f"Apple {index + 1} is near cabinet {index + 2} on table 7.", valid=True)
        state = policy.audit()
        logical_nonprompt = state["logical_tokens"] - protected
        target = logical_nonprompt if method == "fullkv" else max(1, int(ratio * logical_nonprompt))
        assert state["target_nonprompt_tokens"] == target
        assert state["resident_tokens"] == protected + target
        assert state["logical_tokens"] > previous_logical
        assert state["retained_global_ids"][:protected] == list(range(protected))
        assert state["protected_resident_tokens"] == protected
        assert state["full_attention_kv_bytes_by_device"] == {"cpu": state["full_attention_kv_bytes"]}
        assert state["peak_boundary_kv_bytes_by_device"] == {"cpu": state["peak_boundary_kv_bytes"]}
        if method != "fullkv":
            event = state["prune_history"][-1]
            assert type(state["prune_events"]) is int
            assert state["prune_events"] == state["prune_event_count"] == len(state["prune_history"])
            assert event["before_tokens"] > event["after_tokens"]
            assert event["before_kv_bytes"] > event["after_kv_bytes"]
            assert event["before_kv_bytes_by_device"] == {"cpu": event["before_kv_bytes"]}
            assert event["after_kv_bytes_by_device"] == {"cpu": event["after_kv_bytes"]}
            assert event["evicted_count"] == len(event["evicted_global_ids"])
            assert set(event["evicted_global_ids"]).isdisjoint(event["retained_global_ids"])
            per_token_bytes = state["full_attention_kv_bytes"] // state["resident_tokens"]
            assert event["before_kv_bytes"] - event["after_kv_bytes"] == event["evicted_count"] * per_token_bytes
            assert state["peak_boundary_kv_bytes"] >= event["before_kv_bytes"]
            assert state["total_evicted_tokens"] == state["logical_tokens"] - state["resident_tokens"]
            if method.startswith("stepkv"):
                for floor in event["step_floors"]:
                    assert state["steps"][floor["step_id"] - 1]["resident_tokens"] >= floor["floor"]
        else:
            assert state["prune_event_count"] == 0
            assert state["prune_events"] == 0 and state["prune_history"] == []
        previous_logical = state["logical_tokens"]
    json.dumps(policy.audit())
    json.dumps(policy.fingerprint())


def test_prune_event_integer_does_not_count_non_eviction_boundaries(monkeypatch):
    policy = tiny_policy("stepkv", 1.0)
    policy.reset("Return commands.", "A kitchen.")
    action, _ = emit(policy, monkeypatch)
    policy.observe(action, "Apple 1 is on table 2.", valid=True)
    state = policy.audit()
    assert state["boundary_count"] == 1
    assert type(state["prune_events"]) is int and state["prune_events"] == 0
    assert state["prune_event_count"] == state["total_evicted_tokens"] == 0
    assert state["prune_history"] == []
    assert state["resident_tokens"] == state["logical_tokens"]


def selection_fixture(method, ratio=0.2):
    policy = tiny_policy(method=method, ratio=ratio)
    policy._append(list(range(10, 32)))
    policy._protected_tokens = 2
    policy._steps = [
        {"id": 1, "start": 2, "end": 12, "utility": 0.0},
        {"id": 2, "start": 12, "end": 22, "utility": 2.0},
    ]
    policy._hh_scores = torch.zeros(22)
    policy._last_query_scores = torch.zeros(22)
    return policy


def test_step_utility_changes_selection_and_no_step_ablation_preserves_floors():
    policy = selection_fixture("stepkv")
    keep, details = policy._select_keep()
    assert keep == [0, 1, 2, 12, 13, 14]
    assert [entry["floor"] for entry in details["step_floors"]] == [1, 1]
    policy.method = "stepkv_no_step_score"
    ablated, details = policy._select_keep()
    assert ablated == [0, 1, 2, 3, 4, 12]
    assert [entry["floor"] for entry in details["step_floors"]] == [1, 1]


def test_token_saliency_is_used_and_step_floors_use_current_resident_spans():
    policy = selection_fixture("stepkv_no_step_score")
    policy._hh_scores[10] = 100
    policy._hh_scores[19] = 80
    keep, _ = policy._select_keep()
    assert 10 in keep and 19 in keep
    policy._compact(keep)
    keep_again, details = policy._select_keep()
    assert len(keep_again) == len(keep)
    assert sum(entry["resident_before"] for entry in details["step_floors"]) == 4
    assert [entry["floor"] for entry in details["step_floors"]] == [1, 1]
    assert policy._target_nonprompt() == 4  # Not 20% of the already compacted four.


@pytest.mark.parametrize("method", ["stepkv", "stepkv_no_step_score"])
def test_new_attention_can_evict_old_heavy_hitters_under_unchanged_budget(method, monkeypatch):
    policy = selection_fixture(method)
    for step in policy._steps:
        step["utility"] = 0.0
    policy._hh_scores[2:12] = 1e6
    keep, _ = policy._select_keep()
    assert keep == [0, 1, 2, 3, 4, 12]
    policy._compact(keep)
    original = policy._attention_scores

    def favor_new_tokens(attentions, queries, keys):
        _, last = original(attentions, queries, keys)
        mass = torch.zeros(keys)
        mass[-queries:] = torch.arange(1, queries + 1, dtype=torch.float32)
        return mass, last

    monkeypatch.setattr(policy, "_attention_scores", favor_new_tokens)
    policy._append([41, 42, 43, 44])
    policy._steps.append({"id": 3, "start": 22, "end": 26, "utility": 0.0})
    keep, details = policy._select_keep()
    assert policy._target_nonprompt() == 4
    assert [entry["floor"] for entry in details["step_floors"]] == [1, 1, 1]
    assert [policy._global_ids[index] for index in keep] == [0, 1, 2, 12, 24, 25]
    policy._compact(keep)
    assert 3 not in policy._global_ids and 4 not in policy._global_ids


def test_infeasible_step_floors_fail_without_relaxing_the_budget():
    policy = selection_fixture("stepkv", ratio=0.01)
    with pytest.raises(RuntimeError, match="cannot satisfy"):
        policy._select_keep()
    assert len(policy._global_ids) == 22


def test_h2o_window_and_tova_query_scores_are_distinct():
    policy = tiny_policy(method="h2o_step", ratio=0.5)
    policy._append(list(range(10, 112)))
    policy._protected_tokens = 2
    policy._hh_scores = torch.zeros(102)
    policy._last_query_scores = torch.zeros(102)
    policy._hh_scores[10] = 100
    policy._last_query_scores[50] = 100
    heavy, details = policy._select_keep()
    assert details["recent_tokens_reserved"] == 32
    assert set(range(70, 102)).issubset(heavy) and 10 in heavy
    policy.method = "tova_step"
    tova, details = policy._select_keep()
    assert details["recent_tokens_reserved"] == 0
    assert 50 in tova and 101 not in tova
    policy.method = "h2o_step"
    policy.keep_ratio = 0.2
    recent_only, details = policy._select_keep()
    assert details["recent_tokens_reserved"] == 20
    assert recent_only == [0, 1] + list(range(82, 102))


def test_observable_validity_novelty_repeats_and_dynamic_citation_only(monkeypatch):
    policy = tiny_policy(method="stepkv", ratio=0.5)
    policy.reset("Commands only.", "Kitchen.")
    action, _ = emit(policy, monkeypatch, "look")
    with pytest.raises(TypeError):
        policy.observe(action, "Apple 1 on table 2.", valid=True, reward=1.0)
    with pytest.raises(TypeError):
        policy.observe(action, "Apple 1 on table 2.", valid=True, success=True)
    with pytest.raises(TypeError, match="observable bool"):
        policy.observe(action, "Apple 1 on table 2.", valid=1)
    policy.observe(action, "Apple 1 on table 2.", valid=True)
    first = policy.audit()["steps"][0]
    assert first["novelty"] == 1.0 and first["progress"] == 2.0
    assert first["citation"] == 1.0
    assert first["utility"] == pytest.approx(AgentKVPolicy._utility(2.0, 1.0))
    action, generated = emit(policy, monkeypatch, "look")
    before = list(policy._token_ids)
    policy.observe("  LOOK  ", "Apple 1 on table 2.", valid=False)
    current = policy.audit()["steps"]
    assert current[1]["novelty"] == 0
    assert current[1]["repeat_count"] == 1
    assert current[1]["progress"] == pytest.approx(-0.3)
    assert current[1]["action_changed_by_caller"] is True
    assert current[1]["generated_token_ids"] == generated
    assert policy._token_ids[:len(before)] == before
    assert current[0]["citation"] == 2.0
    assert current[0]["utility"] > first["utility"]
    assert policy.fingerprint()["hidden_reward_or_success_used"] is False
    assert policy.fingerprint()["score_inputs"] == [
        "observable valid", "executed action text", "returned observation text",
    ]


def test_context_overflow_counts_evicted_logical_positions():
    policy = tiny_policy(max_context_tokens=5)
    policy._append([11, 12, 13])
    policy._compact([0, 2])
    with pytest.raises(ContextOverflowError, match="Logical context overflow"):
        policy._append([14, 15, 16])
    assert policy._token_ids == [11, 12, 13]
    assert policy._global_ids == [0, 2]
    policy._append([14, 15])
    assert policy._last_forward_positions == [3, 4]
    with pytest.raises(ContextOverflowError):
        policy._append([16])


def test_native_32768_context_is_allowed_without_expanding_action_decode(monkeypatch):
    policy = AgentKVPolicy._from_components(
        tiny_model(max_position_embeddings=32768), TinyChatTokenizer(),
        method="stepkv", keep_ratio=0.2, max_context_tokens=32768, max_new_tokens=128,
    )
    policy._append([11, 12, 13])
    # Simulate already-evicted history after the retained prefix. Qwen2 has no
    # mutable sequence state beyond KV, so the old prefix remains a real cache.
    policy._token_ids.extend([20] * (32736 - len(policy._token_ids)))
    positions = []
    original = policy.model.forward

    def forward(*args, **kwargs):
        positions.append(kwargs["position_ids"][0].tolist())
        return original(*args, **kwargs)

    monkeypatch.setattr(policy.model, "forward", forward)
    logits = policy._append([14] * 32)
    assert positions == [list(range(32736, 32752)), list(range(32752, 32768))]
    assert policy._global_ids == [0, 1, 2] + list(range(32736, 32768))
    assert torch.isfinite(logits).all()
    assert policy.max_new_tokens == policy.fingerprint()["decoding"]["max_new_tokens"] == 128
    assert policy.fingerprint()["max_context_tokens"] == 32768
    with pytest.raises(ContextOverflowError, match="32768"):
        policy._append([15])


def test_attention_area_bound_uses_logical_history_and_preserves_hf_parity(monkeypatch):
    policy = tiny_policy()
    policy.PREFILL_CHUNK_SIZE = 8
    policy.MAX_PREFILL_QUERY_KEY_PRODUCT = 32
    blocks = []
    original = policy.model.forward

    def forward(*args, **kwargs):
        query_count = kwargs["input_ids"].shape[-1]
        start = int(kwargs["position_ids"][0, 0])
        assert query_count * (start + query_count) <= policy.MAX_PREFILL_QUERY_KEY_PRODUCT
        blocks.append(query_count)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(policy.model, "forward", forward)
        logits = policy._append(list(range(11, 31)))
    assert blocks == [4, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    torch.testing.assert_close(logits, teacher_logits(policy), atol=2e-6, rtol=2e-5)
    policy._compact([0, 2, 19])
    assert policy._prefill_chunk_limit(len(policy._token_ids)) == 1
    assert policy._prefill_chunk_limit(len(policy._global_ids)) == 4


def test_default_prefill_area_cap_scales_to_native_context():
    policy = tiny_policy()
    assert policy._prefill_chunk_limit(0) == 64
    assert policy._prefill_chunk_limit(8192) == 32
    assert policy._prefill_chunk_limit(16384) == 16
    assert policy._prefill_chunk_limit(32752) == 16
    assert 16 * 32768 == policy.MAX_PREFILL_QUERY_KEY_PRODUCT


def test_chunked_prefill_uses_bounded_queries_and_absolute_positions(monkeypatch):
    policy = tiny_policy()
    policy.PREFILL_CHUNK_SIZE = 3
    lengths = []
    original = policy.model.forward

    def forward(*args, **kwargs):
        lengths.append(kwargs["input_ids"].shape[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(policy.model, "forward", forward)
    logits = policy._append([12, 13, 14, 15, 16, 17, 18, 19])
    assert lengths == [3, 3, 2]
    torch.testing.assert_close(logits, teacher_logits(policy), atol=2e-6, rtol=2e-5)
    assert policy._last_forward_positions == [6, 7]


def test_default_prefill_chunks_release_all_attention_matrices(monkeypatch):
    policy = tiny_policy()
    references, query_lengths = [], []
    original = policy.model.forward

    def forward(*args, **kwargs):
        outputs = original(*args, **kwargs)
        query_lengths.append(kwargs["input_ids"].shape[-1])
        references.extend(weakref.ref(attention) for attention in outputs.attentions)
        return outputs

    monkeypatch.setattr(policy.model, "forward", forward)
    policy._append([10 + index % 30 for index in range(129)])
    gc.collect()
    assert query_lengths == [64, 64, 1]
    assert len(references) == 3 * policy.text_config.num_hidden_layers
    assert all(reference() is None for reference in references)
    assert policy._hh_scores.shape == policy._last_query_scores.shape == (129,)
    assert policy._hh_scores.device.type == policy._last_query_scores.device.type == "cpu"


@pytest.mark.parametrize("corruption", ["tuple", "sliding", "missing_layer", "length"])
def test_unsupported_or_inconsistent_caches_are_rejected(corruption):
    policy = tiny_policy()
    policy._append([11, 12, 13])
    if corruption == "tuple":
        policy.past_key_values = tuple((layer.keys, layer.values) for layer in policy.past_key_values.layers)
    elif corruption == "sliding":
        policy.past_key_values.layers[0] = DynamicSlidingWindowLayer(sliding_window=2)
    elif corruption == "missing_layer":
        policy.past_key_values.layers.pop()
    else:
        policy.past_key_values.layers[-1].keys = policy.past_key_values.layers[-1].keys[:, :, :2, :]
        policy.past_key_values.layers[-1].values = policy.past_key_values.layers[-1].values[:, :, :2, :]
    with pytest.raises((UnsupportedCacheError, RuntimeError)):
        policy._append([14])
    assert len(policy._token_ids) == 3


def test_empty_attention_fails_and_poisoned_state_cannot_continue(monkeypatch):
    policy = tiny_policy()
    original = policy.model.forward

    def without_attention(*args, **kwargs):
        outputs = original(*args, **kwargs)
        outputs.attentions = ()
        return outputs

    monkeypatch.setattr(policy.model, "forward", without_attention)
    with pytest.raises(RuntimeError, match="Empty attention"):
        policy._append([11, 12])
    with pytest.raises(RuntimeError, match="reset is required"):
        policy._append([13])


def test_partial_attention_cannot_silently_drop_full_layers():
    policy = tiny_policy()
    with pytest.raises(RuntimeError, match="exactly all"):
        policy._attention_scores((torch.ones(1, 4, 1, 2),), 1, 2)
    with pytest.raises(RuntimeError, match="shape"):
        policy._attention_scores((torch.ones(1, 4, 1, 2),) * 3 + (None,), 1, 2)


def test_reset_removes_all_episode_state_and_audit_is_a_copy(monkeypatch):
    policy = tiny_policy("stepkv", 0.2)
    policy.reset("Commands.", "First task.")
    action, _ = emit(policy, monkeypatch)
    policy.observe(action, "Apple 1 on table 2.", valid=True)
    first_cache = policy.past_key_values
    copied = policy.audit()
    copied["boundaries"][0]["retained_global_ids"].clear()
    assert policy.audit()["boundaries"][0]["retained_global_ids"]
    policy.reset("Commands.", "Second task.")
    current = policy.audit()
    assert policy.past_key_values is not first_cache
    assert current["prune_event_count"] == current["boundary_count"] == 0
    assert current["nonprompt_logical_tokens"] == current["nonprompt_resident_tokens"] == 0
    assert current["steps"] == [] and policy._seen_anchors == set()
    torch.testing.assert_close(policy._logits, teacher_logits(policy), atol=2e-6, rtol=2e-5)


def test_public_call_order_is_enforced(monkeypatch):
    policy = tiny_policy()
    with pytest.raises(RuntimeError, match="reset"):
        policy.act()
    policy.reset("Commands.", "Kitchen.")
    with pytest.raises(RuntimeError, match="preceding act"):
        policy.observe("look", "Kitchen.", valid=True)
    emit(policy, monkeypatch)
    with pytest.raises(RuntimeError, match="observe"):
        policy.act()


@pytest.mark.parametrize("method", ["fullkv", "stepkv", "h2o_step", "tova_step", "stepkv_no_step_score"])
def test_fingerprint_stays_stable_across_complete_and_terminal_episodes(method, monkeypatch):
    policy = tiny_policy(method, 0.2)
    fingerprint = json.dumps(policy.fingerprint(), sort_keys=True)
    policy.reset("Only commands.", "First task: find an apple.")
    assert json.dumps(policy.fingerprint(), sort_keys=True) == fingerprint
    action, _ = emit(policy, monkeypatch)
    assert json.dumps(policy.fingerprint(), sort_keys=True) == fingerprint
    policy.observe(action, "Apple 1 on table 2.", valid=True)
    assert json.dumps(policy.fingerprint(), sort_keys=True) == fingerprint
    emit(policy, monkeypatch, "go")
    terminal_audit = policy.audit()
    assert terminal_audit["generated_action_count"] == 2
    assert terminal_audit["completed_step_count"] == terminal_audit["boundary_count"] == 1
    assert terminal_audit["pending_action_count"] == 1
    assert len(terminal_audit["steps"]) == 2
    assert terminal_audit["steps"][-1]["status"] == "awaiting_observation"
    assert terminal_audit["steps"][-1]["utility"] is None
    assert json.dumps(policy.fingerprint(), sort_keys=True) == fingerprint
    policy.reset("A different system prompt.", "A different second task.")
    assert policy.audit()["generated_action_count"] == 0
    assert policy.audit()["system_prompt"] == "A different system prompt."
    assert json.dumps(policy.fingerprint(), sort_keys=True) == fingerprint


def test_unobserved_terminal_action_is_counted_without_mutation_or_fabricated_feedback(monkeypatch):
    policy = tiny_policy("stepkv", 0.2)
    policy.reset("Commands.", "Find an apple.")
    action, ids = emit(policy, monkeypatch)
    tokens, retained, forwards = list(policy._token_ids), list(policy._global_ids), policy._forward_count
    first = policy.audit()
    assert first == policy.audit()
    assert first["generated_action_count"] == first["pending_action_count"] == 1
    assert first["completed_step_count"] == first["boundary_count"] == first["prune_event_count"] == 0
    assert first["steps"][0]["emitted_action"] == action
    assert first["steps"][0]["generated_token_ids"] == ids
    assert first["steps"][0]["environment_return_received"] is False
    for key in ("action", "valid", "progress", "utility", "observation_sha256"):
        assert first["steps"][0][key] is None
    assert policy._token_ids == tokens and policy._global_ids == retained
    assert policy._forward_count == forwards
    json.dumps(first)
    policy.observe(action, "Apple 1 on table 2.", valid=True)
    observed = policy.audit()
    assert observed["generated_action_count"] == observed["completed_step_count"] == 1
    assert observed["pending_action_count"] == 0
    assert len(observed["steps"]) == 1 and observed["steps"][0]["status"] == "completed"


@pytest.mark.parametrize("method,ratio", [("unknown", 0.2), ("stepkv", 0), ("stepkv", 1.1), ("stepkv", float("nan"))])
def test_invalid_options_fail_before_loading(method, ratio):
    with pytest.raises(ValueError):
        AgentKVPolicy("not-a-model", method, ratio, device="cpu")


def test_allocator_limits_are_process_local_and_per_exposed_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=32 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: calls.append((fraction, device)))
    AgentKVPolicy._set_memory_limit(torch.device("cuda:0"), 12.0)
    assert calls == [(0.375, 0)]
    with pytest.raises(ValueError, match="physical memory"):
        AgentKVPolicy._set_memory_limit(torch.device("cuda:0"), 40)
    with pytest.raises(ValueError, match="CUDA device"):
        AgentKVPolicy._set_memory_limit(torch.device("cpu"), 12)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    AgentKVPolicy._set_memory_limit(torch.device("cuda:1"), 12)
    with pytest.raises(ValueError, match="not exposed"):
        AgentKVPolicy._set_memory_limit(torch.device("cuda:2"), 12)
    assert calls == [(0.375, 0), (0.375, 1)]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct"])
def test_loader_is_offline_pinned_and_uses_explicit_single_device(model_id, monkeypatch):
    import importlib
    module = importlib.import_module("models.AgentKVPolicy")
    monkeypatch.setenv("HF_HUB_CACHE", str(AgentKVPolicy.STORAGE_ROOT / "cache" / "unit-test-hub"))
    model = tiny_model()
    calls = []

    def config_load(path, **kwargs):
        calls.append(("config", path, kwargs))
        return model.config

    def tokenizer_load(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return TinyChatTokenizer()

    def model_load(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model

    monkeypatch.setattr(module.AutoConfig, "from_pretrained", config_load)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", model_load)
    policy = AgentKVPolicy(model_id, "fullkv", 1, device="cpu")
    for _, _, kwargs in calls:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["cache_dir"].startswith(str(AgentKVPolicy.STORAGE_ROOT))
        assert kwargs["revision"] == AgentKVPolicy.PINNED_REVISIONS[model_id]
    kwargs = calls[-1][2]
    assert kwargs["device_map"] == {"": "cpu"}
    assert kwargs["attn_implementation"] == "eager"
    assert kwargs["dtype"] is torch.float32
    assert kwargs["offload_buffers"] is False
    assert policy.fingerprint()["requested_revision"] == kwargs["revision"]
    assert policy.device == model.get_input_embeddings().weight.device


def test_fingerprint_discloses_ordinary_attention_and_experiment_semantics():
    policy = tiny_policy("stepkv_no_step_score", 0.2)
    policy.reset("Only native commands; no explanations.", "Put apple in fridge.")
    fingerprint = policy.fingerprint()
    assert fingerprint["step_utility_weight"] == 0
    assert fingerprint["chat_template_kwargs"] == {"enable_thinking": False}
    assert fingerprint["full_attention_layers"] == [0, 1, 2, 3]
    assert fingerprint["model_devices"] == ["cpu"]
    assert fingerprint["cache_layer_devices"] == ["cpu"] * 4
    assert fingerprint["input_device_source"] == "model.get_input_embeddings().weight.device"
    assert "forbidden" in fingerprint["cpu_disk_offload"]
    assert "system_prompt" not in fingerprint and "initial_observation_sha256" not in fingerprint
    assert fingerprint["episode_prompt_record"] == "audit.system_prompt and audit.initial_observation_sha256"
    assert policy.audit()["system_prompt"] == "Only native commands; no explanations."
    assert policy.audit()["initial_observation_sha256"]
    assert fingerprint["implementation_sha256"] and fingerprint["anchor_source_sha256"]
    assert "every decoder layer" in fingerprint["budget_scope"]
    assert any("not official reproductions" in text for text in fingerprint["semantic_differences_from_frozen_qa_engine"])
    assert "includes current step" in fingerprint["citation"]
    json.dumps(fingerprint)


@pytest.mark.parametrize("inferred", [
    {"model": 0, "lm_head": 1},
    {"model": 0, "lm_head": "cpu"},
    {"model": 0, "lm_head": "disk"},
    {"": 0},
])
def test_auto_plans_bounded_two_gpu_loading_and_rejects_any_offload(inferred, monkeypatch):
    import accelerate
    import accelerate.utils
    import importlib
    module = importlib.import_module("models.AgentKVPolicy")
    config = tiny_model().config
    limits, planner_calls = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: limits.append((fraction, device)))

    def empty_model(config, **kwargs):
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["attn_implementation"] == "eager"
        return SimpleNamespace(_no_split_modules=["Qwen2DecoderLayer"], tie_weights=lambda: None)

    def balanced(model, **kwargs):
        assert kwargs["max_memory"] == {0: 40 * 1024 ** 3, 1: 40 * 1024 ** 3}
        planner_calls.append("balanced")
        return {0: 34 * 1024 ** 3, 1: 35 * 1024 ** 3}

    def infer(model, **kwargs):
        assert kwargs["max_memory"] == {0: 34 * 1024 ** 3, 1: 35 * 1024 ** 3}
        assert kwargs["no_split_module_classes"] == ["Qwen2DecoderLayer"]
        assert kwargs["offload_buffers"] is False
        planner_calls.append("infer")
        return inferred

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_config", empty_model)
    monkeypatch.setattr(accelerate.utils, "get_balanced_memory", balanced)
    monkeypatch.setattr(accelerate, "infer_auto_device_map", infer)
    if inferred == {"model": 0, "lm_head": 1}:
        plan = AgentKVPolicy._auto_loading_plan(config, None)
        assert plan["allocator_limits_gib"] == {0: 44.0, 1: 44.0}
        assert plan["weight_ceiling_bytes"] == {0: 40 * 1024 ** 3, 1: 40 * 1024 ** 3}
        assert plan["device_map"] == inferred
    else:
        with pytest.raises(RuntimeError, match="offload"):
            AgentKVPolicy._auto_loading_plan(config, None)
    assert limits == [(0.44, 0), (0.44, 1)]
    assert planner_calls == ["balanced", "infer"]


def test_auto_requires_one_or_two_visible_gpus_and_shared_headroom(monkeypatch):
    config = tiny_model().config
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for count in (0, 3):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        with pytest.raises(ValueError, match="one or two"):
            AgentKVPolicy._auto_loading_plan(config, 44)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    for cap in (4, 45, float("nan"), True):
        with pytest.raises(ValueError):
            AgentKVPolicy._auto_loading_plan(config, cap)


def test_single_gpu_auto_honors_7b_allocator_limit_and_headroom(monkeypatch):
    import accelerate
    import accelerate.utils
    import importlib
    module = importlib.import_module("models.AgentKVPolicy")
    config = tiny_model().config
    limits = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: limits.append((fraction, device)))
    monkeypatch.setattr(
        module.AutoModelForCausalLM, "from_config",
        lambda *args, **kwargs: SimpleNamespace(_no_split_modules=["Qwen2DecoderLayer"], tie_weights=lambda: None),
    )

    def balanced(model, **kwargs):
        assert kwargs["max_memory"] == {0: 24 * 1024 ** 3}
        return kwargs["max_memory"]

    monkeypatch.setattr(accelerate.utils, "get_balanced_memory", balanced)
    monkeypatch.setattr(accelerate, "infer_auto_device_map", lambda *args, **kwargs: {"": 0})
    plan = AgentKVPolicy._auto_loading_plan(config, 28)
    assert plan["max_memory"] == {0: 24 * 1024 ** 3}
    assert plan["allocator_limits_gib"] == {0: 28.0}
    assert plan["placement"] == "explicit_one_gpu_no_offload"
    assert limits == [(0.28, 0)]


def test_auto_constructor_passes_explicit_gpu_map_and_limits_to_loader(monkeypatch):
    import importlib
    module = importlib.import_module("models.AgentKVPolicy")
    monkeypatch.setenv("HF_HUB_CACHE", str(AgentKVPolicy.STORAGE_ROOT / "cache" / "unit-test-hub"))
    config = tiny_model().config
    plan = {
        "device_map": {"model": 0, "lm_head": 1},
        "max_memory": {0: 40 * 1024 ** 3, 1: 40 * 1024 ** 3},
        "allocator_limits_gib": {0: 44.0, 1: 44.0}, "placement": "balanced_explicit_two_gpu_no_offload",
    }
    captured = {}
    monkeypatch.setattr(module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyChatTokenizer())
    monkeypatch.setattr(AgentKVPolicy, "_auto_loading_plan", staticmethod(lambda config, limit: plan))

    def load(*args, **kwargs):
        captured["load"] = kwargs
        return object()

    def initialize(self, *args, **kwargs):
        captured["initialize_args"] = args
        captured["initialize_kwargs"] = kwargs

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(AgentKVPolicy, "_initialize", initialize)
    AgentKVPolicy("Qwen/Qwen2.5-32B-Instruct", "stepkv", 0.2, device="auto", memory_limit_gib=44)
    assert captured["load"]["device_map"] == plan["device_map"]
    assert captured["load"]["max_memory"] == plan["max_memory"]
    assert captured["load"]["dtype"] is torch.bfloat16
    assert captured["load"]["offload_buffers"] is False
    assert captured["initialize_args"][6] == "auto"
    assert captured["initialize_kwargs"]["loading_plan"] == plan


def test_compaction_gathers_each_layer_on_its_own_device_without_gpu_work(monkeypatch):
    policy = tiny_policy()
    policy._append([11, 12, 13, 14])
    moves, gathers = [], []
    original_to = torch.Tensor.to

    class DeviceTensor:
        def __init__(self, tensor, device):
            self.tensor = tensor
            self.device = device

        def index_select(self, dimension, index):
            assert dimension == -2 and index.device == self.device
            gathers.append(str(self.device))
            return DeviceTensor(self.tensor.index_select(dimension, index.cpu), self.device)

    for index, layer in enumerate(policy.past_key_values.layers):
        target = torch.device("cuda", index % 2)
        layer.keys = DeviceTensor(layer.keys, target)
        layer.values = DeviceTensor(layer.values, target)

    def move(tensor, device, *args, **kwargs):
        if isinstance(device, torch.device) and device.type == "cuda":
            moves.append(str(device))
            return SimpleNamespace(device=device, cpu=tensor)
        return original_to(tensor, device, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", move)
    monkeypatch.setattr(policy, "_check_cache", lambda: None)
    monkeypatch.setattr(policy, "_measure_cache", lambda: {})
    policy._compact([0, 2, 3])
    assert moves == ["cuda:0", "cuda:1"]
    assert gathers == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"] * 2
    assert policy._global_ids == [0, 2, 3]
    assert policy._hh_scores.device.type == policy._last_query_scores.device.type == "cpu"


def test_non_full_attention_model_layout_is_explicitly_rejected():
    config = tiny_model().config
    config.layer_types = ["sliding_attention"] + ["full_attention"] * (config.num_hidden_layers - 1)
    with pytest.raises(UnsupportedCacheError, match="layout"):
        AgentKVPolicy._text_config(config)


def test_loader_cache_never_defaults_to_home_or_root_filesystem(monkeypatch):
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert AgentKVPolicy._loader_cache_dir() == str(AgentKVPolicy.STORAGE_ROOT / "cache" / "huggingface" / "hub")
    monkeypatch.setenv("HF_HOME", str(AgentKVPolicy.STORAGE_ROOT / "cache" / "custom-hf"))
    assert AgentKVPolicy._loader_cache_dir().endswith("/cache/custom-hf/hub")
    monkeypatch.setenv("HF_HUB_CACHE", "/home/not-allowed/.cache/huggingface/hub")
    with pytest.raises(ValueError, match="home/root-filesystem cache"):
        AgentKVPolicy._loader_cache_dir()
