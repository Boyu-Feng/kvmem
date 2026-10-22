"""CPU-only original-HF equivalence and asynchronous physical-cache checks."""
import copy
import json
import threading
from types import MethodType

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts.unified_sidequest_runtime import UnifiedSideQuestLLM
from token_tracker import TokenTracker


def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<eos>": 0, "<pad>": 1}
    vocab.update({char: i + 2 for i, char in enumerate(alphabet)})
    merges = [("a", "b"), ("ab", "S"), ("abS", "T"), ("abST", "O"), ("abSTO", "P")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    core = Tokenizer(models.BPE(vocab, merges))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<eos>", pad_token="<pad>")
    tok.chat_template = "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
    return tok


@pytest.fixture
def loaded(monkeypatch):
    tok = tokenizer()
    torch.manual_seed(233)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     eos_token_id=0, pad_token_id=1, max_position_embeddings=4096)
    original = Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    import importlib
    module = importlib.import_module("models.QwenLLMWithKVCache")
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: copy.deepcopy(tok))
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return copy.deepcopy(original)

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load)
    return tok, original, calls


def force_tokens(llm, main_text, aux_text='{"del_cursors": [0]}', aux_barrier=None):
    """Force logits, not generation: every HF/auxiliary forward still runs Qwen."""
    original = llm.model.forward
    tok = llm.tokenizer
    state = dict(main=0, auxiliary=0, main_logits=[], main_inputs=[])
    sequences = {
        "main": tok(main_text, add_special_tokens=False).input_ids + [tok.eos_token_id],
        "auxiliary": tok(aux_text, add_special_tokens=False).input_ids + [tok.eos_token_id],
    }

    def forward(model, *args, **kwargs):
        auxiliary = hasattr(llm, "_thread") and getattr(llm._thread, "auxiliary", False)
        branch = "auxiliary" if auxiliary else "main"
        if auxiliary and aux_barrier is not None:
            entered, release = aux_barrier
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test auxiliary release timed out")
        output = original(*args, **kwargs)
        seq = sequences[branch]
        index = state[branch]
        token = seq[min(index, len(seq) - 1)]
        output.logits.fill_(-100)
        output.logits[:, -1, token] = 100
        state[branch] += 1
        if not auxiliary:
            state["main_logits"].append(output.logits.detach().clone())
            state["main_inputs"].append(kwargs["input_ids"].detach().clone())
        return output

    llm.model.forward = MethodType(forward, llm.model)

    def rewind(text=None):
        state["main"] = 0
        if text is not None:
            sequences["main"] = tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]

    state["rewind"] = rewind
    return state


def assert_cache_equal(left, right):
    assert left.get_seq_length() == right.get_seq_length()
    for a, b in zip(left.layers, right.layers):
        assert torch.equal(a.keys, b.keys)
        assert torch.equal(a.values, b.values)


@pytest.mark.parametrize("text,stops", [
    ("Action 1: search[x]\nObservation hallucination", ["\nObservation"]),
    ("abSTOPtail", ["STOP"]),  # Stop is inside a single BPE token; re-encoding differs.
    ("", ["\nObservation"]),  # First generated token is EOS.
    ("ordinary", None),
])
def test_disabled_is_original_hf_including_tool_prefill_and_crop(loaded, text, stops):
    original = QwenLLMWithKVCache("tiny", token_tracker=TokenTracker())
    unified = UnifiedSideQuestLLM("tiny", token_tracker=TokenTracker(), enabled=False)
    a = force_tokens(original, text)
    b = force_tokens(unified, text)
    try:
        ra = original.generate_first("Question: tiny\nThought 1:", 80, stops)
        rb = unified.generate_first("Question: tiny\nThought 1:", 80, stops)
        assert ra[0] == rb[0]
        for left, right in zip(ra[1:], rb[1:]):
            for alayer, blayer in zip(left, right):
                assert all(torch.equal(x, y) for x, y in zip(alayer, blayer))
        assert_cache_equal(original.past_key_values, unified.past_key_values)
        a["rewind"]()
        b["rewind"]()
        tool = "\nObservation 1: real evidence\nThought 2:"
        assert original.generate_incremental(tool, 80, stops) == unified.generate_incremental(tool, 80, stops)
        assert original.current_cache_len == unified.current_cache_len
        assert original._all_token_ids == unified._all_token_ids
        assert original._global_token_id_log == unified._global_token_id_log
        assert original.token_tracker.__dict__ == unified.token_tracker.__dict__
        assert_cache_equal(original.past_key_values, unified.past_key_values)
        assert len(a["main_logits"]) == len(b["main_logits"])
        assert all(torch.equal(x, y) for x, y in zip(a["main_logits"], b["main_logits"]))
        assert all(torch.equal(x, y) for x, y in zip(a["main_inputs"], b["main_inputs"]))
        assert unified.pending is None
        assert not unified.audit()["aux_events"]
        assert all(call["dtype"] is torch.bfloat16 for call in loaded[2])
        json.dumps(unified.audit())
    finally:
        unified.close()


def test_disabled_unforced_logits_are_bitwise_original(loaded):
    original = QwenLLMWithKVCache("tiny")
    unified = UnifiedSideQuestLLM("tiny", enabled=False)
    seen = [[], []]
    hooks = [
        llm.model.register_forward_hook(
            lambda m, a, out, i=i: seen[i].append(out.logits.detach().clone()))
        for i, llm in enumerate((original, unified))
    ]
    try:
        assert original.generate_first("Question: q", 6)[0] == unified.generate_first("Question: q", 6)[0]
        tool = "\nObservation 1: evidence\nThought 2:"
        assert original.generate_incremental(tool, 6) == unified.generate_incremental(tool, 6)
        assert_cache_equal(original.past_key_values, unified.past_key_values)
        assert all(torch.equal(x, y) for x, y in zip(*seen))
        assert len(seen[0]) == len(seen[1])
    finally:
        for hook in hooks:
            hook.remove()
        unified.close()


def test_enabled_stop_inside_bpe_token_does_not_short_circuit_original_256_cap(loaded):
    original = QwenLLMWithKVCache("tiny")
    llm = UnifiedSideQuestLLM("tiny")
    # The stop is in token one; EOS occurs beyond the unchanged HF 256 cap.
    text = "abSTOP" + "x" * 300
    a = force_tokens(original, text)
    b = force_tokens(llm, text)
    try:
        assert original.generate_first("Question: q", 256, ["STOP"])[0] == llm.generate_first(
            "Question: q", 256, ["STOP"])[0] == "ab"
        assert len(a["main_inputs"]) == len(b["main_inputs"]) == 256
        assert_cache_equal(original.past_key_values, llm.past_key_values)
        a["rewind"]()
        b["rewind"]()
        tool = "\nObservation 1: evidence\nThought 2:"
        assert original.generate_incremental(tool, 256, ["STOP"]) == llm.generate_incremental(
            tool, 256, ["STOP"]) == "ab"
        assert len(a["main_inputs"]) == len(b["main_inputs"]) == 512
        assert_cache_equal(original.past_key_values, llm.past_key_values)
        assert original.current_cache_len == llm.current_cache_len
    finally:
        llm.close()


def test_enabled_real_async_late_stop_and_next_turn_physical_eviction(loaded):
    control = QwenLLMWithKVCache("tiny")
    llm = UnifiedSideQuestLLM("tiny", {"sidequest_interval": 1})
    entered, release = threading.Event(), threading.Event()
    text = "Action 1: search[x]\nObservation hallucinated late tail"
    control_state = force_tokens(control, text)
    state = force_tokens(llm, text, aux_barrier=(entered, release))
    stops = ["\nObservation"]
    try:
        assert control.generate_first("Question: q\nThought 1:", 100, stops)[0] == llm.generate_first(
            "Question: q\nThought 1:", 100, stops)[0]
        control_state["rewind"]()
        state["rewind"]()
        tool = "\nObservation 1: first evidence\nThought 2:"
        assert control.generate_incremental(tool, 100, stops) == llm.generate_incremental(tool, 100, stops)
        assert entered.wait(10)
        # Main finishes while the real auxiliary Qwen forward is blocked.
        assert llm.pending is not None and not llm.pending[0].done()
        assert_cache_equal(control.past_key_values, llm.past_key_values)
        assert len(control_state["main_inputs"]) == len(state["main_inputs"])
        assert all(torch.equal(x, y) for x, y in zip(
            control_state["main_inputs"], state["main_inputs"]))
        assert len(state["main_inputs"]) > 2 * len(llm.tokenizer(
            "Action 1: search[x]", add_special_tokens=False).input_ids)
        live = llm.live_caches()
        assert live["pending_prefix"] is live["auxiliary"]
        assert live["main"] is llm.past_key_values
        before = llm.past_key_values.get_seq_length()
        main_keys = llm.past_key_values.layers[0].keys.clone()
        release.set()
        llm.collect_pending(apply=True, wait=True)
        assert llm.audit()["staged_decisions"] == 1
        assert llm.past_key_values.get_seq_length() == before
        assert torch.equal(main_keys, llm.past_key_values.layers[0].keys)
        state["rewind"]()
        # Disable further spawning, not eviction of the already-ready decision.
        llm.interval = 100
        llm.generate_incremental("\nObservation 2: second evidence\nThought 3:", 100, stops)
        event = llm.audit()["aux_events"][0]
        assert event["parse_error"] is None
        assert event["del_cursors"] == [0]
        assert event["applied"] and event["after_tokens"] < event["before_tokens"]
        assert event["apply_turn"] > event["spawn_turn"]
        assert llm.cursors[0]["deleted"]
        assert min(start for start, _ in event["selection_intervals"]) >= llm.prompt_length
        assert not any(start <= gid < end for start, end in event["selection_intervals"]
                       for gid in event["kept_global_ids"])
        audit = llm.audit()
        assert len(audit["final_global_ids"]) == llm.past_key_values.get_seq_length()
        assert all(layer.keys.shape[2] == layer.values.shape[2] == len(audit["final_global_ids"])
                   for layer in llm.past_key_values.layers)
        assert any(e["branch"] == "auxiliary" for e in audit["forward_positions"])
        for forward in audit["forward_positions"]:
            if forward["attention_mask_shape"] is not None:
                assert forward["attention_mask_shape"] == [
                    1, forward["resident_tokens"] + forward["input_tokens"]]
            if forward["cache_position"] is not None:
                assert forward["cache_position"] == forward["physical_positions"]
        assert set(llm.live_caches()) == {"main"}
        json.dumps(audit)
    finally:
        release.set()
        llm.close()


def test_close_drains_without_applying_and_runtime_failures_propagate(loaded, monkeypatch):
    llm = UnifiedSideQuestLLM("tiny", {"sidequest_interval": 1})
    state = force_tokens(llm, "Action 1: search[x]\nObservation tail")
    try:
        llm.generate_first("Question: q\nThought 1:", 80, ["\nObservation"])
        state["rewind"]()

        def broken(*args):
            raise RuntimeError("auxiliary failure")

        monkeypatch.setattr(llm, "_auxiliary", broken)
        with pytest.raises(RuntimeError, match="auxiliary failure"):
            # Failure may become visible on a real forward or on explicit drain.
            llm.generate_incremental("\nObservation 1: evidence\nThought 2:", 80, ["\nObservation"])
            llm.collect_pending(apply=False, wait=True)
        assert "auxiliary failure" in llm.aux_events[0]["runtime_error"]
        assert not llm.aux_events[0]["applied"]
    finally:
        llm.close()
    assert not llm._hooks[0].id in llm.model._forward_pre_hooks
    llm.close()
    with pytest.raises(RuntimeError, match="closed"):
        llm.generate_first("Question: q", 4)


def test_close_discards_completed_auxiliary_without_post_answer_eviction(loaded):
    llm = UnifiedSideQuestLLM("tiny", {"sidequest_interval": 1})
    state = force_tokens(llm, "Action 1: search[x]\nObservation tail")
    try:
        llm.generate_first("Question: q\nThought 1:", 80, ["\nObservation"])
        state["rewind"]()
        llm.generate_incremental("\nObservation 1: evidence\nThought 2:", 80, ["\nObservation"])
        before = llm.past_key_values.layers[0].keys.clone()
        llm.close()
        event = llm.audit()["aux_events"][0]
        assert event["unused_at_episode_end"]
        assert not event["applied"]
        assert torch.equal(before, llm.past_key_values.layers[0].keys)
        assert llm.pending is None
        assert llm.audit()["staged_decisions"] == 0
        assert set(llm.live_caches()) == {"main"}
    finally:
        llm.close()


def test_invalid_combined_pruning_is_rejected(loaded):
    with pytest.raises(ValueError, match="combined"):
        UnifiedSideQuestLLM("tiny", {"pruning_mode": "ours"})


def test_parent_drain_marks_staged_result_unused_without_eviction(loaded):
    llm = UnifiedSideQuestLLM("tiny", {"sidequest_interval": 1})
    state = force_tokens(llm, "Action 1: search[x]\nObservation tail")
    try:
        llm.generate_first("Question: q\nThought 1:", 256, ["\nObservation"])
        state["rewind"]()
        llm.generate_incremental("\nObservation 1: evidence\nThought 2:", 256, ["\nObservation"])
        llm.collect_pending(apply=True, wait=True)
        assert llm.audit()["staged_decisions"] == 1
        before = llm.past_key_values.layers[0].keys.clone()
        from scripts.unified_latency_runtime import drain_auxiliary
        drain_auxiliary(llm)
        assert llm.audit()["staged_decisions"] == 0
        assert llm.audit()["aux_events"][0]["unused_at_episode_end"]
        assert torch.equal(before, llm.past_key_values.layers[0].keys)
    finally:
        llm.close()


def test_shared_generation_audit_monitor_and_drained_episode_reset(loaded):
    from scripts.unified_latency_runtime import GenerationAudit, UnifiedMonitor, drain_auxiliary
    llm = UnifiedSideQuestLLM("tiny", {"sidequest_interval": 1})
    state = force_tokens(llm, "Action 1: search[x]\nObservation tail")
    try:
        with UnifiedMonitor(llm) as monitor:
            for name, phase in llm.DIAGNOSTIC_HOOKS:
                monitor.wrap(llm, name, phase)
            with GenerationAudit(llm) as generation:
                llm.generate_first("Question: q\nThought 1:", 256, ["\nObservation"])
                state["rewind"]()
                llm.generate_incremental(
                    "\nObservation 1: evidence\nThought 2:", 256, ["\nObservation"])
                drain_auxiliary(llm)
            assert [call["max_new_tokens"] for call in generation.calls] == [256, 255]
            assert monitor.snapshot()["auxiliary_events"]
            assert not llm.pending
            state["rewind"]()
            # Parent drains each clean/diagnostic pass before starting the next:
            # no old auxiliary forward can race the diagnostic episode reset.
            with GenerationAudit(llm):
                llm.generate_first("Question: second\nThought 1:", 256, ["\nObservation"])
                drain_auxiliary(llm)
            assert not monitor.snapshot()["auxiliary_events"]
            assert not llm.audit()["aux_events"]
    finally:
        llm.close()
