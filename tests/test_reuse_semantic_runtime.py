"""CPU-only provider injection; no downloads, GPU contexts or main model hooks."""
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from kv_cache.anchor_utils import extract_anchor_terms
from scripts import reuse_semantic_runtime as runtime


@pytest.fixture
def workspace():
    path = Path.cwd() / f".reuse-semantic-tests-{os.getpid()}"
    path.mkdir(exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class FakeTokenizer:
    eos_token_id = 1
    all_special_ids = [1]

    def get_vocab(self):
        vocab = {chr(code): code + 2 for code in range(128)}
        vocab["٣"] = ord("٣") + 2
        return vocab

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None,
                 return_tensors=None, return_offsets_mapping=False):
        ids = [ord(char) + 2 for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids = [2] + ids + [1]
        if truncation:
            ids = ids[:max_length]
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        if return_tensors:
            result = {key: torch.tensor([value]) for key, value in result.items()}
        return result

    def apply_chat_template(self, messages, **kwargs):
        self.messages = copy.deepcopy(messages)
        return json.dumps(messages)

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(chr(int(item) - 2) for item in ids if int(item) > 2)


class FakeEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        self.calls = []

    def forward(self, **inputs):
        self.calls.append(inputs)
        vector = torch.tensor([3., 4., 0.])
        hidden = vector.expand(1, inputs["input_ids"].shape[-1], 3).clone()
        hidden[:, 1:, :] = torch.tensor([0., 0., 99.])
        return SimpleNamespace(last_hidden_state=hidden)


def ready_manifest(path):
    root = path / "encoder"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"test weights, never loaded")
    ready = {
        "version": 1, "repo": "BAAI/bge-small-en-v1.5", "revision": "abc123pinned",
        "path": str(root), "config_hashes": {"config.json": runtime._file_sha(root / "config.json")},
        "weight_inventory": [{"name": "model.safetensors", "size": (root / "model.safetensors").stat().st_size}],
        "pooling": "cls", "normalize": True, "max_tokens": 512, "dtype": "float32", "device": "cpu",
    }
    manifest = path / "ready.json"
    manifest.write_text(json.dumps(ready))
    return manifest


class FakeQwen(torch.nn.Module):
    def __init__(self, response="prose", eos=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(5, dtype=torch.bfloat16))
        self.config = SimpleNamespace(max_position_embeddings=8192)
        self.generation_config = SimpleNamespace(eos_token_id=1)
        self.response = response
        self.eos = eos
        self.calls = []
        self.fail = None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        self()
        tail = [ord(char) + 2 for char in self.response] + ([1] if self.eos else [])
        seq = torch.cat([kwargs["input_ids"], torch.tensor([tail])], dim=1)
        return SimpleNamespace(sequences=seq, past_key_values=((torch.ones(2), torch.ones(3)),))

    def forward(self):
        return SimpleNamespace(logits=torch.ones(1))


def paraphraser(path, **kwargs):
    model = kwargs.pop("model", None)
    cache_path = kwargs.pop("cache_path", path / "paraphrases.json")
    tokenizer = kwargs.pop("tokenizer", FakeTokenizer())
    if model is None:
        model = ProseSourceModel(tokenizer)
    return runtime.LocalQwenParaphraser(
        path, cache_path, device="cpu", tokenizer=tokenizer,
        model=model, model_fingerprint="fixed-test-model", **kwargs)


class FakeEncoder:
    def __init__(self):
        self.cache = {}
        self.reset_episode()

    def reset_episode(self):
        self.requests = []

    def encode(self, text):
        self.requests.append(text)
        if text not in self.cache:
            self.cache[text] = torch.tensor([1., 0.]) if "trees" in text else torch.tensor([0., 1.])
        return self.cache[text]

    def audit(self):
        return {"requests": self.requests, "cache_tensor_bytes": runtime._tensor_bytes(self.cache)}


class FakeParaphraser:
    def __init__(self):
        self.cache = {}
        self.reset_episode()

    def reset_episode(self):
        self.requests = []

    def paraphrase(self, text):
        self.requests.append(text)
        self.cache.setdefault(text, text.replace("trees", "woodland").replace("grown", "cultivated"))
        return self.cache[text]

    def audit(self):
        return {"requests": self.requests, "cache_entries": len(self.cache)}


@pytest.mark.parametrize("current", ["", "unrelated", "trees", "trees grown 2001", "123 trees grown other"])
def test_lexical_original_exact_existing_formula(current):
    scorer = runtime.RuntimeCitationScorer("lexical_original")
    meta = {0: {"anchors": set()}, 1: {"anchors": {"trees", "2001"}},
            2: {"anchors": {"grown", "other", "longer"}}}
    originals = copy.deepcopy(meta)
    anchors = extract_anchor_terms(current)
    expected = {}
    for step, item in meta.items():
        overlap = len(anchors & item["anchors"])
        if overlap:
            expected[step] = min(1., 1.5 * overlap / max(1, min(len(anchors), len(item["anchors"]))))
    assert scorer.increments(current, meta) == expected
    assert meta == originals
    assert scorer.encoder is scorer.paraphraser is None
    assert json.dumps(scorer.audit())


def test_lexical_original_full_anchors_not_paraphrase_prefix():
    scorer = runtime.RuntimeCitationScorer("lexical_original")
    current = "stopword " * 600 + "zebras"
    assert scorer.increments(current, {3: {"anchors": {"zebras"}}}) == {3: 1.}


@pytest.mark.parametrize("cos,expected", [(-1., 0.), (.49, 0.), (.5, 0.), (.6, .3),
                                        (.75, .75), (.9, 1.), (1.000001, 1.)])
def test_fixed_mapping(cos, expected):
    assert runtime.mapped_increment(cos) == pytest.approx(expected)


def test_mapping_rejects_nan():
    with pytest.raises(ValueError, match="Nonfinite"):
        runtime.mapped_increment(float("nan"))


def test_no_citation_loads_nothing():
    scorer = runtime.RuntimeCitationScorer("no_citation")
    assert scorer.increments("trees", {3: {"anchors": {"trees"}}}) == {}
    assert scorer.audit()["pairs"] == 0
    assert scorer.encoder is scorer.paraphraser is None


def test_semantic_register_reset_cache_and_audit():
    encoder = FakeEncoder()
    scorer = runtime.RuntimeCitationScorer("semantic_original", encoder=encoder)
    scorer.register_step(0, "trees original", {"trees"})
    scorer.register_step(1, "unrelated", {"unrelated"})
    assert scorer.increments("trees current", {0: {"anchors": set()}, 1: {"anchors": set()}}) == {0: 1.}
    snapshot = scorer.audit()
    assert [p["raw_cosine"] for p in snapshot["events"][0]["pairs"]] == [1., 0.]
    assert snapshot["encoder"]["cache_tensor_bytes"] == 24
    snapshot["events"][0]["pairs"].clear()
    assert len(scorer.audit()["events"][0]["pairs"]) == 2
    scorer.reset_episode()
    assert not scorer.history and not scorer.events and scorer.calls == 0
    assert len(encoder.cache) == 3 and not encoder.requests
    with pytest.raises(ValueError, match="Missing original"):
        scorer.increments("trees", {0: {"anchors": set()}})


def test_paraphrase_both_sides_no_meta_or_messages_mutation():
    provider = FakeParaphraser()
    scorer = runtime.RuntimeCitationScorer("lexical_paraphrase", paraphraser=provider)
    original = "trees grown 1000"
    scorer.register_step(3, original, {"trees", "grown", "1000"})
    main_messages = [{"role": "user", "content": original}]
    meta = {3: {"anchors": {"trees", "grown", "1000"}, "gold": "never transmit",
                "reward": 42, "expert": "forbidden"}}
    before = copy.deepcopy((main_messages, meta))
    assert scorer.increments(original, meta) == {3: 1.}
    assert provider.requests == [original, original]
    assert (main_messages, meta) == before
    event = scorer.audit()["events"][0]
    assert event["current_scored_sha"] != event["current_original_sha"]
    assert event["pairs"][0]["history_scored_sha"] != event["pairs"][0]["history_original_sha"]
    scorer.reset_episode()
    assert provider.cache and not provider.requests


def test_semantic_paraphrase_uses_rewrites_only():
    para, enc = FakeParaphraser(), FakeEncoder()
    scorer = runtime.RuntimeCitationScorer("semantic_paraphrase", encoder=enc, paraphraser=para)
    scorer.register_step(2, "trees grown", {"trees", "grown"})
    assert scorer.increments("trees", {2: {"anchors": {"trees"}}}) == {2: 1.}
    assert enc.requests == ["woodland", "woodland cultivated"]


def test_encoder_cpu_fp32_cls_truncation_cache_and_threads(workspace):
    before = torch.get_num_threads()
    model = FakeEmbeddingModel()
    encoder = runtime.EmbeddingEncoder(ready_manifest(workspace), tokenizer=FakeTokenizer(), model=model)
    vector = encoder.encode("x" * 600)
    assert vector.tolist() == pytest.approx([.6, .8, 0.])
    assert vector.device.type == "cpu" and vector.dtype == torch.float32
    assert model.weight.dtype == torch.float32 and not model.weight.requires_grad
    assert not model.training and torch.get_num_threads() == before
    assert encoder.encode("x" * 600) is vector and len(model.calls) == 1
    audit = encoder.audit()
    assert audit["truncations"] == 1 and audit["hits"] == 1 and audit["misses"] == 1
    assert audit["cache_tensor_bytes"] == audit["model_tensor_bytes"] == 12
    assert audit["events"][0]["input_tokens"] == 602
    assert audit["events"][0]["scored_tokens"] == 512
    encoder.reset_episode()
    assert not encoder.events and len(encoder.cache) == 1


def test_encoder_manifest_rejects_changed_config(workspace):
    path = ready_manifest(workspace)
    (workspace / "encoder/config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime.EmbeddingEncoder(path, tokenizer=FakeTokenizer(), model=FakeEmbeddingModel())


@pytest.mark.parametrize("bad", ['{}', '{"paraphrase":""}', '{"paraphrase":" "}',
                                 '{"paraphrase":5}', '{"paraphrase":"a","gold":"x"}',
                                 '{"paraphrase":"a","paraphrase":"b"}',
                                 'prefix {"paraphrase":"a"}', '{"paraphrase":"a"} trailing',
                                 '{"paraphrase": "unterminated'])
def test_strict_parse(bad):
    with pytest.raises(ValueError):
        runtime.parse_paraphrase(bad)


def test_local_qwen_greedy_cache_numeric_normalization_and_accounting(workspace):
    provider = paraphraser(workspace)
    original = "There are 1000 trees."
    result = provider.paraphrase(original)
    assert result == original
    call = provider.model.calls[0]
    assert call["do_sample"] is False and call["num_beams"] == 1
    assert 2 <= call["max_new_tokens"] <= 512 and call["use_cache"] is True
    assert not provider.model.training and not provider.model.weight.requires_grad
    request = provider.tokenizer.messages
    assert "1000" not in json.loads(request[1]["content"])["source"]
    assert all(word not in json.dumps(request) for word in ["gold", "reward", "expert", "HotpotQA"])
    assert provider.paraphrase(original) == result and len(provider.model.calls) == 2
    assert len(provider.events[0]["attempts"]) == 1
    assert provider.events[0]["attempts"][0]["accepted"]
    audit = provider.audit()
    assert audit["hits"] == audit["misses"] == 1
    assert audit["events"][0]["attempts"][0]["span_attempts"][0]["auxiliary_kv_tensor_bytes"] == 20
    assert audit["model_tensor_bytes"] == 10
    assert audit["auxiliary_forward_calls"] == audit["auxiliary_generate_calls"] == 2
    assert audit["cache_json_bytes"] > 0
    assert not list(workspace.glob("*.stage"))
    provider.reset_episode()
    assert not provider.events and provider.cache
    restored = paraphraser(workspace)
    assert restored.paraphrase(original) == result and not restored.model.calls
    assert restored.audit()["hits"] == 1


@pytest.mark.parametrize("response,eos,error", [
    ("", True, ValueError),
    ("There are 99 trees.", True, ValueError),
    ("There are 1000 trees.", False, ValueError),
    ("Invalid\x01text", True, ValueError),
])
def test_paraphrase_failure_explicit_no_fallback(workspace, response, eos, error):
    provider = paraphraser(workspace, model=FakeQwen(response=response, eos=eos))
    with pytest.raises(error):
        provider.paraphrase("Original prose.")
    assert not provider.cache and not provider.cache_path.exists()
    assert provider.audit()["events"][0]["error"]


def test_timeout_propagates_and_audits(workspace):
    model = FakeQwen()
    model.fail = TimeoutError("test deadline")
    provider = paraphraser(workspace, model=model)
    with pytest.raises(TimeoutError, match="deadline"):
        provider.paraphrase("There are 1000 trees.")
    assert not provider.cache
    assert provider.audit()["events"][0]["error"].startswith("TimeoutError:")


def test_unchanged_allowed_but_recorded(workspace):
    text = "There are 1000 trees."
    provider = prose_provider(workspace)
    assert provider.paraphrase(text) == text
    entry = next(iter(provider.cache.values()))
    assert entry["unchanged"] and entry["anchor_overlap"] == 1.
    provider.paraphrase(text)
    stress = provider.audit()["realized_stress"]
    assert stress["unique_texts"] == 1
    assert stress["rewritten_fraction_bounded_source"] == 0.
    assert stress["changed_fraction_vs_full_original_including_truncation"] == 0.
    assert stress["mean_bounded_anchor_replacement_fraction"] == 0.
    assert not stress["meaning_equivalence_verified"]
    provider.reset_episode()
    assert provider.audit()["realized_stress"]["unique_texts"] == 0
    assert provider.audit()["realized_stress"]["rewritten_fraction_bounded_source"] is None


def test_paraphrase_prefix_audit_and_span_failure(workspace):
    text = "x" * 520
    provider = paraphraser(workspace, model=FakeQwen("A different sentence."))
    provider.paraphrase(text)
    entry = next(iter(provider.cache.values()))
    assert entry["original"] == text and entry["source_prefix"] == text[:512]
    assert entry["source_tokens"] == 520 and entry["truncated"]
    assert provider.audit()["truncations"] == 1
    stats = entry["rewrite_audit"]
    assert stats["bounded_input"]["retained_tokens"] == 512
    assert stats["bounded_input"]["retained_token_fraction"] == pytest.approx(512 / 520)
    assert stats["bounded_input"]["omitted_characters"] == 8
    assert stats["bounded_input"]["source_prefix_sha"] == runtime.text_sha(text[:512])
    assert provider.audit()["realized_stress"]["truncated_fraction"] == 1.
    with pytest.raises(ValueError, match="bisects"):
        provider.paraphrase("x" * 510 + " 1234567890")


def test_cache_hash_validation_and_other_protocol_not_reused(workspace, monkeypatch):
    original = "There are 1000 trees."
    provider = paraphraser(workspace)
    provider.paraphrase(original)
    first_key = next(iter(provider.cache))
    monkeypatch.setattr(runtime, "PROTOCOL_SHA256", "another-protocol")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        paraphraser(workspace)
    restored = paraphraser(workspace, cache_path=workspace / "new-protocol.json")
    restored.paraphrase(original)
    assert len(restored.model.calls) == 2 and len(restored.cache) == 1
    second_key = next(iter(restored.cache))
    assert second_key != first_key
    payload = json.loads(restored.cache_path.read_text())
    payload["entries"][second_key]["paraphrase"] = "corrupted"
    restored.cache_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        paraphraser(workspace, cache_path=restored.cache_path)


def test_lazy_missing_assets_and_duplicate_steps_explicit():
    scorer = runtime.RuntimeCitationScorer("semantic_original")
    scorer.register_step(1, "trees", {"trees"})
    scorer.register_step(1, "trees", {"trees"})
    with pytest.raises(ValueError, match="cannot be replaced"):
        scorer.register_step(1, "changed", {"trees"})
    with pytest.raises(ValueError, match="manifest"):
        scorer.increments("trees", {1: {"anchors": {"trees"}}})
    assert scorer.audit()["events"][0]["error"]
    para = runtime.RuntimeCitationScorer("lexical_paraphrase")
    with pytest.raises(ValueError, match="local Qwen"):
        para.increments("trees", {})
    with pytest.raises(ValueError, match="Unknown"):
        runtime.RuntimeCitationScorer("other")


def test_static_fingerprints_and_numeric_negation_formats():
    assert len(runtime.RuntimeCitationScorer.source_fingerprint()) == 64
    assert len(runtime.RuntimeCitationScorer.protocol_fingerprint()) == 64
    assert runtime._numeric_mentions("-3.0 and 1,000.00") == runtime._numeric_mentions("−3 and 1000")
    assert runtime._numeric_mentions("2000-01-02") == runtime._numeric_mentions("2000/1/2")
    assert runtime._numeric_mentions("A320 and F-16") == runtime._numeric_mentions("320, 16")


def test_local_only_loader_settings_and_lazy_conditions(workspace, monkeypatch):
    import transformers
    loads = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda path, **kwargs: (loads.append(("tokenizer", kwargs)) or FakeTokenizer()))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        lambda path, **kwargs: (loads.append(("encoder", kwargs)) or FakeEmbeddingModel()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda path, **kwargs: (loads.append(("qwen", kwargs)) or FakeQwen()))
    manifest = ready_manifest(workspace)
    root = workspace / "qwen"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"fake model bytes")
    scorer = runtime.RuntimeCitationScorer("semantic_original", manifest, root, workspace / "cache.json",
                                           device="cpu")
    assert loads == []
    scorer.register_step(0, "trees", {"trees"})
    scorer.increments("trees", {0: {"anchors": {"trees"}}})
    assert [kind for kind, _ in loads] == ["tokenizer", "encoder"]
    assert loads[1][1]["dtype"] == torch.float32 and scorer.paraphraser is None
    loads.clear()
    scorer = runtime.RuntimeCitationScorer("lexical_paraphrase", manifest, root, workspace / "cache.json",
                                           device="cpu")
    scorer.register_step(0, "There are 1000 trees.", {"trees", "1000"})
    scorer.increments("There are 1000 trees.", {0: {"anchors": {"trees", "1000"}}})
    assert [kind for kind, _ in loads] == ["tokenizer", "qwen"]
    assert loads[1][1]["dtype"] == torch.bfloat16
    assert loads[1][1]["device_map"] == {"": "cpu"}
    assert loads[1][1]["attn_implementation"] == "sdpa" and scorer.encoder is None
    assert all(kwargs["local_files_only"] and not kwargs["trust_remote_code"] for _, kwargs in loads)


def test_deadline_stopping_criterion_raises_explicitly(workspace, monkeypatch):
    model = FakeQwen()
    generate = model.generate

    def check_deadline(**kwargs):
        monkeypatch.setattr(runtime.time, "perf_counter", lambda: 999999999999.)
        kwargs["stopping_criteria"][0](kwargs["input_ids"], None)
        return generate(**kwargs)

    model.generate = check_deadline
    provider = paraphraser(workspace, model=model, timeout_seconds=1)
    with pytest.raises(TimeoutError, match="exceeded deadline"):
        provider.paraphrase("There are 1000 trees.")
    assert not provider.cache and provider.audit()["events"][0]["error"]


def test_truncation_not_counted_as_rewrite_and_realized_anchor_stress():
    original = "trees zebras"
    stats = runtime._rewrite_audit(original, "trees", "trees", 600, 512)
    assert stats["unchanged_bounded_source"]
    assert stats["changed_vs_full_original"] and stats["bounded_input"]["truncated"]
    assert stats["bounded_anchor_replacement_fraction"] == 0.
    assert stats["anchor_overlap_bounded_source"] == 1.
    rewritten = runtime._rewrite_audit(original, "trees", "woodland", 600, 512)
    assert not rewritten["unchanged_bounded_source"]
    assert rewritten["bounded_anchor_replacement_fraction"] == 1.
    assert rewritten["anchor_overlap_bounded_source"] == 0.
    assert not rewritten["meaning_equivalence_verified"]


def test_parent_asset_prep_schema_and_manifest_provenance(workspace):
    manifest = ready_manifest(workspace)
    ready = json.loads(manifest.read_text())
    root = Path(ready["path"])
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
        (root / name).write_text("{}")
        ready["config_hashes"][name] = runtime._file_sha(root / name)
    ready["weight_inventory"] = {"model.safetensors": {
        "bytes": (root / "model.safetensors").stat().st_size,
        "sha256": runtime._file_sha(root / "model.safetensors"),
    }}
    ready["preparation_source_sha256"] = "f" * 64
    manifest.write_text(json.dumps(ready))
    encoder = runtime.EmbeddingEncoder(manifest, tokenizer=FakeTokenizer(), model=FakeEmbeddingModel())
    assert encoder.fingerprint == runtime._json_sha(ready)
    assert len(ready["config_hashes"]) == 5


@pytest.mark.parametrize("condition", runtime.CONDITIONS)
def test_static_fingerprint_before_after_scoring_reset_and_caller_mutation(workspace, condition):
    manifest = ready_manifest(workspace)
    root = workspace / "qwen"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"fake frozen weights")
    scorer = runtime.RuntimeCitationScorer(
        condition, manifest, root, workspace / "paraphrases.json", device="cpu",
        encoder=FakeEncoder() if condition.startswith("semantic_") else None,
        paraphraser=FakeParaphraser() if condition.endswith("_paraphrase") else None)
    before = scorer.fingerprint()
    assert before["condition"] == condition
    assert before["source_sha"] == runtime.SOURCE_SHA256
    assert before["protocol_sha"] == runtime.PROTOCOL_SHA256
    assert before["encoder_ready"]["sha256"] == runtime._file_sha(manifest)
    assert before["model"]["config_hashes"]["config.json"] == runtime._file_sha(root / "config.json")
    assert before["paraphrase_cache_path"] == str((workspace / "paraphrases.json").resolve())
    assert not before["paraphrase_generation"]["condition_in_cache_key"]
    def fingerprint_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(fingerprint_keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(fingerprint_keys(v) for v in value))
        return set()

    keys = fingerprint_keys(before)
    for forbidden in ["cache_entries", "cache_contents", "seconds", "timers", "calls", "loaded_state"]:
        assert forbidden not in keys
    scorer.register_step(1, "trees grown", {"trees", "grown"})
    assert scorer.fingerprint() == before
    scorer.increments("trees", {1: {"anchors": {"trees", "grown"}}})
    assert scorer.fingerprint() == before
    scorer.reset_episode()
    assert scorer.fingerprint() == before
    modified = scorer.fingerprint()
    modified["paraphrase_generation"]["max_new_tokens"] = 1
    modified["model"]["config_hashes"].clear()
    assert scorer.fingerprint() == before
    (workspace / "paraphrases.json").write_text('{"cache_contents":"not provenance"}')
    assert scorer.fingerprint() == before


def test_fingerprint_cpu_plan_no_provider_loads_and_stable_lazy_loading(workspace, monkeypatch):
    import transformers
    manifest = ready_manifest(workspace)
    loads = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda path, **kwargs: (loads.append(("tokenizer", kwargs)) or FakeTokenizer()))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        lambda path, **kwargs: (loads.append(("encoder", kwargs)) or FakeEmbeddingModel()))

    def forbidden(*args, **kwargs):
        raise AssertionError("Fingerprint must not touch GPU or load a Qwen model")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    scorer = runtime.RuntimeCitationScorer("semantic_original", encoder_ready=manifest, device="cuda:0")
    before = scorer.fingerprint()
    assert before == scorer.fingerprint() and loads == [] and scorer.encoder is None
    scorer.register_step(0, "trees", {"trees"})
    scorer.increments("trees", {0: {"anchors": {"trees"}}})
    assert len(loads) == 2
    assert scorer.fingerprint() == before and len(loads) == 2
    scorer.reset_episode()
    assert scorer.fingerprint() == before and len(loads) == 2


class SequenceQwen(FakeQwen):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses

    def generate(self, **kwargs):
        self.response = self.responses[len(self.calls)]
        return super().generate(**kwargs)


def diagnostic_attempts(provider):
    records = ([json.loads(line) for line in provider.diagnostics_path.read_text().splitlines()]
               if provider.diagnostics_path.exists() else [])
    for record in records:
        assert record["attempt_sha"] == runtime._diagnostic_sha(record["attempt"])
        assert record["attempt_sha_encoding"] == "canonical_ascii_json_v1"
        assert record["attempt"]["retry_strategy"] == "numeric_diff_guided_v1"
        assert record["attempt"]["max_attempts"] == 3
        assert record["attempt"]["protocol_sha"] == runtime.PROTOCOL_SHA256
    return [record["attempt"] for record in records]


def logical_diagnostic_attempts(provider):
    return [json.loads(line)["attempt"] for line in provider.logical_diagnostics_path.read_text().splitlines()]


@pytest.mark.parametrize("accepted_attempt", [2, 3])
def test_numeric_correction_accepts_second_or_third_and_accounts_every_attempt(workspace, accepted_attempt):
    source = "Original prose."
    invalid = "Invented 2."
    valid = "Reworded prose."
    model = SequenceQwen([invalid] * (accepted_attempt - 1) + [valid])
    provider = paraphraser(workspace, model=model)
    assert provider.paraphrase(source) == valid
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == len(model.calls) == accepted_attempt
    assert len({a["request_id"] for a in attempts}) == 1
    assert [a["attempt"] for a in attempts] == list(range(1, accepted_attempt + 1))
    for attempt in attempts[:-1]:
        assert attempt["numeric_missing"] == {} and attempt["numeric_extra"] == {"2": 1}
        assert attempt["rejection_type"] == "NumericMismatch" and not attempt["accepted"]
    assert attempts[-1]["accepted"] and attempts[-1]["numeric_missing"] == {}
    assert attempts[-1]["rejection_reason"] is None
    correction = json.loads(attempts[1]["request"][1]["content"])
    assert correction["source"] == source
    assert correction["numeric_correction"]["extra"] == {"2": 1}
    assert correction["numeric_correction"]["required_numeric_multiset"] == dict(runtime._numeric_mentions(source))
    assert all(a["source_prefix"] == source and a["retained_source_tokens"] <= 512 for a in attempts)
    assert all(a["eos_validated"] and a["generated_tokens"] > 0 for a in attempts)
    assert all(a["seconds"] >= 0 and a["auxiliary_kv_tensor_bytes"] == 20 for a in attempts)
    assert not any(word in json.dumps(attempts[1]["request"]) for word in ["gold", "reward", "expert", "F1"])
    audit = provider.audit()
    assert audit["auxiliary_generate_calls"] == audit["auxiliary_forward_calls"] == accepted_attempt
    assert audit["generated_tokens_total"] == sum(a["generated_tokens"] for a in attempts)
    assert audit["attempt_wall_seconds_sum"] == sum(a["seconds"] for a in attempts)
    assert audit["attempt_named_tensor_bytes_sum"] == sum(
        a["input_tensor_bytes"] + a["output_tensor_bytes"] + a["auxiliary_kv_tensor_bytes"] for a in attempts)
    entry = next(iter(provider.cache.values()))
    assert entry["accepted_attempt"] == accepted_attempt
    assert [s for a in entry["attempts"] for s in a["span_attempts"]] == attempts
    restored = paraphraser(workspace)
    assert restored.paraphrase(source) == valid
    assert not restored.model.calls and diagnostic_attempts(restored) == attempts
    provider.reset_episode()
    assert provider.audit()["auxiliary_forward_calls"] == provider.audit()["attempts"] == 0
    assert provider.paraphrase(source) == valid
    assert len(model.calls) == accepted_attempt and diagnostic_attempts(provider) == attempts


def test_all_three_numeric_mismatches_fail_and_persist_without_cache(workspace):
    model = SequenceQwen(["There are 99 trees."] * 3)
    provider = paraphraser(workspace, model=model)
    with pytest.raises(runtime.NumericMismatch):
        provider.paraphrase("Original prose.")
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == len(model.calls) == 3
    assert all(not a["accepted"] and a["rejection_type"] == "NumericMismatch" for a in attempts)
    assert all(a["numeric_missing"] == {} and a["numeric_extra"] == {"99": 1} for a in attempts)
    assert all(a["raw_response"] and a["eos_validation"] == "valid" for a in attempts)
    assert not provider.cache and not provider.cache_path.exists()
    assert provider.audit()["auxiliary_forward_calls"] == provider.audit()["auxiliary_generate_calls"] == 3


@pytest.mark.parametrize("invalid,eos", [
    ("", True), (" \n\t", True),
    ("1000\x00", True), ("1000", False),
])
def test_nonnumeric_invalid_is_terminal_with_persisted_diagnostic(workspace, invalid, eos):
    model = SequenceQwen([invalid])
    model.eos = eos
    provider = paraphraser(workspace, model=model, tokenizer=ControlPreservingTokenizer())
    with pytest.raises(ValueError) as caught:
        provider.paraphrase("original")
    assert not isinstance(caught.value, runtime.NumericMismatch)
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == len(model.calls) == 1
    assert attempts[0]["raw_response"] == invalid
    assert attempts[0]["rejection_reason"] and not attempts[0]["accepted"]
    assert attempts[0]["numeric_expected"] == {}
    assert attempts[0]["numeric_actual"] is None
    assert not provider.cache and not provider.cache_path.exists()


def test_numeric_repeat_multiplicity_correction_and_static_retry_fingerprint(workspace):
    source = "In 2010 and again in 2010."
    provider = prose_provider(workspace)
    model = provider.model
    scorer = runtime.RuntimeCitationScorer("lexical_paraphrase", paraphraser=provider,
                                           paraphrase_cache=provider.cache_path, device="cpu")
    before = scorer.fingerprint()
    assert before["paraphrase_generation"]["protocol_version"] == 11
    assert before["paraphrase_generation"]["max_attempts"] == 3
    assert before["paraphrase_generation"]["retry_strategy"] == "numeric_diff_guided_v1"
    scorer.register_step(0, source, extract_anchor_terms(source))
    scorer.increments(source, {0: {"anchors": extract_anchor_terms(source)}})
    anchors = [s for s in provider.events[0]["source_layout"]["protected_spans"] if s["kind"] == "numerical_anchor"]
    assert len(anchors) == 2 and all(s["raw_literal"] == "2010" for s in anchors)
    assert all(s["required_count"] == 2 for s in anchors)
    assert scorer.fingerprint() == before
    scorer.reset_episode()
    assert scorer.fingerprint() == before and len(provider.cache) == 1
    assert provider.paraphrase(source) == source
    assert len(model.calls) == 2


def test_old_cache_schema_is_rejected_without_accepting_entries(workspace):
    (workspace / "paraphrases.json").write_text(json.dumps({"version": 1, "entries": {}}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        paraphraser(workspace)


def test_empty_payload_during_numeric_correction_stops_without_third_attempt(workspace):
    model = SequenceQwen(["99", "", "1000"])
    provider = paraphraser(workspace, model=model)
    with pytest.raises(ValueError, match="nonempty"):
        provider.paraphrase("source")
    attempts = diagnostic_attempts(provider)
    assert len(model.calls) == len(attempts) == 2
    assert attempts[0]["rejection_type"] == "NumericMismatch"
    assert attempts[1]["rejection_type"] == "ValueError"
    assert not provider.cache and not provider.cache_path.exists()


def test_exception_message_alone_cannot_trigger_retry(workspace):
    model = FakeQwen()
    model.fail = ValueError("Paraphrase numeric mentions changed")
    provider = paraphraser(workspace, model=model)
    with pytest.raises(ValueError, match="numeric mentions changed") as caught:
        provider.paraphrase("source")
    assert not isinstance(caught.value, runtime.NumericMismatch)
    assert len(model.calls) == len(diagnostic_attempts(provider)) == 1


def test_shared_policy_id_binds_fingerprint_protocol_cache_and_diagnostics(workspace):
    provider = paraphraser(workspace)
    lexical = runtime.RuntimeCitationScorer("lexical_paraphrase", paraphraser=provider)
    semantic = runtime.RuntimeCitationScorer("semantic_paraphrase", paraphraser=provider, encoder=FakeEncoder())
    lexical_policy = lexical.fingerprint()["paraphrase_generation"]
    semantic_policy = semantic.fingerprint()["paraphrase_generation"]
    assert lexical_policy == semantic_policy
    assert lexical_policy["retry_strategy"] == runtime.PROTOCOL["retry_strategy"] == "numeric_diff_guided_v1"
    assert lexical_policy["retry_strategy_description"]
    assert runtime.PROTOCOL_SHA256 == runtime._json_sha(runtime.PROTOCOL)
    provider.paraphrase("There are 1000 trees.")
    entry = next(iter(provider.cache.values()))
    assert entry["protocol_sha"] == lexical_policy["protocol_sha"]
    assert diagnostic_attempts(provider)[0]["retry_strategy"] == lexical_policy["retry_strategy"]
    assert provider.audit()["retry_strategy"] == lexical_policy["retry_strategy"]


def test_real_v3_heading_omission_context_and_distinct_final_correction(workspace):
    source = (
        "Jalachhayam (2010); Veena Vaadanam in 2008; a film in 2012. "
        "Jacques Feyder died in 1948.\n\nAction 2: Search[Sathish Kalathil]"
    )
    missing_heading = (
        "Jalachhayam was released in 2010, Veena Vaadanam in 2008, "
        "and a film in 2012. Jacques Feyder died in 1948."
    )
    corrected = missing_heading + "\n\nAction 2: Search[Sathish Kalathil]"
    provider = prose_provider(workspace)
    assert provider.paraphrase(source) == source
    attempts = diagnostic_attempts(provider)
    assert all(a["eos_validated"] and not a["truncated"] for a in attempts)
    assert all(a["numeric_missing"] == a["numeric_extra"] == {} for a in attempts)
    assert all(not runtime._numeric_mentions(json.loads(a["request"][1]["content"])["source"]) for a in attempts)
    assert any(span["text"] == "Action 2: Search[Sathish Kalathil]"
               for span in provider.events[0]["source_layout"]["protected_spans"])
    assert not next(iter(provider.cache.values()))["rewrite_audit"]["meaning_equivalence_verified"]


def test_numeric_contexts_are_source_only_bounded_and_preserve_repeat_contexts():
    source = "x" * 150 + " 2010 " + "y" * 150 + "\nAgain 2010.\nAction 2: Search[Name]"
    contexts = runtime._numeric_source_contexts(source, dict(runtime._numeric_mentions("2010")))
    assert len(contexts) == 2
    for context in contexts:
        start, end = context["context_span"]
        num_start, num_end = context["number_span"]
        assert source[start:end] == context["source_excerpt"]
        assert source[num_start:num_end] == context["raw"]
        assert num_start - start <= 96 and end - num_end <= 96
        assert "\n" not in context["source_excerpt"]
        assert context["raw"] == "2010"
    assert runtime._numeric_source_contexts(source, {}) == []


def test_protected_headers_and_balanced_commands_not_observation_bodies():
    source = (
        'Observation 1: A long body with facts and numbers 2010.\n'
        'Thought 2: This wording may change.\n'
        'Action 3: Search["Title with ] bracket" [nested]]\n'
        "Action 4: Lookup[Schindler's List]"
    )
    spans = runtime._protected_react_spans(source)
    texts = [span["text"] for span in spans]
    assert "Observation 1:" in texts and "Thought 2:" in texts
    assert 'Action 3: Search["Title with ] bracket" [nested]]' in texts
    assert "Action 4: Lookup[Schindler's List]" in texts
    assert all("A long body" not in text and "This wording" not in text for text in texts)
    for span in spans:
        start, end = span["span"]
        assert source[start:end] == span["text"]
    assert runtime._protected_span_diff(source, spans) == []
    assert runtime._protected_span_diff(source.replace("Search", "Lookup"), spans)
    literal_parenthesis = "Action 2: Search[An unfinished title (film]"
    assert runtime._protected_react_spans(literal_parenthesis)[-1]["text"] == literal_parenthesis
    incomplete = runtime._protected_react_spans("Action 1: Search[unfinished")
    assert incomplete[-1]["kind"] == "command_fragment" and not incomplete[-1]["complete"]
    assert incomplete[-1]["text"] == "Action 1: Search[unfinished"


def test_protected_only_mismatch_is_terminal_and_durable_not_numeric_retry(workspace):
    source = "Narrative.\nAction: Search[Sathish Kalathil]"
    model = SequenceQwen(["Narrative.\nAction: Search[Sathish Kalathil]"])
    provider = paraphraser(workspace, model=model)
    with pytest.raises(runtime.ProtectedSpanMismatch):
        provider.paraphrase(source)
    attempts = diagnostic_attempts(provider)
    assert len(model.calls) == len(attempts) == 1
    assert attempts[0]["numeric_missing"] == attempts[0]["numeric_extra"] == {}
    assert logical_diagnostic_attempts(provider)[-1]["rejection_type"] == "ProtectedSpanMismatch"
    assert not provider.cache and not provider.cache_path.exists()


def test_actual_v3_historical_source_numeric_precedence_and_verbatim_react_protection():
    source = (
        "Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam. "
        "He is also Story Writer, and Lyricist. His experimental works are well known and appropriately "
        "discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed "
        "according to its experimental approach. Actually, his first venture was in 2008 named Veena "
        "Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in "
        "movie. In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about "
        "Municipal solid waste..\nJacques Feyder's birth year is not directly mentioned, but his death "
        "year is 1948. I need to find Sathish Kalathil's birth year.\n\nAction 2: Search[Sathish Kalathil]"
    )
    rejected = (
        "Sathish Kalathil (മലയാളം: ) is an Indian film and documentary director and producer working "
        "in Malayalam. He is also known as a story writer and lyricist. His experimental films are "
        "recognized and discussed within the Malayalam cinema industry. His debut film, 'Jalachhayam' "
        "(2010), received attention for its experimental approach. His initial foray into filmmaking "
        "was in 2008 with the documentary 'Veena Vaadanam', which focused on art (painting). In 2012, "
        "he directed 'Laloorinu Parayanullathu', a socially committed documentary about municipal solid "
        "waste. Jacques Feyder's death year is 1948, but his birth year is not specified. I need to find "
        "Sathish Kalathil's birth year."
    )
    spans = runtime._protected_react_spans(source)
    assert [s["text"] for s in spans] == ["Action 2:", "Action 2: Search[Sathish Kalathil]"]
    expected, actual = runtime._numeric_mentions(source), runtime._numeric_mentions(rejected)
    mismatch = runtime.NumericMismatch(expected, actual)
    assert mismatch.missing == {"2": 1} and mismatch.extra == {}
    assert runtime._protected_span_diff(rejected, spans)
    contexts = runtime._numeric_source_contexts(source, mismatch.missing)
    assert contexts[0]["source_excerpt"] == "Action 2: Search[Sathish Kalathil]"
    assert runtime.PROTOCOL["retry_strategy"] == "numeric_diff_guided_v1"
    assert runtime.PROTOCOL["prompt_revision"] == 8


NATIVE_V4_MALFORMED_SOURCE = (
    'Invalid action.\nThought 2: Now that I know the show is "American Dad!", I need to find out '
    'who did the voice of Francine in the show.\nAction 2: Lookup[WObservation 5: Wendy SchSchaal '
    'is an American actress and voice2'
)


def test_real_v4_native_malformed_command_is_preserved_not_repaired(workspace):
    assert runtime.text_sha(NATIVE_V4_MALFORMED_SOURCE) == (
        "154fe666bb727b25bb53a0a31a123a298373bcec0e80f4c059288d9cf6e27ffd")
    fragment = NATIVE_V4_MALFORMED_SOURCE.splitlines()[-1]
    output = NATIVE_V4_MALFORMED_SOURCE.replace("Now that I know", "Since I know")
    body = output.splitlines()[1].split("Thought 2: ", 1)[1]
    provider = paraphraser(workspace, model=SequenceQwen(["Invalid action.", body]))
    assert provider.paraphrase(NATIVE_V4_MALFORMED_SOURCE) == output
    attempt = diagnostic_attempts(provider)[0]
    protected = provider.events[0]["source_layout"]["protected_spans"]
    partial = next(span for span in protected if span["kind"] == "command_fragment")
    assert partial["text"] == fragment and partial["reason"] == "incomplete_at_prefix_end"
    assert not partial["source_truncated"] and not partial["complete"]
    assert attempt["protected_spans_validated"] and attempt["numeric_missing"] == {}
    assert provider.audit()["auxiliary_generate_calls"] == 2
    for synthetic in [output + "]", output + " missing]"]:
        assert runtime._protected_span_diff(synthetic, protected)
    restored = paraphraser(workspace)
    assert restored.paraphrase(NATIVE_V4_MALFORMED_SOURCE) == output and not restored.model.calls
    assert all(fragment not in json.loads(a["request"][1]["content"])["source"]
               for a in diagnostic_attempts(provider))


def test_truncated_balanced_command_keeps_only_actual_prefix_fragment(workspace):
    opening = "Action 2: Search["
    source = "Context " + "p" * 440 + "\n" + opening + "x" * 80 + "]"
    prefix = source[:512]
    fragment = prefix[prefix.index(opening):]
    rewritten = "A brief context.\n" + fragment
    provider = paraphraser(workspace, model=FakeQwen("A brief context."))
    assert provider.paraphrase(source) == rewritten
    attempt = diagnostic_attempts(provider)[0]
    assert attempt["truncated"] and attempt["retained_source_tokens"] == 512
    protected = provider.events[0]["source_layout"]["protected_spans"]
    partial = next(span for span in protected if span["kind"] == "command_fragment")
    assert partial["source_truncated"] and partial["text"] == fragment
    assert partial["reason"] == "incomplete_at_prefix_end"
    assert not partial["text"].endswith("]") and attempt["protected_spans_validated"]
    assert runtime._protected_span_diff(rewritten + "]", protected)


def test_native_fragment_stops_before_multiline_observation_body():
    source = (
        "Action 2: Lookup[unfinished\nObservation 5: The native tool returned a result.\n"
        "More observation prose with a stray closing bracket ].\nThought 6: Plan the next move."
    )
    protected = runtime._protected_react_spans(source)
    partial = next(span for span in protected if span["kind"] == "command_fragment")
    assert partial["text"] == "Action 2: Lookup[unfinished"
    assert partial["reason"] == "next_react_header_before_close"
    assert not partial["source_truncated"]
    assert all("native tool" not in span["text"] and "More observation" not in span["text"] for span in protected)
    rewritten = source.replace("The native tool returned a result.", "A result arrived from the tool.")
    assert runtime._protected_span_diff(rewritten, protected) == []
    assert runtime._protected_span_diff(rewritten.replace("[unfinished\n", "[unfinished]\n"), protected)


def test_fragment_counts_do_not_count_balanced_commands_with_same_prefix():
    source = "Action 2: Search[query]\nAction 2: Search[query"
    protected = runtime._protected_react_spans(source)
    partial = next(span for span in protected if span["kind"] == "command_fragment")
    assert partial["required_count"] == 1
    assert runtime._protected_span_diff(source, protected) == []


@pytest.mark.parametrize("source", ["Action 2: Lo", "Action 2:", "Action 2: Lookup without brackets"])
def test_native_partial_command_syntax_is_literal_nonempty_and_not_completed(source):
    protected = runtime._protected_react_spans(source)
    partial = next(span for span in protected if span["kind"] == "command_fragment")
    assert partial["text"] == source and partial["required_count"] == 1
    assert partial["reason"] == "missing_or_incomplete_command_syntax"
    assert runtime._protected_span_diff(source, protected) == []
    assert runtime._protected_span_diff(source + " completed]", protected)


@pytest.mark.parametrize("condition", ["lexical_original", "no_citation", "semantic_original"])
def test_original_score_paths_never_call_paraphrase_parser_for_native_malformed_text(monkeypatch, condition):
    def forbidden(*args, **kwargs):
        raise AssertionError("Original score paths must not run the paraphrase parser")

    monkeypatch.setattr(runtime, "_protected_react_spans", forbidden)
    encoder = FakeEncoder() if condition == "semantic_original" else None
    scorer = runtime.RuntimeCitationScorer(condition, encoder=encoder)
    source = NATIVE_V4_MALFORMED_SOURCE
    anchors = extract_anchor_terms(source)
    scorer.register_step(5, source, anchors)
    assert scorer.increments(source, {5: {"anchors": anchors}}) == (
        {} if condition == "no_citation" else {5: 1.})
    assert scorer.paraphraser is None


def test_actual_v5_literal_newline_transport_preserves_content_raw_and_cache(workspace):
    source = (
        '\nObservation 1: Wendy Schaal (born July 2, 1954) is an American actress. She is currently '
        'the voice of Francine Smith in the animated series "American Dad!".\nThought 2:'
    )
    raw = """{
  "paraphrase": "Observation 1: Wendy Schaal, born on July 2, 1954, is an American actress. Currently, she voices Francine Smith in the animated series 'American Dad!'.
Thought 2:"
}"""
    assert raw[170] == "\n"
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    prose = ("Wendy Schaal, born on July 2, 1954, is an American actress. Currently, "
             "she voices Francine Smith in the animated series 'American Dad!'.")
    expected_value = "\nObservation 1: " + prose + "\nThought 2:"
    provider = prose_provider(workspace)
    expected_value = source
    value = provider.paraphrase(source)
    assert value == expected_value
    attempt = diagnostic_attempts(provider)[0]
    actual_span = attempt["source_prefix"]
    assert attempt["raw_response"] == actual_span and attempt["raw_response_sha"] == runtime.text_sha(actual_span)
    assert attempt["raw_response_utf8_bytes"] == len(actual_span.encode())
    assert attempt["numeric_missing"] == attempt["numeric_extra"] == {}
    assert attempt["protected_spans_validated"] and attempt["accepted"]
    transport = attempt["payload_transport"]
    assert not transport["normalization_applied"] and transport["decoded_string_content_preserved"]
    assert transport["raw_response_sha"] == runtime.text_sha(actual_span)
    assert json.loads(transport["canonical_response"]) == {"paraphrase": actual_span}
    assert provider.audit()["json_transport_normalizations"] == 0
    assert provider.audit()["auxiliary_generate_calls"] == len(diagnostic_attempts(provider))
    entry = next(iter(provider.cache.values()))
    assert entry["raw_response"] == expected_value and entry["paraphrase"] == value
    assert not entry["rewrite_audit"]["meaning_equivalence_verified"]
    restored = paraphraser(workspace)
    assert restored.paraphrase(source) == value and not restored.model.calls
    assert diagnostic_attempts(restored)[0]["raw_response"] == actual_span


@pytest.mark.parametrize("separator", ["\n", "\r", "\t", "\r\n"])
def test_escaped_and_literal_whitespace_decode_to_identical_content(separator):
    value = "alpha" + separator + "beta"
    escaped = json.dumps({"paraphrase": value})
    literal = '{"paraphrase":"' + value + '"}'
    parsed_escaped, escaped_meta = runtime._decode_paraphrase_transport(escaped)
    parsed_literal, literal_meta = runtime._decode_paraphrase_transport(literal)
    assert parsed_escaped == parsed_literal == value
    assert not escaped_meta["normalization_applied"] and literal_meta["normalization_applied"]
    assert escaped_meta["canonical_response"] == literal_meta["canonical_response"]
    assert json.loads(literal_meta["canonical_response"])["paraphrase"] == value
    assert runtime.parse_paraphrase(literal) == value


@pytest.mark.parametrize("control", [chr(i) for i in range(32) if i not in {9, 10, 13}] + ["\x7f", "\x85"])
def test_other_raw_and_escaped_controls_are_rejected(control):
    value = "alpha" + control + "beta"
    for response in ['{"paraphrase":"' + value + '"}', json.dumps({"paraphrase": value})]:
        with pytest.raises(ValueError, match="Forbidden"):
            runtime.parse_paraphrase(response)


@pytest.mark.parametrize("bad", [
    '{"paraphrase":"alpha\nbeta","paraphrase":"duplicate"}',
    '{"paraphrase":"alpha\tbeta","extra":"wrong schema"}',
    '{"paraphrase":"alpha\n"bad quote"}',
    '{"paraphrase":"alpha\rbeta"',
    '{"paraphrase":["alpha\nbeta"]}',
])
def test_transport_tolerance_cannot_repair_structure_schema_or_duplicate_fields(bad):
    with pytest.raises(ValueError):
        runtime.parse_paraphrase(bad)


@pytest.mark.parametrize("non_json", [
    "paraphrase: |\n  alpha\n  beta",
    "{'paraphrase': 'alpha'}",
    '{"paraphrase":"alpha\nbeta",}',
])
def test_whitespace_transport_remains_json_only_not_yaml_or_python_literals(non_json):
    with pytest.raises(json.JSONDecodeError):
        runtime.parse_paraphrase(non_json)


class ControlPreservingTokenizer(FakeTokenizer):
    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(chr(int(item) - 2) for item in ids if int(item) != self.eos_token_id)


def test_nul_transport_is_terminal_and_raw_bytes_remain_durable(workspace):
    raw = "1000\x00"
    provider = paraphraser(workspace, model=FakeQwen(raw), tokenizer=ControlPreservingTokenizer())
    with pytest.raises(ValueError, match="Forbidden"):
        provider.paraphrase("source")
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == len(provider.model.calls) == 1
    assert attempts[0]["raw_response"] == raw and attempts[0]["raw_response_sha"] == runtime.text_sha(raw)
    assert attempts[0]["json_transport"] is None and not attempts[0]["accepted"]
    assert not provider.cache and not provider.cache_path.exists()


def test_literal_newline_text_still_requires_complete_eos(workspace):
    raw = "alpha\nbeta"
    provider = paraphraser(workspace, model=FakeQwen(raw, eos=False))
    with pytest.raises(ValueError, match="without EOS"):
        provider.paraphrase("alpha\nbeta")
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == 1 and attempts[0]["raw_response"] == raw
    assert not attempts[0]["eos_validated"] and attempts[0]["json_transport"] is None
    assert not provider.cache


def test_plain_text_transport_policy_is_static_in_shared_fingerprint():
    lexical = runtime.RuntimeCitationScorer("lexical_paraphrase")
    semantic = runtime.RuntimeCitationScorer("semantic_paraphrase")
    left = lexical.fingerprint()["paraphrase_generation"]
    right = semantic.fingerprint()["paraphrase_generation"]
    assert left["output_transport_policy"] == right["output_transport_policy"] == runtime.PROTOCOL["output_transport_policy"]
    assert left["output_transport"] == right["output_transport"] == "plain_utf8_eos_v1"
    assert left["retry_strategy"] == "numeric_diff_guided_v1" and left["max_attempts"] == 3
    assert left["protocol_sha"] == runtime._json_sha(runtime.PROTOCOL)
    assert lexical.encoder is lexical.paraphraser is semantic.encoder is semantic.paraphraser is None


@pytest.mark.parametrize("failure", ["numeric", "protected"])
def test_literal_whitespace_transport_never_relaxes_numeric_or_protected_validation(workspace, failure):
    source = "A value.\nAction: Search[Name]"
    value = "A value 8.\n" if failure == "numeric" else "A value.\nAction: Search[Name]"
    raw = value
    provider = paraphraser(workspace, model=FakeQwen(raw))
    error = runtime.NumericMismatch if failure == "numeric" else runtime.ProtectedSpanMismatch
    with pytest.raises(error):
        provider.paraphrase(source)
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == (3 if failure == "numeric" else 1)
    assert all(not a["payload_transport"]["normalization_applied"] for a in attempts)
    assert not logical_diagnostic_attempts(provider)[-1]["accepted"]
    assert all(a["raw_response"] == raw for a in attempts)
    assert not provider.cache and not provider.cache_path.exists()


def test_fixed_prompt_requires_plain_text_no_json_encoding_on_every_attempt(workspace):
    source = 'Observation 1: A quoted "name".\nThought 2:'
    invalid = 'A quoted "name".\nThought 3:'
    valid = 'A quoted "name".'
    provider = paraphraser(workspace, model=SequenceQwen([invalid, invalid, valid]))
    assert provider.paraphrase(source) == source
    attempts = diagnostic_attempts(provider)
    prompt = runtime.PROTOCOL["output_transport_prompt"]
    assert "plain UTF-8 paraphrased text" in prompt
    assert "do not JSON-escape" in prompt
    assert "No JSON envelope" in prompt
    assert len(attempts) == 3
    for attempt in attempts:
        assert prompt in attempt["request"][0]["content"]
        assert attempt["prompt_revision"] == 8
    fingerprint = runtime.RuntimeCitationScorer("lexical_paraphrase").fingerprint()["paraphrase_generation"]
    assert fingerprint["output_transport_prompt"] == prompt and fingerprint["prompt_revision"] == 8
    assert fingerprint["output_transport"] == "plain_utf8_eos_v1"
    assert fingerprint["retry_strategy"] == "numeric_diff_guided_v1"


def test_actual_v6_failure_plain_boundary_never_parses_model_json(workspace, monkeypatch):
    source = "Invalid action.\nI will be: Search[scientific journal Bronx"
    failed_v6 = '{"paraphrase":"Invalid action.\\nI will be: Search[scientific journal Bronx"]'
    with pytest.raises(json.JSONDecodeError) as caught:
        runtime.parse_paraphrase(failed_v6)
    assert caught.value.pos == 75

    def forbidden(*args, **kwargs):
        raise AssertionError("Production must never parse a JSON model-output envelope")

    monkeypatch.setattr(runtime, "parse_paraphrase", forbidden)
    monkeypatch.setattr(runtime, "_decode_paraphrase_transport", forbidden)
    provider = paraphraser(workspace, model=FakeQwen(source))
    assert provider.paraphrase(source) == source
    attempt = diagnostic_attempts(provider)[0]
    assert attempt["raw_response"] == source and attempt["output_transport"] == "plain_utf8_eos_v1"
    assert attempt["payload_transport"]["normalization_applied"] is False
    assert attempt["json_transport"] is None and attempt["accepted"]
    assert attempt["generated_token_ids"][-1] == provider.tokenizer.eos_token_id
    assert attempt["payload_token_offset"] == provider.model.calls[0]["input_ids"].shape[-1]
    restored = paraphraser(workspace)
    assert restored.paraphrase(source) == source and not restored.model.calls
    assert json.loads(next(iter(restored.cache.values()))["payload_transport"]["canonical_response"]) == {
        "paraphrase": source}


class BoundaryTokenizer(ControlPreservingTokenizer):
    def decode(self, ids, **kwargs):
        self.decode_calls = getattr(self, "decode_calls", []) + [(list(map(int, ids)), kwargs)]
        return super().decode(ids, **kwargs)


def test_plain_payload_preserves_unicode_quotes_controls_and_offsets_exactly(workspace):
    source = '  Δ “quotation” Café 北京 😀\r\nAction 2: Search[Å "日本"]\t\n'
    prose = 'Δ “quotation” Café 北京 😀'
    tokenizer = BoundaryTokenizer()
    provider = paraphraser(workspace, model=FakeQwen(prose), tokenizer=tokenizer)
    tokenizer.decode_calls = []
    assert provider.paraphrase(source) == source
    ids, kwargs = tokenizer.decode_calls[0]
    assert ids == [ord(char) + 2 for char in prose]
    assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    attempt = diagnostic_attempts(provider)[0]
    assert attempt["raw_response"] == prose
    assert attempt["raw_response_utf8_valid"]
    assert attempt["raw_response_utf8_bytes"] == len(prose.encode("utf-8"))
    assert attempt["raw_response_sha"] == runtime.text_sha(prose)
    assert attempt["protected_spans_validated"] and attempt["numeric_missing"] == {}
    assert json.loads(attempt["payload_transport"]["canonical_response"])["paraphrase"] == prose
    assert not next(iter(provider.cache.values()))["rewrite_audit"]["meaning_equivalence_verified"]


@pytest.mark.parametrize("control", [chr(i) for i in range(32) if i not in {9, 10, 13}] + ["\x7f", "\x85"])
def test_plain_payload_forbidden_controls_are_terminal_and_losslessly_durable(workspace, control):
    value = "1000" + control
    provider = paraphraser(workspace, model=FakeQwen(value), tokenizer=ControlPreservingTokenizer())
    with pytest.raises(ValueError, match="Forbidden"):
        provider.paraphrase("source")
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == 1 and attempts[0]["raw_response"] == value
    assert attempts[0]["raw_response_sha"] == runtime.text_sha(value)
    assert attempts[0]["auxiliary_forward_calls"] == 1
    assert not provider.cache and not provider.cache_path.exists()


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_non_utf8_scalar_is_terminal_with_complete_raw_diagnostics(workspace, surrogate):
    value = "bad" + surrogate
    provider = paraphraser(workspace, model=FakeQwen(value), tokenizer=ControlPreservingTokenizer())
    with pytest.raises(ValueError, match="strict UTF8"):
        provider.paraphrase("source")
    attempt = diagnostic_attempts(provider)[0]
    assert attempt["raw_response"] == value
    assert attempt["raw_response_utf8_valid"] is False
    assert attempt["raw_response_sha"] is None
    assert attempt["generated_token_ids"] == [ord(c) + 2 for c in value] + [1]
    assert attempt["auxiliary_forward_calls"] == 1 and attempt["auxiliary_kv_tensor_bytes"] == 20
    assert not provider.cache


@pytest.mark.parametrize("value", ["", " ", "\n\r\t"])
def test_plain_empty_payload_is_terminal_without_original_fallback(workspace, value):
    provider = paraphraser(workspace, model=FakeQwen(value))
    with pytest.raises(ValueError, match="nonempty"):
        provider.paraphrase("original")
    attempts = diagnostic_attempts(provider)
    assert len(attempts) == 1 and attempts[0]["raw_response"] == value
    assert attempts[0]["eos_validated"] and not provider.cache


def test_plain_output_cap_is_enforced_even_with_scripted_model(workspace):
    value = "p" * 512
    provider = paraphraser(workspace, model=FakeQwen(value))
    with pytest.raises(ValueError, match="exceeds max_new_tokens"):
        provider.paraphrase("source")
    attempt = diagnostic_attempts(provider)[0]
    assert attempt["raw_response"] == value and attempt["generated_tokens"] == 513
    assert attempt["eos_validated"] and not provider.cache


def test_plain_paraphrase_conditions_share_same_frozen_cache(workspace):
    source = 'trees "quoted"\nnext line'
    lexical_provider = paraphraser(workspace, model=FakeQwen(source))
    lexical = runtime.RuntimeCitationScorer("lexical_paraphrase", paraphraser=lexical_provider)
    lexical.register_step(0, source, extract_anchor_terms(source))
    lexical.increments(source, {0: {"anchors": extract_anchor_terms(source)}})
    semantic_provider = paraphraser(workspace, model=FakeQwen("must never be generated"))
    semantic = runtime.RuntimeCitationScorer(
        "semantic_paraphrase", paraphraser=semantic_provider, encoder=FakeEncoder())
    semantic.register_step(0, source, extract_anchor_terms(source))
    semantic.increments(source, {0: {"anchors": extract_anchor_terms(source)}})
    assert not semantic_provider.model.calls
    assert lexical_provider._key(source) == semantic_provider._key(source)
    assert lexical_provider.cache == semantic_provider.cache
    assert (lexical.fingerprint()["paraphrase_generation"] ==
            semantic.fingerprint()["paraphrase_generation"])


class ProseSourceModel(FakeQwen):
    def __init__(self, tokenizer, transform=lambda text: text):
        super().__init__()
        self.tokenizer = tokenizer
        self.transform = transform
        self.source_inputs = []

    def generate(self, **kwargs):
        source = json.loads(self.tokenizer.messages[1]["content"])["source"]
        self.source_inputs.append(source)
        self.response = self.transform(source)
        return super().generate(**kwargs)


def prose_provider(path, transform=lambda text: text):
    tokenizer = FakeTokenizer()
    model = ProseSourceModel(tokenizer, transform)
    return paraphraser(path, model=model, tokenizer=tokenizer)


V7_ORDINAL_ARTIFACT = """{'1': ", '2': ", '3': ", '4': "}"""
V7_ACTUAL_SOURCE = (
    "\nObservation 4: Pierre Charles Bouvier " + V7_ORDINAL_ARTIFACT +
    " (born 9 May 1979) is a Canadian singer, songwriter, musician, composer and actor who is best "
    "known as the lead singer and guitarist of the rock band Simple Plan..\nThought 5:"
)


def test_real_v7_ordinal_artifact_fixed_before_generation_facts_still_rewritten(workspace):
    provider = prose_provider(workspace, lambda text: text.replace("best known", "recognized"))
    result = provider.paraphrase(V7_ACTUAL_SOURCE)
    assert result == V7_ACTUAL_SOURCE.replace("best known", "recognized")
    assert V7_ORDINAL_ARTIFACT in result
    assert provider.model.source_inputs[0] == "Pierre Charles Bouvier"
    assert all(not runtime._numeric_mentions(text) for text in provider.model.source_inputs)
    assert all(V7_ORDINAL_ARTIFACT not in text and "Observation 4:" not in text and "Thought 5:" not in text
               for text in provider.model.source_inputs)
    layout = provider.events[0]["source_layout"]
    artifact = next(span for span in layout["protected_spans"] if span["kind"] == "opaque_source_artifact")
    start, end = artifact["span"]
    assert V7_ACTUAL_SOURCE[start:end] == artifact["text"] == V7_ORDINAL_ARTIFACT
    assert artifact["raw_literal"] == V7_ORDINAL_ARTIFACT
    assert artifact["recognition_policy"] == "opaque_source_placeholder_map_v1"
    assert artifact["structural_numeric_mentions"] == {"1": 1, "2": 1, "3": 1, "4": 1}
    assert "no factual prose or numeric values" in artifact["recognition_reason"]
    assert runtime._numeric_mentions(result) == runtime._numeric_mentions(V7_ACTUAL_SOURCE)
    spans = diagnostic_attempts(provider)
    logical = logical_diagnostic_attempts(provider)
    assert len(spans) >= 2 and len(logical) == 1 and logical[0]["accepted"]
    assert provider.audit()["auxiliary_forward_calls"] == sum(s["auxiliary_forward_calls"] for s in spans) == len(spans)
    assert provider.audit()["generated_tokens_total"] == sum(s["generated_tokens"] for s in spans)
    assert logical[0]["generated_tokens"] == sum(s["generated_tokens"] for s in spans)
    assert sum(call["max_new_tokens"] for call in provider.model.calls) <= 512
    assert next(iter(provider.cache.values()))["composite_tokens_including_eos"] <= 512
    assert not next(iter(provider.cache.values()))["layout_audit"]["no_llm_rewrite"]
    assert not next(iter(provider.cache.values()))["rewrite_audit"]["meaning_equivalence_verified"]
    restored = paraphraser(workspace)
    assert restored.paraphrase(V7_ACTUAL_SOURCE) == result and not restored.model.calls


@pytest.mark.parametrize("factual", [
    "{'1': 'Pierre', '2': 'Charles'}", "{'1': 9, '2': 1979}", "{'1979': '', '1980': ''}",
    "Born 9 May 1979.", "1979. A factual year at the start of a line.", "1. May 1979",
])
def test_factual_values_and_years_never_protected_as_ordinal_scaffolding(factual):
    layout = runtime._source_layout(factual)
    assert all(span["kind"] == "numerical_anchor" for span in layout["protected_spans"])
    assert runtime._reconstruct_layout(layout, [slot["source"] for slot in layout["slots"]]) == factual


def test_search_result_list_labels_preserved_by_layout_not_requested_from_model(workspace):
    source = "Observation 3:\n1. Film born 1979\n2) Music from 2008\n3: Novel in 2012\nThought 4:"
    provider = prose_provider(workspace, lambda text: text.replace("Film", "Cinema").replace("Music", "Songs"))
    result = provider.paraphrase(source)
    assert result == source.replace("Film", "Cinema").replace("Music", "Songs")
    assert provider.model.source_inputs == ["Film born", "Music from", "Novel in"]
    assert all(not text.startswith(("1.", "2)", "3:")) for text in provider.model.source_inputs)
    assert runtime._numeric_mentions(result) == runtime._numeric_mentions(source)
    spans = diagnostic_attempts(provider)
    assert len(spans) == 3 and all(span["allocated_max_new_tokens"] < 512 for span in spans)
    assert sum(span["allocated_max_new_tokens"] for span in spans) <= 512


@pytest.mark.parametrize("source", [
    "Action 2: Search[Name]", "Action 2: Lookup[unfinished", "Observation 4:\nThought 5:",
    V7_ORDINAL_ARTIFACT,
])
def test_source_only_metadata_is_explicit_identity_no_fake_qwen_forward(workspace, source):
    model = FakeQwen()
    model.fail = AssertionError("Metadata-only source must not be sent to Qwen")
    provider = paraphraser(workspace, model=model)
    assert provider.paraphrase(source) == source
    assert not model.calls and not diagnostic_attempts(provider)
    identity = logical_diagnostic_attempts(provider)[0]
    assert identity["record_scope"] == "no_rewritable_prose" and identity["accepted"]
    assert identity["no_llm_rewrite"] and identity["generated_tokens"] == identity["auxiliary_forward_calls"] == 0
    entry = next(iter(provider.cache.values()))
    assert entry["generation_mode"] == "no_rewritable_prose" and entry["accepted_attempt"] == 0
    assert entry["layout_audit"]["no_llm_rewrite"] and entry["layout_audit"]["prose_slots"] == 0
    assert provider.audit()["no_rewritable_prose_requests"] == 1
    assert provider.audit()["realized_stress"]["mean_prose_anchor_overlap"] is None
    restored = paraphraser(workspace)
    assert restored.paraphrase(source) == source and not restored.model.calls


def test_bad_factual_numbers_retry_only_three_logical_sources_with_all_span_costs(workspace):
    source = "Observation 1: Name " + V7_ORDINAL_ARTIFACT + " born 9 May 1979.\nThought 2:"
    provider = prose_provider(workspace, lambda text: text + " 1980" if "May" in text else text)
    with pytest.raises(runtime.NumericMismatch):
        provider.paraphrase(source)
    spans, logical = diagnostic_attempts(provider), logical_diagnostic_attempts(provider)
    assert len(logical) == 3 and len(spans) >= 6
    assert all(a["rejection_type"] == "NumericMismatch" and not a["accepted"] for a in logical)
    assert all(s["numeric_missing"] == {} and s["numeric_extra"] == {"1.98E+3": 1}
               for s in spans if not s["accepted"])
    assert provider.audit()["auxiliary_forward_calls"] == len(provider.model.calls) == len(spans)
    assert provider.audit()["logical_source_attempts"] == 3
    for attempt in logical:
        actual = [s for s in spans if s["attempt"] == attempt["attempt"]]
        assert sum(s["allocated_max_new_tokens"] for s in actual) <= 512
        assert attempt["generated_tokens"] == sum(s["generated_tokens"] for s in actual)
        assert attempt["auxiliary_forward_calls"] == sum(s["auxiliary_forward_calls"] for s in actual)
    assert not provider.cache and not provider.cache_path.exists()


def test_aggregate_composite_budget_checked_without_clipping_generated_slots(workspace):
    provider = prose_provider(workspace)
    layout = runtime._source_layout("one")
    # Check the final guard independently of model generation's per-slot cap.
    with pytest.raises(ValueError, match="composite exceeds 512"):
        provider._check_composite("one", layout, {0: "x" * 512})


def test_slot_cap_is_explicit_and_never_bypassed_by_copying_extra_prose():
    source = "\n".join(f"{i}. factual prose" for i in range(1, 18))
    with pytest.raises(ValueError, match="too many rewritable"):
        runtime._source_layout(source)


def test_layout_cache_reset_and_paired_conditions_share_exact_reconstruction(workspace):
    source = "Observation 1: prose born 1979.\nThought 2:"
    provider = prose_provider(workspace, lambda text: text.replace("prose", "account"))
    result = provider.paraphrase(source)
    provider.reset_episode()
    assert provider.paraphrase(source) == result and provider.audit()["auxiliary_forward_calls"] == 0
    other = paraphraser(workspace)
    assert other.paraphrase(source) == result and not other.model.calls
    a = runtime.RuntimeCitationScorer("lexical_paraphrase", paraphraser=provider)
    b = runtime.RuntimeCitationScorer("semantic_paraphrase", paraphraser=other, encoder=FakeEncoder())
    assert a.fingerprint()["paraphrase_generation"] == b.fingerprint()["paraphrase_generation"]
    assert a.fingerprint()["paraphrase_generation"]["layout_policy"] == runtime.PROTOCOL["layout_policy"]


@pytest.mark.parametrize("artifact", [
    V7_ORDINAL_ARTIFACT, """['1': ", '2': ", '3': ", '4': "]""",
    "{'1': '', '2': ''}",
])
def test_opaque_artifact_boundaries_do_not_sweep_neighboring_factual_prose(artifact):
    source = "Pierre Charles Bouvier " + artifact + " (born 9 May 1979) sings for Simple Plan."
    layout = runtime._source_layout(source)
    protected = next(span for span in layout["protected_spans"] if span["kind"] == "opaque_source_artifact")
    assert protected["raw_literal"] == protected["text"] == artifact
    start, end = protected["span"]
    assert source[start:end] == artifact
    assert source[:start] == "Pierre Charles Bouvier "
    assert source[end:] == " (born 9 May 1979) sings for Simple Plan."
    assert layout["slots"][0]["source"] == "Pierre Charles Bouvier"
    assert runtime._reconstruct_layout(layout, [slot["source"] for slot in layout["slots"]]) == source
    assert all(not runtime._numeric_mentions(slot["source"]) for slot in layout["slots"])
    assert "9" not in protected["structural_numeric_mentions"]
    assert "1979" not in protected["structural_numeric_mentions"]


@pytest.mark.parametrize("factual", [
    "[1, 2, 3, 4]", "{'1': '1979', '2': '1980'}",
    "{'1': 9, '2': 1979}", "['1979': '', '1980': '']",
])
def test_numeric_or_prose_values_are_not_opaque_placeholder_artifacts(factual):
    layout = runtime._source_layout(factual)
    assert not any(span["kind"] == "opaque_source_artifact" for span in layout["protected_spans"])
    assert runtime._reconstruct_layout(layout, [slot["source"] for slot in layout["slots"]]) == factual


class ChunkTokenizer(FakeTokenizer):
    """Lossless injected codec; lets long real source excerpts fit the CPU source bound."""
    def __init__(self):
        self.pieces = {}
        self.ids = {}

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None,
                 return_tensors=None, return_offsets_mapping=False):
        chunks = [text[i:i + 4] for i in range(0, len(text), 4)]
        ids = []
        for chunk in chunks:
            if chunk not in self.ids:
                index = len(self.ids) + 10
                self.ids[chunk] = index
                self.pieces[index] = chunk
            ids.append(self.ids[chunk])
        offsets = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
        if add_special_tokens:
            ids = [2] + ids + [1]
        if truncation:
            ids = ids[:max_length]
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        if return_tensors:
            result = {key: torch.tensor([value], dtype=torch.long) for key, value in result.items()}
        return result

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(int(index), "<EOS>") for index in ids)


class ChunkProseModel(ProseSourceModel):
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self()
        source = json.loads(self.tokenizer.messages[1]["content"])["source"]
        self.source_inputs.append(source)
        response = self.transform(source)
        tail = self.tokenizer(response, add_special_tokens=False)["input_ids"] + [1]
        return SimpleNamespace(
            sequences=torch.cat([kwargs["input_ids"], torch.tensor([tail], dtype=torch.long)], dim=1),
            past_key_values=((torch.ones(2), torch.ones(3)),))


V8_ACTUAL_SOURCE = (
    'The Secret Circle is a supernatural, young-adult series of books created by L. J. Smith. The '
    'narrative follows the character of Cassie Blake as she is initiated into a "Circle" of eleven '
    'other teenage witches and the danger that ensues when they accidentally unleash a dark force '
    'upon their town. Matters are further complicated when Cassie finds herself in a love triangle '
    'that threatens to tear the Circle apart..\nTheObservation 5: 1: Geoff LaTulippe is an American '
    'writer and screenwriter. His best-known work is the television series "The Secret Circle," '
    'which was created for The CW by Kevin Williamson.\nThought 6: The information provided does '
    'not mention who directed Geoff LaTulippe\'s best-known work. I need to find more details about '
    '"The Secret Circle" and its director.\nAction 2: Search[The Secret Circle]'
)


def test_actual_v8_malformed_inline_reference_numeric_anchors_not_inferred_metadata(workspace):
    tokenizer = ChunkTokenizer()
    model = ChunkProseModel(tokenizer, lambda text: text.replace("narrative", "story"))
    provider = paraphraser(workspace, model=model, tokenizer=tokenizer)
    result = provider.paraphrase(V8_ACTUAL_SOURCE)
    assert result == V8_ACTUAL_SOURCE.replace("narrative", "story")
    assert "TheObservation 5: 1:" in result
    layout = provider.events[0]["source_layout"]
    anchors = [s for s in layout["protected_spans"] if s["kind"] == "numerical_anchor"]
    inline = [s for s in anchors if s["text"] in {"5", "1"}]
    assert len(inline) == 2 and all(s["inside_structural_span"] for s in inline)
    assert any(s["text"] == "TheObservation 5: 1:" and s["kind"] == "header"
               for s in layout["protected_spans"])
    assert all("TheObservation" not in text for text in model.source_inputs)
    assert all(s["preservation_role"] == "numeric_literal_not_metadata" and s["semantic_type"] == "not_inferred"
               for s in inline)
    for anchor in anchors:
        start, end = anchor["span"]
        assert V8_ACTUAL_SOURCE[start:end] == anchor["raw_literal"]
    assert len(layout["slots"]) == 3 < runtime.PROTOCOL["layout_policy"]["max_prose_slots"]
    assert all(not runtime._numeric_mentions(text) for text in model.source_inputs)
    assert runtime._numeric_mentions(result) == runtime._numeric_mentions(V8_ACTUAL_SOURCE)
    assert sum(call["max_new_tokens"] for call in model.calls) <= 512
    entry = next(iter(provider.cache.values()))
    assert entry["composite_tokens_including_eos"] <= 512
    stats = entry["layout_audit"]
    assert stats["numerical_anchor_character_fraction"] > 0
    assert stats["metadata_character_fraction"] > 0
    assert stats["numerical_anchor_semantics"].startswith("not_inferred")
    assert not entry["rewrite_audit"]["meaning_equivalence_verified"]
    assert provider.audit()["auxiliary_forward_calls"] == len(model.calls) == 3
    restored = paraphraser(workspace)
    assert restored.paraphrase(V8_ACTUAL_SOURCE) == result and not restored.model.calls


@pytest.mark.parametrize("source", [
    "Born 9 May 1979 and died 2 January 2010.",
    "Placed 1st in 2010, 2nd in 2010 and 3rd in 2008.",
    "Values -3.00 and 1,000.00; repeated 1,000.00.",
    "Version v2 and model A320 contain literal digits.",
])
def test_dates_ordinals_formats_and_repeats_are_exact_raw_numeric_anchors(workspace, source):
    provider = prose_provider(workspace)
    result = provider.paraphrase(source)
    assert result == source
    matches = list(runtime.re.finditer(runtime._NUMBER_PATTERN, source))
    anchors = [s for s in provider.events[0]["source_layout"]["protected_spans"] if s["kind"] == "numerical_anchor"]
    assert [(s["numeric_span"], s["numeric_literal"]) for s in anchors] == [
        ([m.start(), m.end()], m.group()) for m in matches]
    assert all(source[s["span"][0]:s["span"][1]] == s["raw_literal"] for s in anchors)
    assert all(s["semantic_type"] == "not_inferred" and s["preservation_role"] == "numeric_literal_not_metadata"
               for s in anchors)
    assert all(not runtime._numeric_mentions(text) for text in provider.model.source_inputs)
    assert provider.audit()["generated_tokens_total"] == sum(s["generated_tokens"] for s in diagnostic_attempts(provider))


def test_large_numeric_only_source_is_explicit_non_prose_identity_with_zero_forwards(workspace):
    source = " ".join(str(i) for i in range(1, 101))
    provider = paraphraser(workspace, model=FakeQwen("must never run"))
    assert provider.paraphrase(source) == source
    assert not provider.model.calls and not diagnostic_attempts(provider)
    entry = next(iter(provider.cache.values()))
    assert entry["generation_mode"] == "no_rewritable_prose" and entry["accepted_attempt"] == 0
    assert len([s for s in entry["protected_spans"] if s["kind"] == "numerical_anchor"]) == 100
    assert entry["layout_audit"]["metadata_characters"] == 0
    assert entry["layout_audit"]["numerical_anchor_characters"] > 0
    assert entry["layout_audit"]["prose_characters"] == 0 and entry["layout_audit"]["no_llm_rewrite"]
    assert entry["composite_tokens_including_eos"] <= 512


def test_many_numeric_anchors_never_bypass_fixed_prose_slot_cap_by_copying_prose():
    source = " ".join("word " + str(i) for i in range(1, 18))
    with pytest.raises(ValueError, match="too many rewritable prose slots"):
        runtime._source_layout(source)


def test_numeric_literal_format_cannot_be_silently_normalized_in_composite(workspace):
    provider = prose_provider(workspace)
    source = "Value 1,000.00."
    layout = runtime._source_layout(source)
    fake_outputs = {s["slot_id"]: s["source"] for s in layout["slots"]}
    composite = runtime._reconstruct_layout(layout, fake_outputs)
    # Whole normalized numeric equality is supplemented by exact raw anchor equality.
    normalized = composite.replace("1,000.00", "1000")
    assert runtime._numeric_mentions(normalized) == runtime._numeric_mentions(source)
    assert runtime._protected_span_diff(normalized, layout["protected_spans"])


def test_public_tokenizer_only_planner_matches_production_without_any_model_load(workspace, monkeypatch):
    import transformers

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure preflight must not load weights or initialize CUDA")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    tokenizer = FakeTokenizer()
    source = "Observation 1: prose born 9 May 1979.\nThought 2:"
    plan = runtime.plan_source_layout(source, tokenizer)
    provider = paraphraser(workspace, tokenizer=tokenizer, model=ProseSourceModel(tokenizer))
    assert provider.paraphrase(source) == source
    event = provider.events[0]
    assert plan["source_prefix"] == event["source_prefix"]
    assert plan["layout"] == event["source_layout"] and plan["budget"] == event["budget"]
    assert plan["retained_source_tokens"] <= 512
    assert "no generated-prose or meaning-equivalence" in plan["scope"]


def test_public_planner_rejects_known_infeasible_sources_explicitly():
    with pytest.raises(ValueError, match="bisects"):
        runtime.plan_source_layout("x" * 510 + " 123456", FakeTokenizer())
    with pytest.raises(ValueError, match="identity exceeds"):
        runtime.plan_source_layout("1" * 512, FakeTokenizer())
    with pytest.raises(ValueError, match="too many rewritable"):
        runtime.plan_source_layout(" ".join("word " + str(i) for i in range(1, 18)), FakeTokenizer())


def test_public_planner_lossless_offsets_on_previous_native_failures():
    tokenizer = ChunkTokenizer()
    for source in [NATIVE_V4_MALFORMED_SOURCE, V7_ACTUAL_SOURCE, V8_ACTUAL_SOURCE]:
        plan = runtime.plan_source_layout(source, tokenizer)
        layout = plan["layout"]
        assert runtime._reconstruct_layout(layout, [s["source"] for s in layout["slots"]]) == plan["source_prefix"]
        for span in layout["protected_spans"]:
            start, end = span["span"]
            assert plan["source_prefix"][start:end] == span["text"]
        assert sum(plan["budget"]["slot_caps"]) <= 512
        assert len(layout["slots"]) <= runtime.PROTOCOL["layout_policy"]["max_prose_slots"]


@pytest.mark.parametrize("marker", ["TheObservation 5: 1:", "Observation 5: 1: 2:", "PreviousThought 6:"])
def test_fused_nested_role_prefix_fixed_before_prose_generation(workspace, marker):
    source = "Narrative.\n" + marker + " Factual neighboring prose born 1979."
    provider = prose_provider(workspace, lambda text: text.replace("neighboring", "adjacent"))
    result = provider.paraphrase(source)
    assert result == source.replace("neighboring", "adjacent")
    layout = provider.events[0]["source_layout"]
    role = next(s for s in layout["protected_spans"] if s["kind"] == "header" and s["text"] == marker)
    assert role["recognition_policy"] == "native_role_prefix_literal_v1"
    assert role["raw_literal"] == marker
    assert all(marker not in text for text in provider.model.source_inputs)
    assert all("Factual neighboring prose" not in s["text"] for s in layout["protected_spans"])


def test_inline_role_prefix_protected_but_command_argument_not_repartitioned():
    source = "A preceding phrase TheObservation 5: 1: a following phrase."
    layout = runtime._source_layout(source)
    assert any(s["kind"] == "header" and s["text"] == "TheObservation 5: 1:"
               for s in layout["protected_spans"])
    command = 'Action 2: Search[TheObservation 5: 1: literal argument]'
    protected = runtime._protected_react_spans(command)
    assert [s["text"] for s in protected] == ["Action 2:", command]
    native = runtime._protected_react_spans(NATIVE_V4_MALFORMED_SOURCE)
    assert any(s["kind"] == "command_fragment" and s["text"] == NATIVE_V4_MALFORMED_SOURCE.splitlines()[-1]
               for s in native)


def test_aux_generate_digit_suppression_only_no_generation_config_or_main_mutation(workspace):
    provider = prose_provider(workspace)
    main = FakeQwen()
    main_before = copy.deepcopy(main.generation_config.__dict__)
    aux_before = copy.deepcopy(provider.model.generation_config.__dict__)
    source = "In the 13th century."
    assert provider.paraphrase(source) == source
    table = provider.prose_digit_constraint
    assert table["ban_count"] == len(table["suppress_tokens"])
    assert table["ban_sha"] == runtime._json_sha(table["suppress_tokens"])
    assert not set(table["suppress_tokens"]) & set(provider.tokenizer.all_special_ids)
    assert all(call["suppress_tokens"] == table["suppress_tokens"] for call in provider.model.calls)
    assert provider.model.generation_config.__dict__ == aux_before
    assert main.generation_config.__dict__ == main_before and not main.calls
    spans = diagnostic_attempts(provider)
    assert all(s["prose_decoding_constraint"]["ban_sha"] == table["ban_sha"] for s in spans)
    assert "13" not in "".join(provider.model.source_inputs)
    assert any(s["kind"] == "numerical_anchor" and s["numeric_literal"] == "13"
               for s in provider.events[0]["source_layout"]["protected_spans"])
    original = runtime.build_prose_digit_constraint(provider.tokenizer)
    original["suppress_tokens"].clear()
    assert runtime.build_prose_digit_constraint(provider.tokenizer)["ban_count"] > 0


def test_actual_local_qwen_vocab_real_logits_processor_blocks_13_preserves_eos():
    from transformers import AutoTokenizer, SuppressTokensLogitsProcessor
    path = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/assets/models/Qwen2.5-7B-Instruct")
    if not (path / "tokenizer.json").is_file():
        pytest.skip("Pinned local Qwen tokenizer assets unavailable; no network/model fallback")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    table = runtime.build_prose_digit_constraint(tokenizer)
    banned = set(table["suppress_tokens"])
    ids = tokenizer.encode("13", add_special_tokens=False)
    assert ids and all(i in banned for i in ids)
    assert not banned & set(tokenizer.all_special_ids)
    scores = torch.zeros(1, len(tokenizer))
    for i in ids:
        scores[0, i] = 100.
    letter = tokenizer.encode("a", add_special_tokens=False)[0]
    scores[0, letter] = 10.
    scores[0, tokenizer.eos_token_id] = 5.
    output = SuppressTokensLogitsProcessor(table["suppress_tokens"])(
        torch.tensor([[tokenizer.eos_token_id]]), scores)
    assert all(torch.isneginf(output[0, i]) for i in ids)
    assert output.argmax(-1).item() == letter
    assert torch.isfinite(output[0, tokenizer.eos_token_id])
    assert all(i in banned for i in tokenizer.encode("１", add_special_tokens=False))
    # Disclosure: these Nd characters use individually non-digit byte fragments.
    split = tokenizer.encode("٣", add_special_tokens=False)
    assert not any(i in banned for i in split)
    assert tokenizer.decode(split, clean_up_tokenization_spaces=False) == "٣"


@pytest.mark.parametrize("new_digit", ["13", "10", "٣", "𝟙"])
def test_scripted_digit_or_unicode_byte_split_bypass_is_rejected_never_posthoc_removed(workspace, new_digit):
    provider = prose_provider(workspace, lambda text: text + " " + new_digit)
    with pytest.raises(runtime.NumericMismatch):
        provider.paraphrase("th century from earlier sources.")
    spans = diagnostic_attempts(provider)
    assert len(spans) == 3
    assert all(s["raw_response"].endswith(new_digit) and not s["accepted"] for s in spans)
    assert all(s["prose_decoding_constraint"]["policy_id"] == "prose_vocab_digit_suppression_v1" for s in spans)
    assert not provider.cache and not provider.cache_path.exists()


V9_FULLA_SOURCE = (
    '\nObservation 2: In Germanic mythology, Fulla (Old Norse, possibly "bountiful") or Volla '
    '(Old High German) is a goddess. In Norse mythology, Fulla is described as wearing a golden '
    'band and as tending to the ashen box and the footwear owned by the goddess Frigg, and, in '
    'addition, Frigg confides in Fulla her secrets. Fulla is attested in the "Poetic Edda", '
    'compiled in the 13th century from earlier traditional sources; the "Prose Edda", written '
    'in the 13th century by Snorri Sturluson; and in skaldic poetry. Volla is attested in the '
    '"Horse Cure" Merseburg Incantation, recorded anonymously in the 10th century in Old High '
    "German, in which she assists in healing the wounded foal of Phol and is referred to as "
    "Frigg's sister. Scholars have proposed theories about the implications of the goddess..\nThought 3:"
)


def test_actual_v9_fulla_ordinal_centuries_bound_without_neighboring_narrative(workspace):
    tokenizer = ChunkTokenizer()
    model = ChunkProseModel(tokenizer)
    provider = paraphraser(workspace, tokenizer=tokenizer, model=model)
    assert provider.paraphrase(V9_FULLA_SOURCE) == V9_FULLA_SOURCE
    layout = provider.events[0]["source_layout"]
    anchors = [s for s in layout["protected_spans"] if s["kind"] == "numerical_anchor"]
    centuries = [s for s in anchors if s["bound_unit"] == "century"]
    assert [s["text"] for s in centuries] == ["13th century", "13th century", "10th century"]
    assert all(s["syntax_type"] == "ordinal_literal" and s["semantic_type"] == "not_inferred" for s in centuries)
    assert all("from earlier" not in s["text"] and "by Snorri" not in s["text"] for s in centuries)
    assert any("from earlier traditional sources" in s for s in model.source_inputs)
    assert all(not s.startswith("th century") and "century" not in s for s in model.source_inputs)
    assert sum(c["max_new_tokens"] for c in model.calls) <= 512
    assert next(iter(provider.cache.values()))["composite_tokens_including_eos"] <= 512


@pytest.mark.parametrize("source,bound", [
    ("During the 13th century from sources.", "13th century"),
    ("At the 1st year after school.", "1st year"),
    ("The 1950s brought changes.", "1950s"),
    ("H2O is water.", "H2O"),
    ("Model B52 was listed.", "B52"),
    ("We measured 3.5kg today.", "3.5kg"),
    ("A 12% increase followed.", "12%"),
    ("Water reached 20°C yesterday.", "20°C"),
    ("A load of 12 kg arrived.", "12 kg"),
    ("Recorded 1979-05-09 in the archive.", "1979-05-09"),
])
def test_generic_numeric_lexical_bindings_exact_source_and_no_narrative_sweep(source, bound):
    layout = runtime._source_layout(source)
    anchors = [s for s in layout["protected_spans"] if s["kind"] == "numerical_anchor"]
    assert any(s["text"] == bound for s in anchors)
    for anchor in anchors:
        start, end = anchor["span"]
        ns, ne = anchor["numeric_span"]
        assert source[start:end] == anchor["text"]
        assert source[ns:ne] == anchor["numeric_literal"]
        assert anchor["semantic_type"] == "not_inferred"
    assert runtime._reconstruct_layout(layout, [s["source"] for s in layout["slots"]]) == source
    assert any(s["source"] for s in layout["slots"])


@pytest.mark.parametrize("source,unit", [("Measured 3.5kg today.", "kg"), ("A 12% increase.", "%"),
                                       ("Water reached 20°C yesterday.", "°C")])
def test_attached_units_are_recorded_and_guarded_without_neighboring_prose(source, unit):
    anchors = [s for s in runtime._source_layout(source)["protected_spans"] if s["kind"] == "numerical_anchor"]
    assert len(anchors) == 1 and anchors[0]["bound_unit"] == unit
    assert "with_attached_unit" in anchors[0]["binding_reason"]


def test_spelled_out_duplicate_bound_century_is_terminal_not_accepted_or_retried(workspace):
    source = "Recorded in the 13th century from earlier sources."
    provider = prose_provider(workspace, lambda text: text + " thirteenth century")
    with pytest.raises(runtime.ProtectedSpanMismatch, match="grammatical units"):
        provider.paraphrase(source)
    spans = diagnostic_attempts(provider)
    assert len(spans) == 1 and spans[0]["numeric_missing"] == spans[0]["numeric_extra"] == {}
    assert spans[0]["lexical_binding_issues"] == ["century"]
    assert logical_diagnostic_attempts(provider)[0]["rejection_type"] == "ProtectedSpanMismatch"
    assert not provider.cache and not provider.cache_path.exists()
