"""Offline CPU contracts for the isolated QA32 runner; never load 32B weights."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import uuid

import pytest
import torch
from transformers import AutoTokenizer, BatchEncoding, Qwen2Config, Qwen2ForCausalLM

import run_all_wiki_experiments_v2 as base
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts import qa32_runtime as runtime
from scripts import run_qa32_rebuttal as runner
from token_tracker import TokenTracker


@pytest.fixture
def output_dir():
    # Project-local scratch only: no tempfile/tmp_path/system temporary directory.
    path = Path(".qa32-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, _name=name, **k:
                            pytest.fail(f"CPU test touched CUDA: {_name}"))
    monkeypatch.setattr(base, "RANDOM_SEED", 233)
    monkeypatch.setattr(base, "NUM_SAMPLES", 500)
    monkeypatch.setattr(base, "MAX_STEPS", 7)


def rows(count=605):
    return [{"id": f"id-{index}", "question": f"Question {index}?", "answer": f"Answer {index}"}
            for index in range(count)]


def cli_args(*extra):
    return runner.parser().parse_args(
        ["--dataset", "hotpotqa", "--method", "stepkv", "--assets", ".", "--output", ".", *extra])


def test_main_and_smoke_use_original_selection_disjoint_windows():
    data = rows()
    main, smoke = cli_args(), cli_args("--purpose", "smoke")
    runner.validate_protocol(main)
    runner.validate_protocol(smoke)
    assert (main.samples, main.sample_start) == (500, 0)
    assert (smoke.samples, smoke.sample_start) == (2, 500)
    chosen = runner.select_legacy_samples(base, data, samples=500, sample_start=0, expected_rows=605)
    heldout = runner.select_legacy_samples(base, data, samples=2, sample_start=500, expected_rows=605)
    order = list(range(len(data)))
    random.Random(233).shuffle(order)
    assert [i for i, _ in chosen] == order[:500]
    assert [i for i, _ in heldout] == order[500:502]
    assert not {row["id"] for _, row in chosen}.intersection(row["id"] for _, row in heldout)


@pytest.mark.parametrize("args", [
    ("--samples", "2"), ("--sample-start", "500"), ("--seed", "42"),
    ("--max-steps", "8"), ("--memory-limit-gib", "43"),
    ("--purpose", "smoke", "--sample-start", "0"),
    ("--purpose", "smoke", "--samples", "500"),
])
def test_protocol_guards(args):
    with pytest.raises(ValueError):
        runner.validate_protocol(cli_args(*args))


@pytest.mark.parametrize("problem", ["missing", "duplicate", "empty-question", "empty-answer", "count", "insufficient"])
def test_selection_rejects_invalid_input(problem):
    data = rows(3)
    if problem == "missing":
        del data[0]["id"]
    elif problem == "duplicate":
        data[1]["id"] = data[0]["id"]
    elif problem == "empty-question":
        data[0]["question"] = ""
    elif problem == "empty-answer":
        data[0]["answer"] = ""
    with pytest.raises(RuntimeError):
        runner.select_legacy_samples(base, data, samples=4 if problem == "insufficient" else 2,
                                     sample_start=0, expected_rows=4 if problem == "count" else 3)


@pytest.fixture
def synthetic_assets(output_dir, monkeypatch):
    spec = runner.read_json(runner.ASSET_SPEC)
    model = output_dir / "model"
    model.mkdir()
    runner.atomic_json(model / "config.json", {
        "model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 64,
        "num_attention_heads": 40, "num_key_value_heads": 8})
    for name in ("generation_config.json", "tokenizer_config.json", "tokenizer.json"):
        runner.atomic_json(model / name, {})
    (model / "weights.safetensors").write_bytes(b"fixture")
    runner.atomic_json(model / "model.safetensors.index.json", {
        "metadata": {"total_size": 7}, "weight_map": {"weight": "weights.safetensors"}})
    runner.atomic_json(output_dir / "model_ready.json", {
        "repo": runner.MODEL_REPO, "revision": runner.MODEL_REVISION, "path": str(model.resolve())})
    wiki = output_dir / "wiki"
    wiki.mkdir()
    (wiki / "titles.json").write_text("[]")
    (wiki / "corpus_texts.jsonl").write_text('"fixture"\n')
    runner.atomic_json(output_dir / "data_ready.json", {
        "hotpotqa": spec["data"]["hotpotqa"], "longrag": spec["data"]["longrag"],
        "wiki_index": str(wiki.resolve())})
    for dataset in ("2wiki", "musique"):
        directory = output_dir / "datasets" / dataset
        runner.atomic_json(directory / "normalized_dev.json", rows(3))
        spec["additional_datasets"][dataset]["sha256"] = runner.file_hash(directory / "normalized_dev.json")
        runner.atomic_json(directory / "ready.json", {
            **spec["additional_datasets"][dataset],
            "path": str((directory / "normalized_dev.json").resolve())})
    path = output_dir / "synthetic_spec.json"
    runner.atomic_json(path, spec)
    monkeypatch.setattr(runner, "ASSET_SPEC", path)
    return output_dir


@pytest.mark.parametrize("dataset", ["2wiki", "musique"])
def test_normalized_asset_revisions_and_payload_hashes(synthetic_assets, dataset):
    model, wiki, data, provenance, generation = runner.load_assets(synthetic_assets, dataset)
    assert model.name == "model" and wiki.name == "wiki"
    assert data == rows(3) and generation == {}
    assert provenance["model_revision"] == runner.MODEL_REVISION
    assert "model.safetensors.index.json" in provenance["model_metadata_sha256"]
    path = synthetic_assets / "datasets" / dataset / "normalized_dev.json"
    runner.atomic_json(path, rows(4))
    with pytest.raises(RuntimeError, match="payload checksum mismatch"):
        runner.load_assets(synthetic_assets, dataset)


@pytest.mark.parametrize("kind", ["model", "corpus", "normalizer", "quantized"])
def test_wrong_asset_revision_or_normalizer_fails_closed(synthetic_assets, kind):
    if kind == "model":
        path = synthetic_assets / "model_ready.json"
        value = runner.read_json(path)
        value["revision"] = "wrong"
    elif kind == "corpus":
        path = synthetic_assets / "data_ready.json"
        value = runner.read_json(path)
        value["longrag"]["revision"] = "wrong"
    elif kind == "normalizer":
        path = synthetic_assets / "datasets/2wiki/ready.json"
        value = runner.read_json(path)
        value["normalizer_sha256"] = "wrong"
    else:
        path = synthetic_assets / "model/config.json"
        value = runner.read_json(path)
        value["quantization_config"] = {"bits": 4}
    runner.atomic_json(path, value)
    with pytest.raises(RuntimeError):
        runner.load_assets(synthetic_assets, "2wiki")


@pytest.mark.parametrize("method", runtime.METHODS)
def test_defaults_are_collected_from_original_without_loading_model(monkeypatch, method):
    module = importlib.import_module("models.QwenLLMWithKVCache")
    monkeypatch.setattr(module, "AutoModelForCausalLM", SimpleNamespace(
        from_pretrained=lambda *a, **k: pytest.fail("Configuration extraction loaded a model")))
    original = base._finalize_react_kv_output
    config = runtime.legacy_config(base, method, .5)
    assert base._finalize_react_kv_output is original
    assert config["cache_ratio"] == .5 and config["protect_prompt"] is True
    assert config["prompt_prefill_keep_ratio"] == 1.0
    assert config["pruning_mode"] == runtime.METHODS[method]
    assert config["observation_window"] == (0 if runtime.METHODS[method] in ("step_aware_h2o", "tova") else 32)
    assert config["attn_mode"] == ("piggyback" if method == "stepkv" else "scoring_forward")
    assert config["step_aware_alpha"] == config["step_aware_beta"] == .8
    assert config["step_reward_weight"] == .85 and config["step_citation_weight"] == .15


def fake_parameter(device="cuda:0", dtype=torch.bfloat16, numel=16):
    return SimpleNamespace(
        device=torch.device(device), dtype=dtype, is_floating_point=lambda: True,
        numel=lambda: numel, element_size=lambda: 2 if dtype == torch.bfloat16 else 4)


def placed_model():
    first, second = fake_parameter(), fake_parameter("cuda:1")
    return SimpleNamespace(
        hf_device_map={"model.layers.0": 0, "model.layers.1": "cuda:1"},
        named_parameters=lambda: iter((("first", first), ("second", second))),
        named_buffers=lambda: iter((("rope", fake_parameter(dtype=torch.float32)),)),
        get_input_embeddings=lambda: SimpleNamespace(weight=first), device=torch.device("cuda:0"),
        model=SimpleNamespace(layers=[SimpleNamespace(parameters=lambda: iter([first])),
                                      SimpleNamespace(parameters=lambda: iter([second]))]),
        config=SimpleNamespace(_attn_implementation="sdpa"))


def test_two_gpu_allocator_policy_without_cuda_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(total_memory=80 * runtime.GIB, name=f"fake-{i}"))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: calls.append((fraction, device)))
    hardware = runtime.configure_cuda()
    assert calls == [(44 / 80, 0), (44 / 80, 1)]
    assert hardware["1"]["allocator_limit_bytes"] == 44 * runtime.GIB
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="Exactly two"):
        runtime.configure_cuda()


def test_factory_override_is_local_identical_and_restored():
    calls, placements = [], []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return placed_model()

    original = SimpleNamespace(from_pretrained=load)
    module = SimpleNamespace(AutoModelForCausalLM=original)
    for _ in runtime.METHODS:
        with runtime.BalancedModelFactory(module, "local-model", placements.append):
            module.AutoModelForCausalLM.from_pretrained(
                "local-model", dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        assert module.AutoModelForCausalLM is original
    assert all(call == calls[0] for call in calls)
    assert calls[0][1] == dict(dtype=torch.bfloat16, device_map="balanced",
                               max_memory={0: "40GiB", 1: "40GiB"},
                               trust_remote_code=True, local_files_only=True)
    assert all(p == placements[0] for p in placements)
    assert placements[0]["layer_devices"] == ["cuda:0", "cuda:1"]
    with pytest.raises(RuntimeError, match="signature changed"):
        with runtime.BalancedModelFactory(module, "local-model"):
            module.AutoModelForCausalLM.from_pretrained("local-model", dtype=torch.float32, device_map="auto")
    assert module.AutoModelForCausalLM is original


@pytest.mark.parametrize("bad", ["cpu-map", "disk-map", "single-gpu", "cpu-parameter", "float32", "ceiling", "cpu-buffer"])
def test_reject_offload_fake_sharding_and_bad_precision(bad):
    model = placed_model()
    if bad in ("cpu-map", "disk-map"):
        model.hf_device_map["model.layers.1"] = bad.split("-")[0]
    elif bad == "single-gpu":
        model.hf_device_map["model.layers.1"] = 0
    elif bad == "cpu-buffer":
        model.named_buffers = lambda: iter((("offloaded", fake_parameter("cpu")),))
    else:
        parameter = fake_parameter(
            "cpu" if bad == "cpu-parameter" else "cuda:1",
            torch.float32 if bad == "float32" else torch.bfloat16,
            numel=41 * runtime.GIB if bad == "ceiling" else 16)
        model.named_parameters = lambda: iter((("first", fake_parameter()), ("bad", parameter)))
    with pytest.raises(RuntimeError):
        runtime.validate_placement(model)


def test_layer_gather_matches_original_and_never_moves_cache(monkeypatch):
    tensor = torch.arange(72).reshape(1, 2, 12, 3)
    keep = torch.tensor([0, 3, 7, 11])
    moves = []
    original = torch.Tensor.to

    def traced(self, *args, **kwargs):
        moves.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", traced)
    actual = runtime.LayerLocalIndex(tensor)[:, :, keep, :]
    torch.testing.assert_close(actual, tensor[:, :, keep, :], rtol=0, atol=0)
    assert len(moves) == 1 and moves[0] is keep
    assert tensor.shape[-2] == 12


def test_cross_device_index_and_score_contract_with_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from kv_cache.kv_cache_manager import KVCacheManager
    with FakeTensorMode():
        key = torch.empty(1, 2, 8, 4, device="cuda:0")
        keep = torch.arange(4, device="cuda:1")
        assert runtime.LayerLocalIndex(key)[:, :, keep, :].device == key.device
        llm = SimpleNamespace(
            kv_manager=KVCacheManager({"pruning_mode": "step_aware_h2o", "num_score_layers": 2}),
            past_key_values=((key, key),))
        attentions = (torch.empty(1, 2, 3, 8, device="cuda:0"),
                      torch.empty(1, 2, 3, 8, device="cuda:1"))
        with runtime.LayerDeviceAdapter(llm):
            scorer = llm.kv_manager.pruning_strategy.h2o_scorer
            scores = scorer.compute_scores(attentions, 1, 7)
            matrix = scorer.compute_attention_matrix(attentions, 1, 7)
            assert scores.device == key.device and scores.shape == (6,)
            assert matrix.device == key.device and matrix.shape == (3, 6)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_colocating_reduced_attention_keeps_exact_arithmetic(dtype):
    from kv_cache.kv_cache_manager import KVCacheManager

    class AdvertisedDevice:
        def __init__(self, tensor, device):
            self.tensor, self.device, self.shape = tensor, torch.device(device), tensor.shape

        def __getitem__(self, index):
            return self.tensor[index]

    torch.manual_seed(233)
    attentions = tuple(torch.rand(1, 2, 3, 8).to(dtype) for _ in range(3))
    llm = SimpleNamespace(
        kv_manager=KVCacheManager({"pruning_mode": "step_aware_h2o", "num_score_layers": 3}),
        past_key_values=((torch.empty(1, 2, 8, 4), torch.empty(1, 2, 8, 4)),))
    scorer = llm.kv_manager.pruning_strategy.h2o_scorer
    scores, matrix = scorer.compute_scores(attentions, 1, 7), scorer.compute_attention_matrix(attentions, 1, 7)
    advertised = tuple(AdvertisedDevice(tensor, f"cuda:{index % 2}")
                       for index, tensor in enumerate(attentions))
    with runtime.LayerDeviceAdapter(llm):
        torch.testing.assert_close(scorer.compute_scores(advertised, 1, 7), scores, rtol=0, atol=0)
        torch.testing.assert_close(scorer.compute_attention_matrix(advertised, 1, 7), matrix, rtol=0, atol=0)


class TinyTokenizer:
    eos_token_id = 63

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1 for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding(dict(input_ids=tensor, attention_mask=torch.ones_like(tensor)))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return "".join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def tiny_models(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=512, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    template = Qwen2ForCausalLM(config).eval()
    template.generation_config.repetition_penalty = 1.05
    module = importlib.import_module("models.QwenLLMWithKVCache")
    monkeypatch.setattr(module, "AutoModelForCausalLM",
                        SimpleNamespace(from_pretrained=lambda *a, **k: copy.deepcopy(template)))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())

    def make(method, ratio=.5):
        return runtime.llm_class_for_method(method)(
            "tiny-offline", runtime.legacy_config(base, method, ratio),
            TokenTracker() if method != "fullkv" else None)
    return make


def assert_same_state(first, second):
    for left, right in zip(runtime.cache_tensor_pairs(first.past_key_values),
                           runtime.cache_tensor_pairs(second.past_key_values)):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert first.current_cache_len == second.current_cache_len
    assert first._all_token_ids == second._all_token_ids
    assert first._global_token_id_log == second._global_token_id_log
    if first.token_tracker:
        assert first.token_tracker.__dict__ == second.token_tracker.__dict__


@pytest.mark.parametrize("method", runtime.METHODS)
@pytest.mark.parametrize("ratio", [.2, .5])
def test_tiny_qwen_adapter_and_monitor_preserve_legacy_generation(tiny_models, method, ratio):
    plain, observed = tiny_models(method, ratio), tiny_models(method, ratio)
    generation = copy.deepcopy(observed.model.generation_config.to_dict())
    forward_calls = []
    hook = plain.model.register_forward_hook(lambda *a: forward_calls.append(1))
    watch = runtime.QAErrorWatch(io.StringIO())
    try:
        with contextlib.redirect_stdout(watch), contextlib.redirect_stderr(watch):
            with runtime.LayerDeviceAdapter(observed) as adapter, runtime.attach_monitor(observed) as monitor:
                assert plain.generate_first("question", max_new_tokens=8)[0] == (
                    observed.generate_first("question", max_new_tokens=8)[0])
                assert_same_state(plain, observed)
                for step, text in enumerate(("observation " * 5, "second observation " * 3), start=2):
                    if plain.token_tracker:
                        plain.token_tracker.set_current_step(step)
                        observed.token_tracker.set_current_step(step)
                    assert plain.generate_incremental(text, max_new_tokens=6) == (
                        observed.generate_incremental(text, max_new_tokens=6))
                    assert_same_state(plain, observed)
                snapshot = monitor.snapshot()
                metrics = runtime.compact_metrics(snapshot)
                assert not adapter.errors and not watch.errors
                expected_bytes = sum(t.numel() * t.element_size()
                                     for pair in runtime.cache_tensor_pairs(plain.past_key_values) for t in pair)
                assert metrics["final"]["total_bytes"] == expected_bytes
                assert metrics["final"]["protected_tokens"] == 6
                assert metrics["final"]["total_bytes"] == (
                    metrics["final"]["protected_bytes"] + metrics["final"]["trajectory_bytes"])
                assert metrics["counts"]["model_forwards"] == len(forward_calls)
                assert metrics["generation_blocks"] == 3
                assert snapshot["timing_scope"]["llm_calls_cuda_synchronized"] is False
                assert len(runtime.cache_layout(observed)) == 3
                if method != "fullkv":
                    assert snapshot["counts"]["pruning_calls"] > 0
                    assert metrics["peak_preprune"] is not None
                else:
                    assert metrics["step_preprune_peaks"] == [None] * 3
                    assert metrics["final"]["tokens"] < observed.current_cache_len
                assert observed.model.generation_config.to_dict() == generation
    finally:
        hook.remove()
    if observed.kv_manager:
        assert "_get_kv" not in observed.kv_manager.pruning_strategy.__dict__


@pytest.mark.parametrize("message,stderr", [
    ("[ERROR] generated_kv is invalid", False),
    ("Scoring forward failed: bad", False),
    ("Error in generate_incremental at step 2", False),
    ("Token tracking failed during H2O prune", False),
    ("Traceback (most recent call last):", True),
    ("generate() returned no past_key_values, fallback forward.", False),
])
def test_swallowed_errors_cannot_be_scored_as_wrong_answers(message, stderr):
    import sys

    def episode(*args, **kwargs):
        print(message, file=sys.stderr if stderr else sys.stdout)
        return "", [], [], {}
    fake_base = SimpleNamespace(_run_react_kv_episode=episode,
                                exact_match=lambda *a: pytest.fail("Scored a failed episode"))
    with pytest.raises(RuntimeError, match="suppressed error"):
        runtime.observed_episode(fake_base, rows(1)[0], None, None, "fullkv", None,
                                 SimpleNamespace(errors=[]), devices=())


def test_real_byte_metrics_ignore_bookkeeping_and_fail_on_instrumentation_error(tiny_models):
    llm = tiny_models("fullkv")
    with runtime.attach_monitor(llm) as monitor:
        llm.generate_first("question", max_new_tokens=5)
        llm.current_cache_len = 999999
        snapshot = monitor.snapshot()
        assert snapshot["kv"]["current"]["tokens"] == 10
        assert snapshot["kv"]["current"]["bookkeeping_tokens"] == 999999
        assert runtime.compact_metrics(snapshot)["final"]["tokens"] == 10
        snapshot["instrumentation_errors"].append("synthetic")
        with pytest.raises(RuntimeError, match="instrumentation failed"):
            runtime.compact_metrics(snapshot)

@pytest.mark.parametrize("method", ["fullkv", *runtime.STEP_METHODS])
def test_original_first_step_finish_and_stop_kwargs_are_preserved(tiny_models, monkeypatch, method):
    llm = tiny_models(method)
    original = llm.model.generate
    observed_kwargs = []

    def short_cpu_generation(*args, **kwargs):
        observed_kwargs.append(dict(kwargs))
        # The stub shortens only this test's expensive 256-token generation.
        return original(*args, **{**kwargs, "max_new_tokens": 6})

    monkeypatch.setattr(llm.model, "generate", short_cpu_generation)
    monkeypatch.setattr(llm.tokenizer, "decode", lambda *a, **k: "Reason\nAction 1: Finish[Answer 0]")
    calls = []
    first = llm.generate_first

    def capture(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return first(prompt, **kwargs)

    monkeypatch.setattr(llm, "generate_first", capture)
    with runtime.LayerDeviceAdapter(llm) as adapter, runtime.attach_monitor(llm) as monitor:
        row, audit = runtime.observed_episode(base, rows(1)[0], llm, None, method, monitor, adapter, devices=())
    assert row["em"] is True and row["f1"] == 1
    assert row["num_steps"] == 0 and row["kv_metrics"]["generation_blocks"] == 1
    assert calls[0][0] == base.REACT_KV_INITIAL_PROMPT.format(
        examples=base.REACT_EXAMPLES, question=rows(1)[0]["question"]) + "Thought 1:"
    assert calls[0][1] == {"max_new_tokens": 256, "stop_strings": ["\nObservation", "\nQuestion:"]}
    assert observed_kwargs[0]["max_new_tokens"] == 256
    assert observed_kwargs[0]["do_sample"] is False
    assert "temperature" not in observed_kwargs[0] and "repetition_penalty" not in observed_kwargs[0]
    assert audit["diagnostics"]["active_cache_layout"][0]["key_device"] == "cpu"
    if method in runtime.STEP_METHODS:
        protocol = runtime.method_protocol(method)
        assert audit["legacy_adapter_audit"]["scope"] == protocol["scope"]
        runner._validate_step_control_audit(
            audit, row, {"method": method, "method_protocol": protocol, "kv_config": llm.kv_config})


def test_stop_crop_with_adapter_matches_unmodified_tiny_qwen(tiny_models):
    preview = tiny_models("fullkv").generate_first("question", max_new_tokens=8)[0]
    stop = preview[3:5]
    plain, observed = tiny_models("fullkv"), tiny_models("fullkv")
    with runtime.LayerDeviceAdapter(observed), runtime.attach_monitor(observed) as monitor:
        expected = plain.generate_first("question", max_new_tokens=8, stop_strings=[stop])
        actual = observed.generate_first("question", max_new_tokens=8, stop_strings=[stop])
        assert expected[0] == actual[0]
        assert_same_state(plain, observed)
        snapshot = monitor.snapshot()
        assert snapshot["counts"]["crop_calls"] == 1
        assert snapshot["kv"]["current"]["tokens"] < snapshot["kv"]["peak_active"]["tokens"]


def test_cli_has_native_token_methods_and_explicit_step_controls():
    assert set(runner.METHODS) == set(runtime.METHODS) == {
        "fullkv", "stepkv", "h2o", "tova", "h2o_step", "tova_step"}
    conditions = [("fullkv", .2)] + [(method, ratio) for method in runner.METHODS
                                    if method != "fullkv" for ratio in (.2, .5)]
    assert len(conditions) == 11 and len(conditions) * len(runner.DATASETS) == 33
    for method, ratio in conditions:
        args = cli_args("--method", method, "--ratio", str(ratio))
        runner.validate_protocol(args)
        assert args.method == method
    for method in ("h2o", "tova"):
        assert runtime.llm_class_for_method(method) is QwenLLMWithKVCache
        assert runtime.method_protocol(method)["pruning_schedule"] == "native_generation_time"


@pytest.mark.parametrize("method", runtime.STEP_METHODS)
def test_step_adapter_hash_scope_and_inherited_decoder_are_preserved(method):
    adapter = runtime.frozen_step_adapter(method)
    protocol = runtime.method_protocol(method)
    assert protocol["scope"] == adapter.LEGACY_SCOPE
    assert protocol["base_mode"] == runtime.METHODS[method]
    assert protocol["pruning_schedule"] == "frozen_completed_block_control"
    assert protocol["config_override"] == adapter.LEGACY_CONFIG_OVERRIDE
    assert runtime.legacy_config(base, method, .2) == runtime.legacy_config(base, runtime.METHODS[method], .2)
    for path, expected in protocol["adapter_source_hashes"].items():
        assert runner.file_hash(runner.ROOT / path) == expected
    for name in ("_decode", "_decode_token_by_token_with_pruning", "_get_attention_for_scoring", "truncate_cache"):
        assert getattr(adapter.LegacyBaselineLLM, name) is getattr(QwenLLMWithKVCache, name)


@pytest.mark.parametrize("method", runtime.STEP_METHODS)
def test_step_adapter_rejects_checksum_drift_without_modifying_source(monkeypatch, method):
    runtime.frozen_step_adapter(method)  # also test that a cached module cannot bypass revalidation
    target = runtime.ROOT / f"rebuttal/adapters/{method}/models/LegacyBaseline.py"
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda path: read_bytes(path) + (b"\n# drift\n" if path == target else b""))
    with pytest.raises(RuntimeError, match="adapter checksum mismatch"):
        runtime.frozen_step_adapter(method)


@pytest.mark.parametrize("method", runtime.STEP_METHODS)
@pytest.mark.parametrize("ratio", [.2, .5])
def test_step_first_block_matches_original_generation_time_baseline(tiny_models, method, ratio):
    native, step = tiny_models(runtime.METHODS[method], ratio), tiny_models(method, ratio)
    with runtime.LayerDeviceAdapter(step), runtime.attach_monitor(step) as monitor:
        assert native.generate_first("question", max_new_tokens=80)[0] == (
            step.generate_first("question", max_new_tokens=80)[0])
        assert_same_state(native, step)
        assert native.kv_manager.pruning_history == step.kv_manager.pruning_history
        audit = step.legacy_audit()
        assert audit["method"] == method
        assert audit["cache_measurements"] and any(event["pruned"] for event in audit["cache_measurements"])
        assert {event["phase"] for event in audit["cache_measurements"]} == {"initial_completed_block"}
        assert all(event["single_token_mode"] for event in audit["cache_measurements"])
        assert audit["suppressed_prune_calls"] == 0
        assert audit["cache_bytes"] == monitor.snapshot()["kv"]["current"]["total_bytes"]


def set_cache_ratio(llm, ratio):
    llm.kv_config["cache_ratio"] = ratio
    llm.kv_manager.cache_ratio = ratio
    llm.kv_manager.target_cache_ratio = ratio
    llm.kv_manager.keep_ratio = ratio


@pytest.mark.parametrize("method", runtime.STEP_METHODS)
@pytest.mark.parametrize("ratio", [.2, .5])
def test_step_control_defers_only_later_pruning_until_after_original_stop_crop(
        tiny_models, monkeypatch, method, ratio):
    native, step, preview = (tiny_models(runtime.METHODS[method], 1), tiny_models(method, 1),
                             tiny_models(runtime.METHODS[method], 1))
    for llm in (native, step, preview):
        llm.generate_first("question", max_new_tokens=12)
    observation = "observation evidence " * 4
    text = preview.generate_incremental(observation, max_new_tokens=20)
    assert len(text) > 4
    stop = text[3:5]
    set_cache_ratio(step, ratio)
    probe_states = []
    probe = step._get_attention_for_scoring

    def after_crop_probe():
        assert not step._step_defer_pruning
        probe_states.append((step.current_cache_len, step.past_key_values.layers[0].keys.shape[-2]))
        return probe()

    monkeypatch.setattr(step, "_get_attention_for_scoring", after_crop_probe)
    for llm in (native, step):
        monkeypatch.setattr(llm.model, "generate",
                            lambda *a, **k: pytest.fail("Must retain original manual token decoder"))
    with runtime.LayerDeviceAdapter(step), runtime.attach_monitor(step) as monitor:
        expected = native.generate_incremental(observation, max_new_tokens=20, stop_strings=[stop])
        actual = step.generate_incremental(observation, max_new_tokens=20, stop_strings=[stop])
        assert actual == expected and len(actual) < len(text)
        assert probe_states[0] == (native.current_cache_len, native.past_key_values.layers[0].keys.shape[-2])
        set_cache_ratio(native, ratio)
        target = native._compute_h2o_budget()
        attentions = native._resolve_scoring_attentions()
        native._do_pruning(single_token_mode=False, attentions_override=attentions)
        for _ in range(max(0, native.current_cache_len - target + 8)):
            if native.current_cache_len <= target:
                break
            before = native.current_cache_len
            native._do_pruning(single_token_mode=True, attentions_override=attentions)
            if native.current_cache_len >= before:
                break
        assert_same_state(native, step)
        audit = step.legacy_audit()
        assert audit["suppressed_prune_calls"] > 0 and audit["suppressed_prefill_probes"] == 1
        assert {event["phase"] for event in audit["cache_measurements"]} == {"incremental_completed_block"}
        assert audit["cache_measurements"][0]["single_token_mode"] is False
        assert monitor.snapshot()["counts"]["crop_calls"] == 1


@pytest.mark.parametrize("method", runtime.STEP_METHODS)
def test_step_control_never_prunes_failed_partial_generation(tiny_models, monkeypatch, method):
    llm = tiny_models(method, 1)
    llm.generate_first("question", max_new_tokens=12)
    set_cache_ratio(llm, .2)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic decode failure")
    monkeypatch.setattr(llm, "_decode", fail)
    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        llm.generate_incremental("observation " * 8, max_new_tokens=16)
    assert llm._step_defer_pruning is False
    assert llm.legacy_audit()["cache_measurements"] == []


def store_inputs(count=2):
    selected = list(enumerate(rows(count)))
    fingerprint = {"sample_ids": [sample["id"] for _, sample in selected], "seed": 233,
                   "source_sha256": "source-v1", "model_revision": runner.MODEL_REVISION}
    quantities = dict(tokens=10, total_bytes=100, protected_tokens=6, protected_bytes=60,
                      trajectory_tokens=4, trajectory_bytes=40)
    metrics = {"boundary_means": quantities, "peak_active": quantities,
               "peak_trajectory": quantities, "peak_preprune": None, "allocator": {}}
    results = [dict(id=sample["id"], index=index, question=sample["question"], gold_answer=sample["answer"],
                    predicted_answer=sample["answer"] if index == 0 else "Other",
                    em=index == 0, f1=1 if index == 0 else .5, kv_metrics=copy.deepcopy(metrics))
               for index, sample in selected]
    return selected, fingerprint, results


def new_store(output_dir, count=2):
    selected, fingerprint, results = store_inputs(count)
    store = runner.ResultStore(output_dir, fingerprint, selected)
    store.set_placement({"hf_device_map": {"layer0": 0, "layer1": 1}})
    return store, selected, fingerprint, results


def test_atomic_resume_audits_and_legacy_result_schema(output_dir):
    store, selected, fingerprint, results = new_store(output_dir)
    store.commit(results[0], {"diagnostics": {"real_tensors": True}})
    resumed = runner.ResultStore(output_dir, fingerprint, selected)
    assert len(resumed.rows) == 1
    resumed.commit(results[1], {"diagnostics": {"real_tensors": True}})
    final = resumed.finalize()
    assert final["summary"]["exact_match"] == 50 and final["summary"]["f1_score"] == 75
    assert [row["id"] for row in final["results"]] == fingerprint["sample_ids"]
    assert runner.ResultStore(output_dir, fingerprint, selected).rows == final["results"]
    audit_index = runner.read_json(output_dir / "kv_audit.json")
    assert audit_index["sample_ids"] == fingerprint["sample_ids"]
    assert not list(output_dir.rglob("*.pending"))


@pytest.mark.parametrize("device_limits", [
    {0: "40GiB"}, {0: "40GiB", 1: "40GiB"}, {0: "40GiB", "1": "40GiB"},
])
def test_manifest_normalizes_numeric_device_keys_on_create_and_resume(output_dir, device_limits):
    selected, fingerprint, results = store_inputs()
    fingerprint["placement_policy"] = {"max_memory": device_limits}
    expected = json.loads(json.dumps(fingerprint))
    store = runner.ResultStore(output_dir, fingerprint, selected)
    assert store.fingerprint == expected
    assert store.fingerprint_hash == runner.digest(expected)
    assert runner.read_json(output_dir / "manifest.json")["fingerprint"] == expected
    assert 0 in fingerprint["placement_policy"]["max_memory"]  # do not mutate the caller
    placement = {"hf_device_map": {"first": 0, "second": 1}, "max_memory": device_limits}
    store.set_placement(placement)
    store.commit(results[0], {})
    resumed = runner.ResultStore(output_dir, fingerprint, selected)
    resumed.set_placement(placement)
    resumed.set_placement(json.loads(json.dumps(placement)))
    assert resumed.rows == store.rows
    changed = copy.deepcopy(expected)
    changed["placement_policy"]["max_memory"]["0"] = "39GiB"
    with pytest.raises(RuntimeError, match="Changed protocol"):
        runner.ResultStore(output_dir, changed, selected)


@pytest.mark.parametrize("change", ["source_sha256", "model_revision", "sample_ids", "seed"])
def test_resume_fingerprint_guard(output_dir, change):
    store, selected, fingerprint, results = new_store(output_dir)
    store.commit(results[0], {})
    fingerprint = {**fingerprint, change: "different"}
    with pytest.raises(RuntimeError, match="Changed protocol"):
        runner.ResultStore(output_dir, fingerprint, selected)


@pytest.mark.parametrize("problem", ["duplicate", "out-of-order", "foreign", "corrupt", "missing-audit"])
def test_resume_rejects_bad_ids_or_corruption(output_dir, problem):
    store, selected, fingerprint, results = new_store(output_dir)
    for row in results:
        store.commit(row, {})
    checkpoint = runner.read_json(output_dir / "checkpoint.json")
    if problem == "duplicate":
        checkpoint["results"][1]["id"] = checkpoint["results"][0]["id"]
    elif problem == "out-of-order":
        checkpoint["results"].reverse()
    elif problem == "foreign":
        checkpoint["results"][0]["id"] = "foreign"
    elif problem == "missing-audit":
        (output_dir / checkpoint["results"][0]["kv_audit"]["path"]).unlink()
    else:
        (output_dir / "checkpoint.json").write_text("{")
        with pytest.raises(json.JSONDecodeError):
            runner.ResultStore(output_dir, fingerprint, selected)
        return
    checkpoint["rows_sha256"] = runner.digest(checkpoint["results"])
    runner.atomic_json(output_dir / "checkpoint.json", checkpoint)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner.ResultStore(output_dir, fingerprint, selected)


def test_audit_before_checkpoint_crash_does_not_commit_or_duplicate_question(output_dir, monkeypatch):
    store, selected, fingerprint, results = new_store(output_dir)
    atomic = runner.atomic_json

    def crash(path, value):
        if Path(path).name == "checkpoint.json":
            raise OSError("simulated checkpoint write failure")
        atomic(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "atomic_json", crash)
        with pytest.raises(OSError):
            store.commit(results[0], {})
    assert store.rows == [] and not (output_dir / "checkpoint.json").exists()
    resumed = runner.ResultStore(output_dir, fingerprint, selected)
    assert resumed.rows == []
    resumed.commit(results[0], {})
    assert len(resumed.rows) == 1
    assert len(list(output_dir.glob("questions/*/kv_audit.json"))) == 1


def test_orphan_results_placement_changes_and_concurrent_writers_are_rejected(output_dir):
    (output_dir / "results.json").write_text("{}")
    selected, fingerprint, _ = store_inputs()
    with pytest.raises(RuntimeError, match="lacks a manifest"):
        runner.ResultStore(output_dir, fingerprint, selected)
    (output_dir / "results.json").unlink()
    store, _, _, _ = new_store(output_dir)
    with pytest.raises(RuntimeError, match="placement changed"):
        store.set_placement({"hf_device_map": {"layer0": 1, "layer1": 0}})
    with runner.output_lock(output_dir), pytest.raises(RuntimeError, match="another runner"):
        with runner.output_lock(output_dir):
            pytest.fail("Acquired an already held run lock")


def test_blocked_state_is_persistent_and_no_wrong_answer_is_created(output_dir, monkeypatch):
    def blocked(args):
        raise RuntimeError("synthetic missing pinned asset")
    monkeypatch.setattr(runner, "run", blocked)
    with pytest.raises(RuntimeError, match="synthetic missing"):
        runner.main(["--dataset", "musique", "--method", "tova", "--purpose", "smoke",
                     "--assets", ".", "--output", str(output_dir)])
    status = runner.read_json(output_dir / "status.json")
    assert status["status"] == "blocked"
    assert "missing pinned asset" in status["reason"]
    assert not (output_dir / "checkpoint.json").exists()
    assert not (output_dir / "results.json").exists()


@pytest.fixture
def completed_controller_run(output_dir, tiny_models):
    def make(evict=False, method=None):
        actual_method = method if method in runtime.STEP_METHODS else ("tova" if evict else "fullkv")
        llm = tiny_models(actual_method)
        llm.model.bfloat16()
        with contextlib.redirect_stdout(io.StringIO()):
            with runtime.LayerDeviceAdapter(llm), runtime.attach_monitor(llm) as monitor:
                llm.generate_first("question", max_new_tokens=8)
                if evict:
                    llm.generate_incremental("observation " * 5, max_new_tokens=6)
                snapshot = monitor.snapshot()
                snapshot["active_cache_layout"] = runtime.cache_layout(llm)
                control_audit = llm.legacy_audit() if actual_method in runtime.STEP_METHODS else None
        # Real CPU BF16 tensor measurements; only placement/allocator metadata
        # is synthetic so the production controller contract needs no GPU.
        devices = ["cuda:0", "cuda:1", "cuda:1"]
        for layer, device in zip(snapshot["active_cache_layout"], devices):
            layer["key_device"] = layer["value_device"] = device
        allocator = {"allocated_bytes": 100, "reserved_bytes": 200,
                     "peak_allocated_bytes": 150, "peak_reserved_bytes": 200}
        snapshot["qa32_allocator"] = {
            boundary: {device: dict(allocator) for device in ("0", "1")}
            for boundary in ("before_episode", "after_episode")}
        metrics = runtime.compact_metrics(snapshot)
        selected, fingerprint, results = store_inputs()
        fingerprint.update(samples=2, method=method or ("tova" if evict else "fullkv"),
                           sample_indices=[index for index, _ in selected],
                           placement_policy={"max_memory": {0: "40GiB", 1: "40GiB"}},
                           selected_content_sha256=runner.digest(selected))
        fingerprint["method_protocol"] = runtime.method_protocol(fingerprint["method"])
        fingerprint["kv_config"] = runtime.legacy_config(base, fingerprint["method"], .5)
        store = runner.ResultStore(output_dir, fingerprint, selected)
        store.set_placement({"hf_device_map": {"first": 0, "remaining": 1}, "layer_devices": devices})
        for row in results:
            row.update(predicted_answer=row["gold_answer"], em=True, f1=1, kv_metrics=metrics)
            audit = {"diagnostics": snapshot,
                     "legacy_step_timings_bookkeeping_only": [{}] * len(snapshot["steps"])}
            if control_audit is not None:
                audit["legacy_adapter_audit"] = control_audit
            store.commit(row, audit)
        final = store.finalize()
        runner.atomic_json(output_dir / "status.json", {
            "status": "complete", "samples": 2, "exact_match": final["summary"]["exact_match"],
            "f1_score": final["summary"]["f1_score"]})
        return store
    return make


@pytest.mark.parametrize("evict", [False, True])
def test_controller_validates_complete_run_read_only(completed_controller_run, output_dir, monkeypatch, evict):
    store = completed_controller_run(evict)
    before = {str(path): (path.stat().st_mtime_ns, runner.file_hash(path))
              for path in output_dir.rglob("*.json")}
    monkeypatch.setattr(runner, "atomic_json", lambda *a, **k: pytest.fail("Validator wrote an output"))
    report = runner.validate_completed_run(output_dir)
    assert report == {
        "completed_samples": 2, "sample_ids": [row["id"] for row in store.rows],
        "has_actual_eviction": evict, "summary": runner.summarize(store.rows)}
    assert before == {str(path): (path.stat().st_mtime_ns, runner.file_hash(path))
                      for path in output_dir.rglob("*.json")}


@pytest.mark.parametrize("method", runtime.STEP_METHODS)
def test_controller_validates_frozen_step_audit_and_real_evictions(completed_controller_run, output_dir, method):
    store = completed_controller_run(evict=True, method=method)
    report = runner.validate_completed_run(output_dir)
    assert report["completed_samples"] == 2 and report["has_actual_eviction"] is True
    row = store.rows[0]
    audit = runner.read_json(output_dir / row["kv_audit"]["path"])
    audit["legacy_adapter_audit"]["scope"] = "not the original scope"
    with pytest.raises(RuntimeError, match="scope"):
        runner._validate_step_control_audit(audit, row, store.fingerprint)


@pytest.mark.parametrize("problem", [
    "blocked", "missing-final", "duplicate-id", "partial", "checkpoint-checksum",
    "summary", "audit-index", "placement-checksum", "missing-audit",
])
def test_controller_rejects_incomplete_or_inconsistent_runs(completed_controller_run, output_dir, problem):
    store = completed_controller_run()
    if problem == "missing-final":
        (output_dir / "results.json").unlink()
    elif problem == "missing-audit":
        (output_dir / store.rows[0]["kv_audit"]["path"]).unlink()
    else:
        filename = {
            "blocked": "status.json", "duplicate-id": "checkpoint.json", "partial": "checkpoint.json",
            "checkpoint-checksum": "checkpoint.json", "summary": "results.json",
            "audit-index": "kv_audit.json", "placement-checksum": "manifest.json",
        }[problem]
        path = output_dir / filename
        value = runner.read_json(path)
        if problem == "blocked":
            value["status"] = "blocked"
        elif problem == "duplicate-id":
            value["results"][1]["id"] = value["results"][0]["id"]
            value["rows_sha256"] = runner.digest(value["results"])
        elif problem == "partial":
            value["results"].pop()
            value["rows_sha256"] = runner.digest(value["results"])
        elif problem == "checkpoint-checksum":
            value["rows_sha256"] = "wrong"
        elif problem == "summary":
            value["summary"]["exact_match"] = 0
        elif problem == "audit-index":
            value["questions"].pop(store.rows[0]["id"])
        else:
            value["actual_placement"]["layer_devices"][0] = "cpu"
        runner.atomic_json(path, value)
    with pytest.raises(RuntimeError, match="Invalid/incomplete QA32"):
        runner.validate_completed_run(output_dir)


@pytest.mark.parametrize("problem", [
    "instrumentation", "failed-step", "failed-prune", "eviction-count",
    "tensor-bytes", "missing-allocator", "compact-metrics",
])
def test_controller_rejects_invalid_tensor_audits_even_with_updated_hashes(
        completed_controller_run, output_dir, problem):
    store = completed_controller_run(evict=True)
    checkpoint = runner.read_json(output_dir / "checkpoint.json")
    row = checkpoint["results"][0]
    audit_path = output_dir / row["kv_audit"]["path"]
    audit = runner.read_json(audit_path)
    snapshot = audit["diagnostics"]
    if problem == "instrumentation":
        snapshot["instrumentation_errors"].append("synthetic failure")
    elif problem == "failed-step":
        snapshot["steps"][0]["succeeded"] = False
    elif problem == "failed-prune":
        snapshot["pruning_events"][0]["succeeded"] = False
    elif problem == "eviction-count":
        snapshot["pruning_events"][0]["actual_tokens_removed"] += 1
    elif problem == "tensor-bytes":
        snapshot["active_cache_layout"][0]["key_bytes"] += 2
    elif problem == "missing-allocator":
        snapshot["qa32_allocator"]["after_episode"].pop("1")
    else:
        snapshot["counts"]["model_forwards"] += 1
    runner.atomic_json(audit_path, audit)
    row["kv_audit"]["sha256"] = runner.file_hash(audit_path)
    checkpoint["rows_sha256"] = runner.digest(checkpoint["results"])
    runner.atomic_json(output_dir / "checkpoint.json", checkpoint)
    final = runner.read_json(output_dir / "results.json")
    final["results"] = checkpoint["results"]
    runner.atomic_json(output_dir / "results.json", final)
    index = runner.read_json(output_dir / "kv_audit.json")
    index["questions"][row["id"]] = row["kv_audit"]
    runner.atomic_json(output_dir / "kv_audit.json", index)
    with pytest.raises(RuntimeError, match="Invalid/incomplete QA32"):
        runner.validate_completed_run(output_dir)


@pytest.mark.parametrize("problem", [None, "missing", "extra", "invalid", "tensor-bytes"])
def test_recent_single_device_audit_uses_real_placement(
        completed_controller_run, output_dir, problem):
    from scripts.qa32_recent_tensor_audit import validate_tensor_audit

    store = completed_controller_run(evict=True)
    row = runner.read_json(output_dir / "checkpoint.json")["results"][0]
    audit = runner.read_json(output_dir / row["kv_audit"]["path"])
    snapshot = audit["diagnostics"]
    for layer in snapshot["active_cache_layout"]:
        layer["key_device"] = layer["value_device"] = "cuda:0"
    for boundary in ("before_episode", "after_episode"):
        snapshot["qa32_allocator"][boundary].pop("1")
    placement = {**store.manifest["actual_placement"],
                 "layer_devices": ["cuda:0"] * len(snapshot["active_cache_layout"]),
                 "parameter_bytes_per_device": {"0": 100}}
    if problem == "missing":
        snapshot["qa32_allocator"]["after_episode"].pop("0")
    elif problem == "extra":
        snapshot["qa32_allocator"]["after_episode"]["1"] = dict(
            snapshot["qa32_allocator"]["after_episode"]["0"])
    elif problem == "invalid":
        snapshot["qa32_allocator"]["after_episode"]["0"]["allocated_bytes"] = 0
    elif problem == "tensor-bytes":
        snapshot["active_cache_layout"][0]["key_bytes"] += 2
    audit["metrics"] = runtime.compact_metrics(snapshot)
    if problem:
        with pytest.raises(RuntimeError):
            validate_tensor_audit(audit, placement)
    else:
        assert validate_tensor_audit(audit, placement)


def test_controller_rejects_fullkv_eviction_and_wrong_json_root(completed_controller_run, output_dir):
    completed_controller_run(evict=True, method="fullkv")
    with pytest.raises(RuntimeError, match="FullKV output contains actual pruning"):
        runner.validate_completed_run(output_dir)
    runner.atomic_json(output_dir / "results.json", None)
    with pytest.raises(RuntimeError, match="Invalid/incomplete QA32"):
        runner.validate_completed_run(output_dir)
