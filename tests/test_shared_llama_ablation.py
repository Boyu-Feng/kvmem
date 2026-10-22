import os

from scripts import run_shared_llama_ablation as shared


def test_only_known_qwen_accuracy_is_allowed(tmp_path):
    directory = tmp_path / "10"
    directory.mkdir()
    command = [
        "/usr/bin/python3", "-u", str(shared.ROOT / "scripts/run_component_ablation_rebuttal.py"),
        "--execute", "--model-ready", "/assets/7b_ready.json",
    ]
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(part) for part in command))
    assert shared.existing_accuracy(10, tmp_path)
    command[-1] = "/assets/llama31_8b_ready.json"
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(part) for part in command))
    assert not shared.existing_accuracy(10, tmp_path)


def test_latency_and_unknown_python_remain_blocked(tmp_path):
    directory = tmp_path / "10"
    directory.mkdir()
    directory.joinpath("cmdline").write_bytes(b"/usr/bin/python3\0latency.py\0")
    directory.joinpath("environ").write_bytes(b"CUDA_VISIBLE_DEVICES=GPU-0\0")
    assert shared.assigned_blocker(0, "GPU-0", tmp_path)
    assert not shared.assigned_blocker(1, "GPU-1", tmp_path)
