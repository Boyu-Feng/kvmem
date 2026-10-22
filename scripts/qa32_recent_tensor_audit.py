"""Device-aware tensor audits without changing the frozen two-GPU QA32 validator."""
import math

from scripts.qa32_runtime import QUANTITIES, compact_metrics


def validate_tensor_audit(audit, placement):
    snapshot = audit["diagnostics"]
    if compact_metrics(snapshot) != audit["metrics"]:
        raise RuntimeError("Compact cache metrics disagree with the native tensor audit")
    layer_devices = placement["layer_devices"]
    layout = snapshot["active_cache_layout"]
    if not layout or len(layout) != len(layer_devices):
        raise RuntimeError("Actual KV layer count disagrees with model placement")
    expected_devices = set(placement.get("parameter_bytes_per_device", {}))
    for device in layer_devices:
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise RuntimeError("Actual model placement must contain indexed CUDA devices")
        expected_devices.add(device[5:])
    per_token_bytes = 0
    for index, layer in enumerate(layout):
        if layer["layer"] != index:
            raise RuntimeError("Actual KV layer IDs are not ordered and unique")
        for kind in ("key", "value"):
            shape = layer[f"{kind}_shape"]
            if (len(shape) != 4 or any(type(size) is not int or size <= 0 for size in shape)
                    or layer[f"{kind}_device"] != layer_devices[index]
                    or layer[f"{kind}_dtype"] != "torch.bfloat16"
                    or layer[f"{kind}_bytes"] != math.prod(shape) * 2):
                raise RuntimeError("Invalid per-layer BF16 K/V tensor payload/device")
            if shape[-2] != snapshot["kv"]["current"]["tokens"]:
                raise RuntimeError("Final tensor shape disagrees with measured resident tokens")
            per_token_bytes += layer[f"{kind}_bytes"] // shape[-2]

    def sample_checked(sample):
        if sample is None:
            return
        if any(type(sample[key]) is not int or sample[key] < 0 for key in QUANTITIES):
            raise RuntimeError("Invalid real KV token/byte counts")
        tokens, protected = sample["tokens"], sample["protected_tokens"]
        if sample["layer_tokens"] != [tokens] * len(layout) and not (
                tokens == 0 and sample["layer_tokens"] == []):
            raise RuntimeError("Nonuniform/missing per-layer resident token counts")
        if (protected > tokens or sample["trajectory_tokens"] != tokens - protected
                or sample["total_bytes"] != per_token_bytes * tokens
                or sample["protected_bytes"] != per_token_bytes * protected
                or sample["trajectory_bytes"] != per_token_bytes * (tokens - protected)):
            raise RuntimeError("Real KV byte totals or protected/trajectory partition disagree")

    for name in ("current", "peak_active", "peak_trajectory", "peak_preprune", "peak_post_step", "peak_probe"):
        sample_checked(snapshot["kv"][name])
    for step in snapshot["steps"]:
        for name in ("boundary", "peak_active", "peak_preprune"):
            sample_checked(step[name])
    if len(snapshot["steps"]) != len(audit["legacy_step_timings_bookkeeping_only"]):
        raise RuntimeError("Audit generation blocks disagree with legacy episode boundaries")
    events = snapshot["pruning_events"]
    if snapshot["counts"]["pruning_calls"] != len(events):
        raise RuntimeError("Pruning callback count disagrees with recorded audit events")
    evicted = False
    for event in events:
        if not event["succeeded"] or event["before"] is None or event["after"] is None:
            raise RuntimeError("Failed/unmeasured pruning callback in completed run")
        sample_checked(event["before"])
        sample_checked(event["after"])
        removed = event["before"]["tokens"] - event["after"]["tokens"]
        if removed < 0 or event["actual_tokens_removed"] != removed:
            raise RuntimeError("Actual eviction count disagrees with before/after tensor lengths")
        evicted |= removed > 0
    for boundary in ("before_episode", "after_episode"):
        allocators = snapshot["qa32_allocator"][boundary]
        if set(allocators) != expected_devices:
            raise RuntimeError("Missing per-device CUDA allocation measurements")
        for allocator in allocators.values():
            values = [allocator[key] for key in ("allocated_bytes", "reserved_bytes",
                                                 "peak_allocated_bytes", "peak_reserved_bytes")]
            if (any(type(value) is not int or value < 0 for value in values)
                    or values[0] == 0 or values[1] < values[0] or values[2] < values[0]
                    or values[3] < max(values[1], values[2])):
                raise RuntimeError("Invalid per-device CUDA allocation measurements")
    return evicted
