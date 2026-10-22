"""Llama attention support without modifying historical frozen adapters."""
from contextlib import contextmanager
from types import MethodType

from rebuttal.adapters.intentkv_rule.adapter import IntentKVRuleLLM, OVERRIDES, SCOPE
from scripts.latency_runtime import Method, frozen_method


class LlamaIntentKVRuleLLM(IntentKVRuleLLM):
    """Same QueryMemory rule, capturing the actual Llama post-RoPE queries."""

    @contextmanager
    def _capture_prefill(self, length, span):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        if self.model.config.model_type != "llama" or not 0 <= span[0] < span[1] <= length:
            raise ValueError("Expected Llama with a nonempty actionable span")
        encodings, hooks = {}, []
        try:
            for index, layer in enumerate(self.model.model.layers):
                attention, state = layer.self_attn, {}
                if attention.__class__.__name__ != "LlamaAttention":
                    raise RuntimeError("Unsupported Llama attention implementation")

                def before(module, args, kwargs, *, i=index, state=state):
                    if i in encodings:
                        return
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                    if hidden is None or hidden.shape[:2] != (1, length) or embeddings is None:
                        raise RuntimeError("Missing actual Llama prefill/RoPE inputs")
                    state["embeddings"] = embeddings

                def projected(module, args, output, *, i=index, state=state, attention=attention):
                    if i in encodings:
                        return
                    query = output.view(1, length, -1, attention.head_dim).transpose(1, 2)
                    cos, sin = state.pop("embeddings")
                    query, _ = apply_rotary_pos_emb(query, query, cos, sin)
                    encodings[i] = query[0, :, span[0]:span[1]].detach().float().mean(dim=1)

                hooks.append(attention.register_forward_pre_hook(before, with_kwargs=True))
                hooks.append(attention.q_proj.register_forward_hook(projected))
            yield encodings
            if len(encodings) != len(self.model.model.layers):
                raise RuntimeError("Incomplete Llama post-RoPE query capture")
        finally:
            for hook in hooks:
                hook.remove()


def llama_lazy_class(module):
    """Reuse frozen recurrence/selection; install it on real Llama attention."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    # This module was just loaded into a private process-local namespace by
    # frozen_method. No on-disk frozen sources or model identity are modified.
    module.apply_rotary_pos_emb = apply_rotary_pos_emb
    module.repeat_kv = repeat_kv

    class LlamaLazyEviction(module.LegacyBaselineLLM):
        RUNTIME_VERSION = "legacy-llama-lazyeviction-hooks-v1"

        def _install_legacy_hooks(self):
            self._legacy_ratio = float(self.kv_config.get("cache_ratio", .2))
            self._legacy_window = int(self.kv_config.get("lazy_window", 175))
            self._legacy_alpha = float(self.kv_config.get("lazy_alpha", .0001))
            if (not 0 < self._legacy_ratio <= 1 or self._legacy_window < 1
                    or self._legacy_alpha < 0 or not self.kv_config.get("protect_prompt", True)):
                raise ValueError("Invalid LazyEviction configuration")
            if self.model.config.model_type != "llama":
                raise ValueError("Expected an actual Llama model")
            for layer in self.model.model.layers:
                attention = layer.self_attn
                if attention.__class__.__name__ != "LlamaAttention":
                    raise RuntimeError("Unsupported Llama attention")
                attention._legacy_lazy_owner = self
                attention.forward = MethodType(module._legacy_lazy_attention, attention)
            self._legacy_pre_handle = self.model.register_forward_pre_hook(
                self._legacy_before_forward, with_kwargs=True)
            self._legacy_post_handle = self.model.register_forward_hook(
                self._legacy_after_forward, with_kwargs=True)

    return LlamaLazyEviction


@contextmanager
def method_spec(name, model_key):
    if name in ("h2o_step", "tova_step", "flowkv_style", "lazyeviction"):
        with frozen_method(name) as spec:
            if name == "lazyeviction" and model_key == "llama31_8b":
                scope = spec.scope.replace(
                    "(28 query versus 4 native KV heads, a 7x head expansion on Qwen2.5-7B).",
                    "(32 query versus 8 native KV heads, a 4x expansion on Llama3.1-8B).")
                yield Method(spec.name, spec.mode, llama_lazy_class(spec.modules[-1]),
                             spec.overrides, spec.modules, scope + " Native Llama rotary helpers.")
            else:
                yield spec
    elif name == "intentkv_rule":
        cls = LlamaIntentKVRuleLLM if model_key == "llama31_8b" else IntentKVRuleLLM
        yield Method("tova", name, cls, OVERRIDES, (), SCOPE)
    else:
        raise ValueError(name)
