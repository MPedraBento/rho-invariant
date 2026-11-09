"""Model weight extraction utilities.

Extracts per-head query/key matrices (W_Q, W_K) and RoPE frequencies from
HuggingFace RoPE decoder models (LLaMA / Qwen2 / Mistral / Gemma).
"""

from __future__ import annotations

import torch

from rho_invariant.invariants import HeadWeights


def _head_dim(model) -> int:
    """Per-head dimension, respecting an explicit config.head_dim if present
    (e.g. Gemma), otherwise hidden_size // num_attention_heads."""
    hd = getattr(model.config, "head_dim", None)
    if hd is not None:
        return int(hd)
    return model.config.hidden_size // model.config.num_attention_heads


def extract_head_weights(model, layer: int, head: int) -> HeadWeights:
    """Extract per-head W_Q, W_K from a RoPE decoder (LLaMA/Qwen2/Mistral/Gemma).

    GQA-aware: the key head paired with query `head` is head * n_kv / n_q.
    """
    model_type = model.config.model_type
    if model_type not in ("llama", "qwen2", "mistral", "gemma", "gemma2"):
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported: llama, qwen2, mistral, gemma, gemma2."
        )

    attn = model.model.layers[layer].self_attn
    n_heads = model.config.num_attention_heads
    d_head = _head_dim(model)
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    kv_head = head * n_kv_heads // n_heads

    s_q, e_q = head * d_head, head * d_head + d_head
    s_kv, e_kv = kv_head * d_head, kv_head * d_head + d_head

    W_Q = attn.q_proj.weight[s_q:e_q, :]   # (d_head, d_model)
    W_K = attn.k_proj.weight[s_kv:e_kv, :]
    return HeadWeights(W_Q=W_Q, W_K=W_K)



def get_rope_frequencies(model) -> torch.Tensor | None:
    """Extract RoPE frequencies (inv_freq) the model ACTUALLY uses.

    RoPE differs across models: base (rope_theta) varies, some use RoPE scaling
    (e.g. Llama-3.1 "rope_type": "llama3" reshapes the frequencies), and some set
    an explicit head_dim (e.g. Gemma). To be correct across all of these we read
    the model's own rotary embedding buffer `inv_freq` when available — this is
    the ground truth that includes any scaling. We fall back to the analytic
    formula only if no such buffer exists.

    Returns the (d_head/2,) frequency tensor, or None for models without RoPE.
    """
    model_type = model.config.model_type
    if model_type == "gpt2":
        return None

    # 1) Ground truth: the model's own rotary embedding inv_freq buffer.
    #    Handles rope_scaling (Llama-3.1), custom base, custom head_dim.
    def _find_inv_freq(m):
        # common locations across HF architectures
        candidates = []
        for path in ("model.rotary_emb", "gpt_neox.rotary_emb"):
            obj = m
            ok = True
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                candidates.append(obj)
        # per-layer rotary (older Llama/Mistral)
        try:
            layer0 = m.model.layers[0].self_attn
            candidates.append(getattr(layer0, "rotary_emb", None))
        except Exception:
            pass
        for c in candidates:
            if c is not None:
                inv = getattr(c, "inv_freq", None)
                if inv is not None:
                    return inv.detach().float().cpu()
        # last resort: scan buffers named inv_freq
        for name, buf in m.named_buffers():
            if name.endswith("inv_freq"):
                return buf.detach().float().cpu()
        return None

    inv_freq = _find_inv_freq(model)
    if inv_freq is not None:
        return inv_freq

    # 2) Fallback: analytic θ_k = base^{-2k/d_head}, respecting explicit head_dim.
    d_head = _head_dim(model)
    rope_base = getattr(model.config, "rope_theta", None)
    if rope_base is None:
        rope_params = getattr(model.config, "rope_parameters", {})
        rope_base = rope_params.get("rope_theta", 10000.0) if isinstance(
            rope_params, dict
        ) else 10000.0
    freqs = 1.0 / (rope_base ** (torch.arange(0, d_head, 2).float() / d_head))
    return freqs
