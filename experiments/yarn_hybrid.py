"""YaRN targeted head-selection.

Applies YaRN RoPE scaling and compares:
  baseline       : no scaling
  yarn_all       : YaRN on all heads (uniform)
  yarn_targeted  : YaRN on the top-50% heads by rho, native on the rest

Reports perplexity at a short and a long context.

Usage:
    uv run experiments/yarn_hybrid.py <hf-model-id>
GPU recommended; eager attention required.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rho_invariant.invariants import compute_G_QK, content_position_ratio
from rho_invariant.positional import _build_rope_rotation_fast
from rho_invariant.utils import extract_head_weights, get_rope_frequencies

# --- Configuration ---
MODEL_NAME = (
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B")
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SCALE_FACTOR = float(os.environ.get("SCALE_FACTOR", "4.0"))
SHORT_CTX = int(os.environ.get("SHORT_CTX", "512"))
LONG_CTX = int(os.environ.get("LONG_CTX", "4096"))
EVAL_STRIDE = 512
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _safe_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def compute_head_rhos(model, freqs) -> dict[tuple[int, int], float]:
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    rho_map = {}
    test_deltas = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    freqs_cpu = freqs.cpu().float()
    print("Computing ρ for all heads...")
    for layer in range(n_layers):
        for head in range(n_heads):
            hw = extract_head_weights(model, layer, head)
            hw.W_Q = hw.W_Q.float().cpu()
            hw.W_K = hw.W_K.float().cpu()
            G_QK_static = compute_G_QK(hw)
            G_QK_deltas = [hw.W_Q.T @ (_build_rope_rotation_fast(freqs_cpu, d) @ hw.W_K) for d in test_deltas]
            rho_map[(layer, head)] = float(content_position_ratio(G_QK_static, G_QK_deltas))
        if (layer + 1) % 4 == 0:
            print(f"  ... layer {layer + 1}/{n_layers}")
    return rho_map


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_yarn_inv_freq(head_dim, rope_theta, scale, device):
    """YaRN: ramp from no-scaling (high freq) to full PI (low freq)."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    n_freq = head_dim // 2
    ramp = torch.linspace(0, 1, n_freq, device=device)
    scaling_factors = 1.0 + (scale - 1.0) * ramp
    return inv_freq / scaling_factors


def reset_model_attention(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_original_forward"):
            attn.forward = attn._original_forward


def apply_method(model, method: str, scaled_heads: set, scale: float):
    """Apply a scaling method. Methods: 'yarn_all', 'targeted_pi', 'yarn_targeted'."""
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    gqa_ratio = n_heads // n_kv_heads
    head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // n_heads)
    rope_theta = getattr(model.config, "rope_theta", 1000000.0)
    device = next(model.parameters()).device

    # Compute the scaled inv_freq depending on method
    native_inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if method == "yarn_all":
        # YaRN on ALL heads
        yarn_inv_freq = get_yarn_inv_freq(head_dim, rope_theta, scale, device)
        attn_temp = 0.1 * math.log(scale) + 1.0
    elif method == "targeted_pi":
        # PI on positional heads only
        pi_inv_freq = native_inv_freq / scale
        attn_temp = 1.0
    elif method == "yarn_targeted":
        # YaRN on positional heads, native on content heads
        yarn_inv_freq = get_yarn_inv_freq(head_dim, rope_theta, scale, device)
        attn_temp = 0.1 * math.log(scale) + 1.0
    else:
        raise ValueError(method)

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward

        def create_forward(l_idx, current_attn, m=method):
            def forward_fn(
                hidden_states, position_embeddings=None,
                attention_mask=None, position_ids=None, **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                q = current_attn.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
                k = current_attn.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
                v = current_attn.v_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

                cos_native, sin_native = position_embeddings

                if position_ids is None:
                    position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
                pos_float = position_ids.float()

                # Compute scaled cos/sin depending on method
                if m == "yarn_all":
                    freqs_s = torch.einsum("b s, d -> b s d", pos_float, yarn_inv_freq)
                    emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
                    cos_method = emb_s.cos().to(hidden_states.dtype)
                    sin_method = emb_s.sin().to(hidden_states.dtype)
                elif m == "targeted_pi":
                    freqs_s = torch.einsum("b s, d -> b s d", pos_float / scale, native_inv_freq)
                    emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
                    cos_method = emb_s.cos().to(hidden_states.dtype)
                    sin_method = emb_s.sin().to(hidden_states.dtype)
                elif m == "yarn_targeted":
                    freqs_s = torch.einsum("b s, d -> b s d", pos_float, yarn_inv_freq)
                    emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
                    cos_method = emb_s.cos().to(hidden_states.dtype)
                    sin_method = emb_s.sin().to(hidden_states.dtype)

                CHUNK = 4
                outputs = []
                for h_start in range(0, n_heads, CHUNK):
                    h_end = min(h_start + CHUNK, n_heads)
                    q_rot_list, k_rot_list, v_list = [], [], []

                    for h in range(h_start, h_end):
                        kv_h = h // gqa_ratio
                        q_h = q[:, h:h+1, :, :]
                        k_h = k[:, kv_h:kv_h+1, :, :]

                        # Decide which cos/sin to use for this head
                        if m == "yarn_all":
                            # All heads get YaRN
                            cos_h = cos_method.unsqueeze(1)
                            sin_h = sin_method.unsqueeze(1)
                        elif m == "targeted_pi":
                            # Only scaled_heads get PI
                            if (l_idx, h) in scaled_heads:
                                cos_h = cos_method.unsqueeze(1)
                                sin_h = sin_method.unsqueeze(1)
                            else:
                                cos_h = cos_native.unsqueeze(1)
                                sin_h = sin_native.unsqueeze(1)
                        elif m == "yarn_targeted":
                            # Only scaled_heads get YaRN, rest stay native
                            if (l_idx, h) in scaled_heads:
                                cos_h = cos_method.unsqueeze(1)
                                sin_h = sin_method.unsqueeze(1)
                            else:
                                cos_h = cos_native.unsqueeze(1)
                                sin_h = sin_native.unsqueeze(1)

                        q_rot_list.append((q_h * cos_h) + (rotate_half(q_h) * sin_h))
                        k_rot_list.append((k_h * cos_h) + (rotate_half(k_h) * sin_h))
                        v_list.append(v[:, kv_h:kv_h+1, :, :])

                    q_rot = torch.cat(q_rot_list, dim=1)
                    k_rot = torch.cat(k_rot_list, dim=1)
                    v_chunk = torch.cat(v_list, dim=1)

                    attn_w = torch.matmul(q_rot, k_rot.transpose(-1, -2)) / math.sqrt(head_dim)
                    if attn_temp != 1.0 and m != "targeted_pi":
                        attn_w = attn_w / attn_temp
                    if attention_mask is not None:
                        attn_w = attn_w + attention_mask
                    attn_w = nn.functional.softmax(attn_w, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
                    outputs.append(torch.matmul(attn_w, v_chunk))
                    del q_rot, k_rot, v_chunk, attn_w, q_rot_list, k_rot_list, v_list

                attn_output = torch.cat(outputs, dim=1).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                del q, k, v, outputs
                return current_attn.o_proj(attn_output), None
            return forward_fn

        attn.forward = create_forward(layer_idx, attn)


def evaluate_perplexity(model, tokenizer, test_text, max_length=512):
    torch.cuda.empty_cache()
    encodings = tokenizer(test_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, EVAL_STRIDE):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.item())
        del input_ids, target_ids, outputs
        torch.cuda.empty_cache()
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return math.exp(sum(nlls) / len(nlls))


def main():
    print("=" * 70)
    print(f"YaRN targeted head-selection: {MODEL_NAME}")
    print(f"Scale: {SCALE_FACTOR}x   device: {DEVICE}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_token = os.environ.get("HF_TOKEN")
    load_kwargs = dict(dtype=DTYPE, device_map=DEVICE, attn_implementation="eager")
    if hf_token:
        load_kwargs["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model.eval()

    freqs = get_rope_frequencies(model)
    rho_map = compute_head_rhos(model, freqs)
    sorted_heads = sorted(rho_map.items(), key=lambda x: x[1])
    top_50 = set(h for h, _ in sorted_heads[len(sorted_heads) // 2:])  # highest rho

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paragraphs = [t for t in dataset["text"] if len(t.strip()) > 100]
    short_text = "\n\n".join(paragraphs[:5])
    long_text = "\n\n".join(paragraphs[:60])

    def run(method, heads, label):
        reset_model_attention(model)
        if method is not None:
            apply_method(model, method, heads, SCALE_FACTOR)
        s = evaluate_perplexity(model, tokenizer, short_text, SHORT_CTX)
        l = evaluate_perplexity(model, tokenizer, long_text, LONG_CTX)
        print(f"  {label:<30} short={s:.3f}  long={l:.3f}")
        return {"short_ppl": s, "long_ppl": l}

    print("\n[1/3] Baseline (no scaling)...")
    base = run(None, set(), "Baseline (S=1)")
    print("[2/3] YaRN (all heads)...")
    yall = run("yarn_all", set(), "YaRN (all heads)")
    print("[3/3] YaRN + Targeted (top-50% by rho)...")
    top = run("yarn_targeted", top_50, "YaRN + Targeted (ours)")

    print("\n" + "=" * 70)
    print(f"  RESULTS (S={SCALE_FACTOR}x)")
    print("=" * 70)
    print(f"  {'Method':<30} | {'Short PPL':<10} | {'Long PPL':<10}")
    print("-" * 70)
    for lab, r in [("Baseline (S=1)", base), ("YaRN (all heads)", yall),
                   ("YaRN + Targeted (ours)", top)]:
        print(f"  {lab:<30} | {r['short_ppl']:<10.3f} | {r['long_ppl']:<10.3f}")
    print("=" * 70)

    results = {
        "model": MODEL_NAME,
        "scale_factor": SCALE_FACTOR,
        "short_ctx": SHORT_CTX,
        "long_ctx": LONG_CTX,
        "baseline": base,
        "yarn_all": yall,
        "yarn_targeted": top,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"yarn_{_safe_name(MODEL_NAME)}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
