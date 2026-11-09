"""Does the weight-only invariant rho predict induction heads?

For every attention head in a RoPE model we compute:
  rho        -- positional capacity, from weights alone (no forward pass)
  induction  -- behavioural induction score, from attention on repeated-random
                sequences [BOS, r, r]: attend from a second-copy token to the
                token that FOLLOWED its first occurrence.

We then report the Spearman correlation between rho and the induction score, and
dump per-head (rho, induction) for the rho-vs-induction figure.

Usage:
    uv run experiments/invariant_taxonomy_pilot.py HuggingFaceTB/SmolLM2-1.7B
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from rho_invariant.invariants import compute_G_QK, content_position_ratio
from rho_invariant.positional import _build_rope_rotation_fast
from rho_invariant.utils import extract_head_weights, get_rope_frequencies

MODEL_NAME = (
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B")
)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype() -> "torch.dtype":
    override = os.environ.get("DTYPE", "").lower()
    if override in ("bf16", "bfloat16"):
        return torch.bfloat16
    if override in ("fp16", "float16", "half"):
        return torch.float16
    if override in ("fp32", "float32"):
        return torch.float32
    if DEVICE == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


DTYPE = _select_dtype()
SEQ_HALF_LEN = 50
N_RANDOM_SEQS = 5
TEST_DELTAS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _safe_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def compute_rho_all_heads(model, freqs) -> dict:
    """rho per head, weights only."""
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    dev = next(model.parameters()).device
    freqs_dev = freqs.to(dev).float()

    _hw0 = extract_head_weights(model, 0, 0)
    if 2 * freqs_dev.shape[0] != _hw0.W_K.shape[0]:
        raise ValueError(
            f"RoPE freq length {freqs_dev.shape[0]} (x2) != head_dim "
            f"{_hw0.W_K.shape[0]} for {model.config.model_type}."
        )

    rho = {}
    print(f"Computing rho (weights only) on {dev}...")
    for layer in range(n_layers):
        for head in range(n_heads):
            hw = extract_head_weights(model, layer, head)
            hw.W_Q = hw.W_Q.float().to(dev)
            hw.W_K = hw.W_K.float().to(dev)
            G0 = compute_G_QK(hw)
            G_deltas = [
                hw.W_Q.T @ (_build_rope_rotation_fast(freqs_dev, d) @ hw.W_K)
                for d in TEST_DELTAS
            ]
            rho[(layer, head)] = float(content_position_ratio(G0, G_deltas))
        if (layer + 1) % 4 == 0:
            print(f"  ... layer {layer + 1}/{n_layers}")
    return rho


def compute_induction_scores(model, tokenizer) -> dict:
    """Induction score per head: attn from second-copy token to one-after its
    first occurrence, on repeated-random sequences [BOS, r, r]."""
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    gqa_ratio = n_heads // n_kv_heads
    vocab_size = model.config.vocab_size

    acc = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}

    def induction_per_head(attn_layer, total_len):
        n_reported = attn_layer.shape[0]
        ks = torch.arange(SEQ_HALF_LEN - 1)
        q_idx = 1 + SEQ_HALF_LEN + ks
        valid = q_idx < total_len
        q_idx = q_idx[valid]
        ind_t = (2 + ks)[valid]  # one-after the first occurrence
        ind = attn_layer[:, q_idx, ind_t].mean(dim=1)  # (n_reported,)
        if n_reported != n_heads:
            ind = ind[torch.arange(n_heads) // gqa_ratio]
        return ind.numpy()

    print(f"Computing induction scores ({N_RANDOM_SEQS} sequences)...")
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    for seq_idx in range(N_RANDOM_SEQS):
        rng = torch.Generator()
        rng.manual_seed(42 + seq_idx)
        r = torch.randint(100, vocab_size - 100, (SEQ_HALF_LEN,), generator=rng)
        input_ids = torch.cat([torch.tensor([bos]), r, r]).unsqueeze(0).to(DEVICE)
        total_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for layer_idx, attn_weights in enumerate(outputs.attentions):
            ind = induction_per_head(attn_weights[0].float().cpu(), total_len)
            for h in range(n_heads):
                acc[(layer_idx, h)].append(float(ind[h]))
        del outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  ... sequence {seq_idx + 1}/{N_RANDOM_SEQS}")

    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    global DEVICE
    print("=" * 70)
    print(f"  rho vs induction — {MODEL_NAME}   device: {DEVICE}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_token = os.environ.get("HF_TOKEN")
    load_kwargs = dict(dtype=DTYPE, attn_implementation="eager")
    if hf_token:
        load_kwargs["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs).to(DEVICE)
    model.eval()

    freqs = get_rope_frequencies(model)
    if freqs is None:
        raise SystemExit("Model has no RoPE frequencies; this pilot requires a RoPE model.")

    rho = compute_rho_all_heads(model, freqs)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    try:
        ind = compute_induction_scores(model, tokenizer)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
            raise
        print("\n  [OOM] retrying induction scoring on CPU...")
        model = model.to("cpu")
        DEVICE = "cpu"
        ind = compute_induction_scores(model, tokenizer)

    keys = sorted(rho.keys())
    rho_arr = np.array([rho[k] for k in keys])
    ind_arr = np.array([ind[k] for k in keys])

    r, p = spearmanr(rho_arr, ind_arr)
    # top-10 induction heads: where do they sit in the rho distribution?
    top10 = np.argsort(-ind_arr)[:10]
    rho_rank = rho_arr.argsort().argsort() / (len(rho_arr) - 1)  # percentile in [0,1]
    top10_max_pct = float(rho_rank[top10].max())

    print("\n" + "=" * 70)
    print(f"  Spearman(rho, induction) = {r:+.3f}  (p = {p:.1e})")
    print(f"  top-10 induction heads all within bottom {100*top10_max_pct:.0f}% of rho")
    print("=" * 70)

    report = {
        "model": MODEL_NAME,
        "n_heads": len(keys),
        "spearman_rho_induction": {"r": float(r), "p": float(p)},
        "top10_induction_max_rho_percentile": top10_max_pct,
        "heads": [
            {"layer": int(k[0]), "head": int(k[1]), "rho": rho[k], "induction": ind[k]}
            for k in keys
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"invariant_taxonomy_pilot_{_safe_name(MODEL_NAME)}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
