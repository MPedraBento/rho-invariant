# A Geometric Invariant of Attention: ρ

Code for **ρ**, a basis-invariant scalar computable from attention weights alone
(no forward pass, no data) that measures a head's **positional capacity** — how
much its query–key geometry couples to rotary position. ρ predicts induction
heads from weights alone across four model families and scales.

Accepted as an extended abstract at the Symmetry and Geometry in Neural
Representations Workshop (NeurReps), NeurIPS 2026.

## The invariant

For a RoPE head, the pre-softmax logit splits into a static content term and a
position-dependent perturbation,

```
s_ij = x_i^T G_QK(Δ) x_j = x_i^T G_QK(0) x_j + x_i^T P(Δ) x_j,
G_QK(Δ) = W_Q^T R(Δ) W_K,   P(Δ) = W_Q^T [R(Δ) − I] W_K.
```

ρ is the averaged relative size of the positional perturbation:

```
ρ = (1/|D|) Σ_{Δ∈D}  ||P(Δ)||_F^2 / ||G_QK(0)||_F^2 ,   D = {2^n : n ≤ log2 L}.
```

**Symmetry.** The static composite `G_QK(0) = W_Q^T W_K` is invariant under the
full head-space reparameterization `(W_Q, W_K) → (M^{-T} W_Q, M W_K)`, `M ∈ GL(d_h)`.
Requiring invariance of the whole RoPE family `G_QK(Δ)` breaks `GL(d_h)` down to the
per-plane group `(ℂ*)^{d_h/2} ≅ (SO(2) × ℝ_{>0})^{d_h/2}`: within each frequency
plane a rotation `R` and a positive scaling `D` act as `W_Q → D^{-1} R W_Q`,
`W_K → D R W_K` and leave `G_QK(Δ)` unchanged. ρ, a function of `G_QK`, inherits this
invariance (and is scale-invariant).

ρ measures **capacity**, not usage: a head with ρ = 0 produces logits with no
positional dependence across the sampled offsets, for any input; a high-ρ head
*can* use position but need not on a given input.

## Result

ρ predicts induction heads across four bias-free RoPE models (where `G_QK = W_Q^T W_K`
is exactly the composite the logit uses):

| Model | Params | Spearman r(ρ, induction) | top-10 induction within |
|---|---|---|---|
| SmolLM2-135M  | 135M | −0.74 | bottom 14% |
| TinyLlama-1.1B | 1.1B | −0.71 | bottom 7% |
| SmolLM2-1.7B  | 1.7B | −0.55 | bottom 17% |
| Mistral-7B    | 7B   | −0.59 | bottom 30% |

All p < 10⁻⁴⁰. As a preliminary proof of concept, applying YaRN to the top-50% of
heads by ρ degrades perplexity less than uniform YaRN (Mistral-7B, within native
context).

## Setup

```bash
uv sync
```

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM
from rho_invariant import (
    extract_head_weights, get_rope_frequencies, compute_G_QK,
    content_position_ratio,
)
from rho_invariant.positional import _build_rope_rotation_fast

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B", dtype=torch.bfloat16)
freqs = get_rope_frequencies(model).float()
deltas = [2**n for n in range(10)]

hw = extract_head_weights(model, layer=0, head=0)
hw.W_Q, hw.W_K = hw.W_Q.float(), hw.W_K.float()
G0 = compute_G_QK(hw)
G_deltas = [hw.W_Q.T @ (_build_rope_rotation_fast(freqs, d) @ hw.W_K) for d in deltas]

rho = content_position_ratio(G0, G_deltas)   # positional capacity (>= 0)
```

## Reproducing

```bash
# per-head rho + induction score + Spearman correlation  (GPU auto-detected)
uv run experiments/invariant_taxonomy_pilot.py HuggingFaceTB/SmolLM2-1.7B

# rho-vs-induction figure (reads the pilot JSON, writes results/figures/)
uv run experiments/plot_taxonomy.py results/invariant_taxonomy_pilot_smollm2_1_7b.json

# YaRN top-50%-by-rho vs uniform (proof of concept)
uv run experiments/yarn_hybrid.py mistralai/Mistral-7B-v0.1

# full sweep over the four bias-free models (see RUNNING_ON_GPU.md)
bash run_all.sh
```

## Layout

- `src/rho_invariant/` — ρ (`content_position_ratio`), the QK composite
  (`compute_G_QK`), and weight/RoPE utilities (`extract_head_weights`,
  `get_rope_frequencies`, `_build_rope_rotation_fast`).
- `experiments/` — `invariant_taxonomy_pilot.py` (ρ + induction),
  `yarn_hybrid.py` (YaRN head selection), `plot_taxonomy.py` (figure).
- `paper/` — LaTeX source, figures, references.

## Citation

M. P. Bento. A Geometric Invariant of Attention Identifies Head Specialization
from Weights Alone. Symmetry and Geometry in Neural Representations Workshop
(NeurReps), NeurIPS 2026.
