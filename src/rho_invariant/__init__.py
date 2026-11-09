"""Basis-invariant characterization of transformer attention heads.

Computable from weights alone, without running a forward pass.
"""

from rho_invariant.invariants import (
    HeadWeights,
    compute_G_QK,
    content_position_ratio,
)
from rho_invariant.positional import _build_rope_rotation_fast
from rho_invariant.utils import (
    extract_head_weights,
    get_rope_frequencies,
)

__version__ = "0.1.0"

__all__ = [
    "HeadWeights",
    "compute_G_QK",
    "content_position_ratio",
    "_build_rope_rotation_fast",
    "extract_head_weights",
    "get_rope_frequencies",
]
