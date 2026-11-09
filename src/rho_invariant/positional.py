"""RoPE rotation matrices.

Provides the RoPE rotation R(Delta) used to form the position-dependent QK
composite G_QK(Delta) = W_Q^T R(Delta) W_K.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def _build_rope_rotation_fast(freqs: "Tensor", delta: int) -> "Tensor":
    """Build the RoPE rotation matrix R(Delta) in R^{d_h x d_h}.

    RoPE pairs dimension k with dimension k + d_half:
        x_rot[k]        = x[k] * cos(w_k Delta) - x[k+d_half] * sin(w_k Delta)
        x_rot[k+d_half] = x[k] * sin(w_k Delta) + x[k+d_half] * cos(w_k Delta)

    Args:
        freqs: RoPE frequencies, shape (d_h/2,)
        delta: relative position offset
    """
    d_half = freqs.shape[0]
    d_h = 2 * d_half
    angles = freqs * delta

    R = torch.zeros(d_h, d_h, device=freqs.device, dtype=freqs.dtype)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    idx = torch.arange(d_half, device=freqs.device)

    R[idx, idx] = cos_a
    R[idx, idx + d_half] = -sin_a
    R[idx + d_half, idx] = sin_a
    R[idx + d_half, idx + d_half] = cos_a

    return R
