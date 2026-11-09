"""Basis-invariant quantity for transformer attention heads.

The static QK composite G_QK(0) = W_Q^T W_K is invariant under the full
head-space reparameterisation (W_Q, W_K) -> (M^-T W_Q, M W_K) for any
M in GL(d_h). Requiring invariance of the whole RoPE family
G_QK(Delta) = W_Q^T R(Delta) W_K breaks GL(d_h) down to the per-plane group
(C*)^{d_h/2} = (SO(2) x R_{>0})^{d_h/2}: within each frequency plane a rotation
R and a positive scaling D act as W_Q -> D^-1 R W_Q, W_K -> D R W_K and leave
G_QK(Delta) unchanged. G_QK is the gauge-invariant object; the scalar below is
a function of it and therefore inherits the (C*)^{d_h/2} invariance:

  content_position_ratio  -- rho, positional capacity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class HeadWeights:
    """Query/key weight matrices for a single attention head.

    Attributes:
        W_Q: Query projection, shape (d_h, d_model)
        W_K: Key projection, shape (d_h, d_model)
    """

    W_Q: Tensor
    W_K: Tensor

    @property
    def d_head(self) -> int:
        return self.W_Q.shape[0]

    @property
    def d_model(self) -> int:
        return self.W_Q.shape[1]


def compute_G_QK(head: HeadWeights) -> "Tensor":
    """Compute the QK composite: G_QK = W_Q^T W_K in R^{d_model x d_model}.

    The attention logit is l(i,j) = x_i^T G_QK x_j (before positional encoding).
    """
    return head.W_Q.T @ head.W_K


def content_position_ratio(
    G_QK_static: "Tensor",
    G_QK_deltas: "list[Tensor]",
) -> float:
    """Compute the positional capacity invariant rho.

        rho = (1/|D|) sum_{Delta in D} ||P(Delta)||^2_F / ||G_QK(0)||^2_F

    where P(Delta) = G_QK(Delta) - G_QK(0) is the positional perturbation.

    Interpretation:
        rho ~ 0  : content-dominated (geometry decoupled from position)
        rho >> 0 : positional capacity (geometry couples to RoPE rotations)

    rho measures structural *capacity* to couple to position, not actual usage
    on any particular input.

    Args:
        G_QK_static: G_QK(0) = W_Q^T W_K (the content-only term)
        G_QK_deltas: list of G_QK(Delta) = W_Q^T R(Delta) W_K for Delta in D
    """
    static_norm_sq = torch.sum(G_QK_static**2)
    if static_norm_sq < 1e-12:
        return float("inf")

    total_mod = 0.0
    for G_delta in G_QK_deltas:
        P_delta = G_delta - G_QK_static
        total_mod += torch.sum(P_delta**2).item()

    L = len(G_QK_deltas)
    if L == 0:
        return 0.0
    return total_mod / (L * static_norm_sq.item())
