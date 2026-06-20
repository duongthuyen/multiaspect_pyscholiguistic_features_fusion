"""Concat+MLP fusion baseline — model definition (architecture only).

Concatenates the three feature branches (semantic 768, affective 34,
handcrafted 26) and classifies through a two-layer MLP. The training loop
lives in scripts/training/concat_train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts import config


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConcatMLP(nn.Module):
    """
    Naive concatenation baseline.

    Concatenates all three feature branches (sem 768, aff 34, hc 26 = 828
    total) and classifies through a two-layer MLP.  No gating, no projection
    per branch.
    """

    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        hidden_dim: int = 256,
        num_labels: int = config.NUM_LABELS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = semantic_dim + affective_dim + handcrafted_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([sem, aff, hc], dim=-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
