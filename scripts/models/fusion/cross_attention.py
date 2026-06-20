"""
CrossAttentionFusion — semantic branch attends to affective + handcrafted.

Architecture
------------
Each branch is projected to a common dimension (256).
The semantic projection serves as the query; the affective and handcrafted
projections are stacked as key-value tokens (2 tokens).
A single multi-head cross-attention layer lets the semantic representation
selectively absorb information from the two auxiliary branches.
The attended output is concatenated with the original semantic projection and
passed to a classifier head.

Q  = semantic_proj          (B, 1, D)
KV = [affective_proj, hc_proj]  (B, 2, D)
attended = MHA(Q, KV, KV)   (B, 1, D)
fused = cat([sem_proj, attended.squeeze(1)])  (B, 2D)
logits = MLP(fused)

Attention weights (B, 1, 2) show how much the semantic representation
draws from affective vs handcrafted — analogous to gate weights in GatedFusion.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from scripts import config
from scripts.models.fusion.blocks import ProjectionBlock


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        projection_dim: int = config.GATED_PROJECTION_DIM,  # 256
        num_heads: int = config.CROSS_ATTN_NUM_HEADS,
        handcrafted_dropout: float = config.GATED_HANDCRAFTED_DROPOUT,
        num_labels: int = config.NUM_LABELS,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.num_labels = num_labels

        # Branch projections
        self.sem_proj  = ProjectionBlock(semantic_dim,     projection_dim, activation="tanh", dropout=0.1)
        self.aff_proj  = ProjectionBlock(affective_dim,    projection_dim, activation="tanh", dropout=0.1)
        self.hc_proj   = ProjectionBlock(handcrafted_dim,  projection_dim, activation="tanh", dropout=handcrafted_dropout)

        # Cross-attention: Q from semantic, K/V from [affective, handcrafted]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Classifier: fused = [sem_proj | attended] → 2*projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim, num_labels),
        )

    def forward(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
        return_attn: bool = False,
    ):
        ps = self.sem_proj(sem)                          # (B, D)
        pa = self.aff_proj(aff)                          # (B, D)
        ph = self.hc_proj(hc)                            # (B, D)

        query = ps.unsqueeze(1)                          # (B, 1, D)
        kv    = torch.stack([pa, ph], dim=1)             # (B, 2, D)

        attended, attn_weights = self.cross_attn(query, kv, kv)  # (B,1,D), (B,1,2)
        attended = attended.squeeze(1)                   # (B, D)

        fused  = torch.cat([ps, attended], dim=-1)       # (B, 2D)
        logits = self.classifier(fused)                  # (B, num_labels)

        if return_attn:
            return logits, attn_weights.squeeze(1)       # weights: (B, 2)  [aff, hc]
        return logits

    def gate_parameters(self):
        """Satisfy the optimizer interface (no dedicated gate params here)."""
        return iter([])
