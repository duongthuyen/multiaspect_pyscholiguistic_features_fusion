"""
End-to-end fusion model: trainable RoBERTa backbone + late-concat fusion head.

Unlike the two-stage pipeline (frozen embeddings → fusion MLP), this model
trains the transformer encoder and fusion layers jointly so the backbone can
learn representations shaped by the combined loss signal.

Optimizer usage — two parameter groups with different LRs:

    optimizer = AdamW([
        {"params": model.backbone_parameters(), "lr": config.E2E_BACKBONE_LR},
        {"params": model.fusion_parameters(),   "lr": config.E2E_FUSION_LR},
    ], weight_decay=config.E2E_WEIGHT_DECAY)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from scripts import config
from scripts.models.fusion.blocks import ClassifierHead, ProjectionBlock


class EndToEndFusionModel(nn.Module):
    """
    Architecture mirrors LateConcatFusion but the semantic branch is a live
    RoBERTa encoder instead of a frozen embedding lookup.

    Inputs:
        input_ids      : (B, seq_len)  — tokenised text
        attention_mask : (B, seq_len)
        affective      : (B, 34)       — pre-extracted, fixed during training
        handcrafted    : (B, 26)       — pre-extracted, StandardScaler-normalised
    """

    def __init__(
        self,
        backbone_name_or_path: str = config.MENTAL_ROBERTA_NAME,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        semantic_proj_dim: int = config.SEMANTIC_PROJECTION_DIM,
        affective_proj_dim: int = config.AFFECTIVE_PROJECTION_DIM,
        handcrafted_proj_dim: int = config.HANDCRAFTED_PROJECTION_DIM,
        num_labels: int = config.NUM_LABELS,
        dropout: float = config.E2E_DROPOUT,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name_or_path)

        self.semantic_proj = ProjectionBlock(
            config.SEMANTIC_DIM, semantic_proj_dim, activation="gelu", dropout=dropout
        )
        self.affective_proj = ProjectionBlock(
            affective_dim, affective_proj_dim, activation="gelu", dropout=dropout
        )
        self.handcrafted_proj = ProjectionBlock(
            handcrafted_dim, handcrafted_proj_dim, activation="gelu", dropout=dropout
        )

        fusion_dim = semantic_proj_dim + affective_proj_dim + handcrafted_proj_dim
        self.classifier = ClassifierHead(
            fusion_dim, hidden_dim=256, num_labels=num_labels
        )

    # ------------------------------------------------------------------
    # Parameter groups — pass these to separate AdamW param groups
    # ------------------------------------------------------------------

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def fusion_parameters(self):
        return (
            list(self.semantic_proj.parameters())
            + list(self.affective_proj.parameters())
            + list(self.handcrafted_proj.parameters())
            + list(self.classifier.parameters())
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        affective: torch.Tensor,
        handcrafted: torch.Tensor,
    ) -> torch.Tensor:
        cls = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]  # CLS token (B, 768)

        fused = torch.cat(
            [
                self.semantic_proj(cls),
                self.affective_proj(affective),
                self.handcrafted_proj(handcrafted),
            ],
            dim=1,
        )
        return self.classifier(fused)
