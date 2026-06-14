"""
Gated Fusion network for multi-branch feature fusion.

Architecture
------------
Three feature branches (semantic 768-d, affective 34-d, handcrafted 26-d)
are each projected to a common dimension.  A gate network reads the raw
concatenated features plus soft class probabilities from an auxiliary head
and emits three branch weights that sum to one via softmax.  The fused
representation is the weighted sum of the projected branches and is passed
to the classifier head.

A gate-diversity penalty (1 / (cv + epsilon)) is added to the training loss
to counteract the tendency of the gate to collapse toward uniform weights,
encouraging differentiated routing across inputs.

Usage
-----
    model = GatedFusion()
    logits = model(sem, aff, hc)
    logits, gates = model(sem, aff, hc, return_gates=True)

Gate weights can be read directly as branch importance fractions (they sum to 1).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import config
from scripts.models.fusion.blocks import ClassifierHead, ProjectionBlock


class GatedFusion(nn.Module):
    """
    Gated fusion with class-conditioned gate and gate-diversity regularization.

    The gate network reads the concatenation of raw semantic, affective, and
    handcrafted features together with soft class probabilities produced by an
    auxiliary head over the semantic branch.  This conditioning makes the gate
    routing sensitive to the predicted class identity of each input.

    A gate-diversity penalty 1/(cv + epsilon) is added to the total loss during
    training, where cv is the coefficient of variation of the summed gate weights
    across the three branches.  This penalises gate uniformity (low cv) and
    encourages the gate to differentiate branch usage across inputs.

    Parameters
    ----------
    semantic_dim : int
        Dimensionality of the semantic (CLS) features.
    affective_dim : int
        Dimensionality of the affective feature vector.
    handcrafted_dim : int
        Dimensionality of the handcrafted (lexical + syntactic + structural) vector.
    projection_dim : int
        Common projection dimension for all three branches.
    gate_hidden_dim : int
        Hidden units in the two-layer gate MLP.
    handcrafted_dropout : float
        Dropout rate applied in the handcrafted projection block.
    num_labels : int
        Number of output classes.
    aux_weight : float
        Weight of the auxiliary semantic classification loss.
    diversity_weight : float
        Weight of the gate-diversity penalty (lambda_div in the thesis).
    """

    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        projection_dim: int = config.GATED_PROJECTION_DIM,
        gate_hidden_dim: int = 128,
        handcrafted_dropout: float = 0.4,
        num_labels: int = config.NUM_LABELS,
        aux_weight: float = 0.3,
        diversity_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self.diversity_weight = diversity_weight
        self.num_labels = num_labels

        # Feature branch projections
        self.semantic_branch = ProjectionBlock(
            semantic_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.affective_branch = ProjectionBlock(
            affective_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.handcrafted_branch = ProjectionBlock(
            handcrafted_dim, projection_dim, activation="tanh", dropout=handcrafted_dropout
        )

        # Auxiliary head: predicts class from semantic features alone.
        # Its softmax output conditions the gate, making routing class-aware.
        self.aux_head = nn.Linear(semantic_dim, num_labels)

        # Gate network: reads raw features + aux class probs -> 3 branch weights
        gate_input_dim = semantic_dim + affective_dim + handcrafted_dim + num_labels
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gate_hidden_dim, 3),
        )

        # Final classifier on the gated fused representation
        self.classifier = ClassifierHead(
            projection_dim, hidden_dim=256, num_labels=num_labels
        )

    def gate_parameters(self) -> Iterator[nn.Parameter]:
        """Gate network parameters (used for dedicated weight decay in optimizer)."""
        return self.gate_network.parameters()

    def _forward_full(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass returning logits, gates, and auxiliary logits.

        Returns
        -------
        logits : Tensor of shape (batch, num_labels)
        gates : Tensor of shape (batch, 3) -- branch weights summing to 1
        aux_logits : Tensor of shape (batch, num_labels) -- for auxiliary loss
        """
        sem_proj = self.semantic_branch(sem)
        aff_proj = self.affective_branch(aff)
        hc_proj = self.handcrafted_branch(hc)

        aux_logits = self.aux_head(sem)
        aux_probs = F.softmax(aux_logits, dim=-1)

        gate_input = torch.cat([sem, aff, hc, aux_probs], dim=-1)
        gates = F.softmax(self.gate_network(gate_input), dim=-1)

        fused = (
            gates[:, 0:1] * sem_proj
            + gates[:, 1:2] * aff_proj
            + gates[:, 2:3] * hc_proj
        )
        logits = self.classifier(fused)
        return logits, gates, aux_logits

    def forward(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits, gates, _ = self._forward_full(sem, aff, hc)
        return (logits, gates) if return_gates else logits

    def training_step(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the full training loss:
            L = L_main + aux_weight * L_aux + diversity_weight * L_div

        L_div = 1 / (cv + epsilon) penalises low gate coefficient of variation,
        i.e. penalises near-uniform gate weights across the three branches.

        Returns
        -------
        (total_loss, logits, gates)
        """
        logits, gates, aux_logits = self._forward_full(sem, aff, hc)
        main_loss = criterion(logits, labels)
        aux_loss = criterion(aux_logits, labels)

        # Gate-diversity penalty: high when gates are uniform, low when differentiated.
        importance = gates.sum(dim=0)                           # (3,)
        cv = importance.std() / (importance.mean() + 1e-8)
        diversity_loss = 1.0 / (cv + 1e-8)

        total = main_loss + self.aux_weight * aux_loss + self.diversity_weight * diversity_loss
        return total, logits, gates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_GATED_VARIANTS = ("gated_fusion",)

_VARIANTS: dict[str, type[nn.Module]] = {
    "gated_fusion": GatedFusion,
}


def build_gated_model(variant: str, cfg: dict | None = None) -> nn.Module:
    """Construct the GatedFusion model.

    Parameters
    ----------
    variant : str
        Must be ``"gated_fusion"``.
    cfg : dict, optional
        Hyperparameter overrides (projection_dim, gate_hidden_dim, etc.).

    Returns
    -------
    GatedFusion instance.
    """
    cfg = cfg or {}
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown gated variant {variant!r}. Only 'gated_fusion' is supported."
        )
    return GatedFusion(
        semantic_dim=config.SEMANTIC_DIM,
        affective_dim=config.AFFECTIVE_DIM,
        handcrafted_dim=config.HANDCRAFTED_DIM,
        projection_dim=int(cfg.get("projection_dim", config.GATED_PROJECTION_DIM)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 128)),
        handcrafted_dropout=float(cfg.get("handcrafted_dropout", 0.4)),
        num_labels=config.NUM_LABELS,
        aux_weight=float(cfg.get("aux_weight", 0.3)),
        diversity_weight=float(cfg.get("diversity_weight", config.GATED_DIVERSITY_WEIGHT)),
    )
