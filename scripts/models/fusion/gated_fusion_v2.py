"""
Improved gated fusion variants — four targeted fixes for gate collapse.

Diagnostic: original GatedFusion learned a fixed global 39/25/36
(semantic/affective/handcrafted) split regardless of input content.
Root cause: each branch gated only itself, giving zero cross-branch
context. The gate could only learn "how useful am I on average" —
not "how useful am I for THIS input compared to my peers."

Shared architectural decisions (all four variants):
  Gate INPUT  — raw concatenated features: semantic(768) + affective(34)
                + handcrafted(26) = 828 dims. Using raw inputs (not
                projected) gives the gate the full information before any
                lossy projection occurs.
  Gate MLP    — Linear(gate_input_dim, 128) → ReLU → Dropout(0.3)
                → Linear(128, 3) → softmax. Two layers as spec'd.
  Branches    — same projection as original: Linear → LayerNorm → tanh
                → Dropout. Handcrafted dropout is 0.4 (training param,
                higher than semantic/affective to prevent the small 26-dim
                branch from overfitting via memorisation).

Variant A  content_gate   Gate MLP over raw features → 3 scalar weights.
Variant B  class_aware    A + soft class probs (from aux head on semantic raw)
                          added to gate input. Loss adds aux CE × aux_weight.
Variant C  load_balance   B + CV² load-balance loss on per-batch branch importance.
Variant D  per_class_gate 6 gate heads (one per class), mixed by aux probs.
                          Effective gates = sum_c(p_c × gate_c). Loss adds aux CE.

Public interface (identical across all variants):
    forward(sem, aff, hc, return_gates=False) → logits | (logits, gates)
    training_step(sem, aff, hc, labels, criterion) → (total_loss, logits, gates)
    gate_parameters() → Iterator[Parameter]   # used for optimizer weight-decay group
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import config
from scripts.models.fusion.blocks import ClassifierHead, ProjectionBlock

# Raw feature dimension going into every gate MLP.
_GATE_INPUT_DIM = config.SEMANTIC_DIM + config.AFFECTIVE_DIM + config.HANDCRAFTED_DIM  # 828


# ---------------------------------------------------------------------------
# Variant A — content_gate
# ---------------------------------------------------------------------------

class ContentGatedFusion(nn.Module):
    """
    Gate driven by raw concatenated features — cross-branch baseline fix.

    The original gate computed a 256-dim sigmoid over each branch's own
    projection then softmaxed the three results. That gate could only learn
    a fixed average usefulness per branch because it never compared branches
    against each other. Here a 2-layer MLP sees all raw features at once and
    outputs 3 scalar branch weights per sample.
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
    ):
        super().__init__()
        gate_input_dim = semantic_dim + affective_dim + handcrafted_dim

        self.semantic_branch = ProjectionBlock(
            semantic_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.affective_branch = ProjectionBlock(
            affective_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.handcrafted_branch = ProjectionBlock(
            handcrafted_dim, projection_dim, activation="tanh", dropout=handcrafted_dropout
        )
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.classifier = ClassifierHead(
            projection_dim, hidden_dim=256, num_labels=num_labels
        )

    def gate_parameters(self):
        """Yield gate-network parameters for a dedicated weight-decay group."""
        return self.gate_network.parameters()

    def _project(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.semantic_branch(sem),
            self.affective_branch(aff),
            self.handcrafted_branch(hc),
        )

    def _gate_and_fuse(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
        sem_proj: torch.Tensor,
        aff_proj: torch.Tensor,
        hc_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([sem, aff, hc], dim=-1)       # (B, 828)
        gates = F.softmax(self.gate_network(gate_input), dim=-1)  # (B, 3)
        fused = (
            gates[:, 0:1] * sem_proj
            + gates[:, 1:2] * aff_proj
            + gates[:, 2:3] * hc_proj
        )
        return fused, gates

    def forward(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sem_proj, aff_proj, hc_proj = self._project(sem, aff, hc)
        fused, gates = self._gate_and_fuse(sem, aff, hc, sem_proj, aff_proj, hc_proj)
        logits = self.classifier(fused)
        return (logits, gates) if return_gates else logits

    def training_step(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, gates = self.forward(sem, aff, hc, return_gates=True)
        loss = criterion(logits, labels)
        return loss, logits, gates


# ---------------------------------------------------------------------------
# Variant B — class_aware
# ---------------------------------------------------------------------------

class ClassAwareGatedFusion(nn.Module):
    """
    Gate conditioned on soft class predictions from the semantic branch.

    Root cause fixed: the gate had no signal about what class the model
    thinks this sample belongs to. Knowing the predicted class should change
    routing — PTSD posts should up-weight handcrafted (past_ratio), Anxiety
    posts should up-weight affective (nervousness).

    A lightweight auxiliary head on the raw semantic CLS (768 → 6) produces
    soft class probabilities. These are appended to the gate input so the gate
    MLP can learn class-conditional routing. The aux head is supervised by an
    auxiliary CE loss (weight=aux_weight, default 0.3) so it produces
    meaningful class signals.
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
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.num_labels = num_labels
        gate_input_dim = semantic_dim + affective_dim + handcrafted_dim + num_labels

        self.semantic_branch = ProjectionBlock(
            semantic_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.affective_branch = ProjectionBlock(
            affective_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.handcrafted_branch = ProjectionBlock(
            handcrafted_dim, projection_dim, activation="tanh", dropout=handcrafted_dropout
        )
        # Auxiliary classifier on raw semantic CLS token (768 → num_labels).
        # Explicitly supervised so it learns meaningful class probabilities
        # early in training, giving the gate a useful conditioning signal.
        self.aux_head = nn.Linear(semantic_dim, num_labels)

        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.classifier = ClassifierHead(
            projection_dim, hidden_dim=256, num_labels=num_labels
        )

    def gate_parameters(self):
        return self.gate_network.parameters()

    def _project(self, sem, aff, hc):
        return (
            self.semantic_branch(sem),
            self.affective_branch(aff),
            self.handcrafted_branch(hc),
        )

    def _forward_full(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, gates, aux_logits). Used for training and internally."""
        sem_proj, aff_proj, hc_proj = self._project(sem, aff, hc)

        aux_logits = self.aux_head(sem)                    # (B, C) — raw sem CLS
        aux_probs = F.softmax(aux_logits, dim=-1)          # (B, C)

        gate_input = torch.cat([sem, aff, hc, aux_probs], dim=-1)  # (B, 834)
        gates = F.softmax(self.gate_network(gate_input), dim=-1)    # (B, 3)
        fused = (
            gates[:, 0:1] * sem_proj
            + gates[:, 1:2] * aff_proj
            + gates[:, 2:3] * hc_proj
        )
        logits = self.classifier(fused)
        return logits, gates, aux_logits

    def forward(self, sem, aff, hc, return_gates=False):
        logits, gates, _ = self._forward_full(sem, aff, hc)
        return (logits, gates) if return_gates else logits

    def training_step(self, sem, aff, hc, labels, criterion):
        logits, gates, aux_logits = self._forward_full(sem, aff, hc)
        main_loss = criterion(logits, labels)
        aux_loss = criterion(aux_logits, labels)
        return main_loss + self.aux_weight * aux_loss, logits, gates


# ---------------------------------------------------------------------------
# Variant C — load_balance  (extends B)
# ---------------------------------------------------------------------------

class LoadBalancedFusion(ClassAwareGatedFusion):
    """
    Class-aware gate + Shazeer-style load-balance loss on batch gate importance.

    Root cause fixed (on top of B): even with cross-branch context, the gate
    may still learn to consistently ignore one branch if the task loss lets it.
    The load-balance loss penalises any branch from being systematically
    under-used across a batch.

    Following Shazeer 2017 MoE (adapted for soft routing):
        importance = gate_weights.sum(dim=0)   # total weight per branch in batch
        lb_loss = CV(importance)²              # coefficient of variation squared

    High CV means branches are used unequally across the batch → penalty is
    large. The loss drives the model to spread gate weight across branches
    for different samples, not necessarily to make each sample's gate uniform.

    Total loss = main_CE + aux_weight * aux_CE + lb_weight * lb_loss
    """

    def __init__(
        self,
        lb_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lb_weight = lb_weight

    def training_step(self, sem, aff, hc, labels, criterion):
        logits, gates, aux_logits = self._forward_full(sem, aff, hc)
        main_loss = criterion(logits, labels)
        aux_loss = criterion(aux_logits, labels)

        # Per-batch importance: total gate weight allocated to each branch.
        importance = gates.sum(dim=0)                           # (3,)
        cv = importance.std() / (importance.mean() + 1e-8)
        lb_loss = cv ** 2

        total = main_loss + self.aux_weight * aux_loss + self.lb_weight * lb_loss
        return total, logits, gates


# ---------------------------------------------------------------------------
# Variant D — per_class_gate
# ---------------------------------------------------------------------------

class PerClassGatedFusion(nn.Module):
    """
    Six class-specific gate heads mixed by predicted class probabilities.

    Root cause fixed: a single gate network had to compromise across all
    classes. PTSD needed a high handcrafted weight (past_ratio is its top
    discriminator) while Anxiety needed a high affective weight (nervousness
    dominates), but one gate was forced to average. Here each class has its
    own gate MLP. An auxiliary classifier produces soft class probabilities
    (p₀…p₅); the 6 per-class fused representations are then mixed by those
    probabilities:

        fused_c = Σ_n(gate_c[n] × branch_proj_n)       # per-class fused
        fused   = Σ_c(p_c × fused_c)                   # soft class mixture

    The per-class gate preferences emerge from gradient descent rather than
    being hand-coded. For reporting, "effective gates" = Σ_c(p_c × gate_c)
    is a (B, 3) tensor showing the per-sample effective branch weights.
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
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.num_labels = num_labels
        gate_input_dim = semantic_dim + affective_dim + handcrafted_dim

        self.semantic_branch = ProjectionBlock(
            semantic_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.affective_branch = ProjectionBlock(
            affective_dim, projection_dim, activation="tanh", dropout=0.1
        )
        self.handcrafted_branch = ProjectionBlock(
            handcrafted_dim, projection_dim, activation="tanh", dropout=handcrafted_dropout
        )
        # Auxiliary head: predicts class from raw semantic CLS, provides aux_probs
        # for mixing and aux_CE loss for supervision.
        self.aux_head = nn.Linear(semantic_dim, num_labels)

        # All 6 gate heads implemented as one shared-trunk MLP with a wide
        # output layer.  Output (B, C×3) reshaped to (B, C, 3) then softmax
        # along the branch dim. One trunk reduces parameter count vs 6 separate
        # MLPs while still allowing class-specific gate outputs.
        self.gate_networks = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gate_hidden_dim, num_labels * 3),
        )
        self.classifier = ClassifierHead(
            projection_dim, hidden_dim=256, num_labels=num_labels
        )

    def gate_parameters(self):
        return self.gate_networks.parameters()

    def _forward_full(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, eff_gates (B,3), aux_logits)."""
        sem_proj = self.semantic_branch(sem)   # (B, P)
        aff_proj = self.affective_branch(aff)  # (B, P)
        hc_proj = self.handcrafted_branch(hc)  # (B, P)

        aux_logits = self.aux_head(sem)                  # (B, C)
        aux_probs = F.softmax(aux_logits, dim=-1)        # (B, C)

        gate_input = torch.cat([sem, aff, hc], dim=-1)  # (B, 828)
        raw_gates = self.gate_networks(gate_input)       # (B, C*3)
        B, C = aux_probs.shape
        gates_per_class = F.softmax(
            raw_gates.view(B, C, 3), dim=-1
        )  # (B, C, 3)

        # Stack branch projections: (B, 3, P)
        proj_stack = torch.stack([sem_proj, aff_proj, hc_proj], dim=1)

        # Per-class fused: (B, C, P) = einsum over branch dim
        # gates_per_class: (B, C, 3), proj_stack: (B, 3, P)
        per_class_fused = torch.einsum("bcn,bnp->bcp", gates_per_class, proj_stack)

        # Soft class mixture: (B, P) = einsum over class dim
        fused = torch.einsum("bc,bcp->bp", aux_probs, per_class_fused)
        logits = self.classifier(fused)

        # Effective gates for analysis: (B, 3)
        eff_gates = torch.einsum("bc,bcn->bn", aux_probs, gates_per_class)
        return logits, eff_gates, aux_logits

    def forward(self, sem, aff, hc, return_gates=False):
        logits, eff_gates, _ = self._forward_full(sem, aff, hc)
        return (logits, eff_gates) if return_gates else logits

    def training_step(self, sem, aff, hc, labels, criterion):
        logits, eff_gates, aux_logits = self._forward_full(sem, aff, hc)
        main_loss = criterion(logits, labels)
        aux_loss = criterion(aux_logits, labels)
        return main_loss + self.aux_weight * aux_loss, logits, eff_gates


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, type[nn.Module]] = {
    "content_gate": ContentGatedFusion,
    "class_aware": ClassAwareGatedFusion,
    "load_balance": LoadBalancedFusion,
    "per_class_gate": PerClassGatedFusion,
}


def build_v2_model(variant: str, cfg: dict) -> nn.Module:
    """
    Construct the requested variant from a flat config dict.

    All variants accept the base kwargs; variant-specific kwargs (aux_weight,
    lb_weight) are only passed when the variant supports them.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(_VARIANTS)}")

    base = dict(
        semantic_dim=config.SEMANTIC_DIM,
        affective_dim=config.AFFECTIVE_DIM,
        handcrafted_dim=config.HANDCRAFTED_DIM,
        projection_dim=int(cfg.get("projection_dim", config.GATED_PROJECTION_DIM)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 128)),
        handcrafted_dropout=float(cfg.get("handcrafted_dropout", 0.4)),
        num_labels=config.NUM_LABELS,
    )

    if variant == "content_gate":
        return ContentGatedFusion(**base)
    if variant == "class_aware":
        return ClassAwareGatedFusion(**base, aux_weight=float(cfg.get("aux_weight", 0.3)))
    if variant == "load_balance":
        return LoadBalancedFusion(
            lb_weight=float(cfg.get("lb_weight", 0.01)),
            aux_weight=float(cfg.get("aux_weight", 0.3)),
            **base,
        )
    # per_class_gate
    return PerClassGatedFusion(**base, aux_weight=float(cfg.get("aux_weight", 0.3)))
