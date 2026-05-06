import torch.nn as nn

from scripts import config
from scripts.models.fusion.gated import GatedFusion
from scripts.models.fusion.late_concat import LateConcatFusion


def build_fusion_model(fusion_type: str | None = None) -> nn.Module:
    selected = (fusion_type or config.FUSION_TYPE).lower()
    if selected == "concat":
        return LateConcatFusion()
    if selected == "gated":
        return GatedFusion()
    raise ValueError("fusion_type must be 'concat' or 'gated'")


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
