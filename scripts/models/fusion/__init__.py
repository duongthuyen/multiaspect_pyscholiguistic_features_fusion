"""
Gated fusion package — lazy imports so ``import scripts.models.fusion``
does NOT pull in torch at import time (keeps CLI --help fast).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for static analysis, never executed at runtime
    from scripts.models.fusion.gated import (  # noqa: F401
        GatedFusion,
        build_gated_model,
    )

__all__ = [
    "GatedFusion",
    "build_gated_model",
    "count_parameters",
]

_LAZY = {
    "GatedFusion":       ("scripts.models.fusion.gated", "GatedFusion"),
    "build_gated_model": ("scripts.models.fusion.gated", "build_gated_model"),
}


def __getattr__(name: str):
    """Resolve lazy imports on first attribute access."""
    if name in _LAZY:
        import importlib
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        value = getattr(mod, attr)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def count_parameters(model) -> dict:
    """Return trainable and total parameter counts for *model*."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
