import torch
from .model import DeiTForPruning

# Simple builder that maps backbone name to dimensions
# DeiT variants (standard):
# - tiny:   embed_dim=192, num_heads=3, depth=12
# - small:  embed_dim=384, num_heads=6, depth=12
# - base:   embed_dim=768, num_heads=12, depth=12

_DEIT_CFG = {
    'deit_tiny':  dict(embed_dim=192,  num_heads=3,  depth=12),
    'deit_small': dict(embed_dim=384,  num_heads=6,  depth=12),
    'deit_base':  dict(embed_dim=768,  num_heads=12, depth=12),
}

def build_deit(backbone: str, num_classes: int, img_size: int = 32,
               decision_points=(3, 6, 9), min_budget: float = 0.3):
    name = backbone.lower()
    if name not in _DEIT_CFG:
        raise ValueError(f"Unknown DeiT backbone: {backbone}")
    cfg = _DEIT_CFG[name]
    model = DeiTForPruning(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        decision_points=decision_points,
        min_budget=min_budget,
    )
    return model.cuda()

