import copy
import torch
import torch.nn as nn
from thop import profile

# THOP-based scalable FLOPs decomposition for DeiT (Phase 1: only heads scale)
# Strategy (aligned with CNN path):
# 1) Pre-compute maximum FLOPs for scalable parts via THOP on dedicated wrappers
# 2) Training time: use (keep_head_ratio) * (max FLOPs) per layer
# 3) Overhead via THOP for gating/aux/budget + residual bucket to align to authoritative global value


def _infer_tokens_and_dim(model: nn.Module, img_size: int = 32):
    model = model.cuda()
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size).cuda()
        # Run only patch + pos to get token count
        x_tokens, (Hp, Wp) = model.patch_embed(x)
        B, N, D = x_tokens.shape[0], x_tokens.shape[1], x_tokens.shape[2]
        return N + 1, D  # +1 for CLS


class _AttentionLinearWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.q = block.attn.q
        self.k = block.attn.k
        self.v = block.attn.v
        self.proj = block.attn.proj

    def forward(self, x):
        # x: (B, N, D)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        o = self.proj(v)  # ignore attn matmul here; focus on linear terms scalable by heads
        return o

class _AttentionScalableWrapper(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
    def forward(self, x):
        # Lightweight forward; FLOPs are provided by custom_ops on the whole module
        return x

def _count_attention_module(m, inputs, output):
    # Count the full scalable attention cost: Q/K/V/O linears + QK^T + Attn@V (softmax cost ignored)
    x = inputs[0]
    B, N, D = x.shape
    H = m.num_heads
    d = D // H if H > 0 else 0
    # Linear costs: 4 * B * N * D * D  (Q,K,V,O)
    linear = 4 * B * N * D * D
    # Matmul costs: 2 * B * H * N * N * d  (QK^T and Attn@V)
    mm = 2 * B * H * N * N * d
    flops = linear + mm
    import torch
    # THOP expects writing into module's counters
    if not hasattr(m, 'total_ops'):
        m.total_ops = torch.zeros(1, dtype=torch.float64)
    if not hasattr(m, 'total_params'):
        m.total_params = torch.zeros(1, dtype=torch.float64)
    m.total_ops += torch.tensor([flops], dtype=torch.float64, device=m.total_ops.device if hasattr(m.total_ops, 'device') else None)
    # params unchanged for this wrapper



class _MLPLinearWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.fc1 = block.mlp.fc1
        self.fc2 = block.mlp.fc2

    def forward(self, x):
        # x: (B, N, D)
        h = self.fc1(x)
        o = self.fc2(h)
        return o


class _PatchWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.patch = model.patch_embed

    def forward(self, x):
        y, _ = self.patch(x)
        return y


class _CLSLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def get_scalable_flops_thop(model: nn.Module, img_size: int = 32):
    # conv1_bn1 initial (patch_embed only); will adjust to absorb fixed core FLOPs
    _conv_pe, _ = profile(copy.deepcopy(_PatchWrapper(model)).cuda(), inputs=(torch.randn(1,3,img_size,img_size).cuda(),), verbose=False)

    # Token/Dim
    N_tokens, D = _infer_tokens_and_dim(model, img_size)
    dummy_tokens = torch.randn(1, N_tokens, D).cuda()

    # Per-block attention scalable FLOPs (max): Q/K/V/O linears + QK^T + Attn@V
    per_block_attn = []
    for blk in model.blocks:
        H = blk.attn.num_heads if hasattr(blk.attn, 'num_heads') else model.num_heads
        w_attn = _AttentionScalableWrapper(H).cuda()
        fl_attn, _ = profile(copy.deepcopy(w_attn), inputs=(dummy_tokens.clone(),), verbose=False,
                             custom_ops={_AttentionScalableWrapper: _count_attention_module})
        per_block_attn.append(fl_attn)

    # Per-block MLP linear FLOPs (max): fc1+fc2 only
    per_block_mlp = []
    for blk in model.blocks:
        w = _MLPLinearWrapper(blk).cuda()
        fl, _ = profile(copy.deepcopy(w), inputs=(dummy_tokens.clone(),), verbose=False)
        per_block_mlp.append(fl)

    # Group into 4 layers (each 3 blocks)
    layers = {
        'layer1': sum(per_block_attn[0:3]),
        'layer2': sum(per_block_attn[3:6]),
        'layer3': sum(per_block_attn[6:9]),
        'layer4': sum(per_block_attn[9:12]),
    }
    layers_mlp = {
        'layer1': sum(per_block_mlp[0:3]),
        'layer2': sum(per_block_mlp[3:6]),
        'layer3': sum(per_block_mlp[6:9]),
        'layer4': sum(per_block_mlp[9:12]),
    }

    # fc (final classifier)
    fc, _ = profile(copy.deepcopy(_CLSLinear(D, model.num_classes)).cuda(), inputs=(torch.randn(1, D).cuda(),), verbose=False)

    # Authoritative core-only FLOPs (no overhead)
    core_only = get_authoritative_scalable_core_flops(model, img_size=img_size)
    # Adjust conv1_bn1 to absorb fixed core FLOPs so that sum(scalable)=core_only
    conv1_bn1 = max(0.0, core_only - fc - sum(layers.values()) - sum(layers_mlp.values()))

    return {
        'conv1_bn1': conv1_bn1,
        'layers': layers,
        'layers_mlp': layers_mlp,
        'fc': fc,
    }


def get_overhead_flops_thop(model: nn.Module, img_size: int = 32):
    # Gating modules: per-block CLS-MLP to heads and ffn
    gating_total = 0
    for blk in model.blocks:
        gm_h = blk.gating_module_heads.cuda()
        fl_h, _ = profile(copy.deepcopy(gm_h), inputs=(torch.randn(1, model.embed_dim).cuda(), 1), verbose=False)
        gating_total += fl_h
        if hasattr(blk, 'gating_module_ffn'):
            gm_f = blk.gating_module_ffn.cuda()
            fl_f, _ = profile(copy.deepcopy(gm_f), inputs=(torch.randn(1, model.embed_dim).cuda(), 1), verbose=False)
            gating_total += fl_f

    # Aux heads (3) and Budget generators (3)
    aux_total = 0
    for aux in [model.aux_head1, model.aux_head2, model.aux_head3]:
        fl, _ = profile(copy.deepcopy(aux).cuda(), inputs=(torch.randn(1, model.embed_dim).cuda(),), verbose=False)
        aux_total += fl

    budget_total = 0
    for bg in [model.budget_generator2, model.budget_generator3, model.budget_generator4]:
        # input is scalar per sample
        fl, _ = profile(copy.deepcopy(bg).cuda(), inputs=(torch.randn(1).cuda(),), verbose=False)
        budget_total += fl

    # Static mask apply: analytical small term; approximate as zero here (will be absorbed by residual)
    static_apply = 0

    return {
        'gating_modules': gating_total,
        'static_mask_apply': static_apply,
        'aux_heads': aux_total,
        'budget_generators': budget_total,
    }


def get_authoritative_scalable_core_flops(model: nn.Module, img_size: int = 32):
    class _CoreOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m.forward_core(x)
    core_model = _CoreOnly(model).cuda()
    flops, _ = profile(copy.deepcopy(core_model), inputs=(torch.randn(1,3,img_size,img_size).cuda(),), verbose=False)
    return flops

