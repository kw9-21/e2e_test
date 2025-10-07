import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .dynamic_gating import HeadGatingModule, FfnGatingModule
from .aux_and_budget import AuxHead, BudgetFactorGenerator, calculate_entropy


class PatchEmbedOverlap(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_chans=3, embed_dim=192, stride_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, (H, W)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, head_mask: torch.Tensor = None):
        B, N, C = x.shape
        d = C // self.num_heads
        q = self.q(x).reshape(B, N, self.num_heads, d).transpose(1, 2)  # (B,H,N,d)
        k = self.k(x).reshape(B, N, self.num_heads, d).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, d).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ v)  # (B,H,N,d)
        if head_mask is not None:
            # head-wise mask at value path
            hm = head_mask.unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
            y = y * hm
        # concat heads
        x = y.transpose(1, 2).reshape(B, N, C)
        if head_mask is not None:
            # apply head-wise mask again on O projection input for stricter isolation
            hm_flat = head_mask.unsqueeze(-1).repeat(1, 1, d).reshape(B, 1, C)  # (B,1,C)
            x = x * hm_flat
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        # Static & Dynamic gating for heads
        self.static_gate_heads = nn.Parameter(torch.zeros(num_heads))  # name includes 'static_gate'
        self.gating_module_heads = HeadGatingModule(embed_dim=dim, num_heads=num_heads)
        # Static & Dynamic gating for FFN (per-unit)
        self.hidden_dim = self.mlp.fc1.out_features
        self.static_gate_ffn = nn.Parameter(torch.zeros(self.hidden_dim))
        self.gating_module_ffn = FfnGatingModule(embed_dim=dim, hidden_dim=self.hidden_dim)


    def forward(self, x, cls_token, k_dynamic=None, is_warmup: bool = False):
        # x: (B, N, D), cls_token: (B, D)
        # Note: In warmup, do NOT apply any mask to forward path.
        B, N, D = x.shape
        # Collect masks
        static_soft = torch.sigmoid(self.static_gate_heads)
        static_hard = (static_soft > 0.5).float()
        if k_dynamic is None:
            dyn_soft = torch.ones(B, self.num_heads, device=x.device)
            dyn_hard = torch.ones(B, self.num_heads, device=x.device)
        else:
            dyn_soft, dyn_hard = self.gating_module_heads(cls_token, k_dynamic)
        block_masks = {
            'static_mask_soft_1': static_soft,
            'static_mask_hard_1': static_hard,
            'dynamic_mask_soft_1': dyn_soft,
            'dynamic_mask_hard_1': dyn_hard,
        }

        # Core computation: Attention with head-wise mask
        if is_warmup:
            head_mask = None
            x = x + self.attn(self.norm1(x), head_mask=head_mask)
        else:
            static_h = static_hard.view(1, -1).expand(B, -1)
            dyn_ste = dyn_soft + (dyn_hard - dyn_soft).detach()
            head_mask = dyn_ste * static_h
            x = x + self.attn(self.norm1(x), head_mask=head_mask)

        # FFN gating (per-unit) with STE, applied at fc1 output and before fc2 input
        hidden = self.hidden_dim
        static_soft_ffn = torch.sigmoid(self.static_gate_ffn)
        static_hard_ffn = (static_soft_ffn > 0.5).float()
        if is_warmup:
            dyn_soft_ffn = torch.ones(B, hidden, device=x.device)
            dyn_hard_ffn = torch.ones(B, hidden, device=x.device)
        else:
            if k_dynamic is None:
                dyn_soft_ffn = torch.ones(B, hidden, device=x.device)
                dyn_hard_ffn = torch.ones(B, hidden, device=x.device)
            else:
                if torch.is_tensor(k_dynamic):
                    b_hat = k_dynamic.float().clamp(min=1, max=self.num_heads) / float(self.num_heads)
                else:
                    b_hat = torch.full((B,), float(k_dynamic) / float(self.num_heads), device=x.device)
                k_ffn = torch.clamp(torch.ceil(b_hat * hidden), min=1, max=hidden).long()
                dyn_soft_ffn, dyn_hard_ffn = self.gating_module_ffn(cls_token, k_ffn)
        block_masks.update({
            'static_mask_soft_ffn': static_soft_ffn,
            'static_mask_hard_ffn': static_hard_ffn,
            'dynamic_mask_soft_ffn': dyn_soft_ffn,
            'dynamic_mask_hard_ffn': dyn_hard_ffn,
        })

        x2 = self.norm2(x)
        # fc1
        f1 = self.mlp.fc1(x2)
        if not is_warmup:
            static_h_ffn = static_hard_ffn.view(1, -1).expand(B, -1)
            dyn_ste_ffn = dyn_soft_ffn + (dyn_hard_ffn - dyn_soft_ffn).detach()
            ffn_mask = dyn_ste_ffn * static_h_ffn  # (B,H)
            f1 = f1 * ffn_mask.unsqueeze(1)
        # act + drop
        f1 = self.mlp.act(f1)
        f1 = self.mlp.drop(f1)
        if not is_warmup:
            f1 = f1 * ffn_mask.unsqueeze(1)
        # fc2 + drop
        f2 = self.mlp.fc2(f1)
        f2 = self.mlp.drop(f2)
        x = x + f2
        return x, block_masks


class DeiTForPruning(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_chans=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 decision_points=(3,6,9), min_budget: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.decision_points = tuple(decision_points)

        # Patch embed + CLS + pos
        self.patch_embed = PatchEmbedOverlap(img_size=img_size, patch_size=patch_size,
                                             in_chans=in_chans, embed_dim=embed_dim, stride_size=patch_size)
        # Max tokens for 32x32 with 16 stride: 4 + 1 cls = 5
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 64, embed_dim))  # oversize pos, we will slice
        self.pos_drop = nn.Dropout(0.0)

        # Blocks
        blocks = [TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)
        # Expose layer1~4 as ModuleList slices for compatibility
        self.layer1 = nn.ModuleList(blocks[0:3])
        self.layer2 = nn.ModuleList(blocks[3:6])
        self.layer3 = nn.ModuleList(blocks[6:9])
        self.layer4 = nn.ModuleList(blocks[9:12])

        # Norm & head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Aux heads at decision points
        self.aux_head1 = AuxHead(embed_dim, num_classes)
        self.aux_head2 = AuxHead(embed_dim, num_classes)
        self.aux_head3 = AuxHead(embed_dim, num_classes)

        # Budget generators: dp1->layer2, dp2->layer3, dp3->layer4
        self.budget_generator2 = BudgetFactorGenerator(min_budget=min_budget)
        self.budget_generator3 = BudgetFactorGenerator(min_budget=min_budget)
        self.budget_generator4 = BudgetFactorGenerator(min_budget=min_budget)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_core(self, x: torch.Tensor):
        # Core-only path: patch -> blocks -> norm -> cls head (no aux/budget/gating usage)
        B = x.shape[0]
        x_tokens, (Hp, Wp) = self.patch_embed(x)
        N = x_tokens.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls, x_tokens), dim=1)
        pos = self.pos_embed[:, :x_tokens.shape[1], :]
        x_tokens = self.pos_drop(x_tokens + pos)
        for blk in self.blocks:
            x_tokens, _ = blk(x_tokens, cls_token=x_tokens[:,0], k_dynamic=None, is_warmup=True)
        x_tokens = self.norm(x_tokens)
        cls_out = x_tokens[:, 0]
        logits = self.head(cls_out)
        return logits

    def forward(self, x: torch.Tensor, is_warmup: bool = False):
        B = x.shape[0]
        x_tokens, (Hp, Wp) = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls, x_tokens), dim=1)
        pos = self.pos_embed[:, :x_tokens.shape[1], :]
        x_tokens = self.pos_drop(x_tokens + pos)

        aux_logits_list: List[torch.Tensor] = []
        budget_factors_list: List[torch.Tensor] = []
        uncertainty_scores_list: List[torch.Tensor] = []
        all_block_masks: List[Dict[str, torch.Tensor]] = []

        # Decision point bookkeeping
        dp_indices = set(self.decision_points)
        latest_budget = {2: None, 3: None, 4: None}  # layer2/3/4

        for idx, blk in enumerate(self.blocks, start=1):
            # compute per-sample k from latest budget if available (or fixed 0.8 for layer1)
            if is_warmup:
                k_dynamic = None  # warmup: no gating applied to forward
            else:
                if 1 <= idx <= 3:
                    # layer1 fixed ratio 0.8 (per-sample vector, identical values)
                    bf = torch.full((B,), 0.8, device=x_tokens.device)
                elif 4 <= idx <= 6:
                    bf = latest_budget[2]
                elif 7 <= idx <= 9:
                    bf = latest_budget[3]
                elif 10 <= idx <= 12:
                    bf = latest_budget[4]
                else:
                    bf = None
                if bf is None:
                    k_dynamic = None
                else:
                    k_dynamic = torch.clamp((bf * self.num_heads).ceil(), 1, self.num_heads).long()

            x_tokens, masks = blk(x_tokens, cls_token=x_tokens[:,0], k_dynamic=k_dynamic, is_warmup=is_warmup)
            all_block_masks.append(masks)

            # Decision point: compute aux -> entropy -> budget for next segment
            if idx in dp_indices:
                cls_now = x_tokens[:, 0].detach()
                if idx == self.decision_points[0]:
                    aux_logits = self.aux_head1(cls_now)
                elif idx == self.decision_points[1]:
                    aux_logits = self.aux_head2(cls_now)
                else:
                    aux_logits = self.aux_head3(cls_now)
                aux_logits_list.append(aux_logits)
                ent = calculate_entropy(aux_logits)
                uncertainty_scores_list.append(ent)
                ent_norm = torch.clamp(ent / math.log(self.num_classes), 0.0, 1.0)
                if idx == self.decision_points[0]:
                    bf = self.budget_generator2(ent_norm)
                    latest_budget[2] = bf
                elif idx == self.decision_points[1]:
                    bf = self.budget_generator3(ent_norm)
                    latest_budget[3] = bf
                else:
                    bf = self.budget_generator4(ent_norm)
                    latest_budget[4] = bf
                budget_factors_list.append(torch.clamp(bf, max=0.99))

        x_tokens = self.norm(x_tokens)
        logits = self.head(x_tokens[:, 0])

        return logits, aux_logits_list, budget_factors_list, uncertainty_scores_list, all_block_masks

