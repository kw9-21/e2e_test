import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadGatingModule(nn.Module):
    """CLS-driven head gating: MLP(D)->H logits. Returns both soft and top-k hard masks.
    Note: In warmup, caller should ignore masks when applying to forward path.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(8, int(embed_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_heads)
        )

    def forward(self, cls_token: torch.Tensor, k):
        # cls_token: (B, D); k can be int, scalar tensor, or per-sample LongTensor of shape (B,)
        logits = self.net(cls_token)  # (B, H)
        soft = torch.sigmoid(logits)  # bounded [0,1]
        B, H = soft.shape
        # If k is None, return all-ones hard mask (caller may ignore in warmup)
        if k is None:
            hard = torch.ones_like(soft)
            return soft, hard
        # Accept scalar int
        if isinstance(k, int):
            kk = max(1, min(k, H))
            topk_vals, topk_idx = torch.topk(soft, k=kk, dim=1)
            hard = torch.zeros_like(soft)
            hard.scatter_(1, topk_idx, 1.0)
            return soft, hard
        # Accept scalar tensor
        if torch.is_tensor(k) and k.dim() == 0:
            kk = int(k.item())
            kk = max(1, min(kk, H))
            topk_vals, topk_idx = torch.topk(soft, k=kk, dim=1)
            hard = torch.zeros_like(soft)
            hard.scatter_(1, topk_idx, 1.0)
            return soft, hard
        # Per-sample k vector (B,)
        if torch.is_tensor(k):
            k_vec = k.to(device=soft.device, dtype=torch.long)
            k_vec = torch.clamp(k_vec, min=1, max=H)
            sorted_idx = torch.argsort(soft, dim=1, descending=True)  # (B,H)
            col_idx = torch.arange(H, device=soft.device).unsqueeze(0).expand(B, H)
            comp = col_idx < k_vec.unsqueeze(1)  # (B,H)
            hard = torch.zeros_like(soft)
            hard.scatter_(1, sorted_idx, comp.float())
            return soft, hard
        # Fallback
        hard = torch.zeros_like(soft)
        return soft, hard

class FfnGatingModule(nn.Module):
    """CLS-driven FFN gating: MLP(D)->H logits for hidden units. Returns soft and hard masks.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(8, int(embed_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden_dim)
        )

    def forward(self, cls_token: torch.Tensor, k):
        # cls_token: (B, D); k: int or (B,)
        logits = self.net(cls_token)  # (B, H)
        soft = torch.sigmoid(logits)
        B, H = soft.shape
        if k is None:
            hard = torch.ones_like(soft)
            return soft, hard
        if isinstance(k, int):
            kk = max(1, min(k, H))
            topk_vals, topk_idx = torch.topk(soft, k=kk, dim=1)
            hard = torch.zeros_like(soft)
            hard.scatter_(1, topk_idx, 1.0)
            return soft, hard
        if torch.is_tensor(k) and k.dim() == 0:
            kk = int(k.item())
            kk = max(1, min(kk, H))
            topk_vals, topk_idx = torch.topk(soft, k=kk, dim=1)
            hard = torch.zeros_like(soft)
            hard.scatter_(1, topk_idx, 1.0)
            return soft, hard
        if torch.is_tensor(k):
            k_vec = k.to(device=soft.device, dtype=torch.long)
            k_vec = torch.clamp(k_vec, min=1, max=H)
            sorted_idx = torch.argsort(soft, dim=1, descending=True)
            col_idx = torch.arange(H, device=soft.device).unsqueeze(0).expand(B, H)
            comp = col_idx < k_vec.unsqueeze(1)
            hard = torch.zeros_like(soft)
            hard.scatter_(1, sorted_idx, comp.float())
            return soft, hard
        hard = torch.zeros_like(soft)
        return soft, hard

