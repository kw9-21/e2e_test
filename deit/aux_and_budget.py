import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxHead(nn.Module):
    """Aux head operating on CLS token.
    Input: (B, D) -> logits (B, num_classes)
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.classifier(cls_token)


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


class BudgetFactorGenerator(nn.Module):
    """Map normalized entropy in [0,1] to budget factor in [min_budget, 1]."""
    def __init__(self, min_budget: float = 0.3, hidden_dim: int = 32):
        super().__init__()
        self.min_budget = float(min_budget)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, uncertainty_score: torch.Tensor) -> torch.Tensor:
        # uncertainty_score: (B,)
        x = uncertainty_score.unsqueeze(-1)
        y = self.net(x)
        base = (torch.tanh(y) + 1.0) / 2.0  # (B,1) in [0,1]
        out = self.min_budget + (1.0 - self.min_budget) * base
        return out.squeeze(-1)

