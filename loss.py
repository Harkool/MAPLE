import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        pos_weight: float = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        prob = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: 
            return loss