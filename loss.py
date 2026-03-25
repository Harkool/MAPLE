import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-label focal loss with optional per-label alpha and pos_weight.

    alpha:
        - scalar, or
        - tensor of shape (C,)
    gamma:
        focusing parameter
    pos_weight:
        tensor of shape (C,) for BCEWithLogits
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight=None,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())

        if pos_weight is None:
            self.pos_weight = None
        else:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        pos_weight = None if self.pos_weight is None else self.pos_weight.to(logits.device)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight,
        )

        prob = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, prob, 1.0 - prob)

        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha = self.alpha.to(logits.device)
            if alpha.ndim == 0:
                alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)
            else:
                alpha = alpha.view(1, -1)
                alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
