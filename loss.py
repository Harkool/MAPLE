import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """BCE-with-logits focal loss.

    Use either alpha-balanced focal loss or pos_weight-weighted focal loss.
    Do not combine alpha and pos_weight, otherwise positives are reweighted twice.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", pos_weight=None):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None and pos_weight is not None:
            raise ValueError("FocalLoss does not allow alpha and pos_weight together; choose one.")

        alpha_t = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        pos_w = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32)

        if alpha_t is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", alpha_t)

        if pos_w is None:
            self.pos_weight = None
        else:
            self.register_buffer("pos_weight", pos_w)

    @staticmethod
    def from_targets(
        targets,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight_max: float = 50.0,
    ):
        """Build pos_weight-weighted focal loss from training-set class ratio.

        This conservative path matches the current training code: derive pos_weight
        from neg / pos and leave alpha disabled.
        """
        y = np.asarray(targets, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        pos = y.sum(axis=0)
        total = float(y.shape[0])
        neg = total - pos

        eps = 1e-8
        pos_weight = neg / np.clip(pos, eps, None)
        pos_weight = np.clip(pos_weight, 1.0, pos_weight_max).astype(np.float32)

        alpha = None
        if pos_weight.shape[0] == 1:
            pos_weight = float(pos_weight[0])

        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, pos_weight=pos_weight), alpha, pos_weight

    def forward(self, logits, targets):
        targets = targets.float()

        pos_weight = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)

        probs = torch.sigmoid(logits).clamp(min=1e-8, max=1 - 1e-8)
        pt = torch.where(targets >= 0.5, probs, 1.0 - probs)

        if self.alpha is None:
            alpha_factor = torch.ones_like(targets)
        else:
            alpha = self.alpha.to(logits.device)
            alpha_factor = torch.where(targets >= 0.5, alpha, 1.0 - alpha)

        loss = alpha_factor * ((1.0 - pt) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
