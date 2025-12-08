# Module/CARE.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFidelityGroupNorm(nn.Module):
    def __init__(self, d_model: int, num_groups: int = 8, preserve_ratio: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.preserve_ratio = preserve_ratio
        self.num_groups = min(num_groups, d_model)
        while d_model % self.num_groups != 0 and self.num_groups > 1:
            self.num_groups -= 1
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.preserve_ratio >= 1.0:
            return x
        residual = x
        B, L, D = x.shape
        if self.num_groups == 1:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / (var + self.eps).sqrt()
        else:
            x = x.view(B, L, self.num_groups, D // self.num_groups)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(B, L, D)
        x = x * self.weight + self.bias
        g = torch.sigmoid(self.gate)
        return self.preserve_ratio * residual + (1 - self.preserve_ratio) * g * x


class MotifExtractor(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model // 4, kernel_size=5, padding=2)
        self.fuse = nn.Linear(d_model // 2, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2)
        f3 = F.gelu(self.conv3(x_t))
        f5 = F.gelu(self.conv5(x_t))
        fused = torch.cat([f3, f5], dim=1).transpose(1, 2)
        enhanced = self.fuse(fused)
        gate = self.gate(residual)
        return residual + gate * enhanced


class ChannelCompressor(nn.Module):
    def __init__(self, d_model: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.excitor = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.GELU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        scale = self.excitor(pooled).unsqueeze(1)
        return x * scale


class CARE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_groups: int = 8,
        preserve_ratio: float = 0.8,
        dropout: float = 0.1,
        use_motif: bool = True,
        use_channel: bool = True,
    ):
        super().__init__()
        self.use_motif = use_motif
        self.use_channel = use_channel

        self.norm1 = HighFidelityGroupNorm(d_model, num_groups, preserve_ratio)
        self.dropout = nn.Dropout(dropout)

        if use_motif:
            self.motif = MotifExtractor(d_model)
        if use_channel:
            self.channel = ChannelCompressor(d_model)

        self.norm_out = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"CARE expects (B, L, D), got {x.shape}")

        residual = x
        x = self.norm1(x)

        if self.use_motif:
            x = self.motif(x)

        if self.use_channel:
            x = self.channel(x)

        x = self.norm_out(x)
        x = self.dropout(x)

        gate = self.gate(residual).expand_as(x)
        return gate * x + (1 - gate) * residual