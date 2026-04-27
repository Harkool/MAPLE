import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedGroupNorm(nn.Module):
    """Optimized normalization layer"""

    def __init__(self, d_model: int, num_groups: int = 4, eps: float = 1e-6, preserve_ratio: float = 0.7):
        super().__init__()
        self.preserve_ratio = preserve_ratio
        self.num_groups = min(num_groups, d_model)

        while d_model % self.num_groups != 0 and self.num_groups > 1:
            self.num_groups -= 1

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm_gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.preserve_ratio >= 1.0:
            return x

        bsz, length, dim = x.shape
        original_x = x

        if self.num_groups == 1:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
        else:
            x = x.view(bsz, length, self.num_groups, dim // self.num_groups)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x.view(bsz, length, dim)

        x = x * self.weight + self.bias
        gate_weight = torch.sigmoid(self.norm_gate)
        return self.preserve_ratio * original_x + (1 - self.preserve_ratio) * gate_weight * x


class ConservativeMotifExtractor(nn.Module):
    """Conservative sequence feature extractor - minimizes information loss"""

    def __init__(self, d_model: int, preserve_original: bool = True):
        super().__init__()
        self.preserve_original = preserve_original

        self.conv3 = nn.Conv1d(d_model, d_model // 4, 3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model // 4, 5, padding=2)

        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not self.preserve_original:
            return x

        original_x = x
        x_t = x.transpose(1, 2)

        feat3 = F.gelu(self.conv3(x_t))
        feat5 = F.gelu(self.conv5(x_t))

        combined_feat = torch.cat([feat3, feat5], dim=1)
        combined_feat = combined_feat.transpose(1, 2)

        enhanced_feat = self.feature_fusion(combined_feat)
        attention_weights = self.attention_gate(original_x)

        output = 0.8 * original_x + 0.2 * attention_weights * enhanced_feat
        return output


class AdaptiveChannelProcessor(nn.Module):
    """Adaptive channel processor - reduces information compression"""

    def __init__(self, d_model: int, min_compression_ratio: int = 2, max_compression_ratio: int = 4):
        super().__init__()
        self.d_model = d_model

        self.compression_controller = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.processors = nn.ModuleList()
        for ratio in [min_compression_ratio, max_compression_ratio]:
            hidden_dim = max(32, d_model // ratio)
            processor = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, d_model),
            )
            self.processors.append(processor)

        self.output_fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))

    def forward(self, x):
        compression_weight = self.compression_controller(x.transpose(1, 2))

        processed_features = []
        for processor in self.processors:
            processed = processor(x)
            processed_features.append(processed)

        mixed_processed = compression_weight.unsqueeze(1) * processed_features[0] + (1 - compression_weight.unsqueeze(1)) * processed_features[1]
        combined = torch.cat([x, mixed_processed], dim=-1)
        output = self.output_fusion(combined)
        return output


class CARE(nn.Module):
    """CARE module (renamed from legacy ScConv)."""

    def __init__(
        self,
        d_model: int,
        num_groups: int = 2,
        dropout: float = 0.05,
        preserve_ratio: float = 0.8,
        enable_motif_extraction: bool = True,
        enable_channel_processing: bool = True,
    ):
        super().__init__()
        self.enable_motif_extraction = enable_motif_extraction
        self.enable_channel_processing = enable_channel_processing

        self.norm1 = OptimizedGroupNorm(d_model, num_groups, preserve_ratio=preserve_ratio)

        if enable_motif_extraction:
            self.motif_extractor = ConservativeMotifExtractor(d_model, preserve_original=True)

        if enable_channel_processing:
            self.norm2 = nn.LayerNorm(d_model)
            self.channel_processor = AdaptiveChannelProcessor(d_model)

        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.final_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, L, D], got {x.shape}")

        original_input = x

        if hasattr(self, "norm1"):
            x = self.norm1(x)

        if self.enable_motif_extraction and hasattr(self, "motif_extractor"):
            residual1 = x
            x = self.motif_extractor(x)
            x = x + residual1

        if self.enable_channel_processing and hasattr(self, "channel_processor"):
            residual2 = x
            x = self.norm2(x)
            x = self.channel_processor(x)
            x = x + residual2

        x = self.output_norm(x)
        x = self.dropout(x)

        gate_weight = self.final_gate(original_input)
        output = gate_weight * x + (1 - gate_weight) * original_input
        return output


# Backward compatibility alias
ScConv = CARE


class BypassCARE(nn.Module):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        return self.dropout(self.norm(x))


class MinimalCARE(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        enhanced = self.enhance(x)
        return self.norm(x + 0.1 * enhanced)


__all__ = ["CARE", "ScConv", "BypassCARE", "MinimalCARE"]
