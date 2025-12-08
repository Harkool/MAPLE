import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.learned_pos_bias = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.length_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        pos_encoding = (self.pe[:, :L, :] + self.learned_pos_bias[:, :L, :]) * self.length_scale
        return x + pos_encoding


class SSMCore(nn.Module):
    def __init__(self, dim: int, state_size: int = 32):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        self.gate = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_temp = x_norm.transpose(1, 2)
        conv_out = self.output_proj(self.activation(self.conv(x_temp)).transpose(1, 2))
        gate = torch.sigmoid(self.gate(x_norm))
        gated_features = gate * conv_out
        return x + gated_features


class BiDirectionalMamba(nn.Module):
    def __init__(self, dim: int, state_size: int = 32):
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Linear(dim, dim * 2)
        self.forward_ssm = SSMCore(dim, state_size)
        self.backward_ssm = SSMCore(dim, state_size)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())
        self.output_proj = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.proj_in(x).chunk(2, dim=-1)
        
        y_forward = self.forward_ssm(x_proj)
        
        x_reversed = torch.flip(x_proj, dims=[1])
        y_backward = torch.flip(self.backward_ssm(x_reversed), dims=[1])
        
        concat_features = torch.cat([y_forward, y_backward], dim=-1)
        y_fused = self.fusion(concat_features)
        
        output = y_fused * torch.sigmoid(gate)
        
        return self.output_proj(output)


class FFN(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class ResidualProBiMambaBlock(nn.Module):
    def __init__(self, dim: int = 1280, state_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.pos_encoding = PositionalEncoding(dim)
        self.bimamba = BiDirectionalMamba(dim, state_size)
        self.ffn = FFN(dim, expansion_factor=2, dropout=dropout)
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = self.pos_encoding(x)
        mamba_out = self.bimamba(self.norm1(x_pos))
        ffn_out = self.ffn(self.norm2(x_pos))
        w = torch.sigmoid(self.fusion_weight)
        fused_output = w * mamba_out + (1 - w) * ffn_out
        return x_pos + self.dropout(fused_output)


class ProBiMamba(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 1280, num_layers: int = 6, state_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            ResidualProBiMambaBlock(hidden_dim, state_size, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.use_checkpoint = torch.__version__ >= "1.6.0"

    def forward(self, esm_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(esm_features)
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
        enhanced_features = self.final_norm(x)
        return enhanced_features