import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

class ChemicalAwarePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.aa_type_embedding = nn.Embedding(22, dim)
        proj_dim = max(dim // 8, 32)
        self.hydrophobicity_proj = nn.Linear(1, proj_dim)
        self.charge_proj = nn.Linear(1, proj_dim)
        self.polarity_proj = nn.Linear(1, proj_dim)
        self.aromaticity_proj = nn.Linear(1, proj_dim)
        self.chemical_combine = nn.Linear(proj_dim * 4, dim)
        self.learned_pos_bias = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.length_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, aa_types=None, chemical_properties=None):
        B, L, D = x.shape
        pos_encoding = (self.pe[:, :L, :] + self.learned_pos_bias[:, :L, :]) * self.length_scale
        if aa_types is not None:
            aa_embedding = self.aa_type_embedding(aa_types)
            pos_encoding = pos_encoding + aa_embedding
            if chemical_properties is not None:
                chem_features = [
                    self.hydrophobicity_proj(chemical_properties['hydrophobicity'].unsqueeze(-1)),
                    self.charge_proj(chemical_properties['charge'].unsqueeze(-1)),
                    self.polarity_proj(chemical_properties['polarity'].unsqueeze(-1)),
                    self.aromaticity_proj(chemical_properties['aromaticity'].unsqueeze(-1))
                ]
                chemical_encoding = self.chemical_combine(torch.cat(chem_features, dim=-1))
                pos_encoding = pos_encoding + chemical_encoding
        return x + pos_encoding

class SSMCore(nn.Module):
    def __init__(self, dim, state_size=32):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(dim)
        self.activation = nn.ReLU()
        self.gate = nn.Linear(dim, dim)
        self.skip_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, conservation_scores=None):
        B, L, D = x.shape
        residual = x
        x_temp = x.transpose(1, 2)
        conv_out = self.activation(self.norm(self.conv(x_temp))).transpose(1, 2)
        gate = torch.sigmoid(self.gate(x))
        gated_features = gate * conv_out
        return residual * self.skip_weight + gated_features * (1 - self.skip_weight)

class BiDirectionalMamba(nn.Module):
    def __init__(self, dim, state_size=32):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(dim, dim * 2)
        self.forward_ssm = SSMCore(dim, state_size)
        self.backward_ssm = SSMCore(dim, state_size)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.ReLU())
        self.output_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x, conservation_scores=None):
        B, L, D = x.shape
        residual = x
        x_proj, gate = self.input_proj(x).chunk(2, dim=-1)
        y_forward = self.forward_ssm(x_proj, conservation_scores)
        x_reversed = torch.flip(x_proj, dims=[1])
        y_backward = torch.flip(self.backward_ssm(x_reversed, conservation_scores), dims=[1])
        concat_features = torch.cat([y_forward, y_backward], dim=-1)
        y_fused = self.fusion(concat_features)
        output = y_fused * torch.sigmoid(gate)
        return residual + self.output_proj(output)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, functional_types=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class FunctionPredictionHead(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.catalytic_head = nn.Linear(dim // 2, 2)
        self.binding_head = nn.Linear(dim // 2, 8)
        self.modification_head = nn.Linear(dim // 2, 12)
        self.structural_head = nn.Linear(dim // 2, 4)
        self.conservation_head = nn.Linear(dim // 2, 1)
        self.function_score_head = nn.Linear(dim // 2, 1)

    def forward(self, x):
        shared_feat = self.shared_features(x)
        return {
            'catalytic': self.catalytic_head(shared_feat),
            'binding': self.binding_head(shared_feat),
            'modification': self.modification_head(shared_feat),
            'structural': self.structural_head(shared_feat),
            'conservation': self.conservation_head(shared_feat),
            'function_score': self.function_score_head(shared_feat)
        }

class BiMambaBlock(nn.Module):
    def __init__(self, dim=1280, state_size=32, num_heads=8, dropout=0.1, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if use_attention:
            self.norm3 = nn.LayerNorm(dim)
        self.pos_encoding = ChemicalAwarePositionalEncoding(dim)
        self.bimamba = BiDirectionalMamba(dim, state_size)
        if use_attention:
            self.attention = Attention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.component_weights = nn.Parameter(torch.ones(3 if use_attention else 2) / (3 if use_attention else 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, aa_types=None, chemical_properties=None, conservation_scores=None, functional_types=None):
        x = self.pos_encoding(x, aa_types, chemical_properties)
        weights = F.softmax(self.component_weights, dim=0)
        mamba_out = self.bimamba(self.norm1(x), conservation_scores)
        if self.use_attention:
            attn_out = self.attention(self.norm2(x), functional_types)
            ffn_out = self.ffn(self.norm3(x))
            output = weights[0] * mamba_out + weights[1] * attn_out + weights[2] * ffn_out
        else:
            ffn_out = self.ffn(self.norm2(x))
            output = weights[0] * mamba_out + weights[1] * ffn_out
        return x + self.dropout(output)

class ProBiMamba(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=1280, num_layers=6, state_size=32, num_heads=8, dropout=0.1, use_attention=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            BiMambaBlock(hidden_dim, state_size, num_heads, dropout, use_attention) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.function_head = FunctionPredictionHead(hidden_dim, dropout)
        pytorch_version = torch.__version__
        major_version, minor_version = map(int, pytorch_version.split('.')[:2])
        self.use_checkpoint = major_version > 1 or (major_version == 1 and minor_version >= 6)

    def forward(self, esm_features, aa_types=None, chemical_properties=None, conservation_scores=None, functional_types=None, return_predictions=True):
        x = self.input_proj(esm_features)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, aa_types, chemical_properties, conservation_scores, functional_types)
            else:
                x = layer(x, aa_types, chemical_properties, conservation_scores, functional_types)
        enhanced_features = self.final_norm(x)
        if return_predictions:
            return enhanced_features, self.function_head(enhanced_features)
        return enhanced_features

    def enable_gradient_checkpointing(self):
        pass

    def disable_gradient_checkpointing(self):
        self.use_checkpoint = False

def create_aa_function_bimamba(config='base'):
    configs = {
        'tiny': {'input_dim': 480, 'hidden_dim': 240, 'num_layers': 2, 'state_size': 16, 'num_heads': 4, 'use_attention': False},
        'small': {'input_dim': 480, 'hidden_dim': 320, 'num_layers': 3, 'state_size': 16, 'num_heads': 8, 'use_attention': False},
        'base': {'input_dim': 480, 'hidden_dim': 480, 'num_layers': 4, 'state_size': 32, 'num_heads': 8, 'use_attention': True}
    }
    return ProBiMamba(**configs[config])



# Backward compatibility alias
ProteinSequenceBiMambaEncoder = ProBiMamba

__all__ = ["ProBiMamba", "ProteinSequenceBiMambaEncoder", "create_aa_function_bimamba"]
