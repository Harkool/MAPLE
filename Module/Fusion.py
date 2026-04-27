import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QuadCrossAttention(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_heads=8, fusion="concat"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fusion = fusion.lower()

        # Build independent Q/K/V and projection layers for each direction
        self.q_proj = nn.ModuleDict()
        self.k_proj = nn.ModuleDict()
        self.v_proj = nn.ModuleDict()
        self.out_proj = nn.ModuleDict()

        directions = ['A', 'B', 'C', 'D']
        for q in directions:
            self.q_proj[q] = nn.Linear(dim, self.hidden_dim)
            self.k_proj[q] = nn.Linear(dim, self.hidden_dim)
            self.v_proj[q] = nn.Linear(dim, self.hidden_dim)
            self.out_proj[q] = nn.Linear(self.hidden_dim, dim)

    def attend(self, q_feat, kv_feats, q_label, kv_labels):
        # q_feat: [B, L, D]
        B, L, _ = q_feat.shape
        q = self.q_proj[q_label](q_feat).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        out_sum = 0
        for kv_label in kv_labels:
            kv = kv_feats[kv_label]
            k = self.k_proj[kv_label](kv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj[kv_label](kv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
            out_sum += self.out_proj[q_label](out)

        return q_feat + out_sum / len(kv_labels)

    def forward(self, xA, xB, xC, xD):
        feats = {'A': xA, 'B': xB, 'C': xC, 'D': xD}
        outA = self.attend(xA, feats, 'A', ['B', 'C', 'D'])
        outB = self.attend(xB, feats, 'B', ['A', 'C', 'D'])
        outC = self.attend(xC, feats, 'C', ['A', 'B', 'D'])
        outD = self.attend(xD, feats, 'D', ['A', 'B', 'C'])

        # Fusion
        if self.fusion == "concat":
            fused = torch.cat([
                outA.mean(dim=1),
                outB.mean(dim=1),
                outC.mean(dim=1),
                outD.mean(dim=1)
            ], dim=-1)
        elif self.fusion == "mean":
            fused = (outA + outB + outC + outD).mean(dim=1)
        elif self.fusion == "sum":
            fused = (outA + outB + outC + outD).sum(dim=1)
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")

        return fused

class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion."""
    def __init__(self, dim1, dim2, hidden_dim=None, num_heads=8):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hidden_dim or min(dim1, dim2)
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        
        # Projection layers
        self.q_proj1 = nn.Linear(dim1, self.hidden_dim)
        self.k_proj2 = nn.Linear(dim2, self.hidden_dim)
        self.v_proj2 = nn.Linear(dim2, self.hidden_dim)
        
        self.q_proj2 = nn.Linear(dim2, self.hidden_dim)
        self.k_proj1 = nn.Linear(dim1, self.hidden_dim)
        self.v_proj1 = nn.Linear(dim1, self.hidden_dim)
        
        self.out_proj1 = nn.Linear(self.hidden_dim, dim1)
        self.out_proj2 = nn.Linear(self.hidden_dim, dim2)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x1, x2):
        B, L, _ = x1.shape
        
        # x1 attend to x2
        q1 = self.q_proj1(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k_proj2(x2).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v_proj2(x2).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        out1 = (attn1 @ v2).transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        out1 = self.out_proj1(out1)
        
        # x2 attend to x1
        q2 = self.q_proj2(x2).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj1(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj1(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        out2 = (attn2 @ v1).transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        out2 = self.out_proj2(out2)
        
        return x1 + out1, x2 + out2


class GatedFusion(nn.Module):
    """
    Gated fusion module: takes x1 and x2 and learns gate weights for fusion.
    """
    def __init__(self, dim1, dim2, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = dim1  # Default output dimension matches x1.
        
        self.output_dim = output_dim
        self.gate_layer = nn.Sequential(
            nn.Linear(dim1 + dim2, output_dim),
            nn.Sigmoid()
        )
        self.proj_layer = nn.Linear(dim1 + dim2, output_dim)

    def forward(self, x1, x2):
        # Concatenate
        combined = torch.cat([x1, x2], dim=-1)  # [B, L, D1+D2]
        gate = self.gate_layer(combined)       # [B, L, D_out]
        fused = self.proj_layer(combined)      # [B, L, D_out]
        return gate * fused + (1 - gate) * x1  # Fusion: weighted information with residual connection to x1.
