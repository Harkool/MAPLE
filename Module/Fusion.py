import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=None, num_heads=8):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hidden_dim or min(dim1, dim2)
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
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
        q1 = self.q_proj1(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k_proj2(x2).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v_proj2(x2).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        out1 = (attn1 @ v2).transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        out1 = self.out_proj1(out1)
        q2 = self.q_proj2(x2).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj1(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj1(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        out2 = (attn2 @ v1).transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        out2 = self.out_proj2(out2)
        return x1 + out1, x2 + out2
