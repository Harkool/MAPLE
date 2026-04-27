import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # query/key/value: [B, L, D]
        bsz, q_len, _ = query.size()
        k_len = key.size(1)

        q = self.w_q(query).view(bsz, q_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(bsz, k_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bsz, k_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            # mask should be [B, 1, 1, L]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.w_o(context)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        a = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(a))
        f = self.ffn(x)
        x = self.norm2(x + self.dropout(f))
        return x


class KnowledgeTransformerEncoder(nn.Module):
    """Trainable encoder used to map raw knowledge descriptors into contextual 256-d features."""

    def __init__(
        self,
        input_dim: int = 56,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 700,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def create_padding_mask(lengths: Optional[torch.Tensor], seq_len: int, device: torch.device):
        if lengths is None:
            return None
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # x: [B, L, input_dim]
        x = self.input_projection(x)
        x = self.dropout(x)
        x = self.positional_encoding(x)

        mask = self.create_padding_mask(lengths, x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.output_projection(x)


class KnowledgeTransformerWithDecoder(nn.Module):
    """Used only for pretraining the standalone transformer via reconstruction loss."""

    def __init__(self, encoder: Optional[KnowledgeTransformerEncoder] = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else KnowledgeTransformerEncoder()
        self.decoder = nn.Linear(self.encoder.d_model, self.encoder.input_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        z = self.encoder(x, lengths=lengths)
        recon = self.decoder(z)
        return z, recon


def load_trained_knowledge_transformer(ckpt_path: str, device: str = "cpu") -> KnowledgeTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config: Dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    encoder = KnowledgeTransformerEncoder(
        input_dim=int(config.get("input_dim", 56)),
        d_model=int(config.get("d_model", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("num_layers", 4)),
        d_ff=int(config.get("d_ff", 512)),
        dropout=float(config.get("dropout", 0.1)),
        max_len=int(config.get("max_len", 700)),
    )

    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        sd = ckpt["encoder_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
        sd = {k[len("encoder.") :]: v for k, v in raw_sd.items() if k.startswith("encoder.")}
        if not sd:
            sd = raw_sd
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys while loading knowledge transformer: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading knowledge transformer: {unexpected[:8]}")

    encoder.eval()
    return encoder
