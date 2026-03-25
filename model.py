import torch
import torch.nn as nn
import torch.nn.functional as F

from Module.CARE import CARE
from Module.Fusion import CrossModalAttention
from Module.ProBiMamba import ProBiMamba
from Module.Transformer import KnowledgeEnhancedTransformerEncoder


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MAPLE(nn.Module):
    def __init__(
        self,
        linsize: int = 1024,
        lindropout: float = 0.8,
        num_labels: int = 1,
        esm_dim: int = 1280,
        knowledge_dim: int = 512,
        knowledge_input_dim: int = 56,
        base_dim: int = 512,
        bimamba_dim: int = 256,
        enable_amp_head: bool = True,
        knowledge_num_heads: int = 8,
        knowledge_num_layers: int = 4,
        knowledge_dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.knowledge_dim = knowledge_dim
        self.knowledge_input_dim = knowledge_input_dim
        self.base_dim = base_dim
        self.bimamba_dim = bimamba_dim
        self.enable_amp_head = enable_amp_head

        self.knowledge_encoder = KnowledgeEnhancedTransformerEncoder(
            input_dim=knowledge_input_dim,
            d_model=knowledge_dim,
            num_heads=knowledge_num_heads,
            num_layers=knowledge_num_layers,
            dropout=knowledge_dropout,
            max_len=max_seq_len,
        )

        self.esm_projector = nn.Sequential(
            nn.Linear(esm_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.knowledge_projector = nn.Sequential(
            nn.Linear(knowledge_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.esm_care = CARE(d_model=base_dim, num_groups=8, preserve_ratio=0.7, dropout=0.1)
        self.knowledge_care = CARE(d_model=base_dim, num_groups=8, preserve_ratio=0.7, dropout=0.1)

        self.esm_bimamba = ProBiMamba(input_dim=base_dim, hidden_dim=bimamba_dim, num_layers=3, state_size=16, dropout=0.1)
        self.knowledge_bimamba = ProBiMamba(input_dim=base_dim, hidden_dim=bimamba_dim, num_layers=3, state_size=16, dropout=0.1)

        self.fusion_care = CrossModalAttention(base_dim, base_dim, num_heads=8)
        self.fusion_bimamba = CrossModalAttention(bimamba_dim, bimamba_dim, num_heads=8)
        self.fusion_cross1 = CrossModalAttention(base_dim, bimamba_dim, num_heads=8)
        self.fusion_cross2 = CrossModalAttention(bimamba_dim, base_dim, num_heads=8)

        final_dim = base_dim * 2 + bimamba_dim * 2
        self.final_norm = nn.LayerNorm(final_dim)
        self.task_classifier = MLPClassifier(final_dim, linsize, num_labels, lindropout)
        self.amp_classifier = MLPClassifier(final_dim, linsize, 1, lindropout) if enable_amp_head else None

    def _masked_mean(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if attention_mask is None:
            return x.mean(dim=1)
        weights = attention_mask.float().unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode(self, esm_features, knowledge_features, attention_mask=None):
        lengths = None
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1)

        contextual_knowledge = self.knowledge_encoder(knowledge_features, lengths=lengths)

        e = self.esm_projector(esm_features)
        k = self.knowledge_projector(contextual_knowledge)

        e_care = self.esm_care(e)
        k_care = self.knowledge_care(k)
        e_mamba = self.esm_bimamba(e)
        k_mamba = self.knowledge_bimamba(k)

        e_care_f, k_care_f = self.fusion_care(e_care, k_care)
        e_mamba_f, k_mamba_f = self.fusion_bimamba(e_mamba, k_mamba)
        e_cross1, k_cross2 = self.fusion_cross1(e_care_f, k_mamba_f)
        e_cross2, k_cross1 = self.fusion_cross2(e_mamba_f, k_care_f)

        pooled = [
            self._masked_mean(e_cross1, attention_mask),
            self._masked_mean(e_cross2, attention_mask),
            self._masked_mean(k_cross1, attention_mask),
            self._masked_mean(k_cross2, attention_mask),
        ]
        final_feat = torch.cat(pooled, dim=-1)
        final_feat = self.final_norm(final_feat)
        return F.dropout(final_feat, p=0.3, training=self.training)

    def forward(self, esm_features, knowledge_features, attention_mask=None, return_embedding=False, return_dict=False):
        final_feat = self.encode(esm_features, knowledge_features, attention_mask)
        task_logits = self.task_classifier(final_feat)
        amp_logits = self.amp_classifier(final_feat) if self.amp_classifier is not None else None

        if return_dict:
            outputs = {
                "task_logits": task_logits,
                "amp_logits": amp_logits,
            }
            if return_embedding:
                outputs["embedding"] = final_feat
            return outputs

        if return_embedding:
            return task_logits, final_feat
        return task_logits
