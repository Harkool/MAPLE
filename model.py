import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.CARE import CARE
from Module.Probimamba import ProBiMamba
from Module.Fusion import CrossModalAttention
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
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return torch.clamp(self.net(x), min=-10.0, max=10.0)
class MAPLE(nn.Module):
    def __init__(
        self,
        linsize: int = 1024,
        lindropout: float = 0.8,
        num_labels: int = 1,
        esm_dim: int = 1280, 
        knowledge_dim: int = 512,
        base_dim: int = 512, 
        bimamba_dim: int = 256, 
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.knowledge_dim = knowledge_dim
        self.base_dim = base_dim
        self.bimamba_dim = bimamba_dim
        self.esm_projector = nn.Sequential(
            nn.Linear(esm_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.knowledge_projector = nn.Sequential(
            nn.Linear(knowledge_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.esm_care = CARE(d_model=base_dim, num_groups=8, preserve_ratio=0.7, dropout=0.1)
        self.esm_bimamba = ProBiMamba(
            input_dim=base_dim,
            hidden_dim=bimamba_dim,
            num_layers=3,
            state_size=16,
            num_heads=8,
            dropout=0.1,
            use_attention=False
        )
        self.knowledge_care = CARE(d_model=base_dim, num_groups=8, preserve_ratio=0.7, dropout=0.1)
        self.knowledge_bimamba = ProBiMamba(
            input_dim=base_dim,
            hidden_dim=bimamba_dim,
            num_layers=3,
            state_size=16,
            num_heads=8,
            dropout=0.1,
            use_attention=False
        )
        self.esm_bimamba.enable_gradient_checkpointing()
        self.knowledge_bimamba.enable_gradient_checkpointing()
        self.fusion_care   = CrossModalAttention(base_dim, base_dim, num_heads=8)
        self.fusion_bimamba = CrossModalAttention(bimamba_dim, bimamba_dim, num_heads=8)
        self.fusion_cross1 = CrossModalAttention(base_dim, bimamba_dim, num_heads=8)
        self.fusion_cross2 = CrossModalAttention(bimamba_dim, base_dim, num_heads=8)
        final_dim = base_dim * 2 + bimamba_dim * 2
        self.final_norm = nn.LayerNorm(final_dim)
        self.classifier = MLPClassifier(
            input_dim=final_dim,
            hidden_dim=linsize,
            num_classes=num_labels,
            dropout=lindropout
        )

    def forward(self, esm_features, knowledge_features, return_embedding=False):
        e = self.esm_projector(esm_features)
        k = self.knowledge_projector(knowledge_features)
        e_care = self.esm_care(e)
        e_mamba, _ = self.esm_bimamba(e)
        k_care = self.knowledge_care(k)
        k_mamba, _ = self.knowledge_bimamba(k)
        e_care_f, k_care_f = self.fusion_care(e_care, k_care)
        e_mamba_f, k_mamba_f = self.fusion_bimamba(e_mamba, k_mamba)
        e_cross1, k_cross2 = self.fusion_cross1(e_care_f, k_mamba_f)
        e_cross2, k_cross1 = self.fusion_cross2(e_mamba_f, k_care_f)
        p1 = e_cross1.mean(dim=1)
        p2 = e_cross2.mean(dim=1)
        p3 = k_cross1.mean(dim=1)
        p4 = k_cross2.mean(dim=1)
        final_feat = torch.cat([p1, p2, p3, p4], dim=-1) 
        final_feat = self.final_norm(final_feat)
        final_feat = F.dropout(final_feat, p=0.3, training=self.training)
        logits = self.classifier(final_feat)
        if return_embedding:
            return logits, final_feat
        return logits