import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from Module.CARE import CARE
from Module.ProBiMamba import ProBiMamba
from Module.Fusion import CrossModalAttention


class MLPClassifier(nn.Module):
    """Multi-layer classifier compatible with previous checkpoints."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return torch.clamp(self.model(x), min=-10.0, max=10.0)


class MAPLE(nn.Module):
    """Dual-input / four-path architecture with CARE + ProBiMamba."""

    def __init__(
        self,
        linsize: int = 1024,
        lindropout: float = 0.8,
        num_labels: int = 1,
        esm_dim: int = 480,
        knowledge_dim: int = 256,
    ):
        super().__init__()

        self.hidden_size = 480
        self.probimamba_dim = 240
        self.esm_dim = esm_dim
        self.knowledge_dim = knowledge_dim

        print(
            f"[INFO] Build dual-input model with CARE + ProBiMamba: "
            f"ESM dim={esm_dim}, Knowledge dim={knowledge_dim}"
        )

        self.esm_projector = nn.Sequential(
            nn.Linear(esm_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.knowledge_projector = nn.Sequential(
            nn.Linear(knowledge_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ESM branch: CARE + ProBiMamba
        self.esm_care = CARE(d_model=self.hidden_size, num_groups=2, preserve_ratio=0.7, dropout=0.1)
        self.esm_probimamba = ProBiMamba(
            input_dim=self.hidden_size,
            hidden_dim=self.probimamba_dim,
            num_layers=2,
            state_size=16,
            num_heads=4,
            dropout=0.1,
            use_attention=False,
        )
        self.esm_probimamba.enable_gradient_checkpointing()

        # Knowledge branch: CARE + ProBiMamba
        self.knowledge_care = CARE(d_model=self.hidden_size, num_groups=2, preserve_ratio=0.7, dropout=0.1)
        self.knowledge_probimamba = ProBiMamba(
            input_dim=self.hidden_size,
            hidden_dim=self.probimamba_dim,
            num_layers=2,
            state_size=16,
            num_heads=4,
            dropout=0.1,
            use_attention=False,
        )
        self.knowledge_probimamba.enable_gradient_checkpointing()

        self.norm_esm = nn.LayerNorm(self.hidden_size)
        self.norm_knowledge = nn.LayerNorm(self.hidden_size)

        self.norm_esm_care = nn.LayerNorm(self.hidden_size)
        self.norm_esm_probimamba = nn.LayerNorm(self.probimamba_dim)

        self.norm_knowledge_care = nn.LayerNorm(self.hidden_size)
        self.norm_knowledge_probimamba = nn.LayerNorm(self.probimamba_dim)

        # Cross-modal fusion over four paths
        self.fusion_care = CrossModalAttention(self.hidden_size, self.hidden_size, num_heads=8)
        self.fusion_probimamba = CrossModalAttention(self.probimamba_dim, self.probimamba_dim, num_heads=8)
        self.fusion_cross1 = CrossModalAttention(self.hidden_size, self.probimamba_dim, num_heads=8)
        self.fusion_cross2 = CrossModalAttention(self.probimamba_dim, self.hidden_size, num_heads=8)

        final_dim = self.hidden_size + self.probimamba_dim + self.hidden_size + self.probimamba_dim

        self.norm_final = nn.LayerNorm(final_dim)
        self.classify = MLPClassifier(
            input_dim=final_dim,
            hidden_dim=linsize,
            num_classes=num_labels,
            dropout=lindropout,
        )

        print(f"[INFO] Final feature dim: {final_dim}")
        print(
            f"[INFO] Path dims: ESM_CARE={self.hidden_size}, ESM_ProBiMamba={self.probimamba_dim}, "
            f"Knowledge_CARE={self.hidden_size}, Knowledge_ProBiMamba={self.probimamba_dim}"
        )

    def forward(self, esm_features=None, knowledge_features=None, return_embedding=False, return_attention=False):
        if self.training:
            torch.cuda.empty_cache()

        if esm_features is None or knowledge_features is None:
            raise ValueError("Both esm_features and knowledge_features are required.")

        esm_x = self.esm_projector(esm_features)
        knowledge_x = self.knowledge_projector(knowledge_features)

        esm_x = self.norm_esm(esm_x)
        knowledge_x = self.norm_knowledge(knowledge_x)

        esm_care = self.esm_care(esm_x)
        esm_care = self.norm_esm_care(esm_care + esm_x)

        esm_probimamba, _ = self.esm_probimamba(esm_x)
        esm_probimamba = self.norm_esm_probimamba(esm_probimamba)

        knowledge_care = self.knowledge_care(knowledge_x)
        knowledge_care = self.norm_knowledge_care(knowledge_care + knowledge_x)

        knowledge_probimamba, _ = self.knowledge_probimamba(knowledge_x)
        knowledge_probimamba = self.norm_knowledge_probimamba(knowledge_probimamba)

        esm_care_enhanced, knowledge_care_enhanced = self.fusion_care(esm_care, knowledge_care)
        esm_probimamba_enhanced, knowledge_probimamba_enhanced = self.fusion_probimamba(
            esm_probimamba, knowledge_probimamba
        )

        esm_care_cross, knowledge_probimamba_cross = self.fusion_cross1(
            esm_care_enhanced, knowledge_probimamba_enhanced
        )
        esm_probimamba_cross, knowledge_care_cross = self.fusion_cross2(
            esm_probimamba_enhanced, knowledge_care_enhanced
        )

        esm_care_pooled = esm_care_cross.mean(dim=1)
        esm_probimamba_pooled = esm_probimamba_cross.mean(dim=1)
        knowledge_care_pooled = knowledge_care_cross.mean(dim=1)
        knowledge_probimamba_pooled = knowledge_probimamba_cross.mean(dim=1)

        final_features = torch.cat(
            [
                esm_care_pooled,
                esm_probimamba_pooled,
                knowledge_care_pooled,
                knowledge_probimamba_pooled,
            ],
            dim=-1,
        )

        final_features = self.norm_final(final_features)
        final_features = F.dropout(final_features, p=0.3, training=self.training)

        logits = self.classify(final_features)

        results = [logits]
        if return_embedding:
            results.append(final_features)
        if return_attention:
            results.append(None)
        return results[0] if len(results) == 1 else tuple(results)


def _remap_legacy_keys_to_new(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map old module names (ScConv/ProMamba) to new names (CARE/ProBiMamba)."""

    prefix_map = {
        "esm_scconv.": "esm_care.",
        "knowledge_scconv.": "knowledge_care.",
        "esm_bimamba.": "esm_probimamba.",
        "knowledge_bimamba.": "knowledge_probimamba.",
        "norm_esm_scconv.": "norm_esm_care.",
        "norm_knowledge_scconv.": "norm_knowledge_care.",
        "norm_esm_bimamba.": "norm_esm_probimamba.",
        "norm_knowledge_bimamba.": "norm_knowledge_probimamba.",
        "fusion_scconv.": "fusion_care.",
        "fusion_bimamba.": "fusion_probimamba.",
    }

    remapped = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in prefix_map.items():
            if nk.startswith(old):
                nk = new + nk[len(old) :]
                break
        remapped[nk] = v
    return remapped


def safe_load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load checkpoint with legacy-to-new key remapping and strict shape filtering."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    state_dict = _remap_legacy_keys_to_new(state_dict)

    model_dict = model.state_dict()
    filtered_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_dict and value.shape == model_dict[key].shape
    }

    print(f"[INFO] Matched pretrained params: {len(filtered_dict)}/{len(state_dict)}")
    print(f"[INFO] Model total params: {len(model_dict)}")

    model.load_state_dict(filtered_dict, strict=False)
    return model
