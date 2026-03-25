from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
UNKNOWN_AA_INDEX = len(AA_ORDER)

HYDROPHOBICITY = torch.tensor(
    [0.62, -2.53, -0.78, -0.90, 0.29, -0.85, -0.74, 0.48, -0.40, 1.38,
     1.06, -1.50, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08],
    dtype=torch.float32,
)
CHARGE = torch.tensor(
    [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0,
     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
HELIX = torch.tensor(
    [1.42, 0.98, 0.67, 1.01, 0.70, 1.11, 1.51, 0.57, 1.00, 1.08,
     1.21, 1.16, 1.45, 1.13, 0.57, 0.77, 0.83, 1.08, 0.69, 1.06],
    dtype=torch.float32,
)
MOLECULAR_WEIGHTS = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147, "G": 75,
    "H": 155, "I": 131, "L": 131, "K": 146, "M": 149, "F": 165, "P": 115, "S": 105,
    "T": 119, "W": 204, "Y": 181, "V": 117,
}
BOMAN = {
    "A": 0.61, "R": 0.69, "N": 0.89, "D": 1.15, "C": 1.07, "Q": 0.97, "E": 1.10, "G": 0.84,
    "H": 1.05, "I": -0.31, "L": -0.56, "K": 0.46, "M": -0.23, "F": -0.58, "P": 2.23,
    "S": 0.99, "T": 0.77, "W": 0.37, "Y": 0.24, "V": -0.18,
}


class KnowledgeEnhancedSequenceEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        default_config = {
            "hidden_dim": 128,
            "output_dim": 64,
            "dropout": 0.1,
            "max_seq_length": 1024,
        }
        if config is not None:
            default_config.update(config)
        self.config = default_config
        self.output_dim = self.config["output_dim"]
        self.max_len = self.config["max_seq_length"]

        self.register_buffer("aa_onehot", torch.eye(len(AA_ORDER), dtype=torch.float32))
        self.register_buffer("hydrophobicity", HYDROPHOBICITY)
        self.register_buffer("charge", CHARGE)
        self.register_buffer("helix", HELIX)

        input_dim = 20 + 5 + 15 + 6 + 10
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.config["hidden_dim"]),
            nn.LayerNorm(self.config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(self.config["hidden_dim"], self.config["output_dim"]),
        )

    def forward(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(sequences, str):
            sequences = [sequences]

        device = next(self.parameters()).device
        batch_feats = [self._encode_single_sequence(seq) for seq in sequences]
        padded = torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True, padding_value=0.0)
        return self.proj(padded.to(device))

    def _aa_to_idx(self, aa: str) -> int:
        return AA_TO_IDX.get(aa, UNKNOWN_AA_INDEX)

    def _aa_weight(self, aa: str) -> float:
        return MOLECULAR_WEIGHTS.get(aa, 110.0)

    def _encode_single_sequence(self, seq: str) -> torch.Tensor:
        seq = seq.upper()[: self.max_len]
        if not seq:
            seq = "X"

        indices = torch.tensor([self._aa_to_idx(aa) for aa in seq], dtype=torch.long)
        clamped = indices.clamp(max=len(AA_ORDER) - 1)
        valid_mask = (indices != UNKNOWN_AA_INDEX).float().unsqueeze(1)
        length = len(seq)

        one_hot = self.aa_onehot[clamped] * valid_mask
        hydro = self.hydrophobicity[clamped] * valid_mask.squeeze(1)
        charge = self.charge[clamped] * valid_mask.squeeze(1)
        weight = torch.tensor([self._aa_weight(aa) for aa in seq], dtype=torch.float32) / 200.0
        helix = self.helix[clamped] * valid_mask.squeeze(1)
        aromatic = torch.isin(clamped, torch.tensor([4, 17, 18])).float() * valid_mask.squeeze(1)
        phys = torch.stack([hydro, charge, weight, helix, aromatic], dim=1)

        global_f = torch.tensor(self._global_features(seq), dtype=torch.float32).unsqueeze(0).repeat(length, 1)
        pos = torch.arange(length, dtype=torch.float32) / max(length - 1, 1)
        edge_distance = torch.minimum(
            torch.arange(length, dtype=torch.float32),
            torch.arange(length - 1, -1, -1, dtype=torch.float32),
        ) / max(length - 1, 1)
        pos_feat = torch.stack(
            [
                pos,
                (pos < 0.2).float(),
                (pos > 0.8).float(),
                ((pos >= 0.3) & (pos <= 0.7)).float(),
                edge_distance,
                torch.sin(2 * np.pi * pos),
            ],
            dim=1,
        )
        window_feat = self._window_features_vectorized(indices)
        return torch.cat([one_hot, phys, window_feat, pos_feat, global_f], dim=1)

    def _global_features(self, seq: str) -> List[float]:
        length = max(len(seq), 1)
        net_charge = sum({"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.5}.get(aa, 0) for aa in seq)
        hydro_values = [self.hydrophobicity[self._aa_to_idx(aa)].item() for aa in seq if aa in AA_TO_IDX]
        avg_hydro = sum(hydro_values) / max(len(hydro_values), 1)
        p_i = 7.0 + (seq.count("R") + seq.count("K") - seq.count("D") - seq.count("E")) * 0.5
        mol_weight = sum(self._aa_weight(aa) for aa in seq)
        unique = len(set(seq))
        pos_ratio = sum(aa in "RKH" for aa in seq) / length
        hydro_ratio = sum(aa in "AILMFPWV" for aa in seq) / length
        arom_ratio = sum(aa in "FWY" for aa in seq) / length
        boman = sum(BOMAN.get(aa, 0.0) for aa in seq) / length
        return [
            length / 100.0,
            net_charge / length,
            avg_hydro,
            p_i / 14.0,
            mol_weight / 10000.0,
            unique / 20.0,
            pos_ratio,
            hydro_ratio,
            arom_ratio,
            boman / 3.0,
        ]

    def _window_features_vectorized(self, indices: torch.Tensor, window: int = 5) -> torch.Tensor:
        length = len(indices)
        pad = window // 2
        padded_idx = F.pad(indices, (pad, pad), value=UNKNOWN_AA_INDEX)
        windows = padded_idx.unfold(0, window, 1)
        valid_windows = windows != UNKNOWN_AA_INDEX
        clamped_windows = windows.clamp(max=len(AA_ORDER) - 1)

        feats = []
        groups = [
            "AILMFPWV", "RKDE", "NQSTYC", "FWY", "RKH", "DE", "AGSC", "RWFY",
        ]
        for aas in groups:
            group_indices = torch.tensor([AA_TO_IDX[aa] for aa in aas], dtype=torch.long)
            mask = torch.isin(clamped_windows, group_indices) & valid_windows
            feats.append(mask.float().sum(dim=1) / valid_windows.sum(dim=1).clamp_min(1.0))

        hydro_win = (self.hydrophobicity[clamped_windows] * valid_windows).sum(dim=1) / valid_windows.sum(dim=1).clamp_min(1.0)
        charge_win = (self.charge[clamped_windows] * valid_windows).sum(dim=1) / valid_windows.sum(dim=1).clamp_min(1.0)
        complexity = ((clamped_windows != clamped_windows[:, :1]) & valid_windows).float().sum(dim=1) / valid_windows.sum(dim=1).clamp_min(1.0)
        rk_mask = torch.isin(clamped_windows, torch.tensor([AA_TO_IDX["R"], AA_TO_IDX["K"]])) & valid_windows
        hydrophobic_mask = torch.isin(clamped_windows, torch.tensor([AA_TO_IDX[aa] for aa in "AILMFPWV"])) & valid_windows
        mixed_mask = torch.isin(clamped_windows, torch.tensor([AA_TO_IDX[aa] for aa in "RKDESTNQ"])) & valid_windows
        aromatic_mask = torch.isin(clamped_windows, torch.tensor([AA_TO_IDX["F"], AA_TO_IDX["W"], AA_TO_IDX["Y"]])) & valid_windows

        pos_cluster = (rk_mask.sum(dim=1) >= 2).float()
        hydro_cluster = (hydrophobic_mask.sum(dim=1) >= 3).float()
        amph = ((hydrophobic_mask.sum(dim=1) >= 2) & (mixed_mask.sum(dim=1) >= 2)).float()
        arom_cluster = (aromatic_mask.sum(dim=1) >= 2).float()

        return torch.stack(
            feats + [hydro_win, charge_win, complexity, pos_cluster, hydro_cluster, amph, arom_cluster],
            dim=1,
        )
