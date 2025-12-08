# Knowledge.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import pickle
from pathlib import Path
import glob


class KnowledgeEnhancedSequenceEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        default_config = {
            'hidden_dim': 128,
            'output_dim':  64,
            'dropout': 0.1,
            'max_seq_length': 1024
        }
        if config is not None:
            default_config.update(config)
        self.config = default_config

        self.output_dim = self.config['output_dim']
        self.max_len = self.config['max_seq_length']

        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        self.register_buffer('aa_onehot', torch.eye(20))
        self.register_buffer('hydrophobicity', torch.tensor([0.62, -2.53, -0.78, -0.90, 0.29, -0.85, -0.74, 0.48,
                                                           -0.40, 1.38, 1.06, -1.50, 0.64, 1.19, 0.12, -0.18,
                                                           -0.05, 0.81, 0.26, 1.08]))  
        self.register_buffer('charge', torch.tensor([0, 1, 0, -1, 0, 0, -1, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.register_buffer('helix', torch.tensor([1.42, 0.98, 0.67, 1.01, 0.70, 1.11, 1.51, 0.57,
                                                    1.00, 1.08, 1.21, 1.16, 1.45, 1.13, 0.57, 0.77,
                                                    0.83, 1.08, 0.69, 1.06]))

        input_dim = 20 + 5 + 15 + 6 + 10 
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.config['hidden_dim']),
            nn.LayerNorm(self.config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['hidden_dim'], self.config['output_dim'])
        )

    def forward(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(sequences, str):
            sequences = [sequences]
        device = next(self.parameters()).device

        batch_feats = []
        for seq in sequences:
            feat = self._encode_single_sequence(seq)
            batch_feats.append(feat)
        padded = torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True, padding_value=0.0)
        return self.proj(padded.to(device))  # (B, L, output_dim)

    def _encode_single_sequence(self, seq: str) -> torch.Tensor:
        seq = seq.upper()
        L = len(seq)
        if L == 0:
            L = 1
        indices = torch.tensor([self._aa_to_idx(c) for c in seq], dtype=torch.long)
        one_hot = self.aa_onehot[indices]  # (L, 20)
        h = self.hydrophobicity[indices]
        c = self.charge[indices]
        w = torch.tensor([self._aa_weight(c) for c in seq]) / 200.0
        helix = self.helix[indices]
        aromatic = torch.isin(indices, torch.tensor([8,17,18]))  # F,W,Y
        phys = torch.stack([h, c, w, helix, aromatic.float()], dim=1)
        global_f = self._global_features(seq)
        global_f = torch.tensor(global_f, dtype=torch.float).unsqueeze(0).repeat(L, 1)
        pos = torch.arange(L, dtype=torch.float) / L
        pos_feat = torch.stack([
            pos,
            (pos < 0.2).float(),
            (pos > 0.8).float(),
            ((pos >= 0.3) & (pos <= 0.7)).float(),
            torch.minimum(torch.arange(L, dtype=torch.float), torch.arange(L-1, -1, -1, dtype=torch.float)) / L,
            torch.sin(2 * np.pi * pos)
        ], dim=1)
        window_feat = self._window_features_vectorized(seq, indices)
        feat = torch.cat([one_hot, phys, window_feat, pos_feat, global_f], dim=1)  # (L, total)
        return feat
    def _aa_to_idx(self, c: str) -> int:
        return self._aa_to_idx.get(c, 20) if hasattr(self, '_aa_to_idx') else 20
    def _aa_weight(self, c: str) -> float:
        weights = {'A':89,'R':174,'N':132,'D':133,'C':121,'Q':146,'E':147,'G':75,'H':155,
                   'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,'T':119,
                   'W':204,'Y':181,'V':117}
        return weights.get(c, 110)
    def _global_features(self, seq: str) -> List[float]:
        L = len(seq)
        net_charge = sum({'R':1,'K':1,'D':-1,'E':-1,'H':0.5}.get(c,0) for c in seq)
        avg_hydro = sum(self.hydrophobicity[self._aa_to_idx(c)].item() for c in seq) / L
        pI = 7.0 + (seq.count('R')+seq.count('K')-seq.count('D')-seq.count('E')) * 0.5
        mol_wt = sum(self._aa_weight(c) for c in seq)
        unique = len(set(seq))
        pos_r = sum(c in 'RKH' for c in seq) / L
        hydro_r = sum(c in 'AILMFPWV' for c in seq) / L
        arom_r = sum(c in 'FWY' for c in seq) / L
        boman = sum({'A':0.61,'R':0.69,'N':0.89,'D':1.15,'C':1.07,'Q':0.97,'E':1.10,'G':0.84,
                     'H':1.05,'I':-0.31,'L':-0.56,'K':0.46,'M':-0.23,'F':-0.58,'P':2.23,
                     'S':0.99,'T':0.77,'W':0.37,'Y':0.24,'V':-0.18}.get(c,0) for c in seq) / L
        return [L/100.0, net_charge/L, avg_hydro, pI/14, mol_wt/10000, unique/20, pos_r, hydro_r, arom_r, boman/3]
    def _window_features_vectorized(self, seq: str, indices: torch.Tensor, window: int = 5) -> torch.Tensor:
        L = len(seq)
        pad = window // 2
        padded_idx = F.pad(indices, (pad, pad), value=20)
        windows = padded_idx.unfold(0, window, 1)  # (L, window)
        feats = []
        for group, aas in [
            ('hydrophobic', 'AILMFPWV'), ('charged', 'RKDE'), ('polar', 'NQSTYC'),
            ('aromatic', 'FWY'), ('positive', 'RKH'), ('negative', 'DE'),
            ('small', 'AGSC'), ('large', 'RWFY')
        ]:
            mask = torch.isin(windows, torch.tensor([self._aa_to_idx(c) for c in aas]))
            feats.append(mask.float().sum(dim=1) / window)

        h_win = self.hydrophobicity[windows].mean(dim=1)
        c_win = self.charge[windows].mean(dim=1)
        complexity = (windows != windows[:, [0]]).float().sum(dim=1) / (window - 1)
        pos_cluster = (windows.isin(torch.tensor([self._aa_to_idx('R'), self._aa_to_idx('K')])).sum(dim=1) >= 2).float()
        hydro_cluster = (torch.isin(windows, torch.tensor([self._aa_to_idx(c) for c in 'AILMFPWV'])).sum(dim=1) >= 3).float()
        amph = ((torch.isin(windows, torch.tensor([self._aa_to_idx(c) for c in 'AILMFPWV'])).sum(dim=1) >= 2) &
                (torch.isin(windows, torch.tensor([self._aa_to_idx(c) for c in 'RKDESTNQ'])).sum(dim=1) >= 2)).float()
        arom_cluster = (windows.isin(torch.tensor([self._aa_to_idx('F'), self._aa_to_idx('W'), self._aa_to_idx('Y')])).sum(dim=1) >= 2).float()
        return torch.stack(feats + [h_win, c_win, complexity, pos_cluster, hydro_cluster, amph, arom_cluster], dim=1)