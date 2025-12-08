# data.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re
from typing import Union, List, Dict
from ESMembedding import ESMEmbedder
from Knowledge import KnowledgeEnhancedSequenceEncoder
from Transformer import KnowledgeEnhancedTransformerEncoder


def create_knowledge_enhanced_transformer(config_name: str = "base"):
    configs = {
        "base": {
            "input_dim": 64,
            "d_model": 512,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_len": 1024
        }
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    return KnowledgeEnhancedTransformerEncoder(**configs[config_name])


class UnifiedProteinDataset(Dataset):
    def __init__(
        self,
        csv_file: Union[str, pd.DataFrame],
        sequence_col: str = "sequence",
        label_cols: Union[str, List[str]] = "label",
        max_seq_len: int = 1024,
        device = None,
        transformer_config_name: str = "base",
    ):
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = csv_file.copy()

        df.columns = [c.strip().lower() for c in df.columns]
        sequence_col = sequence_col.lower()
        label_cols = [label_cols] if isinstance(label_cols, str) else [c.lower() for c in label_cols]

        assert sequence_col in df.columns
        for c in label_cols:
            assert c in df.columns

        df["sequence"] = df[sequence_col].astype(str).str.strip().str.upper()
        valid = df["sequence"].apply(lambda x: bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZOU\-]*", x)))
        df = df[valid].reset_index(drop=True)

        self.sequences = df["sequence"].tolist()
        self.labels = df[label_cols].values.astype(float)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = max_seq_len
        self.label_cols = label_cols
        self.num_labels = len(label_cols)

        self.esm_embedder = ESMEmbedder(device=self.device)

        self.base_knowledge_encoder = KnowledgeEnhancedSequenceEncoder(
            config={"hidden_dim": 128, "output_dim": 64, "dropout": 0.1, "max_seq_length": max_seq_len}
        ).to(self.device)
        self.base_knowledge_encoder.eval()

        self.enhanced_knowledge_encoder = create_knowledge_enhanced_transformer(transformer_config_name).to(self.device)
        self.enhanced_knowledge_encoder.eval()
        self.knowledge_dim = self.enhanced_knowledge_encoder.d_model

        for m in (self.esm_embedder.model, self.base_knowledge_encoder, self.enhanced_knowledge_encoder):
            for p in m.parameters():
                p.requires_grad = False

    def __len__(self):
        return len(self.sequences)

    @torch.no_grad()
    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_seq_len]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        seq_len = len(seq)

        esm_out = self.esm_embedder.embed_sequences([seq])
        esm_feat = esm_out[0] if isinstance(esm_out, (list, tuple)) else esm_out

        base_feat = self.base_knowledge_encoder(seq)[0]

        enhanced_feat = self.enhanced_knowledge_encoder(
            base_feat.unsqueeze(0).to(self.device),
            lengths=torch.tensor([seq_len], device=self.device)
        ).squeeze(0)[:seq_len]

        min_len = min(esm_feat.size(0), enhanced_feat.size(0))
        esm_feat = esm_feat[:min_len].to(self.device)
        enhanced_feat = enhanced_feat[:min_len].to(self.device)

        return {
            "esm": esm_feat,
            "knowledge": enhanced_feat,
            "labels": labels,
            "length": min_len
        }

    @staticmethod
    def collate_fn(batch: List[Dict]):
        esm = pad_sequence([b["esm"] for b in batch], batch_first=True)
        know = pad_sequence([b["knowledge"] for b in batch], batch_first=True)
        labels = torch.stack([b["labels"] for b in batch])
        lengths = torch.tensor([b["length"] for b in batch])

        max_l = esm.size(1)
        mask = torch.arange(max_l, device=esm.device)[None, :] < lengths[:, None]

        return {
            "esm": esm,
            "knowledge": know,
            "attention_mask": mask,
            "labels": labels
        }