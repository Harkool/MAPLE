import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Dict, Any
import esm
from esm import FastaBatchedDataset, pretrained

class ESM2Embedder:
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        repr_layer: int = -1,
        use_half: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_half = use_half and self.device.type == "cuda"
        self.repr_layer = repr_layer

        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model.eval()
        self.model.to(self.device)

        if self.use_half:
            self.model = self.model.half()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = self.model.embed_dim
    @torch.no_grad()
    def embed_sequences(self, sequences: List[str], max_tokens_per_batch: int = 2048) -> List[torch.Tensor]:
        dataset = FastaBatchedDataset(["seq"] * len(sequences), sequences)
        batches = dataset.get_batch_indices(max_tokens_per_batch, extra_toks_per_seq=2)

        results = []
        for batch_idx in batches:
            _, _, toks = self.batch_converter([(f"seq{i}", sequences[i]) for i in batch_idx])
            toks = toks.to(self.device)
            if self.use_half:
                toks = toks.to(torch.bfloat16)

            out = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)
            embeddings = out["representations"][self.repr_layer]

            for i, idx in enumerate(batch_idx):
                seq_len = len(sequences[idx])
                emb = embeddings[i, 1:seq_len + 1].cpu().float()
                results.append(emb)
        return results

class ESM2EmbeddingDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        labels: List[float],
        embedder: Optional[ESM2Embedder] = None,
        precomputed_embeddings: Optional[List[torch.Tensor]] = None,
        device: Optional[str] = None,
    ):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precomputed_embeddings is not None:
            self.embeddings = precomputed_embeddings
        elif embedder is not None:
            self.embeddings = embedder.embed_sequences(sequences)
        else:
            raise ValueError("Must provide embedder or precomputed_embeddings")
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "length": len(self.embeddings[idx])
        }
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        embeddings = [item["embedding"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])
        lengths = torch.tensor([item["length"] for item in batch])
        padded_emb = pad_sequence(embeddings, batch_first=True)
        attention_mask = torch.arange(padded_emb.size(1), device=padded_emb.device)[None, :] < lengths[:, None]
        return {
            "input_ids": padded_emb,
            "attention_mask": attention_mask,
            "labels": labels
        }