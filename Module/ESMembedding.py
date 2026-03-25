import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import esm  # type: ignore
    from esm import FastaBatchedDataset, pretrained  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    esm = None
    FastaBatchedDataset = None
    pretrained = None


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
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


class _FallbackESMModel(nn.Module):
    def __init__(self):
        super().__init__()


class ESM2Embedder:
    """ESM-2 sequence embedder with a deterministic offline fallback.

    The fallback keeps the MAPLE dual-stream design runnable even when the
    optional `fair-esm` package or pretrained weights are unavailable.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        repr_layer: int = 33,
        use_half: bool = True,
        fallback_dim: int = 320,
        prefer_pretrained: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.repr_layer = repr_layer
        self.use_half = use_half and self.device.type == "cuda"
        self.fallback_dim = fallback_dim
        self.using_pretrained = False
        self.model = _FallbackESMModel().to(self.device)
        self.alphabet = None
        self.batch_converter = None
        self.embed_dim = fallback_dim

        if prefer_pretrained and pretrained is not None:
            try:
                self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
                self.model.eval().to(self.device)
                if self.use_half:
                    self.model = self.model.half()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embed_dim = int(self.model.embed_dim)
                self.using_pretrained = True
            except Exception as exc:  # pragma: no cover - network/weights dependent
                print(
                    f"[ESM2Embedder] Falling back to deterministic local features because "
                    f"the pretrained ESM model could not be loaded: {exc}"
                )

    def _fallback_embed_sequence(self, sequence: str) -> torch.Tensor:
        seq = sequence.strip().upper()
        if not seq:
            seq = "X"

        indices = torch.tensor([AA_TO_INDEX.get(aa, len(AA_ORDER)) for aa in seq], dtype=torch.long)
        clamped = indices.clamp(max=len(AA_ORDER) - 1)
        one_hot = F.one_hot(clamped, num_classes=len(AA_ORDER)).float()
        unknown_mask = (indices == len(AA_ORDER)).float().unsqueeze(1)
        one_hot = one_hot * (1.0 - unknown_mask)

        hydro = HYDROPHOBICITY[clamped].unsqueeze(1) * (1.0 - unknown_mask)
        charge = CHARGE[clamped].unsqueeze(1) * (1.0 - unknown_mask)
        helix = HELIX[clamped].unsqueeze(1) * (1.0 - unknown_mask)
        aromatic = torch.isin(clamped, torch.tensor([4, 17, 18])).float().unsqueeze(1) * (1.0 - unknown_mask)
        seq_len = len(seq)
        positions = torch.arange(seq_len, dtype=torch.float32)
        relative_pos = (positions / max(seq_len - 1, 1)).unsqueeze(1)
        sinusoid = torch.stack(
            [
                torch.sin(2 * math.pi * relative_pos.squeeze(1)),
                torch.cos(2 * math.pi * relative_pos.squeeze(1)),
            ],
            dim=1,
        )

        base = torch.cat([one_hot, hydro, charge, helix, aromatic, relative_pos, sinusoid], dim=1)
        repeats = math.ceil(self.fallback_dim / base.size(1))
        expanded = base.repeat(1, repeats)[:, : self.fallback_dim]
        scale = torch.linspace(0.8, 1.2, steps=self.fallback_dim, dtype=torch.float32).unsqueeze(0)
        return expanded * scale

    @torch.no_grad()
    def embed_sequences(self, sequences: List[str], max_tokens_per_batch: int = 2048) -> List[torch.Tensor]:
        if self.using_pretrained:
            assert self.batch_converter is not None
            assert FastaBatchedDataset is not None
            dataset = FastaBatchedDataset([f"seq{i}" for i in range(len(sequences))], sequences)
            batches = dataset.get_batch_indices(max_tokens_per_batch, extra_toks_per_seq=2)

            results = [None] * len(sequences)
            for batch_idx in batches:
                batch = [(f"seq{i}", sequences[i]) for i in batch_idx]
                _, _, toks = self.batch_converter(batch)
                toks = toks.to(self.device)
                out = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)
                embeddings = out["representations"][self.repr_layer]

                for local_idx, seq_idx in enumerate(batch_idx):
                    seq_len = len(sequences[seq_idx])
                    emb = embeddings[local_idx, 1 : seq_len + 1].detach().cpu().float()
                    results[seq_idx] = emb

            return [emb for emb in results if emb is not None]

        return [self._fallback_embed_sequence(sequence) for sequence in sequences]
