import argparse
import hashlib
import math
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm
from Module.knowledge_transformer import load_trained_knowledge_transformer

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Deterministic physicochemical table (5 dims).
HYDRO = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
CHARGE = {"R": 1.0, "K": 1.0, "D": -1.0, "E": -1.0, "H": 0.5}
WEIGHT = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121,
    "Q": 146, "E": 147, "G": 75, "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115,
    "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
}
HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}


def load_esm_model(model_name: str, device: torch.device):
    if model_name != "esm2_t12_35M_UR50D":
        raise ValueError("Simplified generator only supports esm2_t12_35M_UR50D (480-dim).")

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval().to(device)
    return model, alphabet.get_batch_converter(), 480


def embed_sequence_esm(sequence: str, esm_model, batch_converter, device: torch.device) -> torch.Tensor:
    _, _, tokens = batch_converter([("seq", sequence)])
    tokens = tokens.to(device)
    with torch.no_grad():
        layer = esm_model.num_layers if hasattr(esm_model, "num_layers") else 12
        out = esm_model(tokens, repr_layers=[layer])
        reps = out["representations"][layer][0]
        emb = reps[1 : len(sequence) + 1].detach().cpu().float()
    return emb


def _global_descriptor(sequence: str) -> np.ndarray:
    seq = list(sequence)
    n = max(len(seq), 1)
    hyd = np.mean([HYDRO.get(a, 0.0) for a in seq])
    cha = np.mean([CHARGE.get(a, 0.0) for a in seq])
    wei = np.mean([WEIGHT.get(a, 120.0) for a in seq]) / 200.0
    hel = np.mean([HELIX.get(a, 1.0) for a in seq])

    aromatic = sum(a in set("FWY") for a in seq) / n
    polar = sum(a in set("NQSTYC") for a in seq) / n
    hydrophobic = sum(a in set("AILMFPWV") for a in seq) / n
    charged = sum(a in set("RKDEH") for a in seq) / n
    positive = sum(a in set("RKH") for a in seq) / n
    negative = sum(a in set("DE") for a in seq) / n

    return np.array([hyd, cha, wei, hel, aromatic, polar, hydrophobic, charged, positive, negative], dtype=np.float32)


def _position_descriptor(i: int, n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(6, dtype=np.float32)
    p = i / (n - 1)
    return np.array([
        p,
        1.0 - p,
        np.sin(np.pi * p),
        np.cos(np.pi * p),
        np.sin(2 * np.pi * p),
        np.cos(2 * np.pi * p),
    ], dtype=np.float32)


def _local_window_descriptor(sequence: str, i: int, window: int = 3) -> np.ndarray:
    n = len(sequence)
    l = max(0, i - window)
    r = min(n, i + window + 1)
    w = sequence[l:r]
    d = np.zeros(15, dtype=np.float32)

    d[0] = len(w) / (2 * window + 1)
    d[1] = sum(a in set("FWY") for a in w) / max(len(w), 1)
    d[2] = sum(a in set("RKDEH") for a in w) / max(len(w), 1)
    d[3] = sum(a in set("AILMFPWV") for a in w) / max(len(w), 1)
    d[4] = sum(a in set("NQSTYC") for a in w) / max(len(w), 1)
    d[5] = np.mean([HYDRO.get(a, 0.0) for a in w]) if w else 0.0
    d[6] = np.mean([CHARGE.get(a, 0.0) for a in w]) if w else 0.0
    d[7] = np.mean([HELIX.get(a, 1.0) for a in w]) if w else 0.0

    center = sequence[i]
    left = sequence[i - 1] if i - 1 >= 0 else "X"
    right = sequence[i + 1] if i + 1 < n else "X"
    d[8] = float(center in set("RKDEH"))
    d[9] = float(center in set("FWY"))
    d[10] = float(left == center)
    d[11] = float(right == center)
    d[12] = float(left in set("AILMFPWV"))
    d[13] = float(right in set("AILMFPWV"))
    d[14] = float(i == 0 or i == n - 1)

    return d


def build_raw56_knowledge(sequence: str) -> np.ndarray:
    n = len(sequence)
    if n == 0:
        return np.zeros((1, 56), dtype=np.float32)

    g = _global_descriptor(sequence)
    feats = []
    for i, aa in enumerate(sequence):
        onehot = np.zeros(20, dtype=np.float32)
        onehot[AA_TO_IDX.get(aa, 0)] = 1.0

        phys = np.array([
            HYDRO.get(aa, 0.0),
            CHARGE.get(aa, 0.0),
            WEIGHT.get(aa, 120.0) / 200.0,
            HELIX.get(aa, 1.0),
            float(aa in set("FWY")),
        ], dtype=np.float32)

        loc = _local_window_descriptor(sequence, i)
        pos = _position_descriptor(i, n)

        # 20 + 5 + 15 + 6 + 10 = 56
        feats.append(np.concatenate([onehot, phys, loc, pos, g], axis=0))

    return np.stack(feats, axis=0).astype(np.float32)


def expand_knowledge_dim(raw56: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim == 56:
        return raw56
    if target_dim < 56:
        return raw56[:, :target_dim]

    rep = int(math.ceil(target_dim / 56))
    tiled = np.tile(raw56, (1, rep))
    return tiled[:, :target_dim].astype(np.float32)


def encode_knowledge_with_transformer(
    raw56: np.ndarray,
    transformer_encoder: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(raw56, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([raw56.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = transformer_encoder(x, lengths=lengths)[0].detach().cpu().numpy()
    return out.astype(np.float32)


def normalize_sequence(seq: str, max_seq_len: int) -> str:
    return str(seq).strip().upper()[:max_seq_len]


def sequence_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:16]


def build_unified_pkl(
    csv_path: Path,
    output_pkl: Path,
    sequence_col: str,
    label_cols: List[str],
    esm_model_name: str,
    max_seq_len: int,
    device: str,
    knowledge_transformer_ckpt: str = "",
    knowledge_dim: int = 256,
):
    df = pd.read_csv(csv_path)
    if sequence_col not in df.columns:
        raise KeyError(f"Missing sequence column: {sequence_col}")
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        raise KeyError(f"Missing label columns: {missing_labels}")

    run_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    esm_model, batch_converter, esm_dim = load_esm_model(esm_model_name, run_device)
    if esm_dim != 480:
        raise ValueError(f"ESM dim mismatch: generator={esm_dim}. Expected 480.")

    knowledge_encoder = None
    knowledge_source = "deterministic_expand"
    # Only used when no transformer checkpoint is provided.
    target_knowledge_dim = int(knowledge_dim)

    if knowledge_transformer_ckpt:
        if not Path(knowledge_transformer_ckpt).exists():
            raise FileNotFoundError(f"knowledge_transformer_ckpt not found: {knowledge_transformer_ckpt}")
        knowledge_encoder = load_trained_knowledge_transformer(knowledge_transformer_ckpt, device=str(run_device)).to(run_device)
        knowledge_encoder.eval()
        knowledge_source = "trained_transformer"
        print(f"[INFO] Knowledge source: trained_transformer ({knowledge_transformer_ckpt})")
    else:
        print(f"[INFO] Knowledge source: deterministic_expand (target_dim={target_knowledge_dim})")

    features: Dict[str, Dict] = {}
    valid_rows = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating features"):
        seq = normalize_sequence(row[sequence_col], max_seq_len=max_seq_len)
        if not seq:
            continue

        seq_h = sequence_hash(seq)
        esm_feat = embed_sequence_esm(seq, esm_model, batch_converter, run_device)

        raw56 = build_raw56_knowledge(seq)
        if knowledge_encoder is not None:
            kn_feat = encode_knowledge_with_transformer(raw56, knowledge_encoder, run_device)
        else:
            kn_feat = expand_knowledge_dim(raw56, target_dim=target_knowledge_dim)

        min_len = min(esm_feat.shape[0], kn_feat.shape[0])
        if min_len <= 0:
            continue

        labels = [float(row[c]) for c in label_cols]
        features[seq_h] = {
            "hash": seq_h,
            "sequence": seq,
            "labels": labels,
            "esm_features": esm_feat[:min_len].numpy(),
            "enhanced_knowledge_features": kn_feat[:min_len],
            "esm_shape": list(esm_feat[:min_len].shape),
            "enhanced_knowledge_shape": list(kn_feat[:min_len].shape),
        }
        valid_rows += 1

    output = {
        "metadata": {
            "source_csv": str(csv_path),
            "num_samples": valid_rows,
            "sequence_col": sequence_col,
            "label_cols": label_cols,
            "esm_model": esm_model_name,
            "esm_dim": 480,
            "knowledge_dim": target_knowledge_dim,
            "knowledge_base_dim": 56,
            "knowledge_source": knowledge_source,
            "knowledge_transformer_ckpt": knowledge_transformer_ckpt or None,
            "format_version": 3,
        },
        "features": features,
    }

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(output, f)

    print(f"[INFO] Saved: {output_pkl}")
    print(f"[INFO] Samples: {valid_rows}")
    print(f"[INFO] Export dims: esm=480, knowledge={target_knowledge_dim}")


def main():
    parser = argparse.ArgumentParser(
        description="Simplified feature generator for train.py input pkl (esm=480 + explicit knowledge dim)."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_pkl", type=str, required=True)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--label_cols", nargs="+", required=True)
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--max_seq_len", type=int, default=700)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--knowledge_dim", type=int, default=256)
    parser.add_argument(
        "--knowledge_transformer_ckpt",
        type=str,
        default="",
        help="Optional standalone knowledge transformer checkpoint (56->256).",
    )
    args = parser.parse_args()


    build_unified_pkl(
        csv_path=Path(args.input_csv),
        output_pkl=Path(args.output_pkl),
        sequence_col=args.sequence_col,
        label_cols=args.label_cols,
        esm_model_name=args.esm_model,
        max_seq_len=args.max_seq_len,
        device=args.device,
        knowledge_transformer_ckpt=args.knowledge_transformer_ckpt,
        knowledge_dim=args.knowledge_dim,
    )


if __name__ == "__main__":
    main()
