import argparse
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import QuadOutputDataset, quad_output_collate_fn
from model import build_maple_from_checkpoint
from Generate_pkl import build_unified_pkl


DEFAULT_LABELS_14 = [
    "anti_mammalian_cells",
    "antibacterial",
    "antibiofilm",
    "anticancer",
    "antifungal",
    "antigram-negative",
    "antigram-positive",
    "antihiv",
    "antimrsa",
    "antioxidant",
    "antiparasitic",
    "antiviral",
    "cytotoxic",
    "hemolytic",
]


def _find_feature_key(sample: Dict, candidate_keys: List[str]) -> str:
    for key in candidate_keys:
        if key in sample:
            return key
    raise ValueError(f"Missing required feature key from: {candidate_keys}")


def _build_records(raw_features: Dict, label_cols: List[str], esm_key: str, knowledge_key: str) -> pd.DataFrame:
    rows = []
    for seq_hash, content in raw_features.items():
        if esm_key not in content or knowledge_key not in content or "labels" not in content:
            continue
        rec = {
            "hash": seq_hash,
            "sequence": content.get("sequence", ""),
            **dict(zip(label_cols, content["labels"])),
        }
        rows.append(rec)
    if not rows:
        raise ValueError("No valid feature rows found in generated pkl.")
    return pd.DataFrame(rows)


def _prepare_tmp_csv(input_csv: str, sequence_col: str, label_cols: List[str]) -> str:
    df = pd.read_csv(input_csv)
    if sequence_col not in df.columns:
        raise KeyError(f"Missing sequence column: {sequence_col}")

    for c in label_cols:
        if c not in df.columns:
            df[c] = 0.0

    tmp_fd, tmp_csv = tempfile.mkstemp(prefix="predict_input_", suffix=".csv")
    os.close(tmp_fd)
    df.to_csv(tmp_csv, index=False)
    return tmp_csv


def _infer_with_single_checkpoint(checkpoint_path: str, loader, device: torch.device, num_labels: int):
    built = build_maple_from_checkpoint(checkpoint_path, device=device)
    model = built["model"]
    if int(built["num_labels"]) != int(num_labels):
        raise ValueError(
            f"Checkpoint label count mismatch: expected {num_labels}, got {built['num_labels']} for {checkpoint_path}"
        )

    probs_list = []
    with torch.no_grad():
        for esm_feat, kn_feat, _ in tqdm(loader, desc=f"Predicting: {Path(checkpoint_path).name}"):
            esm_feat = esm_feat.to(device)
            kn_feat = kn_feat.to(device)
            logits = model(esm_features=esm_feat, knowledge_features=kn_feat)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs_list)


def run_predict(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        label_cols = args.label_cols if args.label_cols else ckpt_args.get("label_cols", ["label"])
    else:
        label_cols = args.label_cols if args.label_cols else DEFAULT_LABELS_14

    tmp_csv = _prepare_tmp_csv(args.input_csv, args.sequence_col, label_cols)

    tmp_fd, tmp_pkl = tempfile.mkstemp(prefix="predict_features_", suffix=".pkl")
    os.close(tmp_fd)

    try:
        build_unified_pkl(
            csv_path=Path(tmp_csv),
            output_pkl=Path(tmp_pkl),
            sequence_col=args.sequence_col,
            label_cols=label_cols,
            esm_model_name=args.esm_model,
            max_seq_len=args.max_seq_len,
            device=args.device,
            knowledge_transformer_ckpt=args.knowledge_transformer_ckpt,
            knowledge_dim=args.knowledge_dim,
        )

        with open(tmp_pkl, "rb") as f:
            raw_data = pickle.load(f)

        sample_features = next(iter(raw_data["features"].values()))
        esm_key = _find_feature_key(sample_features, ["esm_features", "esm_embeddings", "esm2_embeddings", "sequence_embedding"])
        knowledge_key = _find_feature_key(sample_features, ["enhanced_knowledge_features", "knowledge_features"])

        data_df = _build_records(raw_data["features"], label_cols, esm_key, knowledge_key)
        dataset = QuadOutputDataset(data_df, feature_dict=raw_data["features"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=quad_output_collate_fn, num_workers=0)

        if args.checkpoint:
            probs = _infer_with_single_checkpoint(args.checkpoint, loader, device, num_labels=len(label_cols))
        else:
            all_probs = []
            for label in label_cols:
                ckpt_path = os.path.join(args.label_dir, f"{label}.pt")
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Missing label checkpoint: {ckpt_path}")
                p = _infer_with_single_checkpoint(ckpt_path, loader, device, num_labels=1).reshape(-1)
                all_probs.append(p)
            probs = np.stack(all_probs, axis=1)

        out = pd.DataFrame({"sequence": data_df["sequence"].tolist()})
        for i, label in enumerate(label_cols):
            out[f"prob_{label}"] = probs[:, i]

        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        print(f"[INFO] Saved prediction CSV: {args.output_csv}")
        print(f"[INFO] samples={len(out)}, labels={len(label_cols)}")

    finally:
        for p in [tmp_csv, tmp_pkl]:
            try:
                os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict probabilities from CSV using train checkpoints or per-label checkpoints")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--sequence_col", type=str, default="sequence")

    parser.add_argument("--checkpoint", type=str, default="", help="Single checkpoint path (train.py output).")
    parser.add_argument("--label_dir", type=str, default="Model/label", help="Directory of per-label checkpoints when --checkpoint is not set.")
    parser.add_argument("--label_cols", nargs="+", default=None)

    parser.add_argument("--knowledge_transformer_ckpt", type=str, default="Model/knowledge_transformer.pt")
    parser.add_argument("--knowledge_dim", type=int, default=256)
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--max_seq_len", type=int, default=700)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    run_predict(args)
