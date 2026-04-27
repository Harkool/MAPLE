import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import QuadOutputDataset, quad_output_collate_fn
from model import MAPLE, safe_load_checkpoint


def _find_feature_key(sample: Dict, candidate_keys: List[str]) -> str:
    for key in candidate_keys:
        if key in sample:
            return key
    raise ValueError(f"Missing required feature key from: {candidate_keys}")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_model_name(checkpoint_path: str, checkpoint: Dict) -> str:
    explicit_name = checkpoint.get("model_name")
    if explicit_name:
        return str(explicit_name)

    ckpt_args = checkpoint.get("args", {})
    data_pkl = ckpt_args.get("data_pkl")
    if data_pkl:
        return os.path.splitext(os.path.basename(data_pkl))[0]

    checkpoint_dir = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if checkpoint_stem.startswith("best_") and checkpoint_stem.endswith("_model"):
        return checkpoint_dir or "model"
    return checkpoint_stem


def _build_records(raw_features: Dict, label_cols: List[str], esm_key: str, knowledge_key: str) -> pd.DataFrame:
    rows = []
    for seq_hash, content in raw_features.items():
        if esm_key not in content or knowledge_key not in content or "labels" not in content:
            continue
        rec = {"hash": seq_hash, "sequence": content.get("sequence", "")}
        rec.update(dict(zip(label_cols, content["labels"])))
        rows.append(rec)
    if not rows:
        raise ValueError("No valid feature rows found in pkl for evaluation.")
    return pd.DataFrame(rows)


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
    if labels.size < 2:
        return 0.0
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        v = float(matthews_corrcoef(y_true, y_pred))
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro") -> float:
    try:
        v = float(roc_auc_score(y_true, y_prob, average=average))
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro") -> float:
    try:
        v = float(average_precision_score(y_true, y_prob, average=average))
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.reshape(-1)
    y_pred = (y_prob > threshold).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": _specificity(y_true, y_pred),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": _safe_mcc(y_true, y_pred),
        "AUROC": _safe_auroc(y_true, y_prob, average="macro"),
        "AUPRC": _safe_auprc(y_true, y_prob, average="macro"),
    }


def _load_eval_inputs(
    data_pkl: str,
    checkpoint: Dict,
    label_cols: List[str],
) -> Tuple[pd.DataFrame, Dict, str, str]:
    with open(data_pkl, "rb") as f:
        raw_data = pickle.load(f)
    if "features" not in raw_data or not raw_data["features"]:
        raise ValueError("Invalid pkl format: missing non-empty 'features'.")

    sample_features = next(iter(raw_data["features"].values()))
    esm_key = checkpoint.get("esm_key") or _find_feature_key(sample_features, ["esm_features", "esm_embeddings", "esm2_embeddings", "sequence_embedding"])
    knowledge_key = checkpoint.get("knowledge_key") or _find_feature_key(sample_features, ["enhanced_knowledge_features", "knowledge_features"])
    data_df = _build_records(raw_data["features"], label_cols, esm_key, knowledge_key)
    return data_df, raw_data["features"], esm_key, knowledge_key


def _run_inference(model, loader: DataLoader, device: torch.device, label_cols: List[str], disable_tqdm: bool) -> Tuple[np.ndarray, np.ndarray]:
    probs_list, labels_list = [], []
    with torch.no_grad():
        for esm_feat, kn_feat, labels in tqdm(loader, desc="Evaluating", disable=disable_tqdm):
            esm_feat = esm_feat.to(device)
            kn_feat = kn_feat.to(device)
            logits = model(esm_features=esm_feat, knowledge_features=kn_feat)
            probs = torch.sigmoid(logits).cpu().numpy()

            if len(label_cols) == 1:
                labels = labels.view(-1, 1)
            labels = labels.numpy()

            probs_list.append(probs)
            labels_list.append(labels)

    return np.vstack(probs_list), np.vstack(labels_list).astype(int)


def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    step: float = 0.01,
) -> Tuple[float, float]:
    metric = metric.lower()
    best_threshold = 0.5
    best_score = -1.0

    threshold = min_threshold
    while threshold <= max_threshold + 1e-12:
        metrics = _binary_metrics(y_true, y_prob, threshold=threshold)
        score = metrics["F1"] if metric == "f1" else metrics["MCC"]
        if score > best_score:
            best_score = float(score)
            best_threshold = float(round(threshold, 6))
        threshold += step

    return best_threshold, best_score


def evaluate_checkpoint(args):
    device = _resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'.")
    model_name = args.model_name or _infer_model_name(args.checkpoint, checkpoint)

    ckpt_args = checkpoint.get("args", {})
    label_cols = args.label_cols if args.label_cols else ckpt_args.get("label_cols")
    if not label_cols:
        raise ValueError("label_cols not found. Provide --label_cols or ensure checkpoint has args.label_cols")

    data_df, feature_dict, _, _ = _load_eval_inputs(args.data_pkl, checkpoint, label_cols)
    dataset = QuadOutputDataset(data_df, feature_dict=feature_dict)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=quad_output_collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MAPLE(
        linsize=ckpt_args.get("hidden_size", 1024),
        lindropout=ckpt_args.get("dropout", 0.8),
        num_labels=len(label_cols),
        esm_dim=checkpoint.get("esm_dim", 480),
        knowledge_dim=checkpoint.get("knowledge_dim", 256),
    ).to(device)
    safe_load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    y_prob, y_true = _run_inference(model, loader, device, label_cols, disable_tqdm=args.quiet)

    threshold_value: float
    threshold_source = "manual"
    threshold_metric = ""
    threshold_score = np.nan
    if isinstance(args.threshold, str) and args.threshold.lower() == "auto":
        if len(label_cols) != 1:
            raise ValueError("--threshold auto is only supported for single-label checkpoints.")
        search_pkl = args.threshold_search_pkl or ckpt_args.get("data_pkl")
        if not search_pkl:
            raise ValueError("Missing threshold search dataset. Pass --threshold_search_pkl or store args.data_pkl in checkpoint.")
        search_df, search_feature_dict, _, _ = _load_eval_inputs(search_pkl, checkpoint, label_cols)
        search_dataset = QuadOutputDataset(search_df, feature_dict=search_feature_dict)
        search_loader = DataLoader(
            search_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=quad_output_collate_fn,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        search_prob, search_true = _run_inference(model, search_loader, device, label_cols, disable_tqdm=args.quiet)
        threshold_value, threshold_score = _find_best_threshold(
            search_true,
            search_prob,
            metric=args.threshold_metric,
            min_threshold=args.threshold_min,
            max_threshold=args.threshold_max,
            step=args.threshold_step,
        )
        threshold_source = os.path.abspath(search_pkl)
        threshold_metric = args.threshold_metric.lower()
    else:
        threshold_value = float(args.threshold)

    y_pred = (y_prob > threshold_value).astype(int)

    rows = []

    if len(label_cols) == 1:
        m = _binary_metrics(y_true, y_prob, threshold=threshold_value)
        for metric_name, value in m.items():
            rows.append(
                {
                    "scope": "task_binary",
                    "label_name": label_cols[0],
                    "metric": metric_name,
                    "value": value,
                }
            )
    else:
        macro = {
            "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro-Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro-Sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro-AUROC": _safe_auroc(y_true, y_prob, average="macro"),
        }
        for metric_name, value in macro.items():
            rows.append(
                {
                    "scope": "multilabel_macro",
                    "label_name": "",
                    "metric": metric_name,
                    "value": value,
                }
            )

        for i, label_name in enumerate(label_cols):
            per = _binary_metrics(y_true[:, i], y_prob[:, i], threshold=threshold_value)
            for metric_name, value in per.items():
                rows.append(
                    {
                        "scope": "multilabel_label",
                        "label_name": label_name,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.insert(0, "model_name", model_name)
    out_df.insert(1, "samples", len(dataset))
    out_df.insert(2, "threshold", float(threshold_value))

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{model_name}_metrics.csv")
    out_df.to_csv(out_csv, index=False)

    print(f"[INFO] Eval done. samples={len(dataset)}, labels={len(label_cols)}")
    print(f"[INFO] Threshold used: {threshold_value:.4f}")
    if threshold_metric:
        print(f"[INFO] Threshold search metric: {threshold_metric}={threshold_score:.6f}")
    print(f"[INFO] Metrics CSV saved: {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate train.py checkpoints and export one metrics CSV")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_pkl", type=str, required=True)
    parser.add_argument("--label_cols", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=str, default="0.5")
    parser.add_argument("--threshold_search_pkl", type=str, default=None)
    parser.add_argument("--threshold_metric", type=str, choices=["f1", "mcc"], default="f1")
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_out")
    args = parser.parse_args()

    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
