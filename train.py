import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, hamming_loss, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from data import QuadOutputDataset, quad_output_collate_fn
from loss import FocalLoss
from model import MAPLE


def _infer_checkpoint_name(args) -> str:
    if len(args.label_cols) == 1 and args.label_cols[0] == "label":
        stem = os.path.splitext(os.path.basename(args.data_pkl))[0]
        return f"{stem}.pt"
    return "maple.pt"

def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, average: str = "micro") -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, average=average))
    except Exception:
        return 0.0


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray, average: str = "micro") -> float:
    try:
        return float(average_precision_score(y_true, y_prob, average=average))
    except Exception:
        return 0.0


def _find_feature_key(sample: Dict, candidate_keys: List[str]) -> Tuple[str, int]:
    for key in candidate_keys:
        if key in sample:
            v = sample[key]
            v = v if isinstance(v, np.ndarray) else np.array(v)
            return key, int(v.shape[-1])
    raise ValueError(f"Missing required feature key from: {candidate_keys}")


def _build_records(raw_features: Dict, label_cols: List[str], esm_key: str, knowledge_key: str) -> pd.DataFrame:
    rows = []
    for seq_hash, content in raw_features.items():
        if esm_key not in content or knowledge_key not in content or "labels" not in content:
            continue
        rec = {"hash": seq_hash, "sequence": content.get("sequence", "")}
        rec.update(dict(zip(label_cols, content["labels"])))
        rows.append(rec)
    if not rows:
        raise ValueError("No valid feature rows found in pkl.")
    return pd.DataFrame(rows)


def _prepare_labels_for_loss(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    labels = labels.float()
    if num_labels == 1:
        return labels.view(-1, 1)
    if labels.dim() == 1:
        return labels.unsqueeze(0)
    return labels


def _aggregate_multitask_loss(loss_tensor: torch.Tensor) -> torch.Tensor:
    """Average over labels first, then over batch."""
    if loss_tensor.ndim == 0:
        return loss_tensor
    if loss_tensor.ndim == 1:
        return loss_tensor.mean()
    return loss_tensor.mean(dim=1).mean()


def train_one_epoch(model, loader, criterion, optimizer, device, num_labels: int):
    model.train()
    total_loss = 0.0

    for esm_feat, kn_feat, labels in tqdm(loader, desc="Training"):
        esm_feat = esm_feat.to(device)
        kn_feat = kn_feat.to(device)
        labels = _prepare_labels_for_loss(labels.to(device), num_labels=num_labels)

        optimizer.zero_grad()
        logits = model(esm_features=esm_feat, knowledge_features=kn_feat)

        loss_tensor = criterion(logits, labels)
        if num_labels > 1:
            loss = _aggregate_multitask_loss(loss_tensor)
        else:
            loss = loss_tensor if loss_tensor.ndim == 0 else loss_tensor.mean()

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_labels: int):
    model.eval()
    preds, trues = [], []

    for esm_feat, kn_feat, labels in tqdm(loader, desc="Evaluating"):
        esm_feat = esm_feat.to(device)
        kn_feat = kn_feat.to(device)

        logits = model(esm_features=esm_feat, knowledge_features=kn_feat)
        probs = torch.sigmoid(logits).cpu().numpy()

        labels = _prepare_labels_for_loss(labels, num_labels=num_labels)
        preds.append(probs)
        trues.append(labels.numpy())

    y_prob = np.vstack(preds)
    y_true = np.vstack(trues)
    y_pred = (y_prob > 0.5).astype(int)

    if num_labels == 1:
        yt = y_true.reshape(-1)
        yp = y_pred.reshape(-1)
        ypr = y_prob.reshape(-1)
        metrics = {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision_micro": float(precision_score(yt, yp, zero_division=0)),
            "recall_micro": float(recall_score(yt, yp, zero_division=0)),
            "f1_micro": float(f1_score(yt, yp, zero_division=0)),
            "roc_auc_micro": _safe_roc_auc(yt, ypr, average="micro"),
            "average_precision_micro": _safe_auprc(yt, ypr, average="micro"),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc_micro": _safe_roc_auc(y_true, y_prob, average="micro"),
            "average_precision_micro": _safe_auprc(y_true, y_prob, average="micro"),
        }

    return metrics


def run_single_training(args):
    print("\n========== Start MAPLE Training ==========")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= gpu_count:
            raise ValueError(f"Requested --gpu {args.gpu}, but only {gpu_count} CUDA device(s) are visible.")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    with open(args.data_pkl, "rb") as f:
        raw_data = pickle.load(f)

    if "features" not in raw_data or not raw_data["features"]:
        raise ValueError("Invalid pkl format: missing non-empty 'features'.")

    sample_features = next(iter(raw_data["features"].values()))
    esm_key, esm_dim = _find_feature_key(sample_features, ["esm_features", "esm_embeddings", "esm2_embeddings", "sequence_embedding"])
    knowledge_key, knowledge_dim = _find_feature_key(sample_features, ["enhanced_knowledge_features", "knowledge_features"])
    raw_df = _build_records(raw_data["features"], args.label_cols, esm_key, knowledge_key)
    dataset = QuadOutputDataset(raw_df, feature_dict=raw_data["features"])
    num_labels = len(args.label_cols)
    train_idx = list(range(len(dataset)))
    train_targets = raw_df.iloc[train_idx][args.label_cols].astype(np.float32).values
    focal_reduction = "mean" if num_labels == 1 else "none"
    criterion, alpha_cfg, pos_weight_cfg = FocalLoss.from_targets(
        train_targets,
        gamma=args.focal_gamma,
        reduction=focal_reduction,
    )
    criterion = criterion.to(device)
    print(f"[INFO] Loss: FocalLoss(gamma={args.focal_gamma}, reduction='{focal_reduction}')")
    print(f"[INFO] Focal alpha={alpha_cfg}")
    print(f"[INFO] Focal pos_weight={pos_weight_cfg}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=quad_output_collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    eval_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=quad_output_collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    model = MAPLE(
        linsize=args.hidden_size,
        lindropout=args.dropout,
        num_labels=num_labels,
        esm_dim=esm_dim,
        knowledge_dim=knowledge_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    best_f1, best_epoch = None, 0
    last_eval_metrics = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, num_labels=num_labels)
        eval_metrics = evaluate(model, eval_loader, device, num_labels=num_labels)
        last_eval_metrics = eval_metrics

        current_f1 = float(eval_metrics["f1_micro"] if num_labels == 1 else eval_metrics["f1_macro"])
        if num_labels == 1:
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Eval F1: {current_f1:.4f} | Eval Acc: {eval_metrics['accuracy']:.4f} | "
                f"Eval P/R: {eval_metrics['precision_micro']:.4f}/{eval_metrics['recall_micro']:.4f}"
            )
        else:
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Eval F1-macro: {current_f1:.4f} | Eval P/R-macro: {eval_metrics['precision_macro']:.4f}/{eval_metrics['recall_macro']:.4f} | "
                f"Eval F1-micro: {eval_metrics['f1_micro']:.4f} | Eval Hamming Score: {1.0 - eval_metrics['hamming_loss']:.4f}"
            )

        scheduler.step(current_f1)

        if best_f1 is None or current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch

            save_name = _infer_checkpoint_name(args)
            model_path = os.path.join(args.save_dir, save_name)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "eval_metrics": eval_metrics,
                    "eval_name": "train",
                    "esm_dim": esm_dim,
                    "knowledge_dim": knowledge_dim,
                    "esm_key": esm_key,
                    "knowledge_key": knowledge_key,
                    "train_indices": train_idx,
                    "val_indices": [],
                    "args": vars(args),
                },
                model_path,
            )
            print(f"✅ Saved Model: {save_name} (F1={best_f1:.4f})")

    print("\n" + "=" * 60)
    print("Training finished")
    print("=" * 60)
    print(f"Best Eval F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Train size: {len(train_idx)}")
    print(f"ESM dim: {esm_dim} | Knowledge dim: {knowledge_dim}")

    results_file = os.path.join(args.save_dir, "quad_output_results.json")
    results_summary = {
        "model_type": "MAAPLE",
        "esm_dim": esm_dim,
        "knowledge_dim": knowledge_dim,
        "esm_key": esm_key,
        "knowledge_key": knowledge_key,
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "total_epochs": int(epoch),
        "train_size": int(len(train_idx)),
        "eval_name": "train",
        "final_metrics": last_eval_metrics or {},
        "args": vars(args),
    }

    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Train MAPLE model")
    parser.add_argument("--data_pkl", type=str, required=True, help="Path to feature .pkl")
    parser.add_argument("--label_cols", nargs="+", required=True, help="Label column names")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler_patience", type=int, default=7)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        result = run_single_training(args)
        print(f"✅ Done. Best F1: {result['best_f1']:.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
