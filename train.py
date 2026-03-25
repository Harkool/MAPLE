import argparse
import json
import os
import random
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data import UnifiedProteinDataset
from loss import FocalLoss
from model import MAPLE


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_binary_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_binary_ap(y_true, y_prob):
    if np.sum(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_prob))


def safe_multilabel_roc_auc_micro(y_true, y_prob):
    y_true_flat = y_true.ravel()
    if len(np.unique(y_true_flat)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob, average="micro"))


def safe_multilabel_ap_micro(y_true, y_prob):
    if np.sum(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_prob, average="micro"))


def compute_class_weights(train_targets: np.ndarray, max_pos_weight: float = 50.0):
    eps = 1e-8
    train_targets = np.asarray(train_targets).astype(np.float32)

    pos_counts = train_targets.sum(axis=0)
    total = train_targets.shape[0]
    neg_counts = total - pos_counts

    alpha = neg_counts / (pos_counts + neg_counts + eps)
    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)

    degenerate = (pos_counts == 0) | (neg_counts == 0)
    alpha[degenerate] = 0.5
    pos_weight[degenerate] = 1.0

    alpha = np.clip(alpha, 0.05, 0.95)
    pos_weight = np.clip(pos_weight, 1.0, max_pos_weight)

    stats = {
        "positive_count": pos_counts.astype(int).tolist(),
        "negative_count": neg_counts.astype(int).tolist(),
        "alpha": alpha.astype(float).tolist(),
        "pos_weight": pos_weight.astype(float).tolist(),
    }

    return (
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(pos_weight, dtype=torch.float32),
        stats,
    )


def aggregate_multilabel_task_loss(loss_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Aggregate multi-label loss as: mean over label-dim per sample, then mean over batch."""
    if loss_tensor.ndim != 2:
        raise ValueError(f"Expected loss_tensor shape [B, L], got {tuple(loss_tensor.shape)}")

    if mask is None:
        per_sample_loss = loss_tensor.mean(dim=1)
        final_loss = per_sample_loss.mean()
        return final_loss

    mask = mask.to(loss_tensor.device).float()
    if mask.shape != loss_tensor.shape:
        raise ValueError(f"Expected mask shape {tuple(loss_tensor.shape)}, got {tuple(mask.shape)}")
    valid_count = mask.sum(dim=1).clamp_min(1)
    per_sample_loss = (loss_tensor * mask).sum(dim=1) / valid_count
    final_loss = per_sample_loss.mean()
    return final_loss


def train_one_epoch(model, loader, task_criterion, amp_criterion, optimizer, device, task_loss_weight, amp_loss_weight):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        esm = batch["esm"].to(device, non_blocking=True)
        know = batch["knowledge"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()
        amp_labels = batch["amp_label"].to(device, non_blocking=True).float()
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(
            esm_features=esm,
            knowledge_features=know,
            attention_mask=attention_mask,
            return_dict=True,
        )
        task_loss_matrix = task_criterion(outputs["task_logits"], labels)
        if labels.dim() > 1 and labels.size(1) > 1:
            if task_loss_matrix.ndim != 2:
                raise ValueError(f"Expected multi-label loss tensor shape [B, L], got {tuple(task_loss_matrix.shape)}")
            task_loss = aggregate_multilabel_task_loss(task_loss_matrix)
        else:
            task_loss = task_loss_matrix.mean()
        amp_loss = amp_criterion(outputs["amp_logits"], amp_labels)
        loss = task_loss_weight * task_loss + amp_loss_weight * amp_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def collect_predictions(model, loader, device):
    model.eval()
    task_prob_list = []
    task_label_list = []
    amp_prob_list = []
    amp_label_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting predictions", leave=False):
            esm = batch["esm"].to(device, non_blocking=True)
            know = batch["knowledge"].to(device, non_blocking=True)
            labels = batch["labels"].float()
            amp_labels = batch["amp_label"].float()
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(
                esm_features=esm,
                knowledge_features=know,
                attention_mask=attention_mask,
                return_dict=True,
            )
            task_probs = torch.sigmoid(outputs["task_logits"]).cpu().numpy()
            amp_probs = torch.sigmoid(outputs["amp_logits"]).cpu().numpy()

            task_prob_list.append(task_probs)
            task_label_list.append(labels.cpu().numpy())
            amp_prob_list.append(amp_probs)
            amp_label_list.append(amp_labels.cpu().numpy())

    y_prob = np.vstack(task_prob_list)
    y_true = np.vstack(task_label_list)
    amp_prob = np.vstack(amp_prob_list).reshape(-1)
    amp_true = np.vstack(amp_label_list).reshape(-1)
    return y_true, y_prob, amp_true, amp_prob


def find_best_threshold_binary_fixed_rule():
    return 0.5


def find_best_thresholds_multilabel_from_training(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
):
    num_labels = y_true.shape[1]
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    candidate_thresholds = np.arange(start, end + 1e-8, step)

    for j in range(num_labels):
        y_true_j = y_true[:, j].astype(int)
        y_prob_j = y_prob[:, j]
        if len(np.unique(y_true_j)) < 2:
            thresholds[j] = 0.5
            continue

        best_threshold = 0.5
        best_score = -1.0
        for threshold in candidate_thresholds:
            y_pred_j = (y_prob_j > threshold).astype(int)
            score = f1_score(y_true_j, y_pred_j, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[j] = best_threshold

    return thresholds


def apply_thresholds(y_prob: np.ndarray, thresholds: Union[float, np.ndarray]):
    if np.isscalar(thresholds):
        return (y_prob > float(thresholds)).astype(int)
    thresholds = np.asarray(thresholds).reshape(1, -1)
    return (y_prob > thresholds).astype(int)


def evaluate(model, loader, device, num_labels, thresholds, amp_threshold: float = 0.5, gate_multitask_with_amp: bool = True):
    y_true, y_prob, amp_true, amp_prob = collect_predictions(model, loader, device)
    amp_pred = (amp_prob > amp_threshold).astype(int)

    amp_metrics = {
        "amp_accuracy": float(accuracy_score(amp_true.astype(int), amp_pred)),
        "amp_precision": float(precision_score(amp_true.astype(int), amp_pred, zero_division=0)),
        "amp_recall": float(recall_score(amp_true.astype(int), amp_pred, zero_division=0)),
        "amp_f1": float(f1_score(amp_true.astype(int), amp_pred, zero_division=0)),
        "amp_roc_auc": safe_binary_roc_auc(amp_true.astype(int), amp_prob),
        "amp_ap": safe_binary_ap(amp_true.astype(int), amp_prob),
        "amp_threshold": float(amp_threshold),
    }

    if num_labels == 1:
        y_true = y_true.reshape(-1).astype(int)
        y_prob = y_prob.reshape(-1)
        threshold = float(thresholds)
        y_pred = (y_prob > threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": safe_binary_roc_auc(y_true, y_prob),
            "ap": safe_binary_ap(y_true, y_prob),
            "threshold": threshold,
            **amp_metrics,
        }
    else:
        y_true = y_true.astype(int)
        y_pred = apply_thresholds(y_prob, thresholds)
        if gate_multitask_with_amp:
            y_pred = y_pred * amp_pred.reshape(-1, 1)

        metrics = {
            "subset_accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "roc_auc_micro": safe_multilabel_roc_auc_micro(y_true, y_prob),
            "ap_micro": safe_multilabel_ap_micro(y_true, y_prob),
            "thresholds": np.asarray(thresholds, dtype=float).tolist(),
            "gate_multitask_with_amp": bool(gate_multitask_with_amp),
            **amp_metrics,
        }

    return metrics


def build_amp_targets(df: pd.DataFrame, label_cols, amp_label_col: Optional[str]):
    if amp_label_col is not None:
        return df[amp_label_col].values.astype(np.float32)
    return (df[label_cols].values.sum(axis=1) > 0).astype(np.float32)


def normalize_thresholds_for_json(thresholds: Union[float, np.ndarray]):
    if np.isscalar(thresholds):
        return float(thresholds)
    return np.asarray(thresholds, dtype=float).tolist()


def save_threshold_file(path: str, task_thresholds: Union[float, np.ndarray], amp_threshold: float):
    payload = {
        "task_thresholds": normalize_thresholds_for_json(task_thresholds),
        "amp_threshold": float(amp_threshold),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_checkpoint_payload(
    model,
    optimizer,
    epoch,
    selection_metric_name,
    selection_metric,
    task_thresholds,
    amp_threshold,
    best_metrics,
    esm_dim,
    knowledge_input_dim,
    knowledge_dim,
    class_stats,
    amp_stats,
    args,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(selection_metric),
        "best_metric_name": selection_metric_name,
        "best_thresholds": normalize_thresholds_for_json(task_thresholds),
        "best_metrics": best_metrics,
        "amp_threshold": float(amp_threshold),
        "esm_dim": int(esm_dim),
        "knowledge_input_dim": int(knowledge_input_dim),
        "knowledge_dim": int(args.knowledge_hidden_dim),
        "class_statistics": class_stats,
        "amp_statistics": amp_stats,
        "args": vars(args),
    }


def run_training(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    existing_outputs = [
        "best_amp_model.pt",
        "best_multilabel_model.pt",
        "best_amp_thresholds.json",
        "best_multilabel_thresholds.json",
    ]
    found_existing = [name for name in existing_outputs if os.path.exists(os.path.join(args.save_dir, name))]
    if found_existing:
        print(
            "[WARNING] Existing output files found in save_dir and may be overwritten or mixed with new results: "
            + ", ".join(found_existing)
        )

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() and args.gpu >= 0 else torch.device("cpu")
    print(f"Using device: {device}")

    label_cols = [c.lower() for c in args.label_cols]
    amp_label_col = None if args.amp_label_col is None else args.amp_label_col.lower()

    print(
        f"Feature cache enabled: {bool(args.use_feature_cache)} | "
        f"cache_dir={args.cache_dir or './feature_cache'} | "
        f"strict_cache={bool(args.strict_cache)}"
    )

    dataset = UnifiedProteinDataset(
        csv_file=args.data_csv,
        sequence_col="sequence",
        label_cols=label_cols,
        amp_label_col=amp_label_col,
        max_seq_len=args.max_seq_len,
        device="cpu",
        prefer_pretrained_esm=not args.disable_pretrained_esm,
        cache_dir=args.cache_dir,
        use_feature_cache=args.use_feature_cache,
        build_cache_if_missing=args.build_cache_if_missing,
        write_cache_on_miss=args.write_cache_on_miss,
        strict_cache=args.strict_cache,
        cache_name=args.cache_name,
    )

    df = dataset.dataframe.copy()

    with torch.no_grad():
        sample = dataset[0]
        esm_dim = sample["esm"].shape[-1]
        knowledge_input_dim = sample["knowledge"].shape[-1]

    indices = np.arange(len(dataset))
    amp_targets = build_amp_targets(df, label_cols, amp_label_col)
    stratify_labels = amp_targets if len(np.unique(amp_targets)) > 1 else None

    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    pin_memory = device.type == "cuda"
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=args.num_workers, pin_memory=pin_memory)
    train_eval_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=args.num_workers, pin_memory=pin_memory)

    train_targets = df.iloc[train_idx][label_cols].values.astype(np.float32)
    alpha, pos_weight, class_stats = compute_class_weights(train_targets=train_targets, max_pos_weight=args.max_pos_weight)

    train_amp_targets = amp_targets[train_idx].reshape(-1, 1)
    _, amp_pos_weight, amp_stats = compute_class_weights(train_targets=train_amp_targets, max_pos_weight=args.max_pos_weight)

    print("Task class statistics from the training split:")
    for name, pos_c, neg_c, a, pw in zip(label_cols, class_stats["positive_count"], class_stats["negative_count"], class_stats["alpha"], class_stats["pos_weight"]):
        print(f"  {name}: pos={pos_c}, neg={neg_c}, alpha={a:.4f}, pos_weight={pw:.4f}")
    print(f"AMP head ({amp_label_col or 'derived_is_amp'}): pos={amp_stats['positive_count'][0]}, neg={amp_stats['negative_count'][0]}, pos_weight={amp_stats['pos_weight'][0]:.4f}")

    model = MAPLE(
        linsize=args.hidden_size,
        lindropout=args.dropout,
        num_labels=len(label_cols),
        esm_dim=esm_dim,
        knowledge_dim=args.knowledge_hidden_dim,
        knowledge_input_dim=knowledge_input_dim,
        base_dim=args.base_dim,
        bimamba_dim=args.bimamba_dim,
        enable_amp_head=True,
        knowledge_num_heads=args.knowledge_num_heads,
        knowledge_num_layers=args.knowledge_num_layers,
        knowledge_dropout=args.knowledge_dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

    task_criterion = FocalLoss(alpha=alpha, gamma=args.gamma, reduction="none", pos_weight=pos_weight)
    amp_criterion = nn.BCEWithLogitsLoss(pos_weight=amp_pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_labels = len(label_cols)
    is_amp_task = num_labels == 1
    multilabel_metric_key = "f1" if is_amp_task else "f1_macro"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.scheduler_patience)

    best_amp_f1 = -float("inf")
    best_amp_metrics = None
    best_amp_epoch = -1

    best_multilabel_metric = -float("inf")
    best_multilabel_metrics = None
    best_multilabel_epoch = -1
    best_multilabel_thresholds = 0.5 if num_labels == 1 else [0.5] * num_labels

    last_val_metrics = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            task_criterion,
            amp_criterion,
            optimizer,
            device,
            args.task_loss_weight,
            args.amp_loss_weight,
        )

        if num_labels == 1:
            thresholds = find_best_threshold_binary_fixed_rule()
        else:
            train_y_true, train_y_prob, _, _ = collect_predictions(model, train_eval_loader, device)
            thresholds = find_best_thresholds_multilabel_from_training(
                train_y_true.astype(int),
                train_y_prob,
                start=args.threshold_start,
                end=args.threshold_end,
                step=args.threshold_step,
            )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_labels=num_labels,
            thresholds=thresholds,
            amp_threshold=args.amp_threshold,
            gate_multitask_with_amp=args.gate_multitask_with_amp,
        )
        last_val_metrics = val_metrics

        current_multilabel_metric = float(val_metrics[multilabel_metric_key])
        current_amp_f1 = float(val_metrics["amp_f1"])
        scheduler.step(current_multilabel_metric)

        threshold_message = f"threshold={float(thresholds):.2f}" if num_labels == 1 else "thresholds=" + ",".join([f"{x:.2f}" for x in thresholds])
        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | {multilabel_metric_key}={current_multilabel_metric:.4f} | amp_f1={current_amp_f1:.4f} | {threshold_message}"
        )

        if current_amp_f1 > best_amp_f1:
            best_amp_f1 = current_amp_f1
            best_amp_metrics = val_metrics.copy()
            best_amp_epoch = epoch
            if is_amp_task:
                amp_checkpoint = build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    selection_metric_name="amp_f1",
                    selection_metric=best_amp_f1,
                    task_thresholds=thresholds,
                    amp_threshold=args.amp_threshold,
                    best_metrics=best_amp_metrics,
                    esm_dim=esm_dim,
                    knowledge_input_dim=knowledge_input_dim,
                    knowledge_dim=args.knowledge_hidden_dim,
                    class_stats=class_stats,
                    amp_stats=amp_stats,
                    args=args,
                )
                amp_ckpt_path = os.path.join(args.save_dir, "best_amp_model.pt")
                torch.save(amp_checkpoint, amp_ckpt_path)
                save_threshold_file(
                    os.path.join(args.save_dir, "best_amp_thresholds.json"),
                    task_thresholds=thresholds,
                    amp_threshold=args.amp_threshold,
                )

        if current_multilabel_metric > best_multilabel_metric:
            best_multilabel_metric = current_multilabel_metric
            best_multilabel_metrics = val_metrics.copy()
            best_multilabel_epoch = epoch
            best_multilabel_thresholds = normalize_thresholds_for_json(thresholds)
            patience_counter = 0

            if not is_amp_task:
                mtl_checkpoint = build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    selection_metric_name=multilabel_metric_key,
                    selection_metric=best_multilabel_metric,
                    task_thresholds=thresholds,
                    amp_threshold=args.amp_threshold,
                    best_metrics=best_multilabel_metrics,
                    esm_dim=esm_dim,
                    knowledge_input_dim=knowledge_input_dim,
                    knowledge_dim=args.knowledge_hidden_dim,
                    class_stats=class_stats,
                    amp_stats=amp_stats,
                    args=args,
                )
                mtl_ckpt_path = os.path.join(args.save_dir, "best_multilabel_model.pt")
                torch.save(mtl_checkpoint, mtl_ckpt_path)
                save_threshold_file(
                    os.path.join(args.save_dir, "best_multilabel_thresholds.json"),
                    task_thresholds=thresholds,
                    amp_threshold=args.amp_threshold,
                )
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    summary = {
        "best_amp": {
            "epoch": int(best_amp_epoch),
            "metric_name": "amp_f1",
            "metric": float(best_amp_f1) if best_amp_metrics is not None else None,
            "checkpoint": "best_amp_model.pt" if is_amp_task else None,
            "threshold_file": "best_amp_thresholds.json" if is_amp_task else None,
            "val_metrics": best_amp_metrics,
        },
        "best_multilabel": {
            "epoch": int(best_multilabel_epoch),
            "metric_name": multilabel_metric_key,
            "metric": float(best_multilabel_metric) if best_multilabel_metrics is not None else None,
            "checkpoint": "best_multilabel_model.pt" if not is_amp_task else None,
            "threshold_file": "best_multilabel_thresholds.json" if not is_amp_task else None,
            "task_thresholds": best_multilabel_thresholds,
            "val_metrics": best_multilabel_metrics,
        },
        "amp_threshold": float(args.amp_threshold),
        "esm_dim": int(esm_dim),
        "knowledge_input_dim": int(knowledge_input_dim),
        "knowledge_dim": int(args.knowledge_hidden_dim),
        "num_labels": int(num_labels),
        "label_names": label_cols,
        "amp_label_name": amp_label_col or "derived_is_amp",
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "class_statistics": class_stats,
        "amp_statistics": amp_stats,
        "last_val_metrics": last_val_metrics,
        "args": vars(args),
    }

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--label_cols", nargs="+", required=True)
    parser.add_argument("--amp_label_col", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--scheduler_patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base_dim", type=int, default=512)
    parser.add_argument("--bimamba_dim", type=int, default=256)
    parser.add_argument("--disable_pretrained_esm", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--knowledge_hidden_dim", type=int, default=512)
    parser.add_argument("--knowledge_num_heads", type=int, default=8)
    parser.add_argument("--knowledge_num_layers", type=int, default=4)
    parser.add_argument("--knowledge_dropout", type=float, default=0.1)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_feature_cache", action="store_true")
    parser.add_argument("--build_cache_if_missing", action="store_true")
    parser.add_argument("--write_cache_on_miss", action="store_true")
    parser.add_argument("--strict_cache", action="store_true")
    parser.add_argument("--cache_name", type=str, default=None)

    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--max_pos_weight", type=float, default=50.0)
    parser.add_argument("--task_loss_weight", type=float, default=1.0)
    parser.add_argument("--amp_loss_weight", type=float, default=1.0)
    parser.add_argument("--amp_threshold", type=float, default=0.5)
    parser.add_argument("--gate_multitask_with_amp", action="store_true")

    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_end", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    args = parser.parse_args()
    run_training(args)
