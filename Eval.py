import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import UnifiedProteinDataset
from model import MAPLE


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true.ravel())) < 2:
        return None
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        return float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    try:
        return float(roc_auc_score(y_true, y_prob, average=average))
    except ValueError:
        return None


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.sum(y_true) == 0:
        return None
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        return float(average_precision_score(y_true.ravel(), y_prob.ravel()))
    try:
        return float(average_precision_score(y_true, y_prob, average=average))
    except ValueError:
        return None


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        labels = np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()]))
        if labels.size < 2:
            return 0.0
        tn, fp, _, _ = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1]).ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    specs = []
    for i in range(y_true.shape[1]):
        labels = np.unique(np.concatenate([y_true[:, i], y_pred[:, i]]))
        if labels.size < 2:
            specs.append(0.0)
            continue
        tn, fp, _, _ = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        specs.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def apply_thresholds(y_prob: np.ndarray, thresholds: Union[float, np.ndarray]) -> np.ndarray:
    if np.isscalar(thresholds):
        return (y_prob > float(thresholds)).astype(np.int32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (y_prob > thresholds).astype(np.int32)


def load_threshold_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_default_threshold_file(model_path: str) -> str:
    base = os.path.basename(model_path)
    directory = os.path.dirname(model_path) or "."
    if "best_amp_model" in base:
        return os.path.join(directory, "best_amp_thresholds.json")
    return os.path.join(directory, "best_multilabel_thresholds.json")


def resolve_thresholds(
    checkpoint: Dict,
    model_path: str,
    num_labels: int,
    threshold_file_arg: Optional[str],
    threshold_arg: Optional[float],
    amp_threshold_arg: Optional[float],
    auto_threshold_on_eval: bool,
) -> Tuple[Union[float, np.ndarray], float, str]:
    # Threshold search on eval/test is intentionally disabled to avoid information leakage.
    del threshold_arg  # Evaluation must not use manual thresholds.

    if auto_threshold_on_eval:
        raise ValueError("`--auto_threshold_on_eval` has been disabled to prevent eval/test-set threshold leakage.")

    threshold_file = threshold_file_arg or infer_default_threshold_file(model_path)
    if os.path.exists(threshold_file):
        payload = load_threshold_file(threshold_file)
        task_thresholds = payload.get("task_thresholds")
        amp_threshold = payload.get("amp_threshold", checkpoint.get("amp_threshold", 0.5))
        if task_thresholds is None:
            raise ValueError(f"Missing `task_thresholds` in threshold file: {threshold_file}")
        if num_labels == 1:
            thresholds = float(np.squeeze(np.asarray(task_thresholds, dtype=np.float32)))
        else:
            thresholds = np.asarray(task_thresholds, dtype=np.float32).reshape(num_labels)
        return thresholds, float(amp_threshold), f"threshold_file:{threshold_file}"

    # Backward-compatible fallback: use thresholds embedded in a training checkpoint.
    saved_thresholds = checkpoint.get("best_thresholds")
    if saved_thresholds is None:
        raise ValueError(
            "No training-derived thresholds were found for evaluation. "
            f"checkpoint='{model_path}'. "
            "Provide --threshold_file saved during training, or use a checkpoint containing `best_thresholds`."
        )

    if num_labels == 1:
        thresholds = float(np.squeeze(np.asarray(saved_thresholds, dtype=np.float32)))
    else:
        thresholds = np.asarray(saved_thresholds, dtype=np.float32).reshape(num_labels)

    amp_threshold = checkpoint.get("amp_threshold", 0.5)
    if amp_threshold_arg is not None:
        amp_threshold = float(amp_threshold_arg)
    return thresholds, float(amp_threshold), "checkpoint_embedded"


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.reshape(-1)
    y_pred = (y_prob > threshold).astype(int)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size < 2:
        specificity = 0.0
    else:
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    mcc = None
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": mcc,
        "auroc": safe_roc_auc(y_true, y_prob, average="macro"),
        "auprc": safe_average_precision(y_true, y_prob, average="macro"),
        "threshold": float(threshold),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Union[float, np.ndarray],
    gate_multitask_with_amp: bool,
    amp_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    y_true = y_true.astype(int)
    y_pred = apply_thresholds(y_prob, thresholds)
    if gate_multitask_with_amp:
        y_pred = y_pred * amp_pred.reshape(-1, 1)

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auroc": safe_roc_auc(y_true, y_prob, average="macro"),
    }


def compute_discrete_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute thresholded/discrete metrics from already-final predictions."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size < 2:
        specificity = 0.0
    else:
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    mcc = None
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = None

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": mcc,
    }


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_cols: List[str],
    thresholds: Union[float, np.ndarray],
) -> List[Dict[str, Optional[float]]]:
    """Compute per-label metrics for multi-label tasks.

    Per-label AUROC/AUPRC can be undefined for degenerate labels (all-0 or all-1) in a split.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred).astype(int)
    threshold_vec = np.asarray(thresholds, dtype=np.float32).reshape(-1)

    # Per-label discrete metrics must use final post-gating predictions.
    # AUROC/AUPRC remain probability-based to preserve ranking behavior.
    # This keeps per-label rows consistent with gated multilabel evaluation.
    rows: List[Dict[str, Optional[float]]] = []
    for i, label_name in enumerate(label_cols):
        discrete = compute_discrete_metrics_from_predictions(y_true[:, i], y_pred[:, i])
        rows.append(
            {
                "label_name": label_name,
                "label_index": i,
                "precision": discrete.get("precision"),
                "sensitivity": discrete.get("sensitivity"),
                "specificity": discrete.get("specificity"),
                "f1": discrete.get("f1"),
                "mcc": discrete.get("mcc"),
                "auroc": safe_roc_auc(y_true[:, i], y_prob[:, i], average="macro"),
                "auprc": safe_average_precision(y_true[:, i], y_prob[:, i], average="macro"),
                "threshold": float(threshold_vec[i]),
            }
        )
    return rows


def percentile_ci(values: List[float], ci: float = 95.0) -> Optional[List[float]]:
    if len(values) == 0:
        return None
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(values, alpha))
    hi = float(np.percentile(values, 100.0 - alpha))
    return [lo, hi]


def bootstrap_binary_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Optional[List[float]]]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "mcc", "auroc", "auprc"]
    values = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_metrics = compute_binary_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            v = sample_metrics.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            values[k].append(float(v))

    ci_dict = {k: percentile_ci(v) for k, v in values.items()}
    ci_dict["valid_bootstrap_counts"] = {k: len(v) for k, v in values.items()}
    return ci_dict


def bootstrap_multilabel_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Union[float, np.ndarray],
    gate_multitask_with_amp: bool,
    amp_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Optional[List[float]]]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    keys = ["macro_f1", "macro_precision", "macro_sensitivity", "macro_auroc"]
    values = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_metrics = compute_multilabel_metrics(
            y_true=y_true[idx],
            y_prob=y_prob[idx],
            thresholds=thresholds,
            gate_multitask_with_amp=gate_multitask_with_amp,
            amp_pred=amp_pred[idx],
        )
        for k in keys:
            v = sample_metrics.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            values[k].append(float(v))

    ci_dict = {k: percentile_ci(v) for k, v in values.items()}
    ci_dict["valid_bootstrap_counts"] = {k: len(v) for k, v in values.items()}
    return ci_dict


def warn_on_low_bootstrap_counts(ci_dict: Dict, n_bootstrap: int, scope: str):
    counts = ci_dict.get("valid_bootstrap_counts", {})
    for metric_name, count in counts.items():
        if count == 0:
            print(
                f"[WARNING] Bootstrap CI for {scope}.{metric_name} has 0 valid resamples "
                f"(n_bootstrap={n_bootstrap})."
            )
        elif count < 0.5 * n_bootstrap:
            print(
                f"[WARNING] Bootstrap CI for {scope}.{metric_name} has low valid resamples "
                f"({count}/{n_bootstrap})."
            )


def should_export_task_binary_scope(num_labels: int, y_true: np.ndarray, amp_true: np.ndarray) -> bool:
    """Decide whether task_binary should be exported in addition to amp_binary.

    In single-label AMP runs, labels and amp_label can be the same target; exporting both
    scopes would duplicate one task under two names. Multi-label runs always keep task scope.
    """
    if num_labels != 1:
        return True
    y_flat = np.asarray(y_true).reshape(-1).astype(int)
    amp_flat = np.asarray(amp_true).reshape(-1).astype(int)
    return not np.array_equal(y_flat, amp_flat)


def _extract_ci_pair(ci_dict: Dict, metric: str) -> Tuple[Optional[float], Optional[float]]:
    ci = ci_dict.get(metric)
    if isinstance(ci, list) and len(ci) == 2:
        return ci[0], ci[1]
    return None, None


def flatten_metrics_to_rows(
    checkpoint_name: str,
    samples: int,
    eval_time: str,
    threshold_source: str,
    gate_multitask_with_amp: bool,
    amp_threshold: float,
    task_thresholds: List[float],
    label_cols: List[str],
    metrics: Dict,
) -> List[Dict]:
    rows: List[Dict] = []

    amp_ci = metrics.get("amp_binary_ci95", {})
    for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1", "mcc", "auroc", "auprc"]:
        ci_low, ci_high = _extract_ci_pair(amp_ci, metric)
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "samples": samples,
                "eval_time": eval_time,
                "threshold_source": threshold_source,
                "gate_multitask_with_amp": gate_multitask_with_amp,
                "scope": "amp_binary",
                "label_name": "",
                "label_index": "",
                "metric": metric,
                "value": metrics.get("amp_binary", {}).get(metric),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "threshold": amp_threshold,
            }
        )

    if "task_binary" in metrics:
        task_ci = metrics.get("task_binary_ci95", {})
        task_threshold = float(task_thresholds[0]) if task_thresholds else None
        for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1", "mcc", "auroc", "auprc"]:
            ci_low, ci_high = _extract_ci_pair(task_ci, metric)
            rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "samples": samples,
                    "eval_time": eval_time,
                    "threshold_source": threshold_source,
                    "gate_multitask_with_amp": gate_multitask_with_amp,
                    "scope": "task_binary",
                    "label_name": "",
                    "label_index": "",
                    "metric": metric,
                    "value": metrics.get("task_binary", {}).get(metric),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "threshold": task_threshold,
                }
            )

    if "multilabel" in metrics:
        mtl_ci = metrics.get("multilabel_ci95", {})
        for metric in ["macro_f1", "macro_precision", "macro_sensitivity", "macro_auroc"]:
            ci_low, ci_high = _extract_ci_pair(mtl_ci, metric)
            rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "samples": samples,
                    "eval_time": eval_time,
                    "threshold_source": threshold_source,
                    "gate_multitask_with_amp": gate_multitask_with_amp,
                    "scope": "multilabel_macro",
                    "label_name": "",
                    "label_index": "",
                    "metric": metric,
                    "value": metrics.get("multilabel", {}).get(metric),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "threshold": "vector_thresholds",
                }
            )

    per_label_metrics = metrics.get("multilabel_per_label", [])
    if per_label_metrics:
        # Keep per-label rows in the same tidy CSV for direct downstream analysis.
        # CI columns are left empty here to avoid large bootstrap refactors.
        for item in per_label_metrics:
            label_name = item.get("label_name", "")
            label_index = item.get("label_index", "")
            label_threshold = item.get("threshold")
            for metric in ["precision", "sensitivity", "specificity", "f1", "mcc", "auroc", "auprc", "threshold"]:
                value = label_threshold if metric == "threshold" else item.get(metric)
                rows.append(
                    {
                        "checkpoint_name": checkpoint_name,
                        "samples": samples,
                        "eval_time": eval_time,
                        "threshold_source": threshold_source,
                        "gate_multitask_with_amp": gate_multitask_with_amp,
                        "scope": "multilabel_label",
                        "label_name": label_name,
                        "label_index": label_index,
                        "metric": metric,
                        "value": value,
                        "ci_low": None,
                        "ci_high": None,
                        "threshold": label_threshold,
                    }
                )

    return rows


def save_metrics_csv(rows: List[Dict], output_csv: str):
    columns = [
        "checkpoint_name",
        "samples",
        "eval_time",
        "threshold_source",
        "gate_multitask_with_amp",
        "scope",
        "label_name",
        "label_index",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "threshold",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)


@torch.no_grad()
def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    model_path = args.checkpoint or args.model_path
    data_csv = args.csv_path or args.data_csv
    checkpoint = torch.load(model_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    label_cols: List[str] = args.label_cols or ckpt_args.get("label_cols")
    if not label_cols:
        raise ValueError("`label_cols` must be provided either via CLI or in the saved checkpoint.")
    amp_label_col = args.amp_label_col or ckpt_args.get("amp_label_col")

    model = MAPLE(
        linsize=ckpt_args.get("hidden_size", 1024),
        lindropout=ckpt_args.get("dropout", 0.8),
        num_labels=len(label_cols),
        esm_dim=checkpoint.get("esm_dim", 1280),
        knowledge_dim=checkpoint.get("knowledge_dim", ckpt_args.get("knowledge_hidden_dim", 512)),
        knowledge_input_dim=checkpoint.get("knowledge_input_dim", ckpt_args.get("knowledge_input_dim", 56)),
        base_dim=ckpt_args.get("base_dim", args.base_dim),
        bimamba_dim=ckpt_args.get("bimamba_dim", args.bimamba_dim),
        enable_amp_head=True,
        knowledge_num_heads=ckpt_args.get("knowledge_num_heads", 8),
        knowledge_num_layers=ckpt_args.get("knowledge_num_layers", 4),
        knowledge_dropout=ckpt_args.get("knowledge_dropout", 0.1),
        max_seq_len=ckpt_args.get("max_seq_len", 1024),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Feature cache enabled: {bool(args.use_feature_cache)} | "
        f"cache_dir={args.cache_dir or './feature_cache'} | "
        f"strict_cache={bool(args.strict_cache)}"
    )

    dataset = UnifiedProteinDataset(
        csv_file=data_csv,
        sequence_col="sequence",
        label_cols=label_cols,
        amp_label_col=amp_label_col,
        max_seq_len=ckpt_args.get("max_seq_len", 1024),
        device="cpu",
        transformer_config_name="base",
        prefer_pretrained_esm=not args.disable_pretrained_esm,
        cache_dir=args.cache_dir,
        use_feature_cache=args.use_feature_cache,
        build_cache_if_missing=args.build_cache_if_missing,
        write_cache_on_miss=args.write_cache_on_miss,
        strict_cache=args.strict_cache,
        cache_name=args.cache_name,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    task_probs, task_labels, amp_probs, amp_labels = [], [], [], []
    for batch in tqdm(loader, desc="Inference"):
        outputs = model(
            esm_features=batch["esm"].to(device),
            knowledge_features=batch["knowledge"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            return_dict=True,
        )
        task_probs.append(torch.sigmoid(outputs["task_logits"]).cpu().numpy())
        task_labels.append(batch["labels"].cpu().numpy())
        amp_probs.append(torch.sigmoid(outputs["amp_logits"]).cpu().numpy())
        amp_labels.append(batch["amp_label"].cpu().numpy())

    y_prob = np.vstack(task_probs)
    y_true = np.vstack(task_labels).astype(int)
    amp_prob = np.vstack(amp_probs).reshape(-1)
    amp_true = np.vstack(amp_labels).reshape(-1).astype(int)
    num_labels = len(label_cols)

    thresholds, amp_threshold, threshold_source = resolve_thresholds(
        checkpoint=checkpoint,
        model_path=model_path,
        num_labels=num_labels,
        threshold_file_arg=args.threshold_file,
        threshold_arg=args.threshold,
        amp_threshold_arg=args.amp_threshold,
        auto_threshold_on_eval=args.auto_threshold_on_eval,
    )

    amp_binary_metrics = compute_binary_metrics(amp_true, amp_prob, amp_threshold)
    amp_pred = (amp_prob > amp_threshold).astype(np.int32)
    export_task_binary = should_export_task_binary_scope(
        num_labels=num_labels,
        y_true=y_true,
        amp_true=amp_true,
    )

    y_pred = apply_thresholds(y_prob, thresholds)
    if args.gate_multitask_with_amp:
        y_pred = y_pred * amp_pred.reshape(-1, 1)

    if num_labels == 1:
        # Binary tasks intentionally report binary AUROC/F1 metrics, not macro-* variants.
        task_metrics_summary = {}
        task_ci_summary = {}
        if export_task_binary:
            task_binary_metrics = compute_binary_metrics(y_true.reshape(-1), y_prob.reshape(-1), float(thresholds))
            task_metrics_summary = {
                "task_binary": task_binary_metrics,
            }
            task_ci_summary = {
                "task_binary_ci95": bootstrap_binary_ci(
                    y_true=y_true.reshape(-1),
                    y_prob=y_prob.reshape(-1),
                    threshold=float(thresholds),
                    n_bootstrap=args.bootstrap_n,
                    seed=args.bootstrap_seed,
                )
            }
    else:
        multilabel_metrics = compute_multilabel_metrics(
            y_true=y_true,
            y_prob=y_prob,
            thresholds=thresholds,
            gate_multitask_with_amp=args.gate_multitask_with_amp,
            amp_pred=amp_pred,
        )
        task_metrics_summary = {
            "multilabel": multilabel_metrics,
        }
        task_ci_summary = {
            "multilabel_ci95": bootstrap_multilabel_ci(
                y_true=y_true,
                y_prob=y_prob,
                thresholds=thresholds,
                gate_multitask_with_amp=args.gate_multitask_with_amp,
                amp_pred=amp_pred,
                n_bootstrap=args.bootstrap_n,
                seed=args.bootstrap_seed,
            )
        }
        # Multi-label evaluation includes both macro-level and per-label metrics in one tidy CSV.
        task_metrics_summary["multilabel_per_label"] = compute_per_label_metrics(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            label_cols=label_cols,
            thresholds=thresholds,
        )

    amp_ci95 = bootstrap_binary_ci(
        y_true=amp_true,
        y_prob=amp_prob,
        threshold=amp_threshold,
        n_bootstrap=args.bootstrap_n,
        seed=args.bootstrap_seed,
    )

    warn_on_low_bootstrap_counts(amp_ci95, args.bootstrap_n, scope="amp_binary")
    if num_labels == 1 and export_task_binary:
        warn_on_low_bootstrap_counts(task_ci_summary["task_binary_ci95"], args.bootstrap_n, scope="task_binary")
    else:
        if num_labels > 1:
            warn_on_low_bootstrap_counts(task_ci_summary["multilabel_ci95"], args.bootstrap_n, scope="multilabel")

    metrics = {
        "threshold_source": threshold_source,
        "thresholds": np.asarray(thresholds, dtype=float).reshape(-1).tolist(),
        "amp_threshold": float(amp_threshold),
        "amp_binary": amp_binary_metrics,
        "amp_binary_ci95": amp_ci95,
        **task_metrics_summary,
        **task_ci_summary,
        "gate_multitask_with_amp": bool(args.gate_multitask_with_amp),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(model_path).replace(".pt", "")
    eval_time = datetime.now().isoformat()

    # Keep evaluation outputs minimal and analysis-ready: one tidy metrics CSV.
    # Raw arrays / prediction dumps remain intentionally removed.
    rows = flatten_metrics_to_rows(
        checkpoint_name=name,
        samples=len(dataset),
        eval_time=eval_time,
        threshold_source=threshold_source,
        gate_multitask_with_amp=bool(args.gate_multitask_with_amp),
        amp_threshold=float(amp_threshold),
        task_thresholds=np.asarray(thresholds, dtype=float).reshape(-1).tolist(),
        label_cols=label_cols,
        metrics=metrics,
    )
    metrics_csv_path = os.path.join(args.output_dir, f"{name}_metrics.csv")
    save_metrics_csv(rows, metrics_csv_path)

    print(f"Threshold source: {threshold_source}")
    print(f"AMP F1: {amp_binary_metrics['f1']:.4f} | AMP AUROC: {amp_binary_metrics['auroc']} | AMP AUPRC: {amp_binary_metrics['auprc']}")
    if num_labels == 1 and not export_task_binary:
        print("task_binary scope skipped because labels and amp_label are identical in single-label mode.")
    if num_labels > 1:
        print(
            "Multi-label Macro-F1: "
            f"{task_metrics_summary['multilabel']['macro_f1']:.4f} | "
            f"Macro-AUROC: {task_metrics_summary['multilabel']['macro_auroc']}"
        )
        print("Per-label metrics saved to metrics csv.")
    print(f"Metrics CSV saved -> {metrics_csv_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--label_cols", nargs="+", default=None)
    parser.add_argument("--amp_label_col", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./eval_out")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold_file", type=str, default=None)
    parser.add_argument("--amp_threshold", type=float, default=None)
    parser.add_argument("--auto_threshold_on_eval", action="store_true")
    parser.add_argument("--gate_multitask_with_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_dim", type=int, default=512)
    parser.add_argument("--bimamba_dim", type=int, default=256)
    parser.add_argument("--disable_pretrained_esm", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_feature_cache", action="store_true")
    parser.add_argument("--build_cache_if_missing", action="store_true")
    parser.add_argument("--write_cache_on_miss", action="store_true")
    parser.add_argument("--strict_cache", action="store_true")
    parser.add_argument("--cache_name", type=str, default=None)
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.checkpoint or args.model_path):
        raise ValueError("Please provide --checkpoint (preferred) or --model_path.")
    if not (args.csv_path or args.data_csv):
        raise ValueError("Please provide --csv_path (preferred) or --data_csv.")
    for path in [args.checkpoint or args.model_path, args.csv_path or args.data_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    evaluate_model(args)
