import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from web_core.ui_text import CHECKPOINT_LABEL_MAP, FUNCTIONAL_LABELS, _normalize_label_key


def resolve_device(device_choice: str, torch_module) -> str:
    choice = (device_choice or "auto").lower()
    cuda_available = bool(torch_module is not None and torch_module.cuda.is_available())
    if choice == "cuda":
        return "cuda" if cuda_available else "cpu"
    if choice == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def candidate_checkpoint_dirs(app_dir: Path, checkpoint_dir: str) -> List[Path]:
    requested = _resolve_local_path(app_dir, checkpoint_dir)
    candidates = [requested]
    if requested.name == "MAPLE_checkpoints":
        candidates.extend([app_dir / "Model", app_dir / "checkpoints"])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def candidate_threshold_files(app_dir: Path, threshold_file: str) -> List[Path]:
    requested = _resolve_local_path(app_dir, threshold_file)
    candidates = [requested]
    if requested.name == "thresholds.json" and requested.parent.name == "MAPLE_checkpoints":
        candidates.extend([app_dir / "Model" / "thresholds.json", app_dir / "checkpoints" / "thresholds.json"])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def load_thresholds(app_dir: Path, threshold_file: str) -> tuple:
    thresholds = {"amp": 0.5}
    thresholds.update({label: 0.5 for label in FUNCTIONAL_LABELS})
    source = "default_0.5"

    resolved_threshold_file = None
    threshold_candidates = candidate_threshold_files(app_dir, threshold_file) if threshold_file else []
    for candidate in threshold_candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            for key, value in loaded.items():
                normalized = _normalize_label_key(key)
                if normalized in thresholds:
                    try:
                        thresholds[normalized] = float(value)
                    except Exception:
                        continue
            resolved_threshold_file = candidate
            source = f"threshold_file:{resolved_threshold_file}"
            break
        except Exception:
            source = "default_0.5"

    return thresholds, source


def _resolve_local_path(app_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return app_dir / path


def _find_amp_checkpoint(checkpoint_dir: str) -> Optional[str]:
    for candidate in [os.path.join(checkpoint_dir, "AMP.pt")]:
        if os.path.exists(candidate):
            return candidate
    return None


def _find_label_checkpoint(checkpoint_dir: str, label: str) -> Optional[str]:
    basename = CHECKPOINT_LABEL_MAP[label]
    candidates = [
        os.path.join(checkpoint_dir, "label", f"{basename}.pt"),
        os.path.join(checkpoint_dir, "label", basename, f"{basename}.pt"),
        os.path.join(checkpoint_dir, f"{basename}.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_checkpoint_model(checkpoint_path: str, device: str):
    from model import build_maple_from_checkpoint

    return build_maple_from_checkpoint(checkpoint_path, device=device)


@st.cache_resource(show_spinner=False)
def try_load_maple_predictor(app_dir: Path, checkpoint_dir: str, device: str):
    resolved_checkpoint_dir = str(candidate_checkpoint_dirs(app_dir, checkpoint_dir)[0])
    predictor = {
        "model_available": False,
        "warning": "",
        "checkpoint_dir": resolved_checkpoint_dir,
        "device": device,
        "amp_checkpoint": None,
        "label_checkpoints": {},
        "knowledge_transformer_ckpt": None,
        "build_unified_pkl": None,
        "QuadOutputDataset": None,
        "quad_output_collate_fn": None,
        "torch": None,
        "predict_module_reused": False,
    }

    try:
        from Generate_pkl import build_unified_pkl
        from data import QuadOutputDataset, quad_output_collate_fn

        predictor["build_unified_pkl"] = build_unified_pkl
        predictor["QuadOutputDataset"] = QuadOutputDataset
        predictor["quad_output_collate_fn"] = quad_output_collate_fn
        predictor["torch"] = __import__("torch")

        try:
            import predict as _predict_module

            predictor["predict_module_reused"] = True
            predictor["predict_module_name"] = _predict_module.__name__
        except Exception:
            predictor["predict_module_reused"] = False

        missing_report = None
        for candidate_dir in candidate_checkpoint_dirs(app_dir, checkpoint_dir):
            candidate_dir_str = str(candidate_dir)
            predictor["checkpoint_dir"] = candidate_dir_str
            predictor["knowledge_transformer_ckpt"] = os.path.join(candidate_dir_str, "knowledge_transformer.pt")
            predictor["amp_checkpoint"] = _find_amp_checkpoint(candidate_dir_str)
            predictor["label_checkpoints"] = {}
            if predictor["amp_checkpoint"] is None:
                missing_report = "AMP checkpoint was not found."
                continue

            missing = []
            for label in FUNCTIONAL_LABELS:
                path = _find_label_checkpoint(candidate_dir_str, label)
                predictor["label_checkpoints"][label] = path
                if path is None:
                    missing.append(label)
            if missing:
                missing_report = "Missing label checkpoints: " + ", ".join(missing)
                continue

            load_checkpoint_model(predictor["amp_checkpoint"], device)
            for label in FUNCTIONAL_LABELS:
                load_checkpoint_model(predictor["label_checkpoints"][label], device)

            predictor["model_available"] = True
            return predictor

        predictor["warning"] = missing_report or "AMP checkpoint was not found."
        return predictor
    except Exception as exc:
        predictor["warning"] = str(exc)
        return predictor


def build_prediction_feature_bundle(unique_sequences: pd.DataFrame, predictor: Dict) -> Tuple[pd.DataFrame, object]:
    build_unified_pkl = predictor["build_unified_pkl"]
    quad_dataset_cls = predictor["QuadOutputDataset"]
    collate_fn = predictor["quad_output_collate_fn"]
    torch_mod = predictor["torch"]

    tmp_df = unique_sequences[["sequence"]].copy()
    tmp_df["label"] = 0.0

    csv_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    pkl_handle = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    csv_path = csv_handle.name
    pkl_path = pkl_handle.name
    csv_handle.close()
    pkl_handle.close()

    try:
        tmp_df.to_csv(csv_path, index=False)
        build_unified_pkl(
            csv_path=Path(csv_path),
            output_pkl=Path(pkl_path),
            sequence_col="sequence",
            label_cols=["label"],
            esm_model_name="esm2_t12_35M_UR50D",
            max_seq_len=700,
            device=predictor["device"],
            knowledge_transformer_ckpt=predictor["knowledge_transformer_ckpt"],
            knowledge_dim=256,
        )

        import pickle

        with open(pkl_path, "rb") as handle:
            raw_data = pickle.load(handle)

        rows = []
        for seq_hash, content in raw_data["features"].items():
            rows.append(
                {
                    "hash": seq_hash,
                    "sequence": content.get("sequence", ""),
                    "label": content.get("labels", [0])[0] if content.get("labels") else 0,
                }
            )
        feature_df = pd.DataFrame(rows)
        dataset = quad_dataset_cls(feature_df, feature_dict=raw_data["features"])
        loader = torch_mod.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        return feature_df, loader
    finally:
        for path in [csv_path, pkl_path]:
            try:
                os.remove(path)
            except Exception:
                pass


def infer_probabilities(loader, checkpoint_path: str, predictor: Dict) -> List[float]:
    torch_mod = predictor["torch"]
    device = predictor["device"]
    loaded = load_checkpoint_model(checkpoint_path, device)
    model = loaded["model"]

    probs = []
    with torch_mod.no_grad():
        for esm_feat, kn_feat, _ in loader:
            esm_feat = esm_feat.to(device)
            kn_feat = kn_feat.to(device)
            logits = model(esm_features=esm_feat, knowledge_features=kn_feat)
            batch_probs = torch_mod.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probs.extend(batch_probs.tolist())
    return probs


def run_maple_prediction(valid_df: pd.DataFrame, predictor, thresholds: dict) -> pd.DataFrame:
    result_columns = ["sequence_id", "P_AMP", "AMP_label"]
    for label in FUNCTIONAL_LABELS:
        result_columns.extend([f"P_{label}", f"{label}_label"])

    if valid_df.empty:
        return pd.DataFrame(columns=result_columns)

    if not predictor or not predictor.get("model_available", False):
        out = valid_df[["sequence_id"]].copy()
        out["P_AMP"] = pd.NA
        out["AMP_label"] = pd.NA
        for label in FUNCTIONAL_LABELS:
            out[f"P_{label}"] = pd.NA
            out[f"{label}_label"] = pd.NA
        return out[result_columns]

    unique_sequences = valid_df[["sequence"]].drop_duplicates().reset_index(drop=True)
    feature_df, loader = build_prediction_feature_bundle(unique_sequences, predictor)

    prediction_lookup = pd.DataFrame({"sequence": feature_df["sequence"].tolist()})

    amp_probs = infer_probabilities(loader, predictor["amp_checkpoint"], predictor)
    threshold_amp = thresholds.get("amp", 0.5)
    prediction_lookup["P_AMP"] = amp_probs
    prediction_lookup["AMP_label"] = [int(prob >= threshold_amp) for prob in amp_probs]

    amp_positive_lookup = prediction_lookup[prediction_lookup["AMP_label"] == 1][["sequence"]].drop_duplicates().reset_index(drop=True)

    for label in FUNCTIONAL_LABELS:
        prediction_lookup[f"P_{label}"] = pd.NA
        prediction_lookup[f"{label}_label"] = pd.NA

    if not amp_positive_lookup.empty:
        amp_feature_df, amp_loader = build_prediction_feature_bundle(amp_positive_lookup, predictor)
        functional_lookup = pd.DataFrame({"sequence": amp_feature_df["sequence"].tolist()})

        for label in FUNCTIONAL_LABELS:
            probs = infer_probabilities(amp_loader, predictor["label_checkpoints"][label], predictor)
            threshold = thresholds.get(label, 0.5)
            functional_lookup[f"P_{label}"] = probs
            functional_lookup[f"{label}_label"] = [int(prob >= threshold) for prob in probs]

        prediction_lookup = prediction_lookup.merge(functional_lookup, on="sequence", how="left", suffixes=("", "_pred"))
        for label in FUNCTIONAL_LABELS:
            prob_col = f"P_{label}"
            label_col = f"{label}_label"
            pred_prob_col = f"{prob_col}_pred"
            pred_label_col = f"{label_col}_pred"
            if pred_prob_col in prediction_lookup.columns:
                prediction_lookup[prob_col] = prediction_lookup[pred_prob_col]
                prediction_lookup.drop(columns=[pred_prob_col], inplace=True)
            if pred_label_col in prediction_lookup.columns:
                prediction_lookup[label_col] = prediction_lookup[pred_label_col]
                prediction_lookup.drop(columns=[pred_label_col], inplace=True)

    merged = valid_df[["sequence_id", "sequence"]].merge(prediction_lookup, on="sequence", how="left")
    return merged[result_columns]
