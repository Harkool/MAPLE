import hashlib
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

CACHE_SCHEMA_VERSION = 2
CACHE_LOGIC_VERSION = "maple_feature_cache_v2_raw_knowledge"


def compute_sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def compute_cache_fingerprint(
    seq_col: str,
    label_cols: List[str],
    amp_label_col: Optional[str],
    max_seq_len: int,
    descriptor_mode: str,
    descriptor_dim: int,
    prefer_pretrained_esm: bool,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": schema_version,
        "logic_version": CACHE_LOGIC_VERSION,
        "seq_col": seq_col,
        "label_cols": list(label_cols),
        "amp_label_col": amp_label_col,
        "max_seq_len": int(max_seq_len),
        "descriptor_mode": descriptor_mode,
        "descriptor_dim": int(descriptor_dim),
        "prefer_pretrained_esm": bool(prefer_pretrained_esm),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def get_default_cache_path(
    data_csv: str,
    cache_dir: str,
    fingerprint: str,
    cache_name: Optional[str] = None,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    if cache_name:
        stem = cache_name
    else:
        stem = os.path.splitext(os.path.basename(data_csv))[0]
    return os.path.join(cache_dir, f"{stem}__{fingerprint}.pt")


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return value


def save_feature_cache(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload_cpu = _to_cpu(payload)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path) or ".",
    )
    os.close(fd)
    try:
        torch.save(payload_cpu, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_feature_cache(path: str) -> Dict:
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid cache format: expected dict, got {type(cache)}")
    if cache.get("version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache version mismatch: expected {CACHE_SCHEMA_VERSION}, got {cache.get('version')}"
        )
    if "rows" not in cache or not isinstance(cache["rows"], list):
        raise ValueError("Invalid cache format: missing `rows` list")
    return cache


def validate_cache_against_dataframe(
    cache: Dict,
    dataframe: pd.DataFrame,
    sequence_col: str,
    label_cols: List[str],
    amp_label_col: Optional[str],
    expected_fingerprint: Optional[str] = None,
) -> Tuple[bool, str]:
    num_samples = len(dataframe)
    if cache.get("num_samples") != num_samples:
        return False, f"sample count mismatch: cache={cache.get('num_samples')} current={num_samples}"

    if cache.get("seq_col") != sequence_col:
        return False, f"seq_col mismatch: cache={cache.get('seq_col')} current={sequence_col}"

    if cache.get("label_cols") != list(label_cols):
        return False, "label_cols mismatch"

    cache_amp_col = cache.get("amp_label_col")
    current_amp_col = amp_label_col or "derived_is_amp"
    if cache_amp_col != current_amp_col:
        return False, f"amp_label_col mismatch: cache={cache_amp_col} current={current_amp_col}"

    if expected_fingerprint is not None and cache.get("fingerprint") != expected_fingerprint:
        return False, (
            f"fingerprint mismatch: cache={cache.get('fingerprint')} "
            f"current={expected_fingerprint}"
        )

    rows = cache.get("rows", [])
    if len(rows) != num_samples:
        return False, f"rows length mismatch: cache_rows={len(rows)} current={num_samples}"

    seq_hashes = [compute_sequence_hash(s) for s in dataframe[sequence_col].astype(str).tolist()]
    for idx, (row, expected_hash) in enumerate(zip(rows, seq_hashes)):
        if row is None:
            return False, f"row {idx} is empty"
        row_hash = row.get("sequence_hash")
        if row_hash != expected_hash:
            return False, f"sequence hash mismatch at row {idx}"

    return True, "ok"
