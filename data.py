import math
import os
import re
import warnings
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm

from feature_cache import (
    CACHE_SCHEMA_VERSION,
    compute_cache_fingerprint,
    compute_sequence_hash,
    get_default_cache_path,
    load_feature_cache,
    save_feature_cache,
    validate_cache_against_dataframe,
)
from Module.ESMembedding import ESM2Embedder
from Module.Knowledge import AA_ORDER, AA_TO_IDX, BOMAN, CHARGE, HELIX, HYDROPHOBICITY, MOLECULAR_WEIGHTS


VALID_SEQUENCE_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWYBXZOU\-]*")
UNKNOWN_AA_INDEX = len(AA_ORDER)


class RawKnowledgeDescriptorEncoder:
    def __init__(self, max_seq_length: int = 1024):
        self.max_len = int(max_seq_length)
        self.output_dim = 56
        self.aa_onehot = torch.eye(len(AA_ORDER), dtype=torch.float32)
        self.hydrophobicity = HYDROPHOBICITY.clone().float()
        self.charge = CHARGE.clone().float()
        self.helix = HELIX.clone().float()

        self.group_tensors = [
            torch.tensor([AA_TO_IDX[aa] for aa in aas], dtype=torch.long)
            for aas in ["AILMFPWV", "RKDE", "NQSTYC", "FWY", "RKH", "DE", "AGSC", "RWFY"]
        ]
        self.rk_indices = torch.tensor([AA_TO_IDX["R"], AA_TO_IDX["K"]], dtype=torch.long)
        self.hydrophobic_indices = torch.tensor([AA_TO_IDX[aa] for aa in "AILMFPWV"], dtype=torch.long)
        self.mixed_indices = torch.tensor([AA_TO_IDX[aa] for aa in "RKDESTNQ"], dtype=torch.long)
        self.aromatic_indices = torch.tensor([AA_TO_IDX["F"], AA_TO_IDX["W"], AA_TO_IDX["Y"]], dtype=torch.long)

    def _aa_to_idx(self, aa: str) -> int:
        return AA_TO_IDX.get(aa, UNKNOWN_AA_INDEX)

    def _aa_weight(self, aa: str) -> float:
        return MOLECULAR_WEIGHTS.get(aa, 110.0)

    def encode_sequence(self, seq: str) -> torch.Tensor:
        seq = seq.upper()[: self.max_len]
        if not seq:
            seq = "X"

        indices = torch.tensor([self._aa_to_idx(aa) for aa in seq], dtype=torch.long)
        clamped = indices.clamp(max=len(AA_ORDER) - 1)
        valid_mask = (indices != UNKNOWN_AA_INDEX).float().unsqueeze(1)
        length = len(seq)

        one_hot = self.aa_onehot[clamped] * valid_mask
        hydro = self.hydrophobicity[clamped] * valid_mask.squeeze(1)
        charge = self.charge[clamped] * valid_mask.squeeze(1)
        weight = torch.tensor([self._aa_weight(aa) for aa in seq], dtype=torch.float32) / 200.0
        helix = self.helix[clamped] * valid_mask.squeeze(1)
        aromatic = torch.isin(clamped, self.aromatic_indices).float() * valid_mask.squeeze(1)
        phys = torch.stack([hydro, charge, weight, helix, aromatic], dim=1)

        global_f = torch.tensor(self._global_features(seq), dtype=torch.float32).unsqueeze(0).repeat(length, 1)
        pos = torch.arange(length, dtype=torch.float32) / max(length - 1, 1)
        edge_distance = torch.minimum(
            torch.arange(length, dtype=torch.float32),
            torch.arange(length - 1, -1, -1, dtype=torch.float32),
        ) / max(length - 1, 1)
        pos_feat = torch.stack(
            [
                pos,
                (pos < 0.2).float(),
                (pos > 0.8).float(),
                ((pos >= 0.3) & (pos <= 0.7)).float(),
                edge_distance,
                torch.sin(2 * math.pi * pos),
            ],
            dim=1,
        )

        window_feat = self._window_features(indices)
        return torch.cat([one_hot, phys, window_feat, pos_feat, global_f], dim=1).float()

    def _global_features(self, seq: str) -> List[float]:
        length = max(len(seq), 1)
        net_charge = sum({"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.5}.get(aa, 0) for aa in seq)
        hydro_values = [self.hydrophobicity[self._aa_to_idx(aa)].item() for aa in seq if aa in AA_TO_IDX]
        avg_hydro = sum(hydro_values) / max(len(hydro_values), 1)
        p_i = 7.0 + (seq.count("R") + seq.count("K") - seq.count("D") - seq.count("E")) * 0.5
        mol_weight = sum(self._aa_weight(aa) for aa in seq)
        unique = len(set(seq))
        pos_ratio = sum(aa in "RKH" for aa in seq) / length
        hydro_ratio = sum(aa in "AILMFPWV" for aa in seq) / length
        arom_ratio = sum(aa in "FWY" for aa in seq) / length
        boman = sum(BOMAN.get(aa, 0.0) for aa in seq) / length
        return [
            length / 100.0,
            net_charge / length,
            avg_hydro,
            p_i / 14.0,
            mol_weight / 10000.0,
            unique / 20.0,
            pos_ratio,
            hydro_ratio,
            arom_ratio,
            boman / 3.0,
        ]

    def _window_features(self, indices: torch.Tensor, window: int = 5) -> torch.Tensor:
        pad = window // 2
        padded_idx = F.pad(indices, (pad, pad), value=UNKNOWN_AA_INDEX)
        windows = padded_idx.unfold(0, window, 1)
        valid_windows = windows != UNKNOWN_AA_INDEX
        clamped_windows = windows.clamp(max=len(AA_ORDER) - 1)

        feats = []
        denom = valid_windows.sum(dim=1).clamp_min(1.0)
        for group_indices in self.group_tensors:
            mask = torch.isin(clamped_windows, group_indices) & valid_windows
            feats.append(mask.float().sum(dim=1) / denom)

        hydro_win = (self.hydrophobicity[clamped_windows] * valid_windows).sum(dim=1) / denom
        charge_win = (self.charge[clamped_windows] * valid_windows).sum(dim=1) / denom
        complexity = ((clamped_windows != clamped_windows[:, :1]) & valid_windows).float().sum(dim=1) / denom

        rk_mask = torch.isin(clamped_windows, self.rk_indices) & valid_windows
        hydrophobic_mask = torch.isin(clamped_windows, self.hydrophobic_indices) & valid_windows
        mixed_mask = torch.isin(clamped_windows, self.mixed_indices) & valid_windows
        aromatic_mask = torch.isin(clamped_windows, self.aromatic_indices) & valid_windows

        pos_cluster = (rk_mask.sum(dim=1) >= 2).float()
        hydro_cluster = (hydrophobic_mask.sum(dim=1) >= 3).float()
        amph = ((hydrophobic_mask.sum(dim=1) >= 2) & (mixed_mask.sum(dim=1) >= 2)).float()
        arom_cluster = (aromatic_mask.sum(dim=1) >= 2).float()

        return torch.stack(
            feats + [hydro_win, charge_win, complexity, pos_cluster, hydro_cluster, amph, arom_cluster],
            dim=1,
        )


class UnifiedProteinDataset(Dataset):
    def __init__(
        self,
        csv_file: Union[str, pd.DataFrame],
        sequence_col: str = "sequence",
        label_cols: Union[str, List[str]] = "label",
        max_seq_len: int = 1024,
        device=None,
        transformer_config_name: str = "base",
        prefer_pretrained_esm: bool = True,
        amp_label_col: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_feature_cache: bool = False,
        build_cache_if_missing: bool = False,
        write_cache_on_miss: bool = False,
        strict_cache: bool = False,
        cache_name: Optional[str] = None,
    ):
        self.csv_source = csv_file if isinstance(csv_file, str) else "dataframe_input"
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = csv_file.copy()

        df.columns = [c.strip().lower() for c in df.columns]
        sequence_col = sequence_col.lower()
        label_cols = [label_cols] if isinstance(label_cols, str) else [c.lower() for c in label_cols]
        amp_label_col = None if amp_label_col is None else amp_label_col.lower()

        if sequence_col not in df.columns:
            raise KeyError(f"Missing sequence column: {sequence_col}")
        missing_labels = [col for col in label_cols if col not in df.columns]
        if missing_labels:
            raise KeyError(f"Missing label columns: {missing_labels}")
        if amp_label_col is not None and amp_label_col not in df.columns:
            raise KeyError(f"Missing AMP label column: {amp_label_col}")

        df[sequence_col] = df[sequence_col].astype(str).str.strip().str.upper()
        valid = df[sequence_col].apply(lambda x: bool(VALID_SEQUENCE_RE.fullmatch(x)))
        df = df[valid].reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid sequences remain after filtering the input CSV.")

        self.dataframe = df.copy()
        self.sequence_col = sequence_col
        self.sequences = df[sequence_col].tolist()
        self.labels = df[label_cols].astype(float).values
        if amp_label_col is None:
            self.amp_labels = (self.labels.sum(axis=1) > 0).astype(float)
        else:
            self.amp_labels = df[amp_label_col].astype(float).values
        self.amp_label_col = amp_label_col or "derived_is_amp"
        self.max_seq_len = int(max_seq_len)
        self.label_cols = label_cols
        self.num_labels = len(label_cols)
        self.device = torch.device(device or "cpu")
        self.transformer_config_name = transformer_config_name
        self.prefer_pretrained_esm = prefer_pretrained_esm

        self.use_feature_cache = bool(use_feature_cache)
        self.build_cache_if_missing = bool(build_cache_if_missing)
        self.write_cache_on_miss = bool(write_cache_on_miss)
        self.strict_cache = bool(strict_cache)
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.cache_fingerprint: Optional[str] = None
        self.cache_path: Optional[str] = None
        self._cache_warned_worker_write = False

        self.feature_cache_rows: Optional[List[Optional[Dict]]] = None
        self._feature_cache_loaded = False

        self.esm_embedder: Optional[ESM2Embedder] = None
        self.raw_knowledge_encoder = RawKnowledgeDescriptorEncoder(max_seq_length=self.max_seq_len)

        if self.use_feature_cache:
            self._init_feature_cache_state()

        # Dataset only constructs deterministic raw knowledge descriptors.
        # Trainable knowledge contextualization is moved into the model.
        self._ensure_feature_extractors()

        if self.use_feature_cache and self.build_cache_if_missing and not self._feature_cache_loaded:
            self.build_feature_cache(overwrite=False)

        if self._feature_cache_loaded and self.feature_cache_rows:
            first_row = self.feature_cache_rows[0]
            self.knowledge_dim = int(first_row["knowledge_feature"].shape[-1])
        else:
            self.knowledge_dim = self.raw_knowledge_encoder.output_dim

    def _init_feature_cache_state(self):
        effective_cache_dir = self.cache_dir or "./feature_cache"
        self.cache_fingerprint = compute_cache_fingerprint(
            seq_col=self.sequence_col,
            label_cols=self.label_cols,
            amp_label_col=self.amp_label_col,
            max_seq_len=self.max_seq_len,
            descriptor_mode="raw_handcrafted_v1",
            descriptor_dim=self.raw_knowledge_encoder.output_dim,
            prefer_pretrained_esm=self.prefer_pretrained_esm,
        )
        self.cache_path = get_default_cache_path(
            data_csv=self.csv_source,
            cache_dir=effective_cache_dir,
            fingerprint=self.cache_fingerprint,
            cache_name=self.cache_name,
        )
        print(f"[FeatureCache] enabled=True path={self.cache_path}")

        if not os.path.exists(self.cache_path):
            if self.strict_cache and not self.build_cache_if_missing:
                raise FileNotFoundError(
                    f"Feature cache not found in strict mode: {self.cache_path}"
                )
            print("[FeatureCache] cache miss -> fallback to online extraction")
            return

        try:
            cache_payload = load_feature_cache(self.cache_path)
            ok, reason = validate_cache_against_dataframe(
                cache=cache_payload,
                dataframe=self.dataframe,
                sequence_col=self.sequence_col,
                label_cols=self.label_cols,
                amp_label_col=self.amp_label_col,
                expected_fingerprint=self.cache_fingerprint,
            )
            if not ok:
                message = f"Feature cache validation failed: {reason}"
                if self.strict_cache:
                    raise ValueError(message)
                warnings.warn(message + ". Falling back to online extraction.")
                return

            self.feature_cache_rows = cache_payload["rows"]
            self._feature_cache_loaded = True
            print(f"[FeatureCache] cache hit -> loaded {len(self.feature_cache_rows)} samples")
        except Exception as exc:
            if self.strict_cache:
                raise
            warnings.warn(f"Failed to load feature cache ({exc}). Falling back to online extraction.")

    def _ensure_feature_extractors(self):
        if self.esm_embedder is not None:
            return
        self.esm_embedder = ESM2Embedder(device=str(self.device), prefer_pretrained=self.prefer_pretrained_esm)
        if hasattr(self.esm_embedder.model, "parameters"):
            for parameter in self.esm_embedder.model.parameters():
                parameter.requires_grad = False

    def _build_cache_row(self, idx: int) -> Dict:
        self._ensure_feature_extractors()
        seq = self.sequences[idx][: self.max_seq_len]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        amp_label = torch.tensor([self.amp_labels[idx]], dtype=torch.float32)

        with torch.no_grad():
            esm_feat = self.esm_embedder.embed_sequences([seq])[0].to(self.device).float()
            raw_knowledge_feat = self.raw_knowledge_encoder.encode_sequence(seq).to(self.device).float()

        min_len = min(esm_feat.size(0), raw_knowledge_feat.size(0))
        return {
            "sequence": seq,
            "sequence_hash": compute_sequence_hash(seq),
            "length": int(min_len),
            "esm_feature": esm_feat[:min_len].detach().cpu().float(),
            "knowledge_feature": raw_knowledge_feat[:min_len].detach().cpu().float(),
            "label": labels.detach().cpu(),
            "amp_label": amp_label.detach().cpu(),
        }

    def _cache_row_to_item(self, row: Dict) -> Dict:
        return {
            "esm": row["esm_feature"].clone(),
            "knowledge": row["knowledge_feature"].clone(),
            "labels": row["label"].clone(),
            "amp_label": row["amp_label"].clone(),
            "length": int(row["length"]),
        }

    def _build_cache_payload(self) -> Dict:
        return {
            "version": CACHE_SCHEMA_VERSION,
            "fingerprint": self.cache_fingerprint,
            "source_csv": self.csv_source,
            "knowledge_mode": "raw_handcrafted_v1",
            "knowledge_dim": int(self.raw_knowledge_encoder.output_dim),
            "num_samples": len(self.sequences),
            "seq_col": self.sequence_col,
            "label_cols": list(self.label_cols),
            "amp_label_col": self.amp_label_col,
            "rows": self.feature_cache_rows,
        }

    def build_feature_cache(self, overwrite: bool = False) -> str:
        if not self.use_feature_cache:
            raise ValueError("Feature cache is disabled for this dataset instance.")
        if self.cache_path is None:
            raise ValueError("Cache path is not initialized.")

        if os.path.exists(self.cache_path) and not overwrite:
            payload = load_feature_cache(self.cache_path)
            ok, reason = validate_cache_against_dataframe(
                cache=payload,
                dataframe=self.dataframe,
                sequence_col=self.sequence_col,
                label_cols=self.label_cols,
                amp_label_col=self.amp_label_col,
                expected_fingerprint=self.cache_fingerprint,
            )
            if ok:
                self.feature_cache_rows = payload["rows"]
                self._feature_cache_loaded = True
                print(f"[FeatureCache] existing cache reused: {self.cache_path}")
                return self.cache_path
            if self.strict_cache:
                raise ValueError(f"Existing cache invalid in strict mode: {reason}")
            warnings.warn(f"Existing cache invalid ({reason}); rebuilding.")

        rows: List[Optional[Dict]] = [None] * len(self.sequences)
        for idx in tqdm(range(len(self.sequences)), desc="Building feature cache"):
            rows[idx] = self._build_cache_row(idx)

        self.feature_cache_rows = rows
        payload = self._build_cache_payload()
        save_feature_cache(self.cache_path, payload)
        self._feature_cache_loaded = True
        print(f"[FeatureCache] cache built -> {self.cache_path} ({len(rows)} samples)")
        return self.cache_path

    def __len__(self):
        return len(self.sequences)

    @torch.no_grad()
    def __getitem__(self, idx):
        seq = self.sequences[idx][: self.max_seq_len]
        seq_hash = compute_sequence_hash(seq)

        if self.feature_cache_rows is not None and idx < len(self.feature_cache_rows):
            row = self.feature_cache_rows[idx]
            if row is not None and row.get("sequence_hash") == seq_hash:
                return self._cache_row_to_item(row)

        row = self._build_cache_row(idx)
        item = self._cache_row_to_item(row)

        # Optional write-back on miss is best-effort and safe only in main process.
        if self.use_feature_cache and self.write_cache_on_miss and self.cache_path is not None:
            worker_info = get_worker_info()
            if worker_info is None:
                if self.feature_cache_rows is None:
                    self.feature_cache_rows = [None] * len(self.sequences)
                self.feature_cache_rows[idx] = row
                save_feature_cache(self.cache_path, self._build_cache_payload())
            elif not self._cache_warned_worker_write:
                self._cache_warned_worker_write = True
                warnings.warn(
                    "write_cache_on_miss=True with DataLoader workers may cause concurrent writes. "
                    "Use offline build_feature_cache.py for robust cache construction."
                )

        return item

    @staticmethod
    def collate_fn(batch: List[Dict]):
        esm = pad_sequence([b["esm"] for b in batch], batch_first=True)
        know = pad_sequence([b["knowledge"] for b in batch], batch_first=True)
        labels = torch.stack([b["labels"] for b in batch])
        amp_labels = torch.stack([b["amp_label"] for b in batch])
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        max_len = esm.size(1)
        mask = torch.arange(max_len, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)
        return {
            "esm": esm,
            "knowledge": know,
            "attention_mask": mask,
            "labels": labels,
            "amp_label": amp_labels,
            "lengths": lengths,
        }
