import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ESM_KEYS = ["esm_features", "esm_embeddings", "esm2_embeddings", "sequence_embedding"]
KNOWLEDGE_KEYS = ["enhanced_knowledge_features", "knowledge_features"]


def _first_available_key(sample, keys):
    for key in keys:
        if key in sample:
            return key
    return None


def _to_tensor(feature):
    return torch.tensor(feature, dtype=torch.float32)


def _truncate(feature, max_seq_length):
    if feature.size(0) > max_seq_length:
        return feature[:max_seq_length]
    return feature


def _log_feature_shape(prefix, sample, keys):
    key = _first_available_key(sample, keys)
    if key is None:
        return
    value = sample[key]
    if not hasattr(value, "shape"):
        value = np.array(value)
    print(f"[INFO] {prefix}shape: {value.shape}, field: {key}")


def _collect_valid_hashes(feature_dict, data_hashes, required_keys):
    valid_hashes = []
    missing_count = 0
    for seq_hash in data_hashes:
        sample = feature_dict.get(seq_hash)
        if sample is None:
            missing_count += 1
            continue
        if all(_first_available_key(sample, key_group) is not None for key_group in required_keys):
            valid_hashes.append(seq_hash)
        else:
            missing_count += 1
    return valid_hashes, missing_count


class Compatible_PKLDataset(Dataset):
    """Compatibility dataset for legacy PKLDataset interface; can use ESM or knowledge features."""

    def __init__(self, data_df, feature_dict, max_seq_length=700, use_esm_features=True):
        self.data_df = data_df
        self.feature_dict = feature_dict
        self.max_seq_length = max_seq_length
        self.use_esm_features = use_esm_features

        required = [ESM_KEYS] if use_esm_features else [KNOWLEDGE_KEYS]
        valid_hashes, missing_count = _collect_valid_hashes(
            feature_dict=self.feature_dict,
            data_hashes=self.data_df["hash"].tolist(),
            required_keys=required,
        )
        self.data_df = self.data_df[self.data_df["hash"].isin(valid_hashes)].reset_index(drop=True)

        feature_type = "ESM-2" if use_esm_features else "Knowledge"
        print(f"[INFO] {feature_type}dataset contains {len(self.data_df)} valid samples")
        if missing_count > 0:
            print(f"[WARNING] {missing_count} samples are missing {feature_type} features")

        if len(self.data_df) > 0:
            sample = self.feature_dict[self.data_df.iloc[0]["hash"]]
            _log_feature_shape(feature_type + " features", sample, ESM_KEYS if use_esm_features else KNOWLEDGE_KEYS)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        seq_hash = row["hash"]
        sample = self.feature_dict[seq_hash]

        keys = ESM_KEYS if self.use_esm_features else KNOWLEDGE_KEYS
        key = _first_available_key(sample, keys)
        if key is None:
            raise ValueError(f"Sequence {seq_hash} missing features: {keys}")

        feature = _truncate(_to_tensor(sample[key]), self.max_seq_length)
        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        # Legacy-compatible return: (esm_features, knowledge_features, labels)
        return feature.clone(), feature.clone(), labels


class CompatiblePKLDataset(Compatible_PKLDataset):
    """Legacy PKLDataset-compatible wrapper using knowledge features only."""

    def __init__(self, data_df, feature_dict, max_seq_length=700):
        super().__init__(
            data_df=data_df,
            feature_dict=feature_dict,
            max_seq_length=max_seq_length,
            use_esm_features=False,
        )


class _BaseDualFeatureDataset(Dataset):
    """Shared implementation returning (esm_features, knowledge_features, labels)."""

    def __init__(self, data_df, feature_dict, max_seq_length=700, dataset_name="Dual-input"):
        self.data_df = data_df
        self.feature_dict = feature_dict
        self.max_seq_length = max_seq_length
        self.dataset_name = dataset_name

        valid_hashes, missing_count = _collect_valid_hashes(
            feature_dict=self.feature_dict,
            data_hashes=self.data_df["hash"].tolist(),
            required_keys=[ESM_KEYS, KNOWLEDGE_KEYS],
        )
        self.data_df = self.data_df[self.data_df["hash"].isin(valid_hashes)].reset_index(drop=True)

        print(f"[INFO] {self.dataset_name}dataset contains {len(self.data_df)} valid samples")
        if missing_count > 0:
            print(f"[WARNING] {self.dataset_name}dataset has {missing_count} samples missing required features")

        if len(self.data_df) > 0:
            sample = self.feature_dict[self.data_df.iloc[0]["hash"]]
            _log_feature_shape("ESM features", sample, ESM_KEYS)
            _log_feature_shape("Knowledge", sample, KNOWLEDGE_KEYS)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        seq_hash = row["hash"]
        sample = self.feature_dict[seq_hash]

        esm_key = _first_available_key(sample, ESM_KEYS)
        knowledge_key = _first_available_key(sample, KNOWLEDGE_KEYS)
        if esm_key is None:
            raise ValueError(f"Sequence {seq_hash} missing ESM features")
        if knowledge_key is None:
            raise ValueError(f"Sequence {seq_hash} missing knowledge features")

        esm_features = _truncate(_to_tensor(sample[esm_key]), self.max_seq_length)
        knowledge_features = _truncate(_to_tensor(sample[knowledge_key]), self.max_seq_length)
        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        return esm_features, knowledge_features, labels


class DualInputDataset(_BaseDualFeatureDataset):
    """Dual-input dataset."""

    def __init__(self, data_df, feature_dict, max_seq_length=700):
        super().__init__(
            data_df=data_df,
            feature_dict=feature_dict,
            max_seq_length=max_seq_length,
            dataset_name="Dual-input",
        )


class QuadOutputDataset(_BaseDualFeatureDataset):
    """Quad-output dual-input dataset (same data interface as DualInputDataset)."""

    def __init__(self, data_df, feature_dict, max_seq_length=700):
        super().__init__(
            data_df=data_df,
            feature_dict=feature_dict,
            max_seq_length=max_seq_length,
            dataset_name="Quad-output",
        )


def compatible_collate_fn(batch):
    """Legacy-compatible collate: (esm_features, knowledge_features, labels)."""
    esm_features, knowledge_features, labels = zip(*batch)
    esm_features = nn.utils.rnn.pad_sequence(esm_features, batch_first=True)
    knowledge_features = nn.utils.rnn.pad_sequence(knowledge_features, batch_first=True)
    labels = torch.stack(labels)
    return esm_features, knowledge_features, labels


def dual_input_collate_fn(batch):
    return compatible_collate_fn(batch)


def quad_output_collate_fn(batch):
    return compatible_collate_fn(batch)
