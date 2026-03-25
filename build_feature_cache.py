import argparse
import os

import pandas as pd
import torch

from data import UnifiedProteinDataset


def main(args):
    csv_obj = args.data_csv
    if args.max_samples is not None:
        df = pd.read_csv(args.data_csv)
        if args.max_samples < len(df):
            df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        csv_obj = df

    dataset = UnifiedProteinDataset(
        csv_file=csv_obj,
        sequence_col=args.seq_col,
        label_cols=args.label_cols,
        amp_label_col=args.amp_label_col,
        device=args.device,
        transformer_config_name=args.transformer_config_name,
        prefer_pretrained_esm=not args.disable_pretrained_esm,
        cache_dir=args.cache_dir,
        use_feature_cache=True,
        build_cache_if_missing=False,
        write_cache_on_miss=False,
        strict_cache=False,
        cache_name=args.cache_name,
    )

    if dataset.cache_path is None:
        raise RuntimeError("Cache path was not initialized.")

    if os.path.exists(dataset.cache_path) and not args.overwrite:
        print(f"[FeatureCache] cache already exists, skip (use --overwrite to rebuild): {dataset.cache_path}")
        return

    with torch.no_grad():
        cache_path = dataset.build_feature_cache(overwrite=args.overwrite)

    sample = dataset[0]
    print(f"[FeatureCache] saved: {cache_path} (stores raw knowledge descriptors)")
    print(f"[FeatureCache] samples: {len(dataset)}")
    print(f"[FeatureCache] esm_dim={sample['esm'].shape[-1]} raw_knowledge_dim={sample['knowledge'].shape[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--seq_col", type=str, default="sequence")
    parser.add_argument("--label_cols", nargs="+", required=True)
    parser.add_argument("--amp_label_col", type=str, default=None)
    parser.add_argument("--cache_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--transformer_config_name", type=str, default="base")
    parser.add_argument("--disable_pretrained_esm", action="store_true")
    args = parser.parse_args()

    main(args)
