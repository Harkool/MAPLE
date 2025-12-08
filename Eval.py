#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    average_precision_score
)
from model import MAPLE
from data import UnifiedProteinDataset
import argparse
def calculate_specificity(y_true, y_pred):
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        tn, fp, _, _ = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    specs = []
    for i in range(y_true.shape[1]):
        tn, fp, _, _ = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specs)

@torch.no_grad()
def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)

    ckpt = torch.load(args.model_path, map_location=device)
    ckpt_args = ckpt.get('args', {})

    esm_dim_saved = ckpt.get('esm_dim', 1280)
    know_dim_saved = ckpt.get('knowledge_dim', 512)

    model = MAPLE(
        linsize=ckpt_args.get('hidden_size', 1024),
        lindropout=ckpt_args.get('dropout', 0.8),
        num_labels=len(args.label_cols),
        esm_dim=esm_dim_saved,
        knowledge_dim=know_dim_saved,
        base_dim=768,
        bimamba_dim=384,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dataset = UnifiedProteinDataset(
        csv_file=args.data_csv,
        sequence_col='sequence',
        label_cols=args.label_cols, 
        device=device,
        transformer_config_name='base'
    )

    with torch.no_grad():
        sample = dataset[0]
        real_esm_dim = sample['esm'].size(1)
        real_know_dim = sample['knowledge'].size(1)

    if real_esm_dim != esm_dim_saved or real_know_dim != know_dim_saved:
        print(f"Dimension mismatch! Updating projectors: ESM {esm_dim_saved}→{real_esm_dim}, Know {know_dim_saved}→{real_know_dim}")
        model.esm_projector = torch.nn.Sequential(
            torch.nn.Linear(real_esm_dim, model.base_dim),
            torch.nn.LayerNorm(model.base_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        ).to(device)
        model.knowledge_projector = torch.nn.Sequential(
            torch.nn.Linear(real_know_dim, model.base_dim),
            torch.nn.LayerNorm(model.base_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        ).to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    all_probs, all_preds, all_labels = [], [], []
    for batch in tqdm(loader, desc="Inference"):
        logits = model(
            esm_features=batch['esm'].to(device),
            knowledge_features=batch['knowledge'].to(device)
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_preds.append((probs > 0.5).astype(np.int32))
        all_labels.append(batch['labels'].numpy())
    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    def safe(m, avg): return m(y_true, y_pred, average=avg, zero_division=0)
    metrics = {
        'auroc': roc_auc_score(y_true, y_prob, average='macro'),
        'aupr': average_precision_score(y_true, y_prob, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': safe(precision_score, 'micro'),
        'recall_micro': safe(recall_score, 'micro'),
        'f1_micro': safe(f1_score, 'micro'),
        'precision_macro': safe(precision_score, 'macro'),
        'recall_macro': safe(recall_score, 'macro'),
        'f1_macro': safe(f1_score, 'macro'),
        'specificity': calculate_specificity(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true.ravel(), y_pred.ravel()) if y_true.shape[1] == 1
               else np.mean([matthews_corrcoef(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])])
    }

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(args.model_path).replace('.pt', '')
    np.save(f"{args.output_dir}/{name}_y_true.npy", y_true)
    np.save(f"{args.output_dir}/{name}_y_prob.npy", y_prob)
    np.save(f"{args.output_dir}/{name}_y_pred.npy", y_pred)
    pd.concat([pd.DataFrame(y_true, columns=args.label_cols),
               pd.DataFrame(y_prob, columns=[f'prob_{c}' for c in args.label_cols])], axis=1)\
      .to_csv(f"{args.output_dir}/{name}_predictions.csv", index=False)
    summary = {**metrics, "eval_time": datetime.now().isoformat(), "samples": len(dataset), "labels": args.label_cols}
    with open(f"{args.output_dir}/{name}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAUROC: {metrics['auroc']:.4f} | AUPR: {metrics['aupr']:.4f} | F1-micro: {metrics['f1_micro']:.4f}")
    print(f"Results saved → {args.output_dir}")
    return metrics
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--label_cols', nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default='./eval_out')
    args = parser.parse_args()
    for p in [args.model_path, args.data_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    evaluate_model(args)