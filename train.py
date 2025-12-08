import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from model import MAPLE
from loss import FocalLoss
from data import UnifiedProteinDataset


def train(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        esm = batch["esm"].to(device)
        know = batch["knowledge"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(esm_features=esm, knowledge_features=know)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device):
    model.eval()
    prob_list = []
    label_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            esm = batch["esm"].to(device)
            know = batch["knowledge"].to(device)
            labels = batch["labels"]

            logits = model(esm_features=esm, knowledge_features=know)
            probs = torch.sigmoid(logits).cpu().numpy()

            prob_list.append(probs)
            label_list.append(labels.numpy())

    y_prob = np.vstack(prob_list)
    y_true = np.vstack(label_list)

    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_prob > 0.5),
            "precision_micro": precision_score(y_true, y_prob > 0.5, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true, y_prob > 0.5, average="micro", zero_division=0),
            "f1_micro": f1_score(y_true, y_prob > 0.5, average="micro", zero_division=0),
            "roc_auc_micro": roc_auc_score(y_true, y_prob, average="micro"),
            "ap_micro": average_precision_score(y_true, y_prob, average="micro"),
        }
    except:
        metrics = {k: 0.0 for k in ["accuracy", "precision_micro", "recall_micro", "f1_micro", "roc_auc_micro", "ap_micro"]}

    return metrics


def run_training(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    df = pd.read_csv(args.data_csv)

    dataset = UnifiedProteinDataset(
        csv_file=df,
        sequence_col="sequence",
        label_cols=args.label_cols,
        device=device,
    )

    with torch.no_grad():
        sample = dataset[0]
        esm_dim = sample["esm"].size(1)
        knowledge_dim = sample["knowledge"].size(1)

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=args.val_ratio, random_state=42, shuffle=True)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=6,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=6,
        pin_memory=True,
    )

    model = MAPLE(
        linsize=args.hidden_size,
        lindropout=args.dropout,
        num_labels=len(args.label_cols),
        esm_dim=esm_dim,
        knowledge_dim=knowledge_dim,
        base_dim=768,
        bimamba_dim=384,
    ).to(device)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.scheduler_patience, verbose=True)

    best_f1 = 0.0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        f1 = val_metrics["f1_micro"]

        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "esm_dim": esm_dim,
                "knowledge_dim": knowledge_dim,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_model.pt"))
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break

    os.makedirs(args.save_dir, exist_ok=True)
    summary = {
        "best_f1_micro": float(best_f1),
        "esm_dim": esm_dim,
        "knowledge_dim": knowledge_dim,
        "num_labels": len(args.label_cols),
        "label_names": args.label_cols,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "final_metrics": val_metrics,
        "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--label_cols", nargs="+", required=True)
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

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_training(args)