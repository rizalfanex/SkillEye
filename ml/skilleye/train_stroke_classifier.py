"""
Train the ST-GCN stroke-type classifier on extracted THETIS skeletons.

Usage:
    python train_stroke_classifier.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/ml/results/stroke_classifier
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stroke_dataset import StrokeDataset, STROKE_CLASSES, load_records, subject_disjoint_split
from stgcn_model import STGCN


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for kpts, labels, _, _ in loader:
            kpts = kpts.to(device)
            logits = model(kpts)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    n = len(STROKE_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    return acc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    records, skipped = load_records(args.skeletons)
    print(f"loaded {len(records)} records ({skipped} skipped: unrecognized category)")

    train_records, val_records, val_subjects = subject_disjoint_split(
        records, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_records)} clips, val: {len(val_records)} clips "
          f"({len(val_subjects)} held-out subjects)")
    print("train label counts:", Counter(r["stroke"] for r in train_records))
    print("val label counts:  ", Counter(r["stroke"] for r in val_records))

    train_ds = StrokeDataset(train_records, augment=True)
    val_ds = StrokeDataset(val_records, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    label_counts = Counter(r["label"] for r in train_records)
    class_weights = torch.tensor(
        [len(train_records) / (len(STROKE_CLASSES) * label_counts.get(i, 1))
         for i in range(len(STROKE_CLASSES))],
        dtype=torch.float32,
    ).to(device)

    model = STGCN(num_classes=len(STROKE_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = -1.0
    best_cm = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for kpts, labels, _, _ in train_loader:
            kpts, labels = kpts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(kpts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_acc, val_cm = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc": val_acc})
        print(f"epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_cm = val_cm
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    metrics = {
        "best_val_accuracy": best_acc,
        "confusion_matrix": best_cm.tolist(),
        "classes": STROKE_CLASSES,
        "val_subjects": sorted(val_subjects),
        "train_clips": len(train_records),
        "val_clips": len(val_records),
        "history": history,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Final ===")
    print(f"best val accuracy: {best_acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(STROKE_CLASSES)
    print(best_cm)


if __name__ == "__main__":
    main()
