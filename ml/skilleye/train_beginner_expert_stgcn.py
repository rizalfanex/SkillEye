"""
Beginner-vs-expert classifier, v2: a proper ST-GCN on the raw skeleton
sequences (same architecture/augmentation as the stroke classifier), instead
of the 9 hand-crafted features + logistic regression from the first pass.

The first pass established (beginner_expert_check.py) that a real
population-level signal exists but a linear model on aggregate stats doesn't
generalize per-subject (58.6% held-out vs 54.6% baseline). A model that can
see the actual temporal trajectory -- not just its mean/variance -- should
do meaningfully better if the signal is really there. This is still just a
sanity-check pilot on THETIS's weak skill-level proxy label, not the final
quality-score model (that still needs Path A/B ground truth per HANDOFF.md).

Usage:
    python train_beginner_expert_stgcn.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/ml/results/beginner_expert_stgcn
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from stroke_dataset import (
    load_records, subject_disjoint_split,
    resample_time, add_velocity, mirror, random_temporal_crop, FIXED_T,
)
from stgcn_model import STGCN

SKILL_CLASSES = ["beginner", "expert"]


class SkillDataset(Dataset):
    def __init__(self, records, fixed_t=FIXED_T, augment=False, jitter_std=0.02):
        self.records = records
        self.fixed_t = fixed_t
        self.augment = augment
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        kpts = rec["kpts"]

        if self.augment:
            kpts = random_temporal_crop(kpts)
            if np.random.rand() < 0.5:
                kpts = mirror(kpts)

        kpts = resample_time(kpts, self.fixed_t)

        if self.augment:
            kpts = kpts + np.random.normal(0, self.jitter_std, kpts.shape).astype(np.float32)

        kpts = add_velocity(kpts)
        kpts = torch.from_numpy(kpts.astype(np.float32)).permute(2, 0, 1).contiguous()
        label = 1 if rec["skill_level"] == "expert" else 0
        return kpts, label, rec["subject_id"], rec["stroke"]


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for kpts, labels, _, _ in loader:
            kpts = kpts.to(device)
            preds = model(kpts).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist() if torch.is_tensor(labels) else list(labels))
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    return acc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    records, _ = load_records(args.skeletons)
    train_records, val_records, val_subjects = subject_disjoint_split(
        records, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_records)} clips, val: {len(val_records)} clips "
          f"({len(val_subjects)} held-out subjects)")

    train_labels = [1 if r["skill_level"] == "expert" else 0 for r in train_records]
    val_labels = [1 if r["skill_level"] == "expert" else 0 for r in val_records]
    print("train label counts:", Counter(train_labels))
    print("val label counts:  ", Counter(val_labels))
    baseline = max(np.mean(val_labels), 1 - np.mean(val_labels))
    print(f"majority-class baseline (val): {baseline:.4f}")

    train_ds = SkillDataset(train_records, augment=True)
    val_ds = SkillDataset(val_records, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    class_weights = torch.tensor(
        [len(train_labels) / (2 * n_neg), len(train_labels) / (2 * n_pos)],
        dtype=torch.float32,
    ).to(device)

    model = STGCN(num_classes=2, base_channels=32).to(device)
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
            kpts = kpts.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long).to(device)
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
        "baseline_accuracy": float(baseline),
        "confusion_matrix": best_cm.tolist(),
        "classes": SKILL_CLASSES,
        "val_subjects": sorted(val_subjects),
        "train_clips": len(train_records),
        "val_clips": len(val_records),
        "history": history,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Final ===")
    print(f"baseline accuracy:  {baseline:.4f}")
    print(f"best val accuracy:  {best_acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):", SKILL_CLASSES)
    print(best_cm)


if __name__ == "__main__":
    main()
