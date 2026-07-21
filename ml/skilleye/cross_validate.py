"""
5-fold subject-disjoint cross-validation for both the stroke classifier and
the beginner/expert classifier.

A single held-out split (what train_stroke_classifier.py / train_beginner_expert_stgcn.py
report) depends on which ~20% of subjects happened to land in validation -- with
only 55 subjects total, that can swing several points either way. Cross-validation
retrains k times over different held-out subject groups and reports mean +/- std,
which is the number that actually holds up to scrutiny.

Usage:
    python cross_validate.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/results/cross_validation --k 5 --epochs 80
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stroke_dataset import StrokeDataset, STROKE_CLASSES, load_records, subject_kfold_split
from train_beginner_expert_stgcn import SkillDataset, SKILL_CLASSES
from stgcn_model import STGCN


def train_and_eval_fold(train_records, val_records, num_classes, label_key,
                         dataset_cls, epochs, batch_size, lr, device):
    # Both StrokeDataset and SkillDataset already emit the correct numeric label
    # (stroke index, or 0/1 skill) as the 2nd tuple element -- no need to
    # recompute it from the 4th (which differs: skill_level vs. stroke type).
    if label_key == "label":
        train_labels = [r["label"] for r in train_records]
    else:
        train_labels = [1 if r["skill_level"] == "expert" else 0 for r in train_records]

    train_ds = dataset_cls(train_records, augment=True)
    val_ds = dataset_cls(val_records, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = torch.tensor(
        [len(train_labels) / (num_classes * max(c, 1)) for c in counts],
        dtype=torch.float32,
    ).to(device)

    model = STGCN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    val_acc_history = []
    best_acc, best_cm = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for kpts, labels, _, _ in train_loader:
            kpts, labels = kpts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(kpts), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for kpts, labels, _, _ in val_loader:
                kpts = kpts.to(device)
                preds = model(kpts).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        acc = float((all_preds == all_labels).mean())
        val_acc_history.append(acc)

        if acc > best_acc:
            best_acc = acc
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(all_labels, all_preds):
                cm[t, p] += 1
            best_cm = cm

    return best_acc, best_cm, val_acc_history


def run_cv(records, num_classes, label_key, dataset_cls, class_names, k, epochs, batch_size, lr, device, seed):
    folds = subject_kfold_split(records, k=k, seed=seed)
    fold_accs = []
    fold_histories = []
    summed_cm = np.zeros((num_classes, num_classes), dtype=int)

    for i, (train_records, val_records, val_subjects) in enumerate(folds):
        print(f"  fold {i+1}/{k}: train={len(train_records)} val={len(val_records)} "
              f"held-out subjects={sorted(val_subjects)}")
        acc, cm, history = train_and_eval_fold(
            train_records, val_records, num_classes, label_key, dataset_cls,
            epochs, batch_size, lr, device)
        print(f"    fold {i+1} best acc: {acc:.4f}")
        fold_accs.append(acc)
        fold_histories.append(history)
        summed_cm += cm

    fold_accs = np.array(fold_accs)
    return {
        "classes": class_names,
        "k": k,
        "fold_accuracies": fold_accs.tolist(),
        "mean_accuracy": float(fold_accs.mean()),
        "std_accuracy": float(fold_accs.std()),
        "summed_confusion_matrix": summed_cm.tolist(),
        "fold_val_acc_histories": fold_histories,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    records, _ = load_records(args.skeletons)
    print(f"loaded {len(records)} records")

    print("\n=== Stroke classifier cross-validation ===")
    stroke_cv = run_cv(records, len(STROKE_CLASSES), "label", StrokeDataset, STROKE_CLASSES,
                        args.k, args.epochs, args.batch_size, args.lr, device, args.seed)
    print(f"stroke classifier: {stroke_cv['mean_accuracy']:.4f} +/- {stroke_cv['std_accuracy']:.4f} "
          f"(folds: {[f'{a:.4f}' for a in stroke_cv['fold_accuracies']]})")
    with open(out_dir / "stroke_classifier_cv.json", "w") as f:
        json.dump(stroke_cv, f, indent=2)

    print("\n=== Beginner/expert cross-validation ===")
    skill_cv = run_cv(records, 2, "skill", SkillDataset, SKILL_CLASSES,
                       args.k, args.epochs, args.batch_size, args.lr, device, args.seed)
    print(f"beginner/expert: {skill_cv['mean_accuracy']:.4f} +/- {skill_cv['std_accuracy']:.4f} "
          f"(folds: {[f'{a:.4f}' for a in skill_cv['fold_accuracies']]})")
    with open(out_dir / "beginner_expert_cv.json", "w") as f:
        json.dump(skill_cv, f, indent=2)

    print("\nsaved ->", out_dir)


if __name__ == "__main__":
    main()
