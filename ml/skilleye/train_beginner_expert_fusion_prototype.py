"""
Trains FusedBeginnerExpertModel (skilleye/imu_fusion.py) on the existing
subject-disjoint split, using SYNTHETIC IMU data derived from the skeleton
itself (synthetic_imu_from_skeleton) -- no real MPU6050 hardware exists yet.

This is a code-path prototype, not a benchmark: the synthetic IMU channel
carries no information beyond what the skeleton branch already sees, so any
accuracy number this script prints or saves is NOT a validated result and
must never be cited alongside results/beginner_expert_stgcn/'s or
results/cross_validation/'s real, cross-validated numbers. See
docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md.

This script writes only to results/imu_fusion_prototype/ -- it never
modifies results/beginner_expert_stgcn/ or results/cross_validation/.

Usage:
    python train_beginner_expert_fusion_prototype.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/ml/results/imu_fusion_prototype
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from stroke_dataset import (
    load_records, subject_disjoint_split,
    resample_time, add_velocity, mirror, random_temporal_crop, FIXED_T,
)
from imu_fusion import synthetic_imu_from_skeleton, FusedBeginnerExpertModel


class FusionSkillDataset(Dataset):
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

        imu = synthetic_imu_from_skeleton(kpts)  # (T, 6), same resampled/augmented clip

        kpts_with_vel = add_velocity(kpts)  # (T, 17, 4)
        skeleton_tensor = torch.from_numpy(
            kpts_with_vel.astype(np.float32)).permute(2, 0, 1).contiguous()  # (4, T, 17)
        imu_tensor = torch.from_numpy(imu.astype(np.float32)).permute(1, 0).contiguous()  # (6, T)

        label = 1 if rec["skill_level"] == "expert" else 0
        return skeleton_tensor, imu_tensor, label, rec["subject_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("NOTE: this is a code-path prototype using SYNTHETIC IMU data derived from the "
          "skeleton itself. Any accuracy number below is NOT a validated result -- see "
          "docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md.")

    records, _ = load_records(args.skeletons)
    train_records, val_records, val_subjects = subject_disjoint_split(
        records, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_records)} clips, val: {len(val_records)} clips "
          f"({len(val_subjects)} held-out subjects)")

    train_ds = FusionSkillDataset(train_records, augment=True)
    val_ds = FusionSkillDataset(val_records, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FusedBeginnerExpertModel(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x_skel, x_imu, labels, _ in train_loader:
            x_skel = x_skel.to(device)
            x_imu = x_imu.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long).to(device)
            optimizer.zero_grad()
            logits = model(x_skel, x_imu)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_skel, x_imu, labels, _ in val_loader:
                x_skel = x_skel.to(device)
                x_imu = x_imu.to(device)
                labels = torch.as_tensor(labels, dtype=torch.long).to(device)
                preds = model(x_skel, x_imu).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
        val_acc = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc_NOT_A_BENCHMARK": val_acc})
        print(f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_acc(diagnostic only)={val_acc:.4f}")

    torch.save(model.state_dict(), out_dir / "prototype_model.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "note": (
                "PROTOTYPE ONLY: trained on synthetic skeleton-derived IMU data, not real "
                "MPU6050 hardware. val_acc_NOT_A_BENCHMARK is a code-path sanity check "
                "(loss should decrease over epochs), not a validated accuracy result -- see "
                "docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md."
            ),
            "history": history,
            "train_clips": len(train_records),
            "val_clips": len(val_records),
        }, f, indent=2)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
