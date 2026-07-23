# IMU (MPU6050) Sensor-Fusion Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a late-fusion prototype that adds a synthetic-IMU-derived branch to the existing beginner/expert `STGCN` classifier, proving the fusion architecture and code path work end to end — without any real MPU6050 hardware, and without claiming any accuracy result (the synthetic signal is redundant with the skeleton branch's own inputs by construction).

**Architecture:** Refactor `STGCN` to expose its pooled pre-classifier features via `extract_features()` (backward compatible — `forward()` is unchanged externally). A new `imu_fusion.py` module adds `synthetic_imu_from_skeleton()` (a clearly-labeled placeholder signal derived from wrist kinematics), `IMUEncoder` (a small 1D-CNN over the 6-channel IMU stream), and `FusedBeginnerExpertModel` (concatenates both branches' pooled features into one classifier head). A new, separate training script trains this on the existing subject-disjoint split, writing to its own `results/imu_fusion_prototype/` directory — the validated `results/beginner_expert_stgcn/` outputs and cross-validation numbers are never touched.

**Tech Stack:** Python, numpy, PyTorch (existing `STGCN`, unmodified externally), pytest. All in the `torch` conda env (verified in a prior session: `torch==2.11.0+cu130`, `torch.cuda.is_available() == True`).

## Global Constraints

- Python interpreter for all commands: `/c/Users/uclla/miniconda3/envs/torch/python` (the `torch` conda env — this repo's established environment).
- All new scripts/tests run with working directory `E:/SkillEye/skilleye` (matches the existing convention — modules imported as bare top-level, not a package).
- Reuse existing constants/functions, do not redefine: `stroke_dataset.{load_records, subject_disjoint_split, resample_time, add_velocity, mirror, random_temporal_crop, FIXED_T}`, `stgcn_model.STGCN`, `quality.phases.dominant_wrist_index`, `quality.keypoints.{L_WRIST, R_WRIST, L_ELBOW, R_ELBOW}`.
- IMU channel order is fixed and must not change: `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]` (indices 0-5), the standard MPU6050 order — chosen specifically so a real sensor log can later be substituted with no remapping. Channels 2, 3, 4 (`accel_z, gyro_x, gyro_y`) are exact zeros in the synthetic generator; this is a hard contract enforced by a test, not an approximation to relax later without updating the design spec.
- No accuracy claim: nothing produced by this plan may be reported as a validated or improved accuracy number. The prototype training script's own output and saved metrics must say so explicitly (see Task 4).
- This is a separate, additive prototype: it must not modify `results/beginner_expert_stgcn/`, `results/cross_validation/`, or any file under those paths, and must not change `STGCN.forward()`'s external behavior for any existing caller (`train_stroke_classifier.py`, `train_beginner_expert_stgcn.py`, `cross_validate.py`, `app.py`).
- Design spec: `docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md`. Follow it for anything not covered by a step below.

---

### Task 1: `STGCN.extract_features` refactor

**Files:**
- Modify: `skilleye/stgcn_model.py`
- Test: `skilleye/test_stgcn_model.py`

**Interfaces:**
- Consumes: nothing new (existing `STGCN` class, `stgcn_model.py:88-115`).
- Produces: `STGCN.extract_features(x: torch.Tensor) -> torch.Tensor` (input `(N, C, T, V)`, output `(N, base_channels*4)`). `STGCN.forward(x)` keeps its existing signature and output shape `(N, num_classes)`, now implemented as `self.fc(self.extract_features(x))`.

- [ ] **Step 1: Write the failing tests**

Create `skilleye/test_stgcn_model.py`:

```python
import torch

from stgcn_model import STGCN


def test_extract_features_shape():
    model = STGCN(num_classes=2, base_channels=32)
    x = torch.randn(4, 4, 64, 17)
    feats = model.extract_features(x)
    assert feats.shape == (4, 128)


def test_forward_equals_fc_of_extract_features():
    model = STGCN(num_classes=2, base_channels=32)
    model.eval()
    x = torch.randn(4, 4, 64, 17)
    with torch.no_grad():
        feats = model.extract_features(x)
        logits_via_features = model.fc(feats)
        logits_direct = model.forward(x)
    assert torch.allclose(logits_via_features, logits_direct)


def test_forward_output_shape_unchanged():
    model = STGCN(num_classes=6, base_channels=32)
    x = torch.randn(2, 4, 64, 17)
    logits = model(x)
    assert logits.shape == (2, 6)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest test_stgcn_model.py -v
```
Expected: `test_extract_features_shape` and `test_forward_equals_fc_of_extract_features` fail with `AttributeError: 'STGCN' object has no attribute 'extract_features'`. `test_forward_output_shape_unchanged` passes already (this documents current behavior, not a regression check yet).

- [ ] **Step 3: Implement the refactor**

In `skilleye/stgcn_model.py`, replace the `STGCN.forward` method (currently lines 104-115):

```python
    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=[2, 3])  # global average pool over T, V
        return self.fc(x)
```

with:

```python
    def extract_features(self, x):
        # x: (N, C, T, V) -> (N, base_channels*4) pooled feature vector
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        for block in self.blocks:
            x = block(x)

        return x.mean(dim=[2, 3])  # global average pool over T, V

    def forward(self, x):
        return self.fc(self.extract_features(x))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest test_stgcn_model.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Confirm no existing caller broke**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -c "
import torch
from stgcn_model import STGCN
from stroke_dataset import STROKE_CLASSES
m = STGCN(num_classes=len(STROKE_CLASSES))
sd = torch.load('E:/SkillEye/results/stroke_classifier_v2/best_model.pt', map_location='cpu')
m.load_state_dict(sd)
m.eval()
x = torch.randn(2, 4, 64, 17)
with torch.no_grad():
    out = m(x)
print('loaded existing checkpoint and ran forward() OK, output shape:', tuple(out.shape))
"
```
Expected: `loaded existing checkpoint and ran forward() OK, output shape: (2, 6)` with no errors — confirms the refactor didn't change the checkpoint's state-dict keys or break loading.

- [ ] **Step 6: Commit**

```bash
cd E:/SkillEye
git add skilleye/stgcn_model.py skilleye/test_stgcn_model.py
git commit -m "Refactor STGCN to expose extract_features() for fusion prototype"
```

---

### Task 2: Synthetic IMU signal generator

**Files:**
- Create: `skilleye/imu_fusion.py`
- Test: `skilleye/test_imu_fusion.py`

**Interfaces:**
- Consumes: `quality.phases.dominant_wrist_index` (existing), `quality.keypoints.{L_WRIST, R_WRIST, L_ELBOW, R_ELBOW}` (existing).
- Produces: `imu_fusion.synthetic_imu_from_skeleton(kpts: np.ndarray) -> np.ndarray` — input `(T, 17, 2)`, output `(T, 6)`, channel order `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`.

- [ ] **Step 1: Write the failing tests**

Create `skilleye/test_imu_fusion.py`:

```python
import numpy as np

from imu_fusion import synthetic_imu_from_skeleton
from quality.keypoints import R_WRIST, R_ELBOW


def make_moving_clip(T=64):
    """Right wrist moves on a small circle around a fixed right elbow (left
    side stays at the origin, so dominant_wrist_index picks the right wrist,
    same convention as quality/test_phases.py)."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    kpts[:, R_ELBOW, 0] = 0.0
    kpts[:, R_ELBOW, 1] = 0.0
    kpts[:, R_WRIST, 0] = np.cos(t * 0.2)
    kpts[:, R_WRIST, 1] = np.sin(t * 0.2)
    return kpts


def test_output_shape():
    kpts = make_moving_clip(64)
    imu = synthetic_imu_from_skeleton(kpts)
    assert imu.shape == (64, 6)


def test_deterministic():
    kpts = make_moving_clip(64)
    imu1 = synthetic_imu_from_skeleton(kpts)
    imu2 = synthetic_imu_from_skeleton(kpts)
    np.testing.assert_array_equal(imu1, imu2)


def test_placeholder_channels_are_exactly_zero():
    kpts = make_moving_clip(64)
    imu = synthetic_imu_from_skeleton(kpts)
    assert np.all(imu[:, 2] == 0.0)  # accel_z
    assert np.all(imu[:, 3] == 0.0)  # gyro_x
    assert np.all(imu[:, 4] == 0.0)  # gyro_y


def test_derived_channels_are_non_constant():
    kpts = make_moving_clip(64)
    imu = synthetic_imu_from_skeleton(kpts)
    assert imu[:, 0].std() > 1e-6  # accel_x
    assert imu[:, 1].std() > 1e-6  # accel_y
    assert imu[:, 5].std() > 1e-6  # gyro_z


def test_static_clip_gives_all_zero_signal():
    kpts = np.zeros((64, 17, 2), dtype=np.float32)
    imu = synthetic_imu_from_skeleton(kpts)
    assert np.all(imu == 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest test_imu_fusion.py -v
```
Expected: `ModuleNotFoundError: No module named 'imu_fusion'`

- [ ] **Step 3: Implement the synthetic IMU generator**

Create `skilleye/imu_fusion.py`:

```python
"""
Late-fusion prototype: adds a racket-mounted-IMU-shaped input branch to the
beginner/expert STGCN classifier. No MPU6050 hardware exists yet (see
docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md), so
synthetic_imu_from_skeleton() below is a clearly-labeled placeholder signal,
not a real sensor reading -- its purpose is to prove the fusion architecture
and code path work, not to claim any accuracy result. When real MPU6050 logs
exist, replacing this function's call site with a real-data loader (resampled
to the same T the same way) is the only change needed.
"""
import numpy as np
import torch
import torch.nn as nn

from quality.phases import dominant_wrist_index
from quality.keypoints import L_WRIST, R_WRIST, L_ELBOW, R_ELBOW
from stgcn_model import STGCN


def synthetic_imu_from_skeleton(kpts):
    """kpts: (T, 17, 2) skeleton, already resampled to the model's fixed T.
    Returns (T, 6): [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z], the
    standard MPU6050 channel order (chosen so a real sensor log can later
    drop in with no remapping).

    Only channels honestly derivable from 2D pose carry a signal:
    accel_x/y (indices 0, 1) from the dominant wrist's 2nd finite difference
    (acceleration), gyro_z (index 5) from the forearm segment's angular
    velocity. accel_z, gyro_x, gyro_y (indices 2, 3, 4) are exact zeros --
    2D pose has no depth/out-of-plane rotation to derive them from honestly.
    This signal is, by construction, redundant with the skeleton branch's own
    position/velocity channels -- it is a code-path placeholder, not an
    independent modality.
    """
    T = kpts.shape[0]
    wrist_idx = dominant_wrist_index(kpts)
    elbow_idx = R_ELBOW if wrist_idx == R_WRIST else L_ELBOW

    wrist_pos = kpts[:, wrist_idx]  # (T, 2)
    velocity = np.zeros_like(wrist_pos)
    velocity[1:] = wrist_pos[1:] - wrist_pos[:-1]
    accel = np.zeros_like(wrist_pos)
    accel[1:] = velocity[1:] - velocity[:-1]

    elbow_pos = kpts[:, elbow_idx]
    forearm_vec = wrist_pos - elbow_pos
    angle = np.arctan2(forearm_vec[:, 1], forearm_vec[:, 0])
    gyro_z = np.zeros((T,), dtype=np.float32)
    gyro_z[1:] = angle[1:] - angle[:-1]

    imu = np.zeros((T, 6), dtype=np.float32)
    imu[:, 0] = accel[:, 0]
    imu[:, 1] = accel[:, 1]
    imu[:, 5] = gyro_z
    return imu
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest test_imu_fusion.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add skilleye/imu_fusion.py skilleye/test_imu_fusion.py
git commit -m "Add synthetic IMU signal generator (placeholder for real MPU6050 data)"
```

---

### Task 3: `IMUEncoder` + `FusedBeginnerExpertModel`

**Files:**
- Modify: `skilleye/imu_fusion.py`
- Modify: `skilleye/test_imu_fusion.py`

**Interfaces:**
- Consumes: `stgcn_model.STGCN` (existing, with `extract_features` from Task 1).
- Produces: `imu_fusion.IMUEncoder(in_channels=6, hidden=32)` — `nn.Module`, `forward(x: (N, 6, T)) -> (N, hidden)`. `imu_fusion.FusedBeginnerExpertModel(num_classes=2, skeleton_channels=4, imu_channels=6, skeleton_base_channels=32, imu_hidden=32)` — `nn.Module`, `forward(x_skeleton: (N, skeleton_channels, T, 17), x_imu: (N, imu_channels, T)) -> (N, num_classes)`.

- [ ] **Step 1: Write the failing tests**

Append to `skilleye/test_imu_fusion.py`:

```python
from imu_fusion import IMUEncoder, FusedBeginnerExpertModel


def test_imu_encoder_output_shape():
    encoder = IMUEncoder()
    x = torch.randn(4, 6, 64)
    out = encoder(x)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()


def test_fused_model_output_shape():
    model = FusedBeginnerExpertModel(num_classes=2)
    x_skeleton = torch.randn(4, 4, 64, 17)
    x_imu = torch.randn(4, 6, 64)
    logits = model(x_skeleton, x_imu)
    assert logits.shape == (4, 2)


def test_fused_model_gradient_reaches_imu_branch():
    model = FusedBeginnerExpertModel(num_classes=2)
    x_skeleton = torch.randn(4, 4, 64, 17)
    x_imu = torch.randn(4, 6, 64)
    labels = torch.tensor([0, 1, 0, 1])

    logits = model(x_skeleton, x_imu)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    imu_grad_norms = [
        p.grad.norm().item() for p in model.imu_branch.parameters() if p.grad is not None
    ]
    assert len(imu_grad_norms) > 0
    assert all(g == g for g in imu_grad_norms)  # no NaNs (NaN != NaN)
    assert sum(imu_grad_norms) > 0.0
```

Add `import torch` at the top of `skilleye/test_imu_fusion.py` if not already present from Task 2 (Task 2's test file only imports `numpy as np` — add `import torch` alongside it).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest test_imu_fusion.py -v
```
Expected: `ImportError: cannot import name 'IMUEncoder' from 'imu_fusion'`

- [ ] **Step 3: Implement `IMUEncoder` and `FusedBeginnerExpertModel`**

Append to `skilleye/imu_fusion.py`:

```python
class IMUEncoder(nn.Module):
    """Small 1D-CNN over the 6-channel IMU stream, pooled to one feature
    vector per clip. Kept small deliberately, same reasoning as STGCN's own
    size: this prototype has no real IMU data volume to justify more
    capacity."""

    def __init__(self, in_channels=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (N, in_channels, T) -> (N, hidden)
        x = self.net(x)
        return x.mean(dim=2)


class FusedBeginnerExpertModel(nn.Module):
    """Late-fusion prototype: the existing STGCN skeleton branch (via
    extract_features) plus an IMUEncoder branch, concatenated before one
    classification head. See module docstring for the synthetic-data
    caveat -- this architecture is real, its current training data is not."""

    def __init__(self, num_classes=2, skeleton_channels=4, imu_channels=6,
                 skeleton_base_channels=32, imu_hidden=32):
        super().__init__()
        self.skeleton_branch = STGCN(
            num_classes=num_classes, in_channels=skeleton_channels,
            base_channels=skeleton_base_channels,
        )
        self.imu_branch = IMUEncoder(in_channels=imu_channels, hidden=imu_hidden)
        self.fc = nn.Linear(skeleton_base_channels * 4 + imu_hidden, num_classes)

    def forward(self, x_skeleton, x_imu):
        skeleton_feats = self.skeleton_branch.extract_features(x_skeleton)
        imu_feats = self.imu_branch(x_imu)
        combined = torch.cat([skeleton_feats, imu_feats], dim=1)
        return self.fc(combined)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest test_imu_fusion.py -v
```
Expected: `8 passed` (5 from Task 2 + 3 new)

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add skilleye/imu_fusion.py skilleye/test_imu_fusion.py
git commit -m "Add IMUEncoder and FusedBeginnerExpertModel (late-fusion prototype)"
```

---

### Task 4: Prototype training script

**Files:**
- Create: `skilleye/train_beginner_expert_fusion_prototype.py`

**Interfaces:**
- Consumes: `stroke_dataset.{load_records, subject_disjoint_split, resample_time, add_velocity, mirror, random_temporal_crop, FIXED_T}` (existing), `imu_fusion.{synthetic_imu_from_skeleton, FusedBeginnerExpertModel}` (Tasks 2-3).
- Produces: `results/imu_fusion_prototype/prototype_model.pt`, `results/imu_fusion_prototype/metrics.json` (shaped `{"note": str, "history": [{"epoch", "train_loss", "val_acc_NOT_A_BENCHMARK"}], "train_clips": int, "val_clips": int}`). No importable interface — this is a standalone script, following the pattern of `train_beginner_expert_stgcn.py`.

- [ ] **Step 1: Implement the training script**

Create `skilleye/train_beginner_expert_fusion_prototype.py`:

```python
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
        --out E:/SkillEye/results/imu_fusion_prototype
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
```

- [ ] **Step 2: Run it for real**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" train_beginner_expert_fusion_prototype.py --skeletons E:/SkillEye/skeletons --out E:/SkillEye/results/imu_fusion_prototype
```
Expected: prints the device, the synthetic-data disclaimer, train/val clip counts, then 5 epoch lines. Confirm `train_loss` on the epoch-5 line is lower than on the epoch-1 line (the loop is training, not stuck/broken) — this is the pass/fail check for this step, not any particular accuracy value. Ends with `saved -> ...imu_fusion_prototype`.

- [ ] **Step 3: If loss does not decrease, debug before proceeding**

If epoch-5 loss is not lower than epoch-1 loss, do not proceed to Task 5 yet. Check, in order: (a) that `imu_fusion.py`'s `FusedBeginnerExpertModel.forward` concatenates both branches before `self.fc` (a bug here can make the model ignore one branch and plateau); (b) that `--lr 1e-3` isn't diverging (print a few more epochs with `--epochs 15` temporarily to see if loss is oscillating vs. genuinely stuck). Loss should decrease with this small, simple setup — if it doesn't after checking those two things, that's a bug to fix, not something to work around by lowering expectations.

- [ ] **Step 4: Verify the output directory does not touch existing results**

```bash
git status --short results/
```
Expected: only `results/imu_fusion_prototype/` appears as untracked/new — no modified or deleted files under `results/beginner_expert_stgcn/` or `results/cross_validation/`.

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add skilleye/train_beginner_expert_fusion_prototype.py results/imu_fusion_prototype/
git commit -m "Add IMU-fusion prototype training script (synthetic data, code-path check only)"
```

---

### Task 5: Documentation — fold the prototype into the README

**Files:**
- Modify: `E:/SkillEye/README.md`

**Interfaces:**
- Consumes: nothing new — this task only writes prose, no code.

- [ ] **Step 1: Add a Methodology subsection**

In `README.md`, after the existing "### 2.6 Quality Scoring System" subsection and before "## 3. Results", add:

```markdown
### 2.7 Sensor-Fusion Extension (Prototype)

A team discussion identified a concrete limitation already described in Section 4.1: a
single frontal 2D camera cannot see racket-face angle or wrist-snap dynamics at contact —
exactly the kind of high-frequency, off-camera-plane signal a racket-mounted MPU6050
(accelerometer + gyroscope) would capture directly. No hardware has been acquired yet, so
this section documents an architecture prototype, not a hardware-validated result.

**Architecture** (`skilleye/imu_fusion.py`): `STGCN` is refactored to expose its pooled
pre-classifier features via `extract_features()` (`skilleye/stgcn_model.py`, backward
compatible — every existing caller is unaffected). A small `IMUEncoder` (1D-CNN over a
6-channel accelerometer+gyroscope stream) is fused with that skeleton branch via
concatenation before one classification head (`FusedBeginnerExpertModel`), targeting the
beginner/expert distinction specifically, since that is the axis this sensor is meant to
inform.

**Synthetic data, stated plainly**: because no MPU6050 hardware exists yet,
`synthetic_imu_from_skeleton()` derives a placeholder signal from the skeleton itself
(wrist acceleration, forearm angular velocity) rather than from a real sensor. This signal
is, by construction, redundant with information the skeleton branch already has access to
— so `train_beginner_expert_fusion_prototype.py`'s output is a demonstration that the
fusion architecture and training loop work end to end, **not an accuracy result**, and its
numbers must not be cited alongside Section 3.2's cross-validated 82.4% ± 3.8%.

**Path to real data**: a documented collection protocol (MPU6050 + ESP32, ~100-200 Hz
logging, a tap-based manual sync event between the video and IMU streams, resampled to the
same fixed frame count skeletons already use) is in
`docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md` — swapping
`synthetic_imu_from_skeleton()`'s call site for a real-data loader is the only code change
needed once hardware and recordings exist.
```

- [ ] **Step 2: Update Conclusion and Future Work item 3**

In `README.md`, "## 5. Conclusion and Future Work" currently has as item 3:

```markdown
3. **Own side-view, higher-frame-rate recordings** — required for the joint-angle-based
   error-detection rules in Section 4.7, and expected to resolve the volley/groundstroke
   confusion identified in Section 4.1.
```

Replace it with:

```markdown
3. **Own side-view, higher-frame-rate recordings, plus real MPU6050 hardware** — required
   for the joint-angle-based error-detection rules in Section 4.7, expected to resolve the
   volley/groundstroke confusion identified in Section 4.1, and needed to move Section
   2.7's sensor-fusion prototype from synthetic to real data. The collection protocol
   (hardware, sampling rate, synchronization method) is scoped in
   `docs/superpowers/specs/2026-07-23-imu-fusion-prototype-design.md`; once real logs
   exist, retraining and cross-validating `FusedBeginnerExpertModel` on them — following
   the same 5-fold rigor as Section 3.2 — is the follow-up this prototype sets up for.
```

- [ ] **Step 3: Update the Reproducibility repo layout**

In `README.md`, in "## 6. Reproducibility", the `skilleye/` block currently has this line:

```
  app.py                           Streamlit demo UI (§2.6) -- run: streamlit run app.py
```

Add these lines immediately after it:

```
  imu_fusion.py                    synthetic IMU signal + fusion model prototype (§2.7)
  train_beginner_expert_fusion_prototype.py   trains the fusion prototype (synthetic data, §2.7)
```

And in the `results/` block, add this line after the `quality_templates/` line:

```
  imu_fusion_prototype/             prototype-only metrics (synthetic IMU data, §2.7) -- not a benchmark
```

- [ ] **Step 4: Commit and push**

```bash
cd E:/SkillEye
git add README.md
git commit -m "Document the IMU sensor-fusion prototype in the README"
git push
```
