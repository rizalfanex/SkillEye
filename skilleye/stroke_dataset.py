"""
Dataset for THETIS skeleton JSONs -> stroke-type classification.

Merges THETIS's 12 sub-classes into the 5 strokes the SkillEye proposal
actually cares about (forehand/backhand/volley/serve, plus smash as its own
class since it's biomechanically an overhead motion, not a serve or a
groundstroke). Splits by subject id, not by clip, since clips from the same
subject/action are highly correlated (same person, same gym, same camera) --
a clip-level split would leak and overstate accuracy.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORY_TO_STROKE = {
    "backhand": "backhand",
    "backhand2hands": "backhand",
    "backhand_slice": "backhand",
    "backhand_volley": "volley",
    "forehand_flat": "forehand",
    "forehand_openstands": "forehand",
    "forehand_slice": "forehand",
    "forehand_volley": "volley",
    "flat_service": "serve",
    "kick_service": "serve",
    "slice_service": "serve",
    "smash": "smash",
}
STROKE_CLASSES = ["backhand", "forehand", "volley", "serve", "smash"]
STROKE_TO_IDX = {s: i for i, s in enumerate(STROKE_CLASSES)}

FIXED_T = 64  # resample every clip to this many frames


def resample_time(kpts, target_t):
    """Linearly resample (T, V, C) along the time axis to (target_t, V, C)."""
    T = kpts.shape[0]
    if T == target_t:
        return kpts
    src_idx = np.linspace(0, T - 1, num=T)
    dst_idx = np.linspace(0, T - 1, num=target_t)
    V, C = kpts.shape[1], kpts.shape[2]
    out = np.empty((target_t, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(dst_idx, src_idx, kpts[:, v, c])
    return out


class StrokeDataset(Dataset):
    def __init__(self, records, fixed_t=FIXED_T):
        """records: list of dicts with keys kpts (T,V,C) ndarray, label, subject_id, skill_level."""
        self.records = records
        self.fixed_t = fixed_t

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        kpts = resample_time(rec["kpts"], self.fixed_t)  # (T, V, C)
        kpts = torch.from_numpy(kpts).permute(2, 0, 1).contiguous()  # (C, T, V)
        return kpts, rec["label"], rec["subject_id"], rec["skill_level"]


def load_records(skeleton_root):
    """Walk the skeleton JSON tree once, merge categories, return raw records
    (kept in memory as plain arrays; the full extracted set is small enough)."""
    root = Path(skeleton_root)
    records = []
    skipped_category = 0
    for json_path in sorted(root.rglob("*.json")):
        with open(json_path) as f:
            d = json.load(f)
        category = d["category"]
        stroke = CATEGORY_TO_STROKE.get(category)
        if stroke is None:
            skipped_category += 1
            continue
        kpts = np.asarray(d["keypoints_normalized"], dtype=np.float32)  # (T, V, C)
        records.append({
            "kpts": kpts,
            "label": STROKE_TO_IDX[stroke],
            "stroke": stroke,
            "subject_id": d["subject_id"],
            "skill_level": d["skill_level"],
            "source": str(json_path),
        })
    return records, skipped_category


def subject_disjoint_split(records, val_frac=0.2, seed=42):
    """Split by subject id so no subject appears in both train and val.
    Stratifies the subject-level split by skill_level so both splits keep a
    beginner/expert mix (THETIS's beginner/expert split is subject-level:
    p1-p31 beginner, p32-p55 expert)."""
    rng = np.random.RandomState(seed)

    subjects_by_level = {"beginner": set(), "expert": set()}
    for r in records:
        subjects_by_level[r["skill_level"]].add(r["subject_id"])

    val_subjects = set()
    for level, subjects in subjects_by_level.items():
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_frac)))
        val_subjects.update(subjects[:n_val])

    train = [r for r in records if r["subject_id"] not in val_subjects]
    val = [r for r in records if r["subject_id"] in val_subjects]
    return train, val, val_subjects
