"""
Builds per-stroke-class expert-motion templates for quality scoring: for each
stroke class, the mean and std of each joint's average angle within each
phase (backswing/contact/follow_through), computed across that class's
expert-labeled clips from the TRAINING side of this project's standard
subject-disjoint split (same split, same seed, as the single-split
classifiers elsewhere in this repo -- not a new split). The held-out
validation subjects from that same split are saved alongside the templates
so the demo UI can restrict itself to clips that were never used to build
the template being compared against.

Usage:
    python build_expert_templates.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/results/quality_templates/templates.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stroke_dataset import load_records, subject_disjoint_split, STROKE_CLASSES
from quality.phases import split_phases, PHASES
from quality.angles import phase_mean_angles


def clip_phase_means(kpts):
    """kpts: (T, 17, 2). Returns {phase: {joint: float or None}}."""
    phases = split_phases(kpts)
    return {phase: phase_mean_angles(phases[phase]) for phase in PHASES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records, _ = load_records(args.skeletons)
    train_records, val_records, val_subjects = subject_disjoint_split(
        records, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_records)} clips, val (held out, demo-eligible): "
          f"{len(val_records)} clips ({len(val_subjects)} subjects)")

    expert_train = [r for r in train_records if r["skill_level"] == "expert"]
    print(f"expert clips in training side: {len(expert_train)}")

    # values[stroke][phase][joint] = list of per-clip phase-mean angles
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in expert_train:
        means = clip_phase_means(r["kpts"])
        for phase in PHASES:
            for joint, value in means[phase].items():
                if value is not None:
                    values[r["stroke"]][phase][joint].append(value)

    templates = {}
    for stroke in STROKE_CLASSES:
        templates[stroke] = {}
        for phase in PHASES:
            templates[stroke][phase] = {}
            joint_lists = values[stroke][phase]
            for joint, joint_values in joint_lists.items():
                arr = np.array(joint_values)
                templates[stroke][phase][joint] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n": int(len(arr)),
                }
            n_clips = len(next(iter(joint_lists.values()), []))
            print(f"  {stroke:16s} {phase:16s}: {n_clips} expert clips contributed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "templates": templates,
            "val_subjects": sorted(val_subjects),
            "phases": PHASES,
        }, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
