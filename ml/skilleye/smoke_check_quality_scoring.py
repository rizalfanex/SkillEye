"""
Smoke check for the quality-scoring system. No formal ground truth exists to
validate scores against yet (see docs/superpowers/specs/
2026-07-09-quality-scoring-and-ui-design.md, "Testing") -- this instead
confirms the one directional claim that must hold for the system to be
credible: held-out expert clips should score higher on average than
held-out beginner clips, per stroke class, against that stroke's expert
template. If any stroke fails this, FLAG_THRESHOLD/SCORE_SCALE or the angle
set (skilleye/quality/angles.py) need revisiting before using this in the
demo -- do not ship a quality scorer that rates beginners above experts.

Usage:
    python smoke_check_quality_scoring.py --skeletons E:/SkillEye/skeletons \
        --templates E:/SkillEye/ml/results/quality_templates/templates.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from stroke_dataset import load_records
from quality.score import score_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--templates", required=True)
    args = ap.parse_args()

    with open(args.templates) as f:
        data = json.load(f)
    templates = data["templates"]
    val_subjects = set(data["val_subjects"])

    records, _ = load_records(args.skeletons)
    val_records = [r for r in records if r["subject_id"] in val_subjects]
    print(f"scoring {len(val_records)} held-out clips ({len(val_subjects)} subjects)")

    scores_by_stroke_skill = defaultdict(list)
    for r in val_records:
        result = score_clip(r["kpts"], r["stroke"], templates)
        scores_by_stroke_skill[(r["stroke"], r["skill_level"])].append(result["overall_score"])

    strokes = sorted({stroke for stroke, _ in scores_by_stroke_skill})
    print(f"\n{'stroke':16s} {'expert mean':>12s} {'beginner mean':>14s} {'experts higher?':>16s}")
    all_ok = True
    for stroke in strokes:
        expert_scores = scores_by_stroke_skill.get((stroke, "expert"), [])
        beginner_scores = scores_by_stroke_skill.get((stroke, "beginner"), [])
        if not expert_scores or not beginner_scores:
            print(f"{stroke:16s} -- insufficient held-out data for one group, skipped --")
            continue
        expert_mean = float(np.mean(expert_scores))
        beginner_mean = float(np.mean(beginner_scores))
        ok = expert_mean > beginner_mean
        all_ok = all_ok and ok
        print(f"{stroke:16s} {expert_mean:12.1f} {beginner_mean:14.1f} {'yes' if ok else 'NO':>16s}")

    print()
    if all_ok:
        print("OVERALL: experts scored higher on every stroke with held-out data.")
    else:
        print("OVERALL: at least one stroke did NOT show experts scoring higher -- "
              "revisit SCORE_SCALE/angle set in skilleye/quality/ before using this in the demo.")


if __name__ == "__main__":
    main()
