"""
Cheap beginner-vs-expert sanity check (HANDOFF.md priority #3).

THETIS's skill-level metadata (p1-p31 beginner, p32-p55 expert) is a weak,
free proxy for motion quality -- not a substitute for real coach ratings,
but a fast way to check whether the extracted skeleton data carries *any*
beginner/expert signal before investing in coach recruitment (Path B).

Approach: compute a handful of interpretable, hand-crafted motion features
per clip (speed, smoothness/jerk, joint-angle variance, reach/extension),
then (1) check per-feature whether beginner and expert clips actually differ
(Welch's t-test), and (2) fit a simple logistic regression (on a
subject-disjoint split, controlling for stroke type) to see whether those
features predict skill level above the majority-class baseline.

This deliberately does NOT reuse the stroke classifier's learned features --
the point is to test the *raw normalized skeleton data* for a quality
signal, independent of what the stroke-type model happened to pick up.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stroke_dataset import load_records, subject_disjoint_split, STROKE_CLASSES

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def joint_angle_series(kpts, a, b, c):
    """Angle at joint b formed by points a-b-c, per frame. kpts: (T, 17, 2)."""
    v1 = kpts[:, a] - kpts[:, b]
    v2 = kpts[:, c] - kpts[:, b]
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = np.clip(n1 * n2, 1e-6, None)
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cos_theta)


def extract_features(kpts):
    """kpts: (T, 17, 2) normalized (hip-anchored, torso-scaled). Returns dict of scalars."""
    vel = np.diff(kpts, axis=0)  # (T-1, 17, 2)
    speed = np.linalg.norm(vel, axis=-1)  # (T-1, 17)
    jerk = np.diff(vel, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=-1)  # (T-2, 17)

    l_elbow_ang = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    r_elbow_ang = joint_angle_series(kpts, R_SHOULDER, R_ELBOW, R_WRIST)
    l_knee_ang = joint_angle_series(kpts, L_HIP, L_KNEE, L_ANKLE)
    r_knee_ang = joint_angle_series(kpts, R_HIP, R_KNEE, R_ANKLE)

    hip_center = (kpts[:, L_HIP] + kpts[:, R_HIP]) / 2  # ~= origin, kept for clarity
    l_wrist_dist = np.linalg.norm(kpts[:, L_WRIST] - hip_center, axis=-1)
    r_wrist_dist = np.linalg.norm(kpts[:, R_WRIST] - hip_center, axis=-1)

    return {
        "mean_speed": float(speed.mean()),
        "std_speed": float(speed.std()),
        "mean_jerk": float(jerk_mag.mean()),
        "l_elbow_angle_var": float(l_elbow_ang.var()),
        "r_elbow_angle_var": float(r_elbow_ang.var()),
        "l_knee_angle_var": float(l_knee_ang.var()),
        "r_knee_angle_var": float(r_knee_ang.var()),
        "l_wrist_extension_range": float(l_wrist_dist.max() - l_wrist_dist.min()),
        "r_wrist_extension_range": float(r_wrist_dist.max() - r_wrist_dist.min()),
    }


FEATURE_NAMES = [
    "mean_speed", "std_speed", "mean_jerk",
    "l_elbow_angle_var", "r_elbow_angle_var",
    "l_knee_angle_var", "r_knee_angle_var",
    "l_wrist_extension_range", "r_wrist_extension_range",
]


def main():
    records, _ = load_records("E:/SkillEye/skeletons")
    print(f"loaded {len(records)} clips")

    for r in records:
        r["features"] = extract_features(r["kpts"])

    # --- 1. per-feature statistical comparison (beginner vs expert, pooled across strokes) ---
    beginner_feats = {k: [] for k in FEATURE_NAMES}
    expert_feats = {k: [] for k in FEATURE_NAMES}
    for r in records:
        bucket = beginner_feats if r["skill_level"] == "beginner" else expert_feats
        for k in FEATURE_NAMES:
            bucket[k].append(r["features"][k])

    print("\n=== Feature comparison: beginner vs expert (Welch's t-test) ===")
    print(f"{'feature':<26}{'beginner_mean':>15}{'expert_mean':>15}{'p_value':>12}")
    for k in FEATURE_NAMES:
        b = np.array(beginner_feats[k])
        e = np.array(expert_feats[k])
        t, p = stats.ttest_ind(b, e, equal_var=False)
        print(f"{k:<26}{b.mean():>15.4f}{e.mean():>15.4f}{p:>12.2e}")

    # --- 2. logistic regression on subject-disjoint split, stroke type as covariate ---
    train_records, val_records, val_subjects = subject_disjoint_split(records, val_frac=0.2, seed=42)

    def to_xy(recs):
        X_feat = np.array([[r["features"][k] for k in FEATURE_NAMES] for r in recs])
        stroke_onehot = np.zeros((len(recs), len(STROKE_CLASSES)))
        for i, r in enumerate(recs):
            stroke_onehot[i, STROKE_CLASSES.index(r["stroke"])] = 1.0
        X = np.hstack([X_feat, stroke_onehot])
        y = np.array([1 if r["skill_level"] == "expert" else 0 for r in recs])
        return X, y

    X_train, y_train = to_xy(train_records)
    X_val, y_val = to_xy(val_records)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train_s, y_train)

    train_acc = clf.score(X_train_s, y_train)
    val_acc = clf.score(X_val_s, y_val)
    baseline = max(y_val.mean(), 1 - y_val.mean())

    print(f"\n=== Logistic regression: predict beginner/expert from motion features + stroke type ===")
    print(f"train clips: {len(train_records)}, val clips: {len(val_records)} ({len(val_subjects)} held-out subjects)")
    print(f"val label distribution: {Counter(y_val.tolist())}")
    print(f"majority-class baseline accuracy: {baseline:.4f}")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"held-out val accuracy: {val_acc:.4f}")

    coef_report = sorted(
        zip(FEATURE_NAMES, clf.coef_[0][:len(FEATURE_NAMES)]),
        key=lambda x: -abs(x[1])
    )
    print("\nfeature coefficients (standardized, sorted by |weight|):")
    for name, w in coef_report:
        print(f"  {name:<26}{w:+.4f}")

    out = {
        "n_clips": len(records),
        "feature_stats": {
            k: {
                "beginner_mean": float(np.mean(beginner_feats[k])),
                "expert_mean": float(np.mean(expert_feats[k])),
                "p_value": float(stats.ttest_ind(beginner_feats[k], expert_feats[k], equal_var=False)[1]),
            } for k in FEATURE_NAMES
        },
        "logreg": {
            "baseline_accuracy": float(baseline),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "val_subjects": sorted(val_subjects),
            "coefficients": {name: float(w) for name, w in coef_report},
        },
    }
    out_path = Path("E:/SkillEye/results/beginner_expert_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
