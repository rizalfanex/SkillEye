"""
Interpretable per-frame joint angles used as the quality-scoring feature set:
elbow flexion (L/R), knee flexion (L/R), and a trunk-rotation proxy (angle
between the shoulder line and the hip line). Extending this set later is a
one-entry addition to JOINT_DEFINITIONS (or a new series function, for
trunk_rotation-style pairs) -- nothing else needs to change.
"""
import numpy as np

from quality.keypoints import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)

# name -> (a, b, c): angle at joint b, formed by points a-b-c
JOINT_DEFINITIONS = {
    "left_elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right_elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    "left_knee": (L_HIP, L_KNEE, L_ANKLE),
    "right_knee": (R_HIP, R_KNEE, R_ANKLE),
}


def joint_angle_series(kpts, a, b, c):
    """Angle at joint b formed by points a-b-c, per frame. kpts: (T, 17, 2).
    Returns (T,) in radians; T=0 input returns a (0,) array."""
    if kpts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    v1 = kpts[:, a] - kpts[:, b]
    v2 = kpts[:, c] - kpts[:, b]
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = np.clip(n1 * n2, 1e-6, None)
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cos_theta)


def trunk_rotation_series(kpts):
    """Angle between the shoulder line (L_SHOULDER->R_SHOULDER) and the hip
    line (L_HIP->R_HIP), per frame -- a proxy for trunk twist. kpts: (T,17,2)."""
    if kpts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    shoulder_vec = kpts[:, R_SHOULDER] - kpts[:, L_SHOULDER]
    hip_vec = kpts[:, R_HIP] - kpts[:, L_HIP]
    dot = (shoulder_vec * hip_vec).sum(axis=-1)
    n1 = np.linalg.norm(shoulder_vec, axis=-1)
    n2 = np.linalg.norm(hip_vec, axis=-1)
    denom = np.clip(n1 * n2, 1e-6, None)
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_all_angles(kpts):
    """kpts: (T, 17, 2), T may be 0. Returns {joint_name: (T,) array} for
    every entry in JOINT_DEFINITIONS plus "trunk_rotation"."""
    result = {name: joint_angle_series(kpts, *idx) for name, idx in JOINT_DEFINITIONS.items()}
    result["trunk_rotation"] = trunk_rotation_series(kpts)
    return result


def phase_mean_angles(kpts):
    """kpts: (T, 17, 2) for a single phase, T may be 0. Returns {joint_name:
    float or None} -- None means the phase had no frames to average."""
    all_angles = compute_all_angles(kpts)
    return {name: (float(series.mean()) if series.shape[0] > 0 else None)
            for name, series in all_angles.items()}
