"""
Core pose-extraction pipeline for SkillEye.

video -> RTMPose (multi-person, per frame) -> single primary-subject track
      -> normalization (hip-anchored, bone-length scaled) -> confidence-based
         interpolation/smoothing -> clean per-clip skeleton sequence

COCO-17 keypoint order (RTMPose 'body' models):
0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist,
11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
"""

import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

CONF_THRESHOLD = 0.3


def select_primary_track(per_frame_keypoints, per_frame_scores):
    """
    per_frame_keypoints: list of (N_i, 17, 2) arrays, one per frame
    per_frame_scores:    list of (N_i, 17) arrays, one per frame

    Returns: keypoints (T, 17, 2), scores (T, 17) for a single consistent
    person, tracked frame-to-frame by nearest hip-center to the previous
    frame's selection (falls back to largest skeleton bbox when no one
    was selected yet, or the track is lost).
    """
    T = len(per_frame_keypoints)
    out_kpts = np.zeros((T, 17, 2), dtype=np.float32)
    out_scores = np.zeros((T, 17), dtype=np.float32)
    prev_hip_center = None

    for t in range(T):
        kpts, scores = per_frame_keypoints[t], per_frame_scores[t]
        n_people = kpts.shape[0] if kpts is not None else 0

        if n_people == 0:
            out_scores[t] = 0.0
            continue

        if prev_hip_center is None:
            # bootstrap: pick the person with the largest visible bbox
            best_idx, best_area = 0, -1.0
            for i in range(n_people):
                valid = scores[i] > CONF_THRESHOLD
                if valid.sum() < 3:
                    continue
                pts = kpts[i][valid]
                area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
                if area > best_area:
                    best_area, best_idx = area, i
        else:
            # track: nearest hip-center to previous frame's pick
            best_idx, best_dist = None, float("inf")
            for i in range(n_people):
                if scores[i, L_HIP] < CONF_THRESHOLD or scores[i, R_HIP] < CONF_THRESHOLD:
                    continue
                hip_center = (kpts[i, L_HIP] + kpts[i, R_HIP]) / 2
                dist = np.linalg.norm(hip_center - prev_hip_center)
                if dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx is None:
                best_idx = 0  # lost track this frame; keep array shape, low confidence anyway

        out_kpts[t] = kpts[best_idx]
        out_scores[t] = scores[best_idx]

        if scores[best_idx, L_HIP] > CONF_THRESHOLD and scores[best_idx, R_HIP] > CONF_THRESHOLD:
            prev_hip_center = (kpts[best_idx, L_HIP] + kpts[best_idx, R_HIP]) / 2

    return out_kpts, out_scores


def interpolate_low_confidence(kpts, scores, conf_threshold=CONF_THRESHOLD):
    """Linearly interpolate joints whose confidence falls below threshold,
    per joint per coordinate, across the temporal axis. Edge frames are
    filled with the nearest valid value (no extrapolation)."""
    T = kpts.shape[0]
    out = kpts.copy()
    valid = scores >= conf_threshold  # (T, 17)

    for j in range(kpts.shape[1]):
        good_t = np.where(valid[:, j])[0]
        if len(good_t) == 0:
            continue  # joint never confidently seen; leave as-is (caller should drop such clips)
        if len(good_t) == T:
            continue  # nothing to fill
        for coord in range(2):
            out[:, j, coord] = np.interp(
                np.arange(T), good_t, kpts[good_t, j, coord]
            )
    return out


def normalize_skeleton(kpts):
    """Hip-anchor the skeleton and scale by torso length (hip-center to
    shoulder-center) per frame, so the model learns motion pattern rather
    than the player's distance from / position relative to the camera."""
    hip_center = (kpts[:, L_HIP] + kpts[:, R_HIP]) / 2
    shoulder_center = (kpts[:, L_SHOULDER] + kpts[:, R_SHOULDER]) / 2
    torso_len = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    torso_len = np.where(torso_len < 1e-3, 1.0, torso_len)  # guard div-by-zero on bad frames

    centered = kpts - hip_center[:, None, :]
    normalized = centered / torso_len[:, None, None]
    return normalized.astype(np.float32)


def clean_clip(per_frame_keypoints, per_frame_scores, min_valid_frac=0.5):
    """
    Full cleaning pass for one clip. Returns (kpts, scores, ok) where ok=False
    means the clip should be dropped (too little of the primary subject visible
    to be usable).
    """
    kpts, scores = select_primary_track(per_frame_keypoints, per_frame_scores)

    valid_frac = (scores >= CONF_THRESHOLD).mean()
    if valid_frac < min_valid_frac:
        return kpts, scores, False

    kpts = interpolate_low_confidence(kpts, scores)
    kpts = normalize_skeleton(kpts)
    return kpts, scores, True
