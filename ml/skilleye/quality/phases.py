"""
Splits a normalized skeleton clip into three swing phases -- backswing,
contact, follow_through -- by finding the frame of peak dominant-wrist
speed (a standard swing-analysis heuristic) and slicing around it.
"""
import numpy as np

from quality.keypoints import L_WRIST, R_WRIST

PHASES = ["backswing", "contact", "follow_through"]
CONTACT_WINDOW = 5  # +/- raw frames around the contact frame


def dominant_wrist_index(kpts):
    """kpts: (T, 17, 2). Returns L_WRIST or R_WRIST, whichever moves more
    overall across the clip -- a proxy for the hitting arm, since THETIS
    doesn't label handedness."""
    l_disp = np.linalg.norm(np.diff(kpts[:, L_WRIST], axis=0), axis=-1).sum()
    r_disp = np.linalg.norm(np.diff(kpts[:, R_WRIST], axis=0), axis=-1).sum()
    return R_WRIST if r_disp >= l_disp else L_WRIST


def detect_contact_frame(kpts):
    """Returns the index of peak dominant-wrist speed. Falls back to the clip
    midpoint if the clip is too short (<3 frames) to compute a speed profile."""
    T = kpts.shape[0]
    if T < 3:
        return T // 2
    wrist = dominant_wrist_index(kpts)
    speed = np.linalg.norm(np.diff(kpts[:, wrist], axis=0), axis=-1)  # (T-1,)
    return int(np.argmax(speed))


def split_phases(kpts, contact_window=CONTACT_WINDOW):
    """kpts: (T, 17, 2). Returns {"backswing": ..., "contact": ...,
    "follow_through": ...}, each a (T_phase, 17, 2) array; any phase may have
    0 frames for very short clips or a contact frame near a clip boundary --
    downstream code (quality.angles) must handle that, not this function."""
    T = kpts.shape[0]
    contact = detect_contact_frame(kpts)
    contact_start = max(0, contact - contact_window)
    contact_end = min(T, contact + contact_window + 1)
    return {
        "backswing": kpts[:contact_start],
        "contact": kpts[contact_start:contact_end],
        "follow_through": kpts[contact_end:],
    }
