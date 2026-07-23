import numpy as np

from quality.score import score_clip, suggestion_text
from quality.keypoints import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)
from quality.phases import PHASES


def make_straight_arm_clip(T=20):
    """Left arm straight (pi rad), right elbow at 90 degrees, both legs
    straight (pi rad), shoulder/hip lines parallel (0 rad trunk rotation) --
    every angle is exact and constant across all frames. The right wrist
    moves radially along the fixed elbow->wrist direction (1,0) with a clear
    mid-clip speed peak, so detect_contact_frame lands away from either
    boundary (giving every phase a nonzero frame count) while the right-elbow
    angle stays exactly pi/2 regardless of that motion (angle is invariant to
    magnitude along a fixed direction)."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)
    kpts[:, L_SHOULDER] = (0, 2)
    kpts[:, L_ELBOW] = (0, 1)
    kpts[:, L_WRIST] = (0, 0)
    kpts[:, R_SHOULDER] = (1, 2)
    kpts[:, R_ELBOW] = (1, 1)

    x = 0.0
    for t in range(T):
        step = 1.0 if t < T // 2 - 1 else (5.0 if t == T // 2 - 1 else 0.2)
        x += step
        kpts[t, R_WRIST] = (1 + x, 1)

    kpts[:, L_HIP] = (0, -1)
    kpts[:, R_HIP] = (1, -1)
    kpts[:, L_KNEE] = (0, -2)
    kpts[:, L_ANKLE] = (0, -3)
    kpts[:, R_KNEE] = (1, -2)
    kpts[:, R_ANKLE] = (1, -3)
    return kpts


def make_template(mean_by_joint, std=0.5):
    """Builds templates["strokeX"][phase][joint] with the same mean/std for
    every phase, so tests can reason about one joint's deviation at a time."""
    return {
        "strokeX": {
            phase: {joint: {"mean": mean, "std": std, "n": 50} for joint, mean in mean_by_joint.items()}
            for phase in PHASES
        }
    }


CLOSE_MEANS = {
    "left_elbow": np.pi, "right_elbow": np.pi / 2,
    "trunk_rotation": 0.0, "left_knee": np.pi, "right_knee": np.pi,
}


def test_score_clip_flags_large_deviation():
    kpts = make_straight_arm_clip()
    means = dict(CLOSE_MEANS, left_elbow=0.5)  # far from the clip's actual pi
    templates = make_template(means, std=0.1)
    result = score_clip(kpts, "strokeX", templates)
    left_elbow_rows = [row for row in result["table"] if row["joint"] == "left_elbow"]
    assert len(left_elbow_rows) == len(PHASES)
    assert all(row["flagged"] for row in left_elbow_rows)
    assert any("left elbow" in s for s in result["suggestions"])


def test_score_clip_does_not_flag_close_match():
    kpts = make_straight_arm_clip()
    templates = make_template(CLOSE_MEANS, std=0.5)
    result = score_clip(kpts, "strokeX", templates)
    assert all(not row["flagged"] for row in result["table"])
    assert result["suggestions"] == []


def test_score_clip_marks_missing_template_entries_as_insufficient_data():
    kpts = make_straight_arm_clip()
    templates = {"strokeX": {phase: {} for phase in PHASES}}  # no joints in template at all
    result = score_clip(kpts, "strokeX", templates)
    assert len(result["table"]) == len(PHASES) * 5  # 5 joints x 3 phases, all present
    assert all(row["z"] is None for row in result["table"])
    assert all(row["note"] == "insufficient template data" for row in result["table"])
    assert all(not row["flagged"] for row in result["table"])
    assert result["suggestions"] == []
    assert result["overall_score"] == 50.0


def test_score_clip_overall_score_is_lower_for_bigger_deviation():
    kpts = make_straight_arm_clip()
    close_templates = make_template(CLOSE_MEANS, std=0.3)
    far_templates = make_template(dict(CLOSE_MEANS, left_elbow=0.2), std=0.3)
    close_score = score_clip(kpts, "strokeX", close_templates)["overall_score"]
    far_score = score_clip(kpts, "strokeX", far_templates)["overall_score"]
    assert far_score < close_score


def test_suggestion_text_mentions_phase_and_joint():
    text = suggestion_text("left_elbow", "backswing", z=2.0)
    assert "left elbow" in text
    assert "backswing" in text.lower()
