"""
Scores a single clip against its predicted stroke's expert template: for each
(phase, joint), a z-score of how far the clip's phase-mean angle sits from
the expert average, flagging |z| > FLAG_THRESHOLD as worth mentioning, plus a
0-100 overall score and human-readable correction suggestions.

No coach-rating ground truth exists yet (see docs/superpowers/specs/
2026-07-09-quality-scoring-and-ui-design.md) -- FLAG_THRESHOLD and
SCORE_SCALE are set to sane defaults and checked with a smoke test
(smoke_check_quality_scoring.py), not formally calibrated against real
quality labels.
"""
import numpy as np

from quality.phases import split_phases, PHASES
from quality.angles import phase_mean_angles

FLAG_THRESHOLD = 1.5
SCORE_SCALE = 15.0

JOINT_DISPLAY_NAMES = {
    "left_elbow": "left elbow",
    "right_elbow": "right elbow",
    "left_knee": "left knee",
    "right_knee": "right knee",
    "trunk_rotation": "trunk rotation",
}
PHASE_DISPLAY_NAMES = {
    "backswing": "backswing",
    "contact": "contact",
    "follow_through": "follow-through",
}
# joint -> (tip when the clip's angle is smaller than the template, tip when larger)
JOINT_TIPS = {
    "left_elbow": (
        "try extending your left arm more",
        "try relaxing your left arm, it's more extended than typical",
    ),
    "right_elbow": (
        "try extending your right arm more",
        "try relaxing your right arm, it's more extended than typical",
    ),
    "left_knee": (
        "try bending your left knee more for a lower stance",
        "try standing a bit taller, your left knee is more bent than typical",
    ),
    "right_knee": (
        "try bending your right knee more for a lower stance",
        "try standing a bit taller, your right knee is more bent than typical",
    ),
    "trunk_rotation": (
        "try rotating your trunk more through the shot",
        "your trunk is rotating more than typical -- make sure that's controlled, not overswinging",
    ),
}


def suggestion_text(joint, phase, z):
    direction = 1 if z > 0 else 0  # 0 = clip's angle smaller than template, 1 = larger
    tip = JOINT_TIPS[joint][direction]
    comparison = "above" if z > 0 else "below"
    return (f"{PHASE_DISPLAY_NAMES[phase].capitalize()}: {JOINT_DISPLAY_NAMES[joint]} is "
            f"{comparison} the typical expert range -- {tip}.")


def score_clip(kpts, stroke_class, templates):
    """kpts: (T, 17, 2) normalized skeleton. stroke_class: a key of
    STROKE_CLASSES. templates: templates[stroke][phase][joint] -> {mean, std, n}
    (the "templates" value from templates.json, not the whole loaded file).

    Returns {"overall_score": float, "table": [{"phase", "joint", "value",
    "z" (float or None), "flagged", "note" (str or None)}, ...],
    "suggestions": [str, ...]}. "z"/"note" are None/"insufficient template
    data" respectively when the template has no usable entry for that
    (phase, joint) -- surfaced rather than silently dropped.
    """
    phases = split_phases(kpts)
    stroke_template = templates.get(stroke_class, {})

    table = []
    abs_z_values = []
    suggestions = []

    for phase in PHASES:
        phase_means = phase_mean_angles(phases[phase])
        phase_template = stroke_template.get(phase, {})
        for joint, value in phase_means.items():
            if value is None:
                continue  # this clip's phase had no frames -- nothing to score for this joint

            entry = phase_template.get(joint)
            if entry is None or entry["std"] <= 1e-6:
                # Surface this rather than silently dropping the row (design spec,
                # Error Handling) -- a missing/degenerate template entry is a fact
                # about template coverage worth showing, not hiding.
                table.append({"phase": phase, "joint": joint, "value": value,
                              "z": None, "flagged": False, "note": "insufficient template data"})
                continue

            z = (value - entry["mean"]) / entry["std"]
            flagged = abs(z) > FLAG_THRESHOLD
            table.append({"phase": phase, "joint": joint, "value": value,
                          "z": z, "flagged": flagged, "note": None})
            abs_z_values.append(abs(z))
            if flagged:
                suggestions.append(suggestion_text(joint, phase, z))

    if abs_z_values:
        overall_score = 100.0 - float(np.clip(np.mean(abs_z_values) * SCORE_SCALE, 0, 100))
    else:
        overall_score = 50.0  # no comparable (phase, joint) data at all

    return {"overall_score": overall_score, "table": table, "suggestions": suggestions}
