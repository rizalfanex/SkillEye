"""
Skill-level-specific error detection rules (Module B): unlike quality/score.py
(which z-scores against an expert-only template -- no notion of "less-skilled"
to compare against), each rule here encodes one specific finding from a paper
that directly compares skilled vs. less-skilled players. Output is its own
set of flags, separate from score.py's 15 z-scored deviations. Design:
docs/superpowers/specs/2026-08-14-skill-level-rules-module-b-design.md.

Reference numbers below were extracted from an automated proxy-read of each
paper's published page (not a manual PDF read) -- worth spot-checking against
the source if precision matters.
"""
import numpy as np

from quality.angles import signed_pelvic_rotation_series, signed_shoulder_pelvis_twist_series
from quality.phases import split_phases

MPS2_PER_G = 9.80665


def g_to_mps2(g):
    """Converts accelerometer g-units (what hardware/firmware/firmware.ino
    reports) to m/s^2 (what Aydin & Aydemir's thresholds below are in)."""
    return g * MPS2_PER_G


# Katsumi, K., Koda, H., & Kida, N. (2026). Analysis of Upper-Limb Movement
# Characteristics in Tennis Volleys Based on Skill-Level Differences:
# Kinematic Features of the Backhand Versus Forehand Volley. Journal of
# Functional Morphology and Kinesiology, 11(2), 203.
# Values: (mean, std) in degrees, backhand volley only.
KATSUMI_BACKHAND_VOLLEY = {
    "pelvic_rotation_deg": {
        "skilled": {"backswing": (-29.6, 12.6), "contact": (-37.1, 24.0)},
        "less_skilled": {"backswing": (-46.7, 18.4), "contact": (-65.7, 17.6)},
    },
    "shoulder_pelvis_twist_deg": {
        "skilled": {"backswing": (-18.9, 13.1), "contact": (-15.5, 17.3)},
        "less_skilled": {"backswing": (-7.7, 26.2), "contact": (10.1, 17.0)},
    },
}


def check_shoulder_pelvis_twist_reversal(twist_backswing_deg, twist_contact_deg):
    """Katsumi et al. (2026): less-skilled backhand-volley players' shoulder-
    pelvis twist flips sign between backswing (-7.7 +/- 26.2 deg, near zero)
    and contact (+10.1 +/- 17.0 deg) -- "separation, which had not been
    established during preparation, may have been produced belatedly" (the
    paper's own explanation) -- while skilled players' stays negative
    throughout (-18.9 -> -15.5 deg). Flags when the two phases' signs differ.
    A value of exactly 0 at either phase has no sign to compare, so that
    case is not flagged (no reversal to detect)."""
    if twist_backswing_deg == 0 or twist_contact_deg == 0:
        return False
    return (twist_backswing_deg > 0) != (twist_contact_deg > 0)


def check_excessive_pelvic_rotation(pelvic_backswing_deg, pelvic_contact_deg):
    """Katsumi et al. (2026): less-skilled backhand-volley players rotate the
    pelvis further (more negative) than skilled players at both backswing
    (skilled -29.6 +/- 12.6 deg vs. less-skilled -46.7 +/- 18.4 deg) and
    contact (skilled -37.1 +/- 24.0 deg vs. less-skilled -65.7 +/- 17.6 deg).
    Flags a phase when the query value passes the midpoint between the two
    groups' means for that phase -- a directional lean, not a diagnostic
    cutoff (the SDs overlap the groups substantially). Assumes the specific
    sign convention observed in the paper (less-skilled = more negative);
    not a generalized comparator."""
    ref = KATSUMI_BACKHAND_VOLLEY["pelvic_rotation_deg"]
    result = {}
    for phase, value_deg in (("backswing", pelvic_backswing_deg), ("contact", pelvic_contact_deg)):
        skilled_mean = ref["skilled"][phase][0]
        less_skilled_mean = ref["less_skilled"][phase][0]
        midpoint = (skilled_mean + less_skilled_mean) / 2.0
        result[phase] = value_deg < midpoint
    return result


# Aydin, E.H., & Aydemir, O. (2026). A Robust Deep Learning Framework for
# Skill Level Discrimination in Tennis Strokes Using Bilateral IMU
# Measurements. Sensors, 26(10), 3273.
# Peak dominant-hand acceleration, (mean, std) in m/s^2, volley.
AYDIN_AYDEMIR_VOLLEY = {
    "elite": (48.12, 26.49),
    "amateur": (57.09, 29.86),
}


def check_volley_swing_effort(peak_accel_mps2):
    """Aydin & Aydemir (2026): elite volleys show *lower* peak dominant-hand
    acceleration (48.12 +/- 26.49 m/s^2) than amateur volleys (57.09 +/- 29.86
    m/s^2) -- "amateurs overcompensate for technical deficiencies with
    excessive, uncontrolled force," while elites use "a controlled,
    abbreviated swing" (the "principle of minimum energy"). Flags when the
    query value passes the midpoint between the two groups' means -- a
    directional lean, not a per-swing classifier (these SDs overlap the
    groups even more than Katsumi's, ~9 m/s^2 mean gap against 26-30 SDs)."""
    elite_mean, _ = AYDIN_AYDEMIR_VOLLEY["elite"]
    amateur_mean, _ = AYDIN_AYDEMIR_VOLLEY["amateur"]
    midpoint = (elite_mean + amateur_mean) / 2.0
    flagged = peak_accel_mps2 > midpoint
    note = ("Your swing uses a lot of force. Try a shorter, more compact swing "
        "instead of swinging harder, so you can control the racket more easily.") if flagged else None
    return {"flagged": flagged, "note": note}


def _phase_mean_signed_deg(series_fn, phase_kpts):
    """series_fn's (T,) radians series for phase_kpts, averaged and converted
    to degrees; None if the phase has no frames (mirrors quality/angles.py's
    phase_mean_angles None-for-empty-phase convention)."""
    series = series_fn(phase_kpts)
    if series.shape[0] == 0:
        return None
    return float(np.degrees(series.mean()))


def evaluate_backhand_volley_skill_rules(kpts):
    """Runs both Katsumi et al. (2026) rules on a backhand_volley clip's
    backswing and contact phases (Katsumi's "backswing"/"impact" mapped onto
    this project's existing split_phases "backswing"/"contact"). kpts:
    (T, 17, 2) normalized skeleton.

    Returns {"flags": [{"rule": str, "phase": str or None, "note": str}, ...]}
    -- empty if a phase has no frames to evaluate (surfaced by omission, same
    as the rest of quality/ does for missing data, rather than raising)."""
    phases = split_phases(kpts)
    backswing_twist = _phase_mean_signed_deg(signed_shoulder_pelvis_twist_series, phases["backswing"])
    contact_twist = _phase_mean_signed_deg(signed_shoulder_pelvis_twist_series, phases["contact"])
    backswing_pelvic = _phase_mean_signed_deg(signed_pelvic_rotation_series, phases["backswing"])
    contact_pelvic = _phase_mean_signed_deg(signed_pelvic_rotation_series, phases["contact"])

    flags = []

    if backswing_twist is not None and contact_twist is not None:
        if check_shoulder_pelvis_twist_reversal(backswing_twist, contact_twist):
            flags.append({
                "rule": "shoulder_pelvis_twist_reversal", "phase": None,
                "note": ("Your shoulders and hips change direction between the backswing "
                         "and contact. Try turning your shoulders a little earlier and "
                         "keep that turn steady as you move forward."),
            })

    if backswing_pelvic is not None and contact_pelvic is not None:
        excessive = check_excessive_pelvic_rotation(backswing_pelvic, contact_pelvic)
        for phase, is_excessive in excessive.items():
            if is_excessive:
                flags.append({
                    "rule": "excessive_pelvic_rotation", "phase": phase,
                    "note": (f"During the {phase}, your hips turn more than needed for "
                             "this backhand volley. Try a smaller, controlled hip turn "
                             "and let your shoulders guide the swing."),
                })

    return {"flags": flags}
