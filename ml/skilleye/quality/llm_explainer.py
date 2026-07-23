"""
Rewrites already-flagged rule-based quality-scoring deviations (quality/score.py's
score_clip() output) into one natural-language paragraph via an LLM (NVIDIA's
build.nvidia.com API, OpenAI-compatible chat completions format). The LLM only
rephrases data the rule-based system already flagged -- it never inspects raw
angles or invents a new diagnosis -- which keeps hallucination risk low and keeps
this feature a communication layer over an already-validated system, not a new
source of truth. See docs/superpowers/specs/2026-07-23-llm-correction-explainer-design.md.

No API key exists yet; set NVIDIA_API_KEY in the environment to use this feature.
Every function here is fully testable without one (see test_llm_explainer.py).
"""
from quality.score import JOINT_DISPLAY_NAMES, PHASE_DISPLAY_NAMES


def build_explanation_prompt(stroke, table):
    """stroke: a STROKE_CLASSES key. table: score_clip()'s "table" list.
    Returns a prompt string grounded ONLY in flagged rows -- the LLM is
    instructed to rephrase these into one paragraph, not diagnose anything
    itself."""
    flagged = [row for row in table if row["flagged"]]
    stroke_label = stroke.replace("_", " ")

    if not flagged:
        return (
            f"A student just hit a {stroke_label}. Compared to an expert template, "
            "no significant deviations were flagged in any phase (backswing, contact, "
            "follow-through) or joint. Write one short, encouraging paragraph (2-3 "
            "sentences) telling the student their technique matched the expert "
            "template well on this swing. Do not invent any specific deviation or "
            "correction -- there isn't one."
        )

    lines = []
    for row in flagged:
        phase_label = PHASE_DISPLAY_NAMES[row["phase"]]
        joint_label = JOINT_DISPLAY_NAMES[row["joint"]]
        direction = "above" if row["z"] > 0 else "below"
        lines.append(f"- {phase_label}: {joint_label} is {direction} the typical expert range")

    deviations = "\n".join(lines)
    return (
        f"A student just hit a {stroke_label}. Compared to an expert template, the "
        f"following deviations were flagged:\n{deviations}\n\n"
        "Write one coherent, encouraging paragraph (3-5 sentences) a coach might say "
        "to the student, combining these specific deviations into natural coaching "
        "advice. Use ONLY the deviations listed above -- do not mention any other "
        "joint or phase, and do not invent any deviation not listed."
    )
