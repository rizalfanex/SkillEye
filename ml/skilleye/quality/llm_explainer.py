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
import os

import requests

from quality.score import JOINT_DISPLAY_NAMES, PHASE_DISPLAY_NAMES


def build_explanation_prompt(stroke, table):
    """stroke: a STROKE_CLASSES key. table: score_clip()'s "table" list.
    Returns a prompt string grounded ONLY in flagged rows -- the LLM is
    instructed to rephrase these into one paragraph, not diagnose anything
    itself, using language a beginner can understand."""
    flagged = [row for row in table if row["flagged"]]
    stroke_label = stroke.replace("_", " ")

    if not flagged:
        return (
            f"A student just hit a {stroke_label}. The result shows no significant deviations "
            "in the backswing, contact, or follow-through. Write one short, encouraging "
            "paragraph (2-3 sentences) in simple language that a beginner can understand. "
            "Tell the student what they did well. Do not mention expert templates, ranges, "
            "measurements, scores, or statistics, and do not invent a specific correction."
        )

    lines = []
    for row in flagged:
        phase_label = PHASE_DISPLAY_NAMES[row["phase"]]
        joint_label = JOINT_DISPLAY_NAMES[row["joint"]]
        direction = "above" if row["z"] > 0 else "below"
        lines.append(f"- {phase_label}: the student's {joint_label} movement needs attention ({direction})")

    deviations = "\n".join(lines)
    return (
        f"A student just hit a {stroke_label}. The following movement areas need attention:\n"
        f"{deviations}\n\n"
        "Write one coherent, encouraging paragraph (3-5 sentences) in simple, "
        "beginner-friendly language. Explain what the student should try to do with "
        "their body and why it helps. Use ONLY the movement areas listed above -- do "
        "not mention any other joint or phase, and do not invent a new problem. Do not "
        "mention expert templates, typical ranges, z-scores, angles, measurements, or "
        "statistics. Avoid technical anatomy terms unless you explain them plainly."
    )


NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "meta/llama-3.1-8b-instruct"
API_KEY_ENV_VAR = "NVIDIA_API_KEY"


class LLMExplanationError(Exception):
    """Raised for any failure generating an LLM explanation -- missing API key,
    network error, timeout, non-200 response, or an unexpected response shape.
    Callers catch this one type and fall back to the existing rule-based
    suggestions list; they never need to catch requests exceptions directly."""


def generate_explanation(stroke, table, api_key=None, timeout=8.0, post_fn=requests.post):
    """Returns a generated paragraph (str). Raises LLMExplanationError on any
    failure -- missing key, network error, timeout, non-200 response, or an
    unexpected response shape. post_fn is injectable so tests never make a
    real network call or need a real key."""
    key = api_key or os.environ.get(API_KEY_ENV_VAR)
    if not key:
        raise LLMExplanationError(
            f"{API_KEY_ENV_VAR} is not set -- cannot generate an AI explanation.")

    try:
        prompt = build_explanation_prompt(stroke, table)
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 200,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        response = post_fn(NVIDIA_API_BASE, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMExplanationError(f"request to NVIDIA API failed: {exc}") from exc
    except Exception as exc:
        raise LLMExplanationError(f"failed to build request or reach NVIDIA API: {exc}") from exc

    if response.status_code != 200:
        raise LLMExplanationError(
            f"NVIDIA API returned status {response.status_code}: {response.text}")

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise LLMExplanationError(f"unexpected response shape from NVIDIA API: {exc}") from exc
