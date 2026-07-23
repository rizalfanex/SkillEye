# LLM-Generated Correction Paragraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `quality/score.py`'s already-flagged rule-based deviations into one natural-language coaching paragraph per swing, via NVIDIA's build.nvidia.com API, with a hard fallback to the existing fixed-template suggestions if the API call fails for any reason.

**Architecture:** A new `quality/llm_explainer.py` module builds a prompt grounded only in already-flagged `(phase, joint, z)` rows (never raw angles, never inventing a new diagnosis) and POSTs it to NVIDIA's OpenAI-compatible chat completions endpoint via `requests`. `app.py` wires this behind a manual button, cached per (stroke, table) so re-running the Streamlit script (e.g. dragging the frame slider) never re-fires the API call, and falls back to the existing suggestions list on any `LLMExplanationError`.

**Tech Stack:** Python, `requests` (already installed — verified `requests==2.32.3` in the `torch` conda env), pytest, Streamlit (existing `app.py`).

## Global Constraints

- Python interpreter for all commands: `/c/Users/uclla/miniconda3/envs/torch/python` (the `torch` conda env — this repo's established environment).
- All new scripts/tests run with working directory `E:/SkillEye/ml/skilleye` (the repo reorganized since an earlier plan was written — code now lives at `ml/skilleye/`, not `skilleye/`; results at `ml/results/`, not `results/`). Modules are imported as bare top-level (e.g. `from quality.score import ...`), not a package.
- No API key exists yet — the user will set `NVIDIA_API_KEY` in the environment themselves later. Every test in this plan must pass with no real network access and no real API key; every function that calls the network must accept an injectable `post_fn` for this reason.
- On any failure calling the LLM (missing key, network error, timeout, non-200 response, unexpected response shape), raise exactly one exception type, `LLMExplanationError` — callers never need to catch `requests` exceptions or `KeyError` directly.
- The LLM must be grounded ONLY in rows `quality/score.py`'s `score_clip()` already flagged (`row["flagged"] is True`) — it must never be given raw per-frame angles or unflagged rows to reason over itself. This is a hard requirement from the design spec (`docs/superpowers/specs/2026-07-23-llm-correction-explainer-design.md`), not a nice-to-have — it's what keeps hallucination risk low.
- This is additive only: the existing rule-based `suggestions` list in `app.py` must remain visible exactly as it is today; the LLM paragraph is an optional addition below it, never a replacement.
- Reuse existing constants, do not redefine: `quality.score.{JOINT_DISPLAY_NAMES, PHASE_DISPLAY_NAMES}` (`ml/skilleye/quality/score.py:21-32`).

---

### Task 1: Prompt construction

**Files:**
- Create: `ml/skilleye/quality/llm_explainer.py`
- Test: `ml/skilleye/quality/test_llm_explainer.py`

**Interfaces:**
- Consumes: `quality.score.{JOINT_DISPLAY_NAMES, PHASE_DISPLAY_NAMES}` (existing, `ml/skilleye/quality/score.py`).
- Produces: `llm_explainer.build_explanation_prompt(stroke: str, table: list[dict]) -> str`. `table` has the same row shape `score_clip()` returns: `{"phase": str, "joint": str, "value": float, "z": float | None, "flagged": bool, "note": str | None}`.

- [ ] **Step 1: Write the failing tests**

Create `ml/skilleye/quality/test_llm_explainer.py`:

```python
from quality.llm_explainer import build_explanation_prompt

FLAGGED_TABLE = [
    {"phase": "backswing", "joint": "left_elbow", "value": 0.5, "z": 2.1, "flagged": True, "note": None},
    {"phase": "contact", "joint": "right_knee", "value": 1.2, "z": -1.8, "flagged": True, "note": None},
    {"phase": "follow_through", "joint": "trunk_rotation", "value": 0.3, "z": 0.4, "flagged": False, "note": None},
]

NO_FLAGS_TABLE = [
    {"phase": "backswing", "joint": "left_elbow", "value": 0.5, "z": 0.2, "flagged": False, "note": None},
]


def test_build_explanation_prompt_mentions_only_flagged_rows():
    prompt = build_explanation_prompt("forehand", FLAGGED_TABLE)
    assert "left elbow" in prompt
    assert "backswing" in prompt
    assert "right knee" in prompt
    assert "contact" in prompt
    assert "trunk rotation" not in prompt
    assert "follow-through" not in prompt


def test_build_explanation_prompt_no_flags_gives_positive_prompt():
    prompt = build_explanation_prompt("forehand", NO_FLAGS_TABLE)
    assert "no significant deviations" in prompt
    assert "left elbow" not in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/ml/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_llm_explainer.py -v
```
Expected: `ModuleNotFoundError: No module named 'quality.llm_explainer'`

- [ ] **Step 3: Implement prompt construction**

Create `ml/skilleye/quality/llm_explainer.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/ml/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_llm_explainer.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add ml/skilleye/quality/llm_explainer.py ml/skilleye/quality/test_llm_explainer.py
git commit -m "Add LLM correction-paragraph prompt construction (flagged rows only)"
```

---

### Task 2: API call + custom exception

**Files:**
- Modify: `ml/skilleye/quality/llm_explainer.py`
- Modify: `ml/skilleye/quality/test_llm_explainer.py`

**Interfaces:**
- Consumes: `llm_explainer.build_explanation_prompt` (Task 1).
- Produces: `llm_explainer.LLMExplanationError` (an `Exception` subclass). `llm_explainer.generate_explanation(stroke: str, table: list[dict], api_key: str | None = None, timeout: float = 8.0, post_fn=requests.post) -> str` — returns the generated paragraph, raises `LLMExplanationError` on any failure. `llm_explainer.NVIDIA_API_BASE` (`str`), `llm_explainer.MODEL_NAME` (`str`), `llm_explainer.API_KEY_ENV_VAR` (`str`, value `"NVIDIA_API_KEY"`).

- [ ] **Step 1: Write the failing tests**

Append to `ml/skilleye/quality/test_llm_explainer.py`:

```python
import pytest
import requests

from quality.llm_explainer import generate_explanation, LLMExplanationError


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


def test_generate_explanation_success_path():
    def fake_post(url, json, headers, timeout):
        return FakeResponse(200, {"choices": [{"message": {"content": " Great swing! "}}]})

    result = generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)
    assert result == "Great swing!"


def test_generate_explanation_raises_on_non_200():
    def fake_post(url, json, headers, timeout):
        return FakeResponse(500, text="internal error")

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


def test_generate_explanation_raises_on_network_error():
    def fake_post(url, json, headers, timeout):
        raise requests.ConnectionError("no network")

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


def test_generate_explanation_raises_on_unexpected_response_shape():
    def fake_post(url, json, headers, timeout):
        return FakeResponse(200, {"unexpected": "shape"})

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


def test_generate_explanation_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    def fake_post(url, json, headers, timeout):
        raise AssertionError("post_fn should never be called without an API key")

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key=None, post_fn=fake_post)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/ml/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_llm_explainer.py -v
```
Expected: `ImportError: cannot import name 'generate_explanation' from 'quality.llm_explainer'`

- [ ] **Step 3: Implement the API call**

Add these imports at the top of `ml/skilleye/quality/llm_explainer.py` (alongside the existing `from quality.score import ...` line):

```python
import os

import requests
```

Append to `ml/skilleye/quality/llm_explainer.py`:

```python
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

    prompt = build_explanation_prompt(stroke, table)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 200,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        response = post_fn(NVIDIA_API_BASE, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMExplanationError(f"request to NVIDIA API failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMExplanationError(
            f"NVIDIA API returned status {response.status_code}: {response.text}")

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMExplanationError(f"unexpected response shape from NVIDIA API: {exc}") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/ml/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_llm_explainer.py -v
```
Expected: `7 passed` (2 from Task 1 + 5 new)

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add ml/skilleye/quality/llm_explainer.py ml/skilleye/quality/test_llm_explainer.py
git commit -m "Add NVIDIA API call for LLM correction paragraphs (LLMExplanationError, injectable post_fn)"
```

---

### Task 3: Streamlit demo integration

**Files:**
- Modify: `ml/skilleye/app.py`

**Interfaces:**
- Consumes: `quality.llm_explainer.{generate_explanation, LLMExplanationError}` (Tasks 1-2).
- Produces: a button-triggered, cached UI addition in the existing Streamlit app; no new importable interface (terminal node of this feature).

- [ ] **Step 1: Add the import**

In `ml/skilleye/app.py`, change this line (currently line 18):

```python
from quality.score import score_clip
```

to:

```python
from quality.score import score_clip
from quality.llm_explainer import generate_explanation, LLMExplanationError
```

- [ ] **Step 2: Add a cached wrapper function**

In `ml/skilleye/app.py`, immediately after the existing `predict_stroke` function (currently ends at line 55, right before `def render_skeleton_frame`), add:

```python
@st.cache_data
def cached_llm_explanation(stroke, table):
    return generate_explanation(stroke, table)
```

- [ ] **Step 3: Add the button to the Correction suggestions section**

In `ml/skilleye/app.py`, the current Correction suggestions section (currently the last block inside `with col2:`, lines 119-124) reads:

```python
        st.subheader("Correction suggestions")
        if result["suggestions"]:
            for s in result["suggestions"]:
                st.write(f"- {s}")
        else:
            st.write("No significant deviations flagged against the expert template.")
```

Replace it with (adds the button below the existing, unchanged suggestions list):

```python
        st.subheader("Correction suggestions")
        if result["suggestions"]:
            for s in result["suggestions"]:
                st.write(f"- {s}")
        else:
            st.write("No significant deviations flagged against the expert template.")

        if st.button("Generate AI explanation"):
            try:
                explanation = cached_llm_explanation(pred_stroke, result["table"])
                st.info(explanation)
            except LLMExplanationError as e:
                st.warning(
                    "AI explanation unavailable -- showing the rule-based suggestions "
                    f"above instead. ({e})")
```

- [ ] **Step 4: Launch it and verify manually**

```bash
cd E:/SkillEye/ml/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m streamlit run app.py --server.headless true
```
Expected: prints a local URL. Open it (or use a tool that can screenshot a local URL) and verify:
- The existing rule-based suggestions list still renders exactly as before (unchanged).
- A "Generate AI explanation" button appears below it.
- Since `NVIDIA_API_KEY` is not set in this environment, clicking the button shows the fallback warning message (mentioning `NVIDIA_API_KEY is not set`), not a raw Python traceback or a blank/crashed page.
- Selecting a different clip/stroke and clicking the button again still shows the same graceful fallback (confirms the try/except path works across different inputs, not just the first one tried).

Stop the process (Ctrl+C) once verified.

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add ml/skilleye/app.py
git commit -m "Wire LLM correction paragraphs into the Streamlit demo (button-triggered, cached, with fallback)"
```

---

### Task 4: Documentation — fold the feature into the README

**Files:**
- Modify: `E:/SkillEye/README.md`

**Interfaces:**
- Consumes: nothing new — this task only writes prose, no code.

- [ ] **Step 1: Extend the Quality Scoring System subsection**

In `README.md`, find the existing paragraph in "### 2.6 Quality Scoring System" that begins with `**Demo UI**` (it currently ends with "Run with `streamlit run app.py` from `ml/skilleye/`."). Immediately after that paragraph, add:

```markdown
**AI-generated explanation (optional)**: a "Generate AI explanation" button in the demo
UI rewrites the already-flagged deviations above into one natural coaching paragraph via
an LLM (NVIDIA's build.nvidia.com API). The LLM is only ever given the (phase, joint,
z-score) rows the rule-based system already flagged — it never inspects raw angles or
introduces a new diagnosis — so this is a communication layer over the existing,
smoke-checked scoring system, not a new source of truth. It requires an `NVIDIA_API_KEY`
environment variable; without one (or if the API call fails for any reason), the button
falls back to a short warning and the rule-based suggestions list above remains the
result, so the demo never depends on network access to function. Design:
`docs/superpowers/specs/2026-07-23-llm-correction-explainer-design.md`.
```

- [ ] **Step 2: Update the Reproducibility repo-layout listing**

In `README.md`, in "## 6. Reproducibility", the `ml/skilleye/` block currently has this line:

```
  app.py                           Streamlit demo UI (§2.6) -- run: streamlit run app.py
```

Add this line immediately after it:

```
  quality/llm_explainer.py         optional LLM-generated correction paragraphs (§2.6, needs NVIDIA_API_KEY)
```

- [ ] **Step 3: Commit and push**

```bash
cd E:/SkillEye
git add README.md
git commit -m "Document the LLM correction-paragraph feature in the README"
git push
```
