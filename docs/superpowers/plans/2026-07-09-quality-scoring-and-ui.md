# Quality Scoring System + Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the rule-based swing-quality scoring system (per-phase, per-joint deviation from expert templates + correction suggestions) and a Streamlit demo UI, closing the biggest gap between SkillEye's proposal and its current repo.

**Architecture:** A new `skilleye/quality/` package (phase detection → joint angles → z-score comparison against precomputed expert templates → text suggestions) sits on top of the existing skeleton-extraction and stroke-classification code without modifying it. Two new top-level scripts (`build_expert_templates.py`, `smoke_check_quality_scoring.py`) and a Streamlit app (`app.py`) consume that package.

**Tech Stack:** Python, numpy, PyTorch (existing `STGCN` model, unmodified), pytest, Streamlit. All already installed in the `torch` conda env — verified: pytest 9.0.2, streamlit 1.58.0.

## Global Constraints

- Python interpreter for all commands: `/c/Users/uclla/miniconda3/envs/torch/python` (the `torch` conda env — this repo's established environment).
- All new scripts/tests run with working directory `E:/SkillEye/skilleye` (matches the existing convention — `stroke_dataset`, `stgcn_model`, etc. are imported as bare top-level modules, not a package).
- COCO-17 keypoint order and existing constants/functions (`STROKE_CLASSES`, `load_records`, `subject_disjoint_split`, `resample_time`, `STGCN`, `COCO17_EDGES`) must be reused, not redefined — see `skilleye/stroke_dataset.py` and `skilleye/stgcn_model.py`.
- Demo-eligible clips are restricted to the held-out validation subjects from the standard `subject_disjoint_split(records, val_frac=0.2, seed=42)` split (same split used elsewhere in this repo) — never a subject used to build the expert templates. This is a hard requirement from the design spec (`docs/superpowers/specs/2026-07-09-quality-scoring-and-ui-design.md`), not a nice-to-have.
- No placeholder ground truth: this system compares against expert-clip statistics, never invents or hardcodes example "ideal" values.

---

### Task 1: `quality` package — keypoint constants + phase detection

**Files:**
- Create: `skilleye/quality/__init__.py`
- Create: `skilleye/quality/keypoints.py`
- Create: `skilleye/quality/phases.py`
- Test: `skilleye/quality/test_phases.py`

**Interfaces:**
- Produces: `keypoints.L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, NOSE, L_EYE, R_EYE, L_EAR, R_EAR` (all `int`).
- Produces: `phases.PHASES` (`list[str]`, `["backswing", "contact", "follow_through"]`).
- Produces: `phases.dominant_wrist_index(kpts: np.ndarray) -> int`
- Produces: `phases.detect_contact_frame(kpts: np.ndarray) -> int`
- Produces: `phases.split_phases(kpts: np.ndarray, contact_window: int = 5) -> dict[str, np.ndarray]` (keys match `PHASES`, values are `(T_phase, 17, 2)` arrays, `T_phase` may be 0).

- [ ] **Step 1: Create the package marker**

Create `skilleye/quality/__init__.py` with this exact content (empty file, just a marker):

```python
```

- [ ] **Step 2: Create the keypoint constants module**

Create `skilleye/quality/keypoints.py`:

```python
"""COCO-17 keypoint index constants, shared by the quality-scoring modules."""

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
```

- [ ] **Step 3: Write the failing tests for phase detection**

Create `skilleye/quality/test_phases.py`:

```python
import numpy as np

from quality.phases import dominant_wrist_index, detect_contact_frame, split_phases, PHASES
from quality.keypoints import R_WRIST


def make_kpts(T, wrist_positions):
    """Builds a minimal (T, 17, 2) array with all joints static at the origin
    except the right wrist, which moves through wrist_positions (list of
    (x, y) tuples, length T)."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)
    for t in range(T):
        kpts[t, R_WRIST] = wrist_positions[t]
    return kpts


def test_dominant_wrist_index_picks_the_moving_wrist():
    T = 10
    positions = [(float(t), 0.0) for t in range(T)]  # right wrist moves, left stays at 0
    kpts = make_kpts(T, positions)
    assert dominant_wrist_index(kpts) == R_WRIST


def test_detect_contact_frame_finds_speed_peak():
    # right wrist accelerates steadily, has one big jump at t=5, then slows down
    T = 11
    positions = []
    x = 0.0
    for t in range(T):
        step = 1.0 if t < 5 else (5.0 if t == 5 else 0.2)
        x += step
        positions.append((x, 0.0))
    kpts = make_kpts(T, positions)
    contact = detect_contact_frame(kpts)
    assert 3 <= contact <= 7  # near the big jump between frame 4 and frame 5


def test_detect_contact_frame_falls_back_on_short_clips():
    kpts = np.zeros((2, 17, 2), dtype=np.float32)
    assert detect_contact_frame(kpts) == 1  # T // 2


def test_split_phases_returns_all_three_phases_covering_the_full_clip():
    T = 20
    positions = [(float(t), 0.0) for t in range(T)]
    kpts = make_kpts(T, positions)
    result = split_phases(kpts)
    assert set(result.keys()) == set(PHASES)
    total_frames = sum(p.shape[0] for p in result.values())
    assert total_frames == T
```

- [ ] **Step 4: Run the tests to verify they fail**

Run (from `E:/SkillEye/skilleye`):
```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_phases.py -v
```
Expected: collection error or `ModuleNotFoundError: No module named 'quality.phases'` — `phases.py` doesn't exist yet.

- [ ] **Step 5: Implement phase detection**

Create `skilleye/quality/phases.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_phases.py -v
```
Expected: `4 passed`

- [ ] **Step 7: Commit**

```bash
cd E:/SkillEye
git add skilleye/quality/__init__.py skilleye/quality/keypoints.py skilleye/quality/phases.py skilleye/quality/test_phases.py
git commit -m "Add quality package: keypoint constants + swing phase detection"
```

---

### Task 2: Joint angle computation

**Files:**
- Create: `skilleye/quality/angles.py`
- Test: `skilleye/quality/test_angles.py`

**Interfaces:**
- Consumes: `quality.keypoints.{L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE}` (Task 1).
- Produces: `angles.JOINT_DEFINITIONS` (`dict[str, tuple[int,int,int]]`, keys `"left_elbow"`, `"right_elbow"`, `"left_knee"`, `"right_knee"`).
- Produces: `angles.joint_angle_series(kpts: np.ndarray, a: int, b: int, c: int) -> np.ndarray` (shape `(T,)`, radians; `T` may be 0).
- Produces: `angles.trunk_rotation_series(kpts: np.ndarray) -> np.ndarray` (shape `(T,)`).
- Produces: `angles.compute_all_angles(kpts: np.ndarray) -> dict[str, np.ndarray]` (keys: `JOINT_DEFINITIONS` keys plus `"trunk_rotation"`).
- Produces: `angles.phase_mean_angles(kpts: np.ndarray) -> dict[str, float | None]` (`None` means the input had 0 frames).

- [ ] **Step 1: Write the failing tests**

Create `skilleye/quality/test_angles.py`:

```python
import numpy as np

from quality.angles import (
    joint_angle_series, trunk_rotation_series, compute_all_angles,
    phase_mean_angles, JOINT_DEFINITIONS,
)
from quality.keypoints import L_SHOULDER, R_SHOULDER, L_ELBOW, L_WRIST, L_HIP, R_HIP


def test_joint_angle_series_right_angle():
    # shoulder at (0,1), elbow at (0,0), wrist at (1,0) -> 90 degrees at the elbow
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (0, 0)
    kpts[0, L_WRIST] = (1, 0)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert result.shape == (1,)
    assert np.isclose(result[0], np.pi / 2, atol=1e-5)


def test_joint_angle_series_straight_arm_is_pi():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (0, 0)
    kpts[0, L_WRIST] = (0, -1)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert np.isclose(result[0], np.pi, atol=1e-5)


def test_joint_angle_series_empty_input():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert result.shape == (0,)


def test_trunk_rotation_series_parallel_lines_is_zero():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 0)
    kpts[0, R_SHOULDER] = (1, 0)
    kpts[0, L_HIP] = (0, -1)
    kpts[0, R_HIP] = (1, -1)
    result = trunk_rotation_series(kpts)
    assert np.isclose(result[0], 0.0, atol=1e-5)


def test_compute_all_angles_has_expected_keys_and_shapes():
    kpts = np.zeros((3, 17, 2), dtype=np.float32)
    result = compute_all_angles(kpts)
    assert set(result.keys()) == set(JOINT_DEFINITIONS.keys()) | {"trunk_rotation"}
    for series in result.values():
        assert series.shape == (3,)


def test_phase_mean_angles_returns_none_for_empty_phase():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = phase_mean_angles(kpts)
    assert all(v is None for v in result.values())


def test_phase_mean_angles_returns_float_for_nonempty_phase():
    kpts = np.zeros((3, 17, 2), dtype=np.float32)
    kpts[:, L_SHOULDER] = (0, 1)
    kpts[:, L_ELBOW] = (0, 0)
    kpts[:, L_WRIST] = (1, 0)
    result = phase_mean_angles(kpts)
    assert isinstance(result["left_elbow"], float)
    assert np.isclose(result["left_elbow"], np.pi / 2, atol=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_angles.py -v
```
Expected: `ModuleNotFoundError: No module named 'quality.angles'`

- [ ] **Step 3: Implement joint angle computation**

Create `skilleye/quality/angles.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_angles.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
cd E:/SkillEye
git add skilleye/quality/angles.py skilleye/quality/test_angles.py
git commit -m "Add quality package: joint angle computation"
```

---

### Task 3: Scoring against expert templates

**Files:**
- Create: `skilleye/quality/score.py`
- Test: `skilleye/quality/test_score.py`

**Interfaces:**
- Consumes: `quality.phases.{PHASES, split_phases}` (Task 1), `quality.angles.phase_mean_angles` (Task 2).
- Consumes (at call time, not import time): a `templates` dict shaped `{stroke: {phase: {joint: {"mean": float, "std": float, "n": int}}}}` — produced by Task 4's `build_expert_templates.py` and loaded from `results/quality_templates/templates.json`'s `"templates"` key.
- Produces: `score.FLAG_THRESHOLD` (`float`, `1.5`), `score.SCORE_SCALE` (`float`, `15`).
- Produces: `score.suggestion_text(joint: str, phase: str, z: float) -> str`.
- Produces: `score.score_clip(kpts: np.ndarray, stroke_class: str, templates: dict) -> dict` with keys `"overall_score"` (`float`), `"table"` (`list[dict]`, each `{"phase": str, "joint": str, "value": float, "z": float | None, "flagged": bool, "note": str | None}` — `z=None`/`note="insufficient template data"` when the template has no usable entry for that (phase, joint), per the design spec's error-handling requirement to surface this rather than silently drop the row), `"suggestions"` (`list[str]`).

- [ ] **Step 1: Write the failing tests**

Create `skilleye/quality/test_score.py`:

```python
import numpy as np

from quality.score import score_clip, suggestion_text
from quality.keypoints import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)
from quality.phases import PHASES


def make_straight_arm_clip(T=20):
    """A static clip: left arm straight (pi rad), right elbow at 90 degrees,
    both legs straight (pi rad), shoulder/hip lines parallel (0 rad trunk
    rotation) -- every angle computation below is well-defined."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)
    kpts[:, L_SHOULDER] = (0, 2)
    kpts[:, L_ELBOW] = (0, 1)
    kpts[:, L_WRIST] = (0, 0)
    kpts[:, R_SHOULDER] = (1, 2)
    kpts[:, R_ELBOW] = (1, 1)
    kpts[:, R_WRIST] = (2, 1)
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_score.py -v
```
Expected: `ModuleNotFoundError: No module named 'quality.score'`

- [ ] **Step 3: Implement scoring**

Create `skilleye/quality/score.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/test_score.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Run the full quality test suite together**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m pytest quality/ -v
```
Expected: `16 passed` (4 from test_phases.py + 7 from test_angles.py + 5 from test_score.py)

- [ ] **Step 6: Commit**

```bash
cd E:/SkillEye
git add skilleye/quality/score.py skilleye/quality/test_score.py
git commit -m "Add quality package: expert-template scoring and correction suggestions"
```

---

### Task 4: Build expert templates (real data, run once)

**Files:**
- Create: `skilleye/build_expert_templates.py`
- Creates at runtime: `results/quality_templates/templates.json`

**Interfaces:**
- Consumes: `stroke_dataset.{load_records, subject_disjoint_split, STROKE_CLASSES}` (existing), `quality.phases.split_phases` (Task 1), `quality.angles.phase_mean_angles` (Task 2).
- Produces: `results/quality_templates/templates.json` shaped `{"templates": {stroke: {phase: {joint: {"mean", "std", "n"}}}}, "val_subjects": [int, ...], "phases": [str, ...]}` — the `"templates"` value is exactly what Task 3's `score_clip` expects as its `templates` argument; the `"val_subjects"` value is exactly what Task 6's `app.py` uses to restrict demo-eligible clips.

- [ ] **Step 1: Implement the template builder**

Create `skilleye/build_expert_templates.py`:

```python
"""
Builds per-stroke-class expert-motion templates for quality scoring: for each
stroke class, the mean and std of each joint's average angle within each
phase (backswing/contact/follow_through), computed across that class's
expert-labeled clips from the TRAINING side of this project's standard
subject-disjoint split (same split, same seed, as the single-split
classifiers elsewhere in this repo -- not a new split). The held-out
validation subjects from that same split are saved alongside the templates
so the demo UI can restrict itself to clips that were never used to build
the template being compared against.

Usage:
    python build_expert_templates.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/results/quality_templates/templates.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stroke_dataset import load_records, subject_disjoint_split, STROKE_CLASSES
from quality.phases import split_phases, PHASES
from quality.angles import phase_mean_angles


def clip_phase_means(kpts):
    """kpts: (T, 17, 2). Returns {phase: {joint: float or None}}."""
    phases = split_phases(kpts)
    return {phase: phase_mean_angles(phases[phase]) for phase in PHASES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records, _ = load_records(args.skeletons)
    train_records, val_records, val_subjects = subject_disjoint_split(
        records, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_records)} clips, val (held out, demo-eligible): "
          f"{len(val_records)} clips ({len(val_subjects)} subjects)")

    expert_train = [r for r in train_records if r["skill_level"] == "expert"]
    print(f"expert clips in training side: {len(expert_train)}")

    # values[stroke][phase][joint] = list of per-clip phase-mean angles
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in expert_train:
        means = clip_phase_means(r["kpts"])
        for phase in PHASES:
            for joint, value in means[phase].items():
                if value is not None:
                    values[r["stroke"]][phase][joint].append(value)

    templates = {}
    for stroke in STROKE_CLASSES:
        templates[stroke] = {}
        for phase in PHASES:
            templates[stroke][phase] = {}
            joint_lists = values[stroke][phase]
            for joint, joint_values in joint_lists.items():
                arr = np.array(joint_values)
                templates[stroke][phase][joint] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n": int(len(arr)),
                }
            n_clips = len(next(iter(joint_lists.values()), []))
            print(f"  {stroke:16s} {phase:16s}: {n_clips} expert clips contributed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "templates": templates,
            "val_subjects": sorted(val_subjects),
            "phases": PHASES,
        }, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" build_expert_templates.py --skeletons E:/SkillEye/skeletons --out E:/SkillEye/results/quality_templates/templates.json
```
Expected: prints train/val clip counts, then one line per (stroke, phase) with a clip count that is a nonzero double-digit number for every one of the 6 strokes x 3 phases (18 lines) — e.g. `backhand          backswing       : 172 expert clips contributed`. Zero anywhere here would mean a bug (every stroke class has expert clips in the training split, as established in the design spec). Ends with `saved -> ...templates.json`.

- [ ] **Step 3: Inspect the output for sanity**

```bash
"/c/Users/uclla/miniconda3/envs/torch/python" -c "
import json
with open('E:/SkillEye/results/quality_templates/templates.json') as f:
    d = json.load(f)
print('strokes:', list(d['templates'].keys()))
print('val subjects:', len(d['val_subjects']))
for stroke, phases in d['templates'].items():
    for phase, joints in phases.items():
        for joint, stats in joints.items():
            assert stats['std'] >= 0, (stroke, phase, joint, stats)
            assert stats['n'] > 0, (stroke, phase, joint, stats)
print('all templates have non-negative std and n > 0 -- OK')
"
```
Expected: lists the 6 stroke classes, a val-subjects count around 11 (20% of 55), and ends with `all templates have non-negative std and n > 0 -- OK` with no `AssertionError`.

- [ ] **Step 4: Commit**

```bash
cd E:/SkillEye
git add skilleye/build_expert_templates.py results/quality_templates/templates.json
git commit -m "Add expert-template builder and generated templates.json"
```

---

### Task 5: Smoke check (experts should score higher than beginners)

**Files:**
- Create: `skilleye/smoke_check_quality_scoring.py`

**Interfaces:**
- Consumes: `stroke_dataset.load_records` (existing), `quality.score.score_clip` (Task 3), `results/quality_templates/templates.json` (Task 4).
- Produces: a printed per-stroke comparison table; no new importable interface (this is a standalone diagnostic script, not a library module).

- [ ] **Step 1: Implement the smoke check**

Create `skilleye/smoke_check_quality_scoring.py`:

```python
"""
Smoke check for the quality-scoring system. No formal ground truth exists to
validate scores against yet (see docs/superpowers/specs/
2026-07-09-quality-scoring-and-ui-design.md, "Testing") -- this instead
confirms the one directional claim that must hold for the system to be
credible: held-out expert clips should score higher on average than
held-out beginner clips, per stroke class, against that stroke's expert
template. If any stroke fails this, FLAG_THRESHOLD/SCORE_SCALE or the angle
set (skilleye/quality/angles.py) need revisiting before using this in the
demo -- do not ship a quality scorer that rates beginners above experts.

Usage:
    python smoke_check_quality_scoring.py --skeletons E:/SkillEye/skeletons \
        --templates E:/SkillEye/results/quality_templates/templates.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from stroke_dataset import load_records
from quality.score import score_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--templates", required=True)
    args = ap.parse_args()

    with open(args.templates) as f:
        data = json.load(f)
    templates = data["templates"]
    val_subjects = set(data["val_subjects"])

    records, _ = load_records(args.skeletons)
    val_records = [r for r in records if r["subject_id"] in val_subjects]
    print(f"scoring {len(val_records)} held-out clips ({len(val_subjects)} subjects)")

    scores_by_stroke_skill = defaultdict(list)
    for r in val_records:
        result = score_clip(r["kpts"], r["stroke"], templates)
        scores_by_stroke_skill[(r["stroke"], r["skill_level"])].append(result["overall_score"])

    strokes = sorted({stroke for stroke, _ in scores_by_stroke_skill})
    print(f"\n{'stroke':16s} {'expert mean':>12s} {'beginner mean':>14s} {'experts higher?':>16s}")
    all_ok = True
    for stroke in strokes:
        expert_scores = scores_by_stroke_skill.get((stroke, "expert"), [])
        beginner_scores = scores_by_stroke_skill.get((stroke, "beginner"), [])
        if not expert_scores or not beginner_scores:
            print(f"{stroke:16s} -- insufficient held-out data for one group, skipped --")
            continue
        expert_mean = float(np.mean(expert_scores))
        beginner_mean = float(np.mean(beginner_scores))
        ok = expert_mean > beginner_mean
        all_ok = all_ok and ok
        print(f"{stroke:16s} {expert_mean:12.1f} {beginner_mean:14.1f} {'yes' if ok else 'NO':>16s}")

    print()
    if all_ok:
        print("OVERALL: experts scored higher on every stroke with held-out data.")
    else:
        print("OVERALL: at least one stroke did NOT show experts scoring higher -- "
              "revisit SCORE_SCALE/angle set in skilleye/quality/ before using this in the demo.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" smoke_check_quality_scoring.py --skeletons E:/SkillEye/skeletons --templates E:/SkillEye/results/quality_templates/templates.json
```
Expected: a table with one row per stroke class, and the final line either `OVERALL: experts scored higher on every stroke with held-out data.` (proceed to Task 6) or the revisit message (see Step 3).

- [ ] **Step 3: If the smoke check fails for any stroke, adjust before proceeding**

If the "OVERALL" line reports a failure, do not proceed to Task 6 yet. Likely causes, in order of likelihood: (a) `FLAG_THRESHOLD`/`SCORE_SCALE` in `skilleye/quality/score.py` need different values — try `SCORE_SCALE = 10.0` or `20.0` and re-run Step 2; (b) one of the four flat joint angles isn't actually discriminative for that stroke and is adding noise — this is diagnosable by temporarily printing per-joint z-scores in the loop above for the failing stroke's clips and checking which joint's mean z-score direction disagrees between the expert and beginner groups. Record whatever change fixed it in the commit message in Step 4. Do not skip this check and ship a scorer that rates beginners above experts.

- [ ] **Step 4: Commit**

```bash
cd E:/SkillEye
git add skilleye/smoke_check_quality_scoring.py
git commit -m "Add quality-scoring smoke check (experts must score higher than beginners)"
```

---

### Task 6: Streamlit demo UI

**Files:**
- Create: `skilleye/app.py`
- Modify: `skilleye/requirements.txt`

**Interfaces:**
- Consumes: `stroke_dataset.{load_records, STROKE_CLASSES, resample_time, add_velocity}` (existing), `stgcn_model.{STGCN, COCO17_EDGES}` (existing), `quality.score.score_clip` (Task 3), `results/quality_templates/templates.json` (Task 4), `results/stroke_classifier_v2/best_model.pt` (existing, already-trained checkpoint).
- Produces: a runnable Streamlit app; no importable interface (terminal node of this feature).

- [ ] **Step 1: Add streamlit to requirements**

Read `skilleye/requirements.txt` first, then add a line. Current content is:
```
rtmlib==0.0.15
opencv-python==4.13.0.90
numpy==2.2.6
onnxruntime-gpu==1.23.2
```
Append these two lines (both already installed in the `torch` env, verified: streamlit 1.58.0, pytest 9.0.2 — this just documents them for anyone setting up the environment fresh):
```
streamlit==1.58.0
pytest==9.0.2
```

- [ ] **Step 2: Implement the app**

Create `skilleye/app.py`:

```python
"""
Streamlit demo: pick a held-out sample swing, see its predicted stroke type,
quality score, per-phase/joint deviation table, and correction suggestions.

Run (from the skilleye/ directory):
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt

from stroke_dataset import load_records, STROKE_CLASSES, resample_time, add_velocity
from stgcn_model import STGCN, COCO17_EDGES
from quality.score import score_clip

SKELETONS_DIR = "E:/SkillEye/skeletons"
TEMPLATES_PATH = "E:/SkillEye/results/quality_templates/templates.json"
STROKE_MODEL_PATH = "E:/SkillEye/results/stroke_classifier_v2/best_model.pt"


@st.cache_resource
def load_stroke_model():
    model = STGCN(num_classes=len(STROKE_CLASSES))
    model.load_state_dict(torch.load(STROKE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_templates():
    with open(TEMPLATES_PATH) as f:
        data = json.load(f)
    return data["templates"], set(data["val_subjects"])


@st.cache_data
def load_demo_records():
    _, val_subjects = load_templates()
    records, _ = load_records(SKELETONS_DIR)
    return [r for r in records if r["subject_id"] in val_subjects]


def predict_stroke(model, kpts):
    resampled = resample_time(kpts, 64)
    with_velocity = add_velocity(resampled)
    tensor = torch.from_numpy(with_velocity.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    return STROKE_CLASSES[pred_idx], float(probs[pred_idx])


def render_skeleton_frame(kpts, frame_idx):
    frame = kpts[frame_idx]
    fig, ax = plt.subplots(figsize=(4, 4))
    for a, b in COCO17_EDGES:
        ax.plot([frame[a, 0], frame[b, 0]], [-frame[a, 1], -frame[b, 1]], color="#2a78d6", linewidth=2)
    ax.scatter(frame[:, 0], -frame[:, 1], color="#184f95", s=15, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def main():
    st.set_page_config(page_title="SkillEye Quality Scoring Demo", layout="wide")
    st.title("SkillEye: Swing Quality Scoring Demo")
    st.caption("Sample clips are held-out validation subjects only -- never used to "
               "build the expert templates being compared against.")

    templates, _ = load_templates()
    records = load_demo_records()

    strokes_available = sorted({r["stroke"] for r in records})
    stroke_choice = st.sidebar.selectbox("Stroke category", strokes_available)

    clips_in_stroke = [r for r in records if r["stroke"] == stroke_choice]
    clip_labels = [
        f"subject {r['subject_id']} ({r['skill_level']}) - {Path(r['source']).stem}"
        for r in clips_in_stroke
    ]
    clip_idx = st.sidebar.selectbox(
        "Sample clip", range(len(clip_labels)), format_func=lambda i: clip_labels[i])
    record = clips_in_stroke[clip_idx]
    kpts = record["kpts"]

    model = load_stroke_model()
    pred_stroke, pred_confidence = predict_stroke(model, kpts)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Skeleton viewer")
        frame_idx = st.slider("Frame", 0, kpts.shape[0] - 1, 0)
        st.pyplot(render_skeleton_frame(kpts, frame_idx))
        st.metric("Predicted stroke", pred_stroke, f"{pred_confidence*100:.0f}% confidence")
        st.caption(f"True label (for reference): {record['stroke']}, {record['skill_level']}")

    with col2:
        result = score_clip(kpts, pred_stroke, templates)
        st.subheader("Quality score")
        st.metric("Overall", f"{result['overall_score']:.0f} / 100")

        st.subheader("Per-phase / per-joint deviation")
        st.table([
            {
                "phase": row["phase"],
                "joint": row["joint"],
                "value (rad)": round(row["value"], 3),
                "z-score": round(row["z"], 2) if row["z"] is not None else row["note"],
                "flagged": "!" if row["flagged"] else "",
            }
            for row in result["table"]
        ])

        st.subheader("Correction suggestions")
        if result["suggestions"]:
            for s in result["suggestions"]:
                st.write(f"- {s}")
        else:
            st.write("No significant deviations flagged against the expert template.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Launch it and verify manually**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" -m streamlit run app.py --server.headless true
```
Expected: prints a local URL (e.g. `http://localhost:8501`). Open it in a browser (or use a tool that can screenshot a local URL) and verify:
- The sidebar has a stroke-category dropdown and a clip dropdown that updates when the category changes.
- The skeleton viewer renders a recognizable stick-figure pose, and the frame slider changes the pose shown.
- "Predicted stroke" shows one of the 6 `STROKE_CLASSES` with a confidence percentage.
- "Overall" quality score renders a number between 0 and 100.
- The per-phase/per-joint table has rows for `backswing`, `contact`, `follow_through` and is non-empty.
- Switching to at least 2 different stroke categories and 2 different clips within a category produces visibly different scores/tables (not a frozen/identical output regardless of selection — this would indicate `kpts`/`record` isn't actually changing with the widget state).

Stop the process (Ctrl+C) once verified.

- [ ] **Step 4: Commit**

```bash
cd E:/SkillEye
git add skilleye/app.py skilleye/requirements.txt
git commit -m "Add Streamlit demo UI for the quality-scoring system"
```

---

### Task 7: Documentation — fold the quality-scoring system into the README

**Files:**
- Modify: `E:/SkillEye/README.md`

**Interfaces:**
- Consumes: the real printed output from Task 5's smoke check (captured when Task 5 was executed) — this task must use the *actual* numbers observed then, not placeholder values.

- [ ] **Step 1: Re-run the smoke check to have its exact current output on hand**

```bash
cd E:/SkillEye/skilleye
"/c/Users/uclla/miniconda3/envs/torch/python" smoke_check_quality_scoring.py --skeletons E:/SkillEye/skeletons --templates E:/SkillEye/results/quality_templates/templates.json
```
Copy the full per-stroke table it prints — Step 3 below uses these exact numbers, not approximations.

- [ ] **Step 2: Add a Methodology subsection**

In `README.md`, after the existing "### 2.5 Evaluation Protocol" subsection and before "## 3. Results", add:

```markdown
### 2.6 Quality Scoring System

The proposal's central promised feature -- a per-swing motion-quality score with
per-joint/phase error detection and rule-based correction suggestions (Section 4.7) -- is
implemented here as a rule-based comparison against expert-clip statistics, not a learned
regression model. No coach-rating ground truth exists yet (Path A/B, Section 5), so a
supervised quality-score model cannot be trained honestly; this is an explicitly-scoped
interim v1, in the same spirit as the v1->v2->v3 iterations already described for the
classifiers above.

**Phase detection** (`skilleye/quality/phases.py`): each clip's dominant wrist (whichever
moves more overall -- a proxy for the hitting arm, since THETIS doesn't label handedness)
is used to find the contact frame (peak wrist speed, a standard swing-analysis heuristic),
splitting the clip into three phases: backswing, a short window around contact, and
follow-through.

**Joint angles** (`skilleye/quality/angles.py`): per phase, four flexion angles
(left/right elbow, left/right knee) plus a trunk-rotation proxy (shoulder-line vs.
hip-line angle) are averaged into one scalar per (phase, joint).

**Expert templates** (`skilleye/build_expert_templates.py`, run once offline): for each
stroke class, the mean and standard deviation of each (phase, joint) scalar across that
class's expert-labeled clips, computed only from the training side of this project's
standard subject-disjoint split (Section 2.5) -- never from the held-out validation
subjects, which remain available as genuinely unseen demo inputs.

**Scoring** (`skilleye/quality/score.py`): a query clip's (phase, joint) scalars are
z-scored against its predicted stroke's template; |z| > 1.5 is flagged and mapped to a
fixed coaching-tip sentence, and the overall 0-100 score is a monotonic function of mean
absolute deviation. `SCORE_SCALE`/`FLAG_THRESHOLD` are sanity-checked (Section 3.3), not
formally calibrated -- that calibration is exactly what Path A/B ground truth would
enable.

**Demo UI** (`skilleye/app.py`, Streamlit): pick a stroke category and a sample clip
(restricted to the held-out validation subjects, so every demo score is computed against
a template that never saw that subject) and see the skeleton, the existing stroke
classifier's prediction, the quality score, the per-phase/joint table, and the correction
suggestions together. Run with `streamlit run app.py` from `skilleye/`.
```

- [ ] **Step 3: Add a Results subsection**

In `README.md`, after the existing "### 3.2 Beginner vs. Expert" subsection and before "## 4. Discussion", add a new subsection. Replace `<PASTE STEP 1 TABLE HERE>` below with the exact table captured in Step 1 (do not paraphrase or round differently than what was printed):

```markdown
### 3.3 Quality Scoring Smoke Check

No formal ground truth exists yet to validate quality scores against (Section 2.6) --
this instead checks the one directional claim the system must satisfy to be credible:
held-out expert clips should score higher on average than held-out beginner clips, per
stroke class, against that stroke's own expert template.

<PASTE STEP 1 TABLE HERE>

This is a sanity check, not a validation -- it confirms the scoring system's direction
is sane on the same subject-level proxy label used in Section 3.2, not that its absolute
scores or flagged joints are correct at the level of a real coach's judgment. That
remains gated on Path A/B ground truth (Section 5).
```

- [ ] **Step 4: Update the Discussion section**

In `README.md`, in "### 4.2 Implications for the Quality-Score System", the paragraph currently ends with "...because Section 3.2's result is a proof of learnability, not a finished quality metric." Add this sentence immediately after it:

```markdown
Section 2.6/3.3's rule-based scorer is a further step in that direction -- a working,
demonstrable system rather than only a proof of concept -- but it is calibrated by a
sanity check, not by the coach ratings or synthetic ground truth that would let its
scores be trusted at face value.
```

- [ ] **Step 5: Update Conclusion and Future Work item 5**

In `README.md`, "## 5. Conclusion and Future Work" currently has as item 5:

```markdown
5. **The quality-score model itself** (proposal Section 4.7) — regression rather than
   classification, built on the foundation established here once ground truth from (2) and/or
   (4) is available.
```

Replace it with:

```markdown
5. **A learned quality-score model** (proposal Section 4.7) — Section 2.6/3.3 delivers a
   working rule-based v1 (phase detection, joint angles, expert-template comparison); once
   ground truth from (2) and/or (4) is available, that data enables replacing or
   augmenting the rule-based scorer with a trained regression model, and calibrating
   `FLAG_THRESHOLD`/`SCORE_SCALE` against real quality judgments instead of a directional
   sanity check.
```

Also update the Abstract (near the top of the file) — it currently ends with "Section 5 lays out the remaining work toward the full quality-score system." Insert this sentence before it:

```markdown
A rule-based v1 of that quality-score system — phase detection, per-joint deviation from
expert-clip templates, and generated correction suggestions — is implemented and
demonstrated through an accompanying Streamlit UI.
```

- [ ] **Step 6: Update the Reproducibility repo layout**

In `README.md`, in "## 6. Reproducibility", the `skilleye/` block currently ends with `generate_figures.py ...` and the `results/` block starts with `RESULTS_SUMMARY.md ...`. Add these lines to the `skilleye/` block (after the `generate_figures.py` line, before `requirements.txt`):

```
  quality/                         phase detection, joint angles, template scoring (§2.6)
  build_expert_templates.py        builds results/quality_templates/templates.json (§2.6)
  smoke_check_quality_scoring.py   experts-score-higher-than-beginners sanity check (§3.3)
  app.py                           Streamlit demo UI (§2.6) -- run: streamlit run app.py
```

And add this line to the `results/` block (after the `figures/` line):

```
  quality_templates/                templates.json: per-stroke expert (phase, joint) statistics (§2.6)
```

- [ ] **Step 7: Commit and push**

```bash
cd E:/SkillEye
git add README.md
git commit -m "Document the quality-scoring system and demo UI in the README"
git push
```
