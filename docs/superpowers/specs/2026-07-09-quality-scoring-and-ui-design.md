# Quality Scoring System + Demo UI — Design

Date: 2026-07-09
Status: Approved for implementation

## Problem

SkillEye's proposal promises a per-swing motion-quality score, per-joint/phase error
detection, and rule-based correction suggestions (proposal Section 4.7). None of this
exists yet — the repo so far only has pose extraction and stroke-type classification,
which are supporting infrastructure and a solved sub-problem, not the project's actual
novel contribution. There is also no working demo/UI, which hurts Practicality (10% of
judging weight) independent of the modeling work.

Real coach-rating ground truth (Path B) and own side-view recordings haven't started —
both have long lead times outside this session's scope. This design has to produce a
legitimate, defensible quality-scoring system using only what already exists: the
extracted THETIS skeletons and their beginner/expert labels.

## Approach: rule-based comparison against expert templates

For each of the 6 stroke classes (Table 1 in README.md), build a reference "template" —
the average per-phase joint-angle curve across that class's expert-labeled clips. Score a
new clip by comparing its own per-phase joint angles against its predicted stroke's
template, as a z-score per joint per phase. This is chosen over a learned regression
model (synthetic perturbation, Path A) because it requires no new training data or
training run, is directly interpretable (every number traces to "your elbow angle vs. the
expert average, in this phase"), and produces natural language correction suggestions
without a separate text-generation step — each flagged (joint, phase, direction) maps to
a fixed coaching-tip template.

This does **not** replace Path A/B — it's an interim, fully-defensible system that
demonstrates the concept end-to-end today, explicitly scoped as v1 in the same spirit as
the stroke/skill classifiers' v1→v2→v3 iterations already in the repo.

## Components

### 1. Phase detection (`quality/phases.py`)

Input: normalized skeleton sequence `(T, 17, 2)`. Determine the dominant wrist (the one
with greater total displacement across the clip — proxy for hitting arm, since THETIS
doesn't label handedness). Find the contact frame as the frame of peak dominant-wrist
speed (standard swing-analysis heuristic). Split into 3 phases:

- **backswing**: clip start → contact frame minus a small buffer
- **contact**: a fixed window of ±5 raw frames around the contact frame (clipped to clip
  boundaries if the contact frame is near the start/end)
- **follow_through**: contact frame plus buffer → clip end

Each phase is independently time-resampled (reusing `resample_time` from
`stroke_dataset.py`) to a fixed frame count so phases of different absolute duration
compare fairly.

### 2. Joint angles (`quality/angles.py`)

Compute a fixed set of interpretable per-frame angles: left/right elbow (shoulder-elbow-wrist),
left/right knee (hip-knee-ankle), and trunk rotation proxy (shoulder-line vs. hip-line
angle). These are chosen because they're standard, coach-legible swing-mechanics
checkpoints, not because they're exhaustive — extending the angle set later is a
one-function change (§ code structure keeps them in one list).

### 3. Expert templates (`build_expert_templates.py`, run once offline)

For each stroke class: take all *expert*-labeled clips of that class from the **training**
side of the same subject-disjoint split used elsewhere in this project
(`subject_disjoint_split` from `stroke_dataset.py`, val_frac=0.2, seed=42 — the same split
already used for the single-split classifiers, not a new one). Run phase detection +
joint angles on each, average across clips per (phase, joint) → template mean + std. Save
as `results/quality_templates/templates.json`. This is precomputed once; the app loads it,
it does not recompute templates live.

**Consistency requirement**: the demo UI (§5) must only offer clips from that same split's
**held-out validation subjects** for scoring — never a subject whose clips were used to
build the template. This follows the subject-disjoint principle already used throughout
this project (README §2.5) and matters here for the same reason: scoring a clip against a
template that includes that clip's own subject would be a clip "grading itself," which
would quietly inflate how well experts appear to score and undermines the credibility of
the whole demo if a judge checks this detail.

### 4. Scoring (`quality/score.py`)

Given a query clip + its predicted stroke class (from the existing trained ST-GCN
classifier, `results/stroke_classifier_v2/best_model.pt`):

1. Run phase detection + joint angles on the query clip.
2. For each (phase, joint): z = (value − template_mean) / template_std.
3. Overall score = `100 − clip(mean(|z|) × scale, 0, 100)` (scale constant tuned so a
   typical expert clip scores roughly 75-95 and a typical beginner clip scores lower —
   validated via the smoke check in Testing, not a formal calibration).
4. Flag any (phase, joint) with `|z| > 1.5` as a deviation.
5. Map each flagged (joint, phase, sign of z) to one of a small set of fixed
   coaching-tip templates (e.g. right elbow, follow_through, negative z → "your right arm
   is cutting the follow-through short — try extending further after contact").

### 5. UI (`app.py`, Streamlit)

- Sidebar: dropdown for stroke category, dropdown for a sample clip within that category
  — restricted to clips from the template split's held-out validation subjects (§3
  consistency requirement), not the full dataset.
- Main panel:
  - Skeleton viewer: matplotlib plot of the skeleton at a frame, with a slider to scrub
    through the clip (simple, no video encoding needed).
  - Predicted stroke type + confidence (from the existing classifier — reuses it, doesn't
    retrain).
  - Overall quality score, large and prominent.
  - Per-phase/per-joint table, deviation flagged rows visually distinct.
  - Correction suggestions as a bullet list.

## Data flow

```
skeleton JSON (existing)
  -> phases.py: detect backswing/contact/follow_through
  -> angles.py: per-frame joint angles per phase
  -> score.py: compare to results/quality_templates/templates.json (built once, offline)
  -> {overall_score, per_joint_phase_table, suggestions}
  -> app.py renders all of the above + existing stroke classifier's prediction
```

## Error handling

- Clips too short to detect a meaningful speed peak (contact frame): fall back to the
  midpoint of the clip as the contact frame rather than crashing.
- Missing/zero-variance template entries (a stroke/joint/phase combination with too few
  expert samples): skip that cell in scoring rather than dividing by zero, and note it as
  "insufficient template data" in the UI rather than silently omitting it.

## Testing

No formal ground truth exists to validate scores against (that's exactly the gap Path
A/B is meant to close later), so testing here is a smoke check, not a formal evaluation:
run the scorer on a sample of held-out expert clips and a sample of held-out beginner
clips (per stroke class) and confirm experts score higher on average than beginners on
their own stroke's template. If they don't, the angle set or z-score scale needs
adjusting before shipping.

## Out of scope for this iteration

- Live video upload / real-time pose extraction in the UI (explicitly deferred per this
  session's scope decision — sample THETIS clips only).
- A learned (Path A synthetic-perturbation) regression scorer — may be added later
  alongside this rule-based one, not instead of it.
- Commercial Value and Cross-Disciplinary Team judging criteria — these live in the
  competition proposal document itself, not in this repository; nothing in this design
  addresses them, and no implementation work here should be represented as doing so.
