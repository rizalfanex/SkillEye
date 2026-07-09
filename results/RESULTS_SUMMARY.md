# SkillEye — technical results (RTX 5060 machine run, 2026-07-09)

Generated overnight while the pipeline ran unattended, for direct use in the CTCI 2026
proposal's Technical Content & Completeness section. Everything below is a real trained
result on the full THETIS dataset, not a projected target. Updated same night with a second
iteration (v2) after reviewing v1 — 6-class stroke split (was 5), added velocity features +
augmentation, and replaced the beginner/expert linear-feature model with a proper ST-GCN.
v1 numbers are kept alongside for comparison, not deleted.

## 1. Skeleton extraction

Full THETIS `VIDEO_RGB` set processed through the RTMPose → primary-subject-tracking →
normalization pipeline (`skeleton_pipeline.py` / `batch_extract.py`):

- **1,980 / 1,980 clips extracted, 0 dropped, 0 failed** (all 12 categories, all 55 subjects).
- 46.5 minutes total (~1.4s/clip) on this machine's RTX 5060 via DirectML.
- Output: `E:\SkillEye\skeletons\<category>\p<id>_<action>_s<seq>.json` — normalized COCO-17
  keypoints + per-joint confidence + metadata (subject id, category, skill level, fps,
  resolution) per clip.

**Environment note:** this machine's `onnxruntime-gpu` (CUDA) build didn't work — it needs
CUDA 12.x cuBLAS/cuDNN, but this machine's PyTorch is a CUDA 13 build, and the network here
couldn't reliably pull the ~550MB replacement CUDA wheels (timed out repeatedly at ~30-50 kB/s).
Switched to `onnxruntime-directml` instead (25MB wheel, uses Windows' built-in DX12 stack, no
separate CUDA install needed) — `batch_extract.py` now takes `--device {cuda,directml,cpu}`.
Model training itself (below) still uses native PyTorch CUDA directly, which works fine on
this machine independent of the onnxruntime issue. Full detail in memory (`rtx5060_directml_fix`).

## 2. Stroke-type classifier (ST-GCN)

First concrete trained-model result, per HANDOFF.md priority. **v2 (current)** keeps
forehand_volley and backhand_volley as separate classes (6-way: `backhand`, `forehand`,
`backhand_volley`, `forehand_volley`, `serve`, `smash`; `serve` still merges the three service
types, which are genuinely one stroke with different spins) — v1 had merged both volleys into
one label, which was the weakest class (55% recall) precisely because it was blending two
distinct motions.

**Model**: compact ST-GCN (6 spatio-temporal blocks, COCO-17 skeleton graph, ~32→128 channels),
4 input channels (2D position + velocity per joint), trained from scratch. v2 adds label-preserving
augmentation (left-right mirror, random temporal crop, small coordinate jitter) — worthwhile with
only 55 subjects total. Code: `stgcn_model.py`, `stroke_dataset.py`, `train_stroke_classifier.py`.

**Split**: subject-disjoint (11 of 55 subjects held out entirely for validation, stratified to
keep a beginner/expert mix on both sides) — clip-level splitting would have leaked, since clips
from the same subject are highly correlated.

**Result: 82.1% held-out validation accuracy** (6-way; majority-class baseline is 25%), 100 epochs,
up from 77.5% (v1, 5-way, no augmentation/velocity). Metrics: `results/stroke_classifier_v2/metrics.json`.

| class            | precision | recall | f1   | support |
|-------------------|-----------|--------|------|---------|
| backhand          | 0.78      | 0.92   | 0.85 | 99      |
| forehand          | 0.82      | 0.81   | 0.82 | 99      |
| backhand_volley   | 0.73      | 0.58   | 0.64 | 33      |
| forehand_volley   | 0.70      | 0.58   | 0.63 | 33      |
| serve             | 0.93      | 0.91   | 0.92 | 99      |
| smash             | 0.79      | 0.79   | 0.79 | 33      |

Serve is strongest (clearly distinct: toss + overhead strike). The two volley classes are still
weakest, but the *reason* is now visible in the confusion matrix rather than being a merging
artifact: forehand_volley is confused with forehand (12 of 33 misses) and backhand_volley with
backhand (9 of 33 misses) — i.e. volleys share arm-swing kinematics with their groundstroke
counterpart, and the near-frontal 2D camera doesn't capture the footwork/court-position
difference that would disambiguate them. That's a real, explainable limitation to state in the
proposal, not a modeling bug — own side-view recordings (already planned, see §4) would likely
fix this by making the forward-court-position cue visible.

*(v1 result — 5-way, no augmentation, kept for comparison: `results/stroke_classifier/metrics.json`.)*

## 3. Beginner-vs-expert check

Per HANDOFF.md priority: before investing in coach recruitment (Path B), check whether THETIS's
skill-level metadata (p1-p31 beginner, p32-p55 expert — a free but weak quality-label proxy)
shows a detectable, *usable* signal in the extracted motion data.

**v1 (hand-crafted features)** — `beginner_expert_check.py`: 9 hand-crafted motion features
(mean/std swing speed, jerk, elbow/knee joint-angle variance, wrist extension range) fed into
logistic regression. Established that population-level differences are real and strong (every
feature differs significantly, Welch's t-test, p<0.02, most p<1e-10, n=1,980) — and notably
**experts show higher speed/jerk/joint-angle-variance than beginners, not lower** (expert swings
read as more dynamic/forceful, not smoother, which should inform how Section 4.7's rule-based
correction logic weighs "smoothness"). But per-subject generalization was weak: only 58.6% held-out
accuracy vs. 54.6% baseline — real signal, not yet a usable classifier.

**v2 (ST-GCN, current)** — `train_beginner_expert_stgcn.py`: same architecture/augmentation as
the stroke classifier above, trained directly on the beginner/expert label instead of hand-crafted
aggregate stats, so it can use the actual temporal trajectory. **Result: 76.0% held-out accuracy**
(vs. 54.6% baseline), a 21-point jump over baseline and a large improvement over v1's 58.6% —
confirming the signal v1 found was real, just not extractable with 9 summary statistics and a
linear model. Confusion matrix (rows=true, cols=pred, `[beginner, expert]`):

|              | pred beginner | pred expert |
|--------------|--------------:|------------:|
| **beginner** | 146           | 70          |
| **expert**   | 25            | 155         |

Recall: beginner 67.6%, expert 86.1%. Precision: beginner 85.4%, expert 68.9%. The model is more
prone to calling a beginner clip "expert" than the reverse — consistent with skill level being a
spectrum, not two clean clusters, and THETIS's label being per-subject, not per-swing (a beginner
subject's better attempts likely look expert-ish, and vice versa).

**Implication for the project**: this is now a genuinely useful early quality-proxy classifier,
not just a sanity check — 76% on a held-out, subject-disjoint split is a meaningful result to
cite in the proposal as evidence the core premise (motion quality is learnable from skeleton
data) works. It's still not a substitute for Path A/B ground truth (subject-level skill label ≠
per-swing quality score), so both tracks should still proceed, but this de-risks them
considerably. Full numbers: `results/beginner_expert_stgcn/metrics.json`
(v1 comparison: `results/beginner_expert_check.json`).

## 4. What's still open (unchanged from HANDOFF.md, not attempted tonight)

1. Team/university/advisor/contact placeholders in the proposal — still blocking submission
   eligibility, independent of all technical work.
2. Path B coach recruitment — longest lead time item, hasn't been started.
3. Own side-view/60fps recordings — still required for the real quality-score model; THETIS's
   near-frontal 17-19fps camera doesn't support the joint-angle-based error detection rules.
4. The actual quality-score regression model (Section 4.7) — today's work (classifier + sanity
   check) sets up for this but doesn't build it yet.
