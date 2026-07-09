# SkillEye — technical results (RTX 5060 machine run, 2026-07-09)

Supplementary detail behind the main [`README.md`](../README.md), which is the primary,
journal-structured write-up (abstract, methodology, results, discussion, references) — read
that first. This file adds the environment/infrastructure notes and the full v1→v2→v3
per-iteration history that didn't fit inline there. Everything below is a real trained result
on the full THETIS dataset, not a projected target. Three iterations, each keeping the previous
one for comparison rather than overwriting it:

- **v1** — first working model for each task.
- **v2** — fixed the biggest weaknesses found in v1 (see each section).
- **v3** — 5-fold subject-disjoint cross-validation of v2, because a single held-out split
  of only 55 subjects can swing several points depending on which subjects land in
  validation. **v3's mean ± std is the number to cite** — it's the one that holds up if
  someone asks "is this just a lucky split?".

![Accuracy across iterations](figures/accuracy_comparison.png)

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

**v2: 82.1% held-out accuracy** on a single subject-disjoint split (6-way; majority-class
baseline is 16.7%), up from v1's 77.5% (5-way, no augmentation/velocity, 25% baseline —
not directly comparable to v2/v3 since the class count differs).

**v3: 81.7% ± 4.9% across 5 folds** (subject-disjoint, stratified by skill level) — confirms
v2's number wasn't a lucky split; the ~5-point spread across folds (72.5%–86.1%) is itself
useful information about how much a single-split number can wobble with only 55 subjects.

![Stroke classifier confusion matrix](figures/stroke_confusion_matrix.png)

Per-class breakdown (v3, summed over all 5 folds' held-out predictions, so every clip is
counted exactly once):

| class            | precision | recall | f1   | support |
|-------------------|-----------|--------|------|---------|
| backhand          | 0.79      | 0.90   | 0.84 | 495     |
| forehand          | 0.87      | 0.79   | 0.83 | 495     |
| backhand_volley   | 0.66      | 0.63   | 0.64 | 165     |
| forehand_volley   | 0.68      | 0.67   | 0.68 | 165     |
| serve             | 0.93      | 0.88   | 0.90 | 495     |
| smash             | 0.73      | 0.78   | 0.75 | 165     |

Serve is strongest (clearly distinct: toss + overhead strike). The two volley classes are still
weakest, but the *reason* is visible in the confusion matrix rather than being a merging
artifact: forehand_volley confuses mainly with forehand and backhand_volley mainly with backhand
— i.e. volleys share arm-swing kinematics with their groundstroke counterpart, and the
near-frontal 2D camera doesn't capture the footwork/court-position difference that would
disambiguate them. That's a real, explainable limitation to state in the proposal, not a
modeling bug — own side-view recordings (already planned, see §4) would likely fix this by
making the forward-court-position cue visible.

![Training curves](figures/training_curves.png)

*(v1 result — 5-way, no augmentation: `results/stroke_classifier/metrics.json`. v2:
`results/stroke_classifier_v2/metrics.json`. v3 full per-fold numbers:
`results/cross_validation/stroke_classifier_cv.json`.)*

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

**v2 (ST-GCN)** — `train_beginner_expert_stgcn.py`: same architecture/augmentation as the stroke
classifier above, trained directly on the beginner/expert label instead of hand-crafted aggregate
stats, so it can use the actual temporal trajectory. Single-split result: 76.0% held-out accuracy
(vs. 54.6% baseline) — already a large jump over v1's 58.6%, confirming the signal v1 found was
real, just not extractable with 9 summary statistics and a linear model.

**v3: 82.4% ± 3.8% across 5 folds** — cross-validation came out *higher* than the single v2 split,
not lower (fold range: 78.5%–89.7%), so this isn't a case of the single split flattering the
model; if anything v2's split was one of the harder ones.

![Beginner/expert confusion matrix](figures/skill_confusion_matrix.png)

Per-class breakdown (v3, summed over all 5 folds): beginner precision 0.85 / recall 0.83
(support 1,116), expert precision 0.79 / recall 0.82 (support 864). Balanced in both directions —
the single-split v2 model had been noticeably better at expert recall (86%) than beginner recall
(68%); cross-validation shows that asymmetry doesn't hold up as a general pattern, it was
specific to which subjects v2 happened to hold out.

**Implication for the project**: this is now a genuinely strong, usable quality-proxy
classifier, not just a sanity check — 82% cross-validated accuracy on subject-disjoint data is
solid evidence the core premise (motion quality is learnable from skeleton data) holds. It's
still not a substitute for Path A/B ground truth (subject-level skill label ≠ per-swing quality
score), so both tracks should still proceed, but this de-risks them considerably. Full numbers:
`results/beginner_expert_stgcn/metrics.json` (v2), `results/cross_validation/beginner_expert_cv.json`
(v3), `results/beginner_expert_check.json` (v1).

## 4. What's still open (unchanged from HANDOFF.md, not attempted tonight)

1. Team/university/advisor/contact placeholders in the proposal — still blocking submission
   eligibility, independent of all technical work.
2. Path B coach recruitment — longest lead time item, hasn't been started.
3. Own side-view/60fps recordings — still required for the real quality-score model; THETIS's
   near-frontal 17-19fps camera doesn't support the joint-angle-based error detection rules.
4. The actual quality-score regression model (Section 4.7) — today's work (classifier + sanity
   check) sets up for this but doesn't build it yet.
