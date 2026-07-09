# SkillEye — technical results (RTX 5060 machine run, 2026-07-09)

Generated overnight while the pipeline ran unattended, for direct use in the CTCI 2026
proposal's Technical Content & Completeness section. Everything below is a real trained
result on the full THETIS dataset, not a projected target.

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

First concrete trained-model result, per HANDOFF.md priority — merges THETIS's 12 sub-classes
into 5 strokes (`backhand`, `forehand`, `volley`, `serve`, `smash`; `volley` merges
forehand_volley + backhand_volley, `serve` merges the three service types).

**Model**: compact ST-GCN (6 spatio-temporal blocks, COCO-17 skeleton graph, ~32→128 channels),
trained from scratch, no pretraining. Code: `stgcn_model.py`, `train_stroke_classifier.py`.

**Split**: subject-disjoint (11 of 55 subjects held out entirely for validation, stratified to
keep a beginner/expert mix on both sides) — clip-level splitting would have leaked, since clips
from the same subject are highly correlated.

**Result: 77.5% held-out validation accuracy** (5-way; majority-class baseline is 25%), 60 epochs.

| class    | precision | recall | f1   | support |
|----------|-----------|--------|------|---------|
| backhand | 0.74      | 0.83   | 0.78 | 99      |
| forehand | 0.74      | 0.79   | 0.76 | 99      |
| volley   | 0.68      | 0.55   | 0.61 | 66      |
| serve    | 0.92      | 0.85   | 0.88 | 99      |
| smash    | 0.75      | 0.82   | 0.78 | 33      |

Serve is the strongest class (clearly distinct motion pattern, toss + overhead strike). Volley
is the weakest — expected, since it merges two visually different motions (forehand volley vs
backhand volley) into one label; a 6-class split keeping them separate would likely score higher
if a cleaner number is needed, at the cost of less data per class. Full confusion matrix and
per-epoch training curve saved at `E:\SkillEye\results\stroke_classifier\metrics.json`.

## 3. Beginner-vs-expert sanity check (cheap pilot)

Per HANDOFF.md priority: before investing in coach recruitment (Path B), check whether THETIS's
skill-level metadata (p1-p31 beginner, p32-p55 expert — a free but weak quality-label proxy)
shows *any* detectable signal in the extracted motion data. Code: `beginner_expert_check.py`.

**Method**: 9 hand-crafted motion features per clip (mean/std swing speed, jerk/smoothness,
elbow and knee joint-angle variance, wrist extension range) — deliberately independent of the
stroke classifier above, to test the raw normalized skeleton data itself.

**Finding 1 — population-level signal is real and strong.** Every single feature differs
significantly between beginner and expert clips (Welch's t-test, all p < 0.02, most p < 1e-10
across n=1,980 clips). Direction is notable and worth flagging in the proposal: **experts show
higher speed, higher jerk, and higher joint-angle variance than beginners**, not lower — in this
dataset, expert swings read as more dynamic/forceful rather than "smoother," which runs counter
to a naive smoothness-only quality heuristic and should inform how Section 4.7's rule-based
correction logic is designed.

**Finding 2 — per-subject generalization is weak with simple features.** A logistic regression
on those 9 features (+ stroke type) reaches 77.9% on training subjects but only **58.6% on
11 held-out subjects** (baseline 54.6%) — barely better than chance. So: real signal exists in
aggregate, but hand-crafted features + a linear model don't yet predict skill level for a *new*
person reliably.

**Implication for the project**: THETIS's skill-level label is a valid weak proxy at the
population level (justifies the overall project premise that motion quality is measurable from
skeleton data), but is not sufficient on its own to build the actual per-swing quality-score
model — supports proceeding with **both** Path A (synthetic perturbation) and Path B (real coach
ratings) as planned, and suggests the eventual quality model will need either richer temporal
features than these 9 hand-crafted ones (e.g., the ST-GCN backbone above, fine-tuned) or the
real coach-rating ground truth to reach practical per-subject accuracy. Full numbers (means,
p-values, coefficients) at `E:\SkillEye\results\beginner_expert_check.json`.

## 4. What's still open (unchanged from HANDOFF.md, not attempted tonight)

1. Team/university/advisor/contact placeholders in the proposal — still blocking submission
   eligibility, independent of all technical work.
2. Path B coach recruitment — longest lead time item, hasn't been started.
3. Own side-view/60fps recordings — still required for the real quality-score model; THETIS's
   near-frontal 17-19fps camera doesn't support the joint-angle-based error detection rules.
4. The actual quality-score regression model (Section 4.7) — today's work (classifier + sanity
   check) sets up for this but doesn't build it yet.
