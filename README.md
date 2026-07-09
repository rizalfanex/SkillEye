# SkillEye: Skeleton-Based Motion Quality Analysis for Racket Sports

Submission for the 8th CTCI Science and Technology Creativity Competition 2026 (Sports theme).

## Abstract

Existing racket-sport analytics apps (e.g. SwingVision) center on ball tracking and match
outcome — where the ball landed, who won the point. SkillEye instead analyzes the player's
*movement*, extracting a normalized 2D skeleton from swing video with RTMPose and modeling it
with a Spatio-Temporal Graph Convolutional Network (ST-GCN), toward the goal of a per-swing
motion-quality score with joint/phase-level error detection and rule-based correction feedback.
This addresses both the "training methods" and "injury prevention" sub-themes of the
competition through one shared representation.

This document reports the technical work completed to date on the [THETIS](https://github.com/THETIS-dataset/dataset)
tennis dataset: a validated pose-extraction pipeline (1,980/1,980 clips, 0 failures), a 6-class
stroke-type classifier (**81.7% ± 4.9%**, 5-fold subject-disjoint cross-validation), and a
beginner-vs-expert motion-quality proxy classifier (**82.4% ± 3.8%**). Both are real trained
results on held-out subjects, not projected targets. Section 4 discusses what these results do
and do not establish, and Section 5 lays out the remaining work toward the full quality-score
system.

## 1. Introduction

### 1.1 Motivation

Amateur and intermediate racket-sport players typically improve technique through in-person
coaching, which is expensive, geographically limited, and infrequent relative to how often
players practice alone. Video-based swing analysis could close this gap, but existing consumer
tools (SwingVision and similar) are built around ball-tracking and scorekeeping — they answer
"did the shot land in" rather than "what did the player's body do." Movement quality is the
input a coach actually reasons about (elbow position at contact, hip rotation timing, follow-through),
and it is not derivable from ball trajectory alone.

### 1.2 Related Work

The technical approach follows two established lines of work rather than proposing new
architectures: **RTMPose** for real-time 2D human pose estimation, and **ST-GCN**
(Yan, Xiong & Lin, 2018) for skeleton-based action recognition, which models a keypoint sequence
as a spatio-temporal graph — spatial edges follow the human skeleton, temporal edges connect the
same joint across frames. The novelty in SkillEye is the application: using this representation
not just to classify *what* stroke was performed (a mature task), but as the foundation for
scoring *how well* it was performed, which is comparatively unexplored for racket sports at
consumer scale. The THETIS dataset itself (Gourgari et al., CVPR 2013 workshop) was built for
action recognition, not quality assessment — Section 2.1 and 4.1 discuss what that means for
how it can and cannot be used here.

### 1.3 Contributions

1. A validated, resumable pose-extraction pipeline from raw racket-sport video to normalized,
   tracked, single-subject skeleton sequences (Section 2.2).
2. A stroke-type classifier trained from scratch on the extracted skeletons, evaluated with
   5-fold subject-disjoint cross-validation rather than a single split (Section 3.1).
3. Evidence, from a dedicated experiment, that motion *quality* (not just stroke type) is
   learnable from this same skeleton representation — using THETIS's beginner/expert metadata
   as a weak proxy label ahead of real coach-rating ground truth (Section 3.2).
4. An honest accounting of what this dataset cannot yet support (Section 4.1) and a concrete
   plan for closing that gap (Section 5).

## 2. Methodology

### 2.1 Dataset

[THETIS](https://github.com/THETIS-dataset/dataset) (Gourgari et al., 2013) provides 1,980 RGB
video clips across 12 tennis action categories, performed by 55 subjects (**p1–p31 labeled
beginner, p32–p55 labeled expert** per the dataset's own metadata), recorded at ~17–19 fps,
640×480, with a Kinect camera facing the subject (near-frontal, not side-on). Only the
`VIDEO_RGB` subset is used — `VIDEO_Skelet2D/3D` uses Kinect's joint definition (not COCO-17,
the format the deployed pipeline actually produces) and covers only 1,217 of 8,374 sequences,
so it would not be representative even if remapped.

Class composition after the merge described in 2.3 (Table 1):

| stroke | THETIS sub-categories merged | clips |
|---|---|---:|
| backhand | backhand, backhand2hands, backhand_slice | 495 |
| forehand | forehand_flat, forehand_openstands, forehand_slice | 495 |
| backhand_volley | backhand_volley | 165 |
| forehand_volley | forehand_volley | 165 |
| serve | flat_service, kick_service, slice_service | 495 |
| smash | smash | 165 |
| **total** | | **1,980** |

By skill label: 1,116 clips from 31 beginner subjects, 864 clips from 24 expert subjects.

**Measured limitations relevant to this project** (checked directly on sample frames, not
assumed from the dataset paper): the ~17-19fps frame rate is below what the proposal's
biomechanical rules assume is needed to resolve fast strokes (e.g. serves), and the
near-frontal camera angle does not capture the side-on view that joint-angle-based
error-detection rules require. THETIS is therefore treated here as suitable for pretraining the
motion representation and for stroke/skill classification (both established, well-precedented
tasks — this is what Sections 2.3–2.4 and 3 use it for), but **not** as a substitute for
purpose-collected side-view footage when the actual per-swing quality-score model is built
(Section 5).

### 2.2 Pose Extraction Pipeline

Each clip is processed frame-by-frame with **RTMPose** (`rtmpose-s_simcc-body7`, lightweight
mode, COCO-17 keypoints) for pose estimation and **YOLOX-tiny** for person detection, both run
via `rtmlib`/ONNX Runtime. Frames may contain more than one person (other players/staff visible
in the gym), so a dedicated tracking step selects a single, temporally consistent subject:

1. **Primary-subject selection** — on the first frame, the person with the largest visible
   bounding box (among detections with ≥3 confidently-visible joints) is chosen as the subject.
   On subsequent frames, the detection with the nearest hip-center to the previous frame's
   selection is kept, which prevents identity swaps between the subject and any other person
   entering the frame (largest-bbox alone is insufficient once multiple people are present).
2. **Confidence-based interpolation** — per joint, frames below a 0.3 confidence threshold are
   linearly interpolated from the nearest confident frames on either side (no extrapolation at
   clip boundaries). Clips where the primary subject is confidently visible in under 50% of
   frames are dropped rather than interpolated across too large a gap.
3. **Normalization** — each frame is re-centered on the hip midpoint and scaled by
   shoulder-to-hip (torso) length, so the model learns the *shape* of the motion rather than the
   subject's distance from or position relative to the camera.

Result: **1,980/1,980 clips extracted successfully, 0 dropped, 0 failed**, ~1.4s/clip. Code:
`skilleye/skeleton_pipeline.py` (steps above), `skilleye/batch_extract.py` (CLI batch runner,
resumable — skips clips that already have output on disk, so an interrupted run can restart
without redoing completed work).

### 2.3 Stroke Classification Model

THETIS's 12 action categories are merged into 6 target classes (Table 1) reflecting the strokes
the proposal is actually built around — `serve` merges the three service sub-types (genuinely
one stroke with different spin), but `forehand_volley` and `backhand_volley` are kept **separate**
rather than merged into a single "volley" class, since they are visually and kinematically
distinct motions (an earlier iteration that merged them made that class disproportionately hard
to classify — see Section 4.1).

**Architecture**: a compact ST-GCN — 6 spatio-temporal blocks operating on the COCO-17 skeleton
graph (symmetric-normalized adjacency, self-loops included), channel widths 32→64→128→256,
global average pooling, linear classification head. Input is a 4-channel tensor per joint per
frame: 2D position plus its temporal finite-difference (velocity), giving the model direct
access to motion speed/direction rather than only inferring it across pooled layers. Each clip
is time-resampled to a fixed 64 frames (linear interpolation) so variable clip lengths (THETIS
clips range from roughly 2 to 8+ seconds) can be batched. Code: `skilleye/stgcn_model.py`.

**Training**: Adam (lr 1e-3, weight decay 1e-4), cosine-annealed learning rate, class-weighted
cross-entropy (classes are imbalanced — see Table 1), 80–100 epochs, batch size 32. With only 55
subjects total, label-preserving data augmentation is applied at train time: left-right mirroring
(negate the x-coordinate and swap left/right joint indices — a mirrored forehand is still a
forehand, just as a left-handed player's would look), random temporal cropping (retain a random
85–100%-length contiguous sub-window before resampling), and small Gaussian coordinate jitter
(σ=0.02, in normalized units). Code: `skilleye/train_stroke_classifier.py`.

### 2.4 Beginner-vs-Expert Motion-Quality Proxy Model

To test whether motion *quality* — not just stroke identity — is learnable from this
representation, a second model is trained on THETIS's beginner/expert subject-level label
(Section 2.1) as a weak proxy for swing quality, ahead of collecting real coach ratings
(Section 5). Two approaches were tried, reported both for transparency:

- **v1 — hand-crafted features + logistic regression** (`skilleye/beginner_expert_check.py`):
  9 interpretable per-clip features (mean/std swing speed, jerk, elbow/knee joint-angle
  variance, wrist extension range) fed into a linear classifier.
- **v2 — ST-GCN** (`skilleye/train_beginner_expert_stgcn.py`): identical architecture and
  augmentation to Section 2.3, but with a binary beginner/expert output head, trained directly
  on the raw skeleton sequence rather than hand-summarized statistics.

### 2.5 Evaluation Protocol

All splits are **subject-disjoint**: no subject's clips appear in both the training and
validation sets. THETIS clips from the same subject are highly correlated (same body, same
camera, same session), so a clip-level random split would leak subject identity into validation
and overstate accuracy — the model would partly be recognizing *people*, not strokes or skill.

Headline numbers use **5-fold subject-disjoint cross-validation** (`skilleye/cross_validate.py`):
subjects are split into 5 folds, stratified so each fold's held-out group keeps a
beginner/expert mix; each fold trains a fresh model on the other 4 folds' subjects (~44) and
evaluates on the held-out fold's subjects (~11). This is reported as **mean ± standard
deviation across folds**, rather than a single split's number, because with only 55 subjects a
single split's held-out accuracy depends materially on *which* subjects happened to be held
out — the fold-to-fold spread reported in Section 3 is itself informative about that variance.

## 3. Results

### 3.1 Stroke Classification

**81.7% ± 4.9%** held-out accuracy (6-way; majority-class baseline 16.7%), fold range
72.5%–86.1%.

![Accuracy across iterations](results/figures/accuracy_comparison.png)

![Stroke classifier confusion matrix](results/figures/stroke_confusion_matrix.png)

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| backhand | 0.79 | 0.90 | 0.84 | 495 |
| forehand | 0.87 | 0.79 | 0.83 | 495 |
| backhand_volley | 0.66 | 0.63 | 0.64 | 165 |
| forehand_volley | 0.68 | 0.67 | 0.68 | 165 |
| serve | 0.93 | 0.88 | 0.90 | 495 |
| smash | 0.73 | 0.78 | 0.75 | 165 |

`serve` is the strongest class — mechanically the most distinct motion (toss + overhead
strike). The volley classes are weakest, and the confusion matrix shows *why*: forehand_volley
is confused primarily with forehand, backhand_volley primarily with backhand — i.e. a volley
shares arm-swing kinematics with its groundstroke counterpart, and the near-frontal 2D camera
does not capture the footwork/court-position difference (volleys are played closer to the net)
that would otherwise disambiguate them. This is discussed further in Section 4.1.

![Training curves](results/figures/training_curves.png)

### 3.2 Beginner vs. Expert

**82.4% ± 3.8%** held-out accuracy (2-way; majority-class baseline 54.6%), fold range
78.5%–89.7%. The v1→v2 progression:

| iteration | method | held-out accuracy |
|---|---|---|
| v1 | 9 hand-crafted features + logistic regression, single split | 58.6% (baseline 54.6%) |
| v2 | ST-GCN on raw skeleton sequence, single split | 76.0% |
| **v3** | **ST-GCN, 5-fold cross-validation** | **82.4% ± 3.8%** |

![Beginner vs expert confusion matrix](results/figures/skill_confusion_matrix.png)

v1 established that *population-level* differences between beginner and expert clips are real
and highly significant (Welch's t-test on all 9 features, p<0.02, most p<1e-10, n=1,980) —
notably, **experts showed higher swing speed, jerk, and joint-angle variance than beginners, not
lower**, i.e. expert swings read as more dynamic/forceful rather than smoother in this dataset,
which should inform how a smoothness-based quality heuristic is weighted. But v1's linear model
on those 9 summary statistics only reached 58.6% held-out accuracy — barely above baseline,
meaning the signal existed in aggregate but wasn't usable per-subject with simple features.

v2 replaced the hand-crafted features with the same ST-GCN architecture used for stroke
classification, trained directly on the raw skeleton sequence, and reached 76.0% on a single
split. v3 cross-validated that result and found it was, if anything, an *unfavorable* split —
the cross-validated mean (82.4%) is higher than the single-split number, with folds ranging
78.5%–89.7% (Table above; per-class breakdown: beginner precision 0.85/recall 0.83, expert
precision 0.79/recall 0.82, summed over all 5 folds' held-out predictions).

## 4. Discussion

### 4.1 Limitations

- **Camera geometry limits stroke disambiguation.** The volley/groundstroke confusion in
  Section 3.1 is a direct consequence of THETIS's near-frontal camera — footwork and
  court-position cues that would resolve it are not visible from that angle. Side-view footage
  (planned, Section 5) should address this directly, not just improve accuracy but make the
  representation richer for the eventual per-phase error detection this project needs.
- **The beginner/expert label is a subject-level proxy, not a per-swing quality score.** THETIS
  labels an entire subject as beginner or expert; it says nothing about whether a specific swing
  within that subject's clips was a particularly good or bad execution. The precision/recall
  asymmetry noted in Section 3.2 (the model calls some beginner clips "expert" and vice versa)
  is consistent with this — a beginner's better attempts plausibly look expert-ish in isolation,
  and the reverse. Section 3.2's result demonstrates that quality-relevant signal exists and is
  learnable, which de-risks the project's core premise; it does not by itself produce the
  continuous, per-swing quality score the finished system needs.
- **Frame rate.** ~17–19fps may under-resolve the fastest phases of a serve or smash relative to
  the 60fps the original proposal assumed necessary; this has not yet been quantified directly
  (e.g. by comparing model behavior on down-sampled higher-fps footage) and is flagged as an
  open question rather than a settled limitation.
- **Sample size for cross-validation.** 55 subjects is enough to show fold-to-fold variance
  (Section 3, 4.9 and 3.8 percentage-point standard deviations) but not enough to make that
  variance estimate itself highly precise — it indicates real single-split fragility rather than
  pinning down an exact confidence interval.

### 4.2 Implications for the Quality-Score System

Taken together, Sections 3.1 and 3.2 support two concrete design decisions for the system
described in the proposal's Section 4.7: (1) stroke-type classification is solved to a
practical standard on this representation and can run as a preprocessing step before
stroke-specific quality rules are applied, and (2) motion quality is measurable from the same
representation in principle, but the production quality-score model will need either
per-swing ground truth (real coach ratings, or synthetic perturbation of known-good motion —
both already planned as Path A/B) or a training signal richer than a subject-level label,
because Section 3.2's result is a proof of learnability, not a finished quality metric.

## 5. Conclusion and Future Work

This work establishes, with real trained and cross-validated results rather than projected
targets, that (1) a resumable, validated pipeline exists from raw racket-sport video to a
normalized skeleton representation, (2) stroke type is classifiable from that representation to
a practical standard (81.7% ± 4.9%, 6-way), and (3) motion quality signal — not just stroke
identity — is present and learnable in the same representation (82.4% ± 3.8% on a weak proxy
label), which supports the project's central premise ahead of collecting stronger ground truth.

Remaining work, in priority order:

1. **Team/university/advisor/contact fields** in the competition proposal — still placeholders;
   blocks submission eligibility independent of all technical progress above.
2. **Path B: coach-rated ground truth** — recruit coaches to blind-rate real swings, to be
   correlated against model output (target r > 0.7). Longest lead time item; should run in
   parallel with, not after, further technical work.
3. **Own side-view, higher-frame-rate recordings** — required for the joint-angle-based
   error-detection rules in Section 4.7, and expected to resolve the volley/groundstroke
   confusion identified in Section 4.1.
4. **Path A: synthetic perturbation** of known-good motion (known joint-angle/timing offsets
   injected into skilled-athlete clips) as a complementary, coach-independent ground-truth
   source for the quality-score model.
5. **The quality-score model itself** (proposal Section 4.7) — regression rather than
   classification, built on the foundation established here once ground truth from (2) and/or
   (4) is available.

## 6. Reproducibility

```
skilleye/                          pipeline and modeling code
  skeleton_pipeline.py             RTMPose output -> tracked subject -> normalized skeleton (§2.2)
  batch_extract.py                 CLI batch runner over a THETIS-shaped folder tree (resumable)
  stroke_dataset.py                category merging (Table 1), subject-disjoint/k-fold splitting
  stgcn_model.py                   ST-GCN architecture, COCO-17 skeleton graph (§2.3)
  train_stroke_classifier.py       stroke classifier, single split
  train_beginner_expert_stgcn.py   beginner/expert ST-GCN, single split (§2.4)
  beginner_expert_check.py         beginner/expert hand-crafted-feature baseline, v1 (§2.4)
  cross_validate.py                5-fold cross-validation for both classifiers (§2.5)
  generate_figures.py              renders results/figures/*.png from the metrics JSONs
  requirements.txt
  README.md                        environment setup notes (incl. GPU-specific gotchas)

results/
  RESULTS_SUMMARY.md                supplementary detail beyond what's inlined above
  figures/                          PNGs embedded above; regenerate with generate_figures.py
  cross_validation/                 §3 headline numbers: per-fold + aggregated metrics
  beginner_expert_check.json        v1 (§2.4): hand-crafted features + logistic regression
  beginner_expert_stgcn/            v2 (§2.4): ST-GCN, single split, trained weights + metrics
  stroke_classifier/                early 5-class iteration, kept for reference
  stroke_classifier_v2/             6-class + augmentation, single split, trained weights + metrics
```

Dataset (not included in this repository — see `skilleye/README.md` for how to fetch it):
[THETIS](https://github.com/THETIS-dataset/dataset), `VIDEO_RGB` + `papers` subsets only.

## References

- Gourgari, S., Goudelis, G., Karpouzis, K., & Kollias, S. (2013). THETIS: Three Dimensional
  Tennis Shots a Human Action Dataset. *CVPR Workshops*.
- Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for
  Skeleton-Based Action Recognition. *AAAI*.
- Jiang, T., et al. (2023). RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose.
  *arXiv:2303.07399*.
