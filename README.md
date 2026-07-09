# SkillEye

Motion-quality analysis for racket-sport swings (forehand / backhand / volley / serve / smash).
Built for the 8th CTCI Science and Technology Creativity Competition 2026 (Sports theme).

**Pipeline**: video → RTMPose (2D skeleton extraction) → ST-GCN (spatio-temporal modeling) →
stroke classification, with the same skeleton representation intended to support per-joint/phase
motion-quality scoring and rule-based correction suggestions. Positions against ball-tracking
apps (e.g. SwingVision) by focusing on movement quality rather than match outcome.

## Results (THETIS dataset, full run)

- **Skeleton extraction**: 1,980/1,980 clips processed, 0 dropped, 0 failed, across all 12
  THETIS action categories and 55 subjects.
- **Stroke-type classifier (ST-GCN)**: **82.1% held-out accuracy** on a subject-disjoint split
  (6 classes: backhand/forehand/backhand_volley/forehand_volley/serve/smash), vs. a 25%
  majority-class baseline.
- **Beginner-vs-expert classifier (ST-GCN)**: **76.0% held-out accuracy** distinguishing
  THETIS's beginner/expert-labeled subjects from raw skeleton motion alone, vs. a 54.6%
  majority-class baseline — a real, usable quality-proxy signal, not just a population-level
  statistical trend (an earlier hand-crafted-feature version only reached 58.6%).

Full write-up with per-class metrics, confusion matrices, and interpretation (including the
earlier iteration kept for comparison): [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

## Repo layout

```
skilleye/           pipeline code
  skeleton_pipeline.py       RTMPose output -> single tracked subject -> normalized skeleton
  batch_extract.py           CLI batch runner over a THETIS-shaped folder tree (resumable)
  stroke_dataset.py          THETIS category merging, subject-disjoint splitting
  stgcn_model.py             compact ST-GCN (COCO-17 skeleton graph)
  train_stroke_classifier.py stroke classifier training + evaluation (v2: 6 classes, augmented)
  train_beginner_expert_stgcn.py  beginner/expert ST-GCN training + evaluation (v2)
  beginner_expert_check.py   beginner/expert hand-crafted-feature sanity check (v1)
  requirements.txt
  README.md                  environment setup notes (incl. GPU-specific gotchas)

results/
  RESULTS_SUMMARY.md         full technical write-up (v1 and v2)
  beginner_expert_check.json       v1: hand-crafted features + logistic regression
  beginner_expert_stgcn/            v2: ST-GCN, trained weights + metrics
  stroke_classifier/                v1: 5-class, trained weights + metrics
  stroke_classifier_v2/             v2: 6-class + augmentation, trained weights + metrics
```

## Dataset

Uses [THETIS](https://github.com/THETIS-dataset/dataset) (`VIDEO_RGB` + `papers` only — not
included in this repo, see `skilleye/README.md` for how to fetch it). THETIS's near-frontal,
17-19fps camera is sufficient for stroke classification and pretraining but not for the
biomechanical joint-angle rules the final quality-score model needs — own side-view/60fps
recordings are planned for that stage.
