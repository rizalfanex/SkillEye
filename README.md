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
- **Stroke-type classifier (ST-GCN)**: **77.5% held-out accuracy** on a subject-disjoint split
  (5 merged classes: backhand/forehand/volley/serve/smash), vs. a 25% majority-class baseline.
- **Beginner-vs-expert sanity check**: motion features (speed, jerk, joint-angle variance, reach)
  differ significantly between THETIS's beginner/expert-labeled subjects at the population level
  (p<0.02 on all 9 features tested, most p<1e-10, n=1,980) — but don't yet generalize to unseen
  subjects with simple features (58.6% vs. 54.6% baseline), motivating the project's two-track
  ground-truth strategy (synthetic perturbation + real coach ratings).

Full write-up with per-class metrics, confusion matrix, and interpretation:
[`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

## Repo layout

```
skilleye/           pipeline code
  skeleton_pipeline.py       RTMPose output -> single tracked subject -> normalized skeleton
  batch_extract.py           CLI batch runner over a THETIS-shaped folder tree (resumable)
  stroke_dataset.py          THETIS category merging, subject-disjoint splitting
  stgcn_model.py             compact ST-GCN (COCO-17 skeleton graph)
  train_stroke_classifier.py training + evaluation
  beginner_expert_check.py   beginner/expert motion-feature sanity check
  requirements.txt
  README.md                  environment setup notes (incl. GPU-specific gotchas)

results/
  RESULTS_SUMMARY.md         full technical write-up
  beginner_expert_check.json
  stroke_classifier/         trained weights + metrics
```

## Dataset

Uses [THETIS](https://github.com/THETIS-dataset/dataset) (`VIDEO_RGB` + `papers` only — not
included in this repo, see `skilleye/README.md` for how to fetch it). THETIS's near-frontal,
17-19fps camera is sufficient for stroke classification and pretraining but not for the
biomechanical joint-angle rules the final quality-score model needs — own side-view/60fps
recordings are planned for that stage.
