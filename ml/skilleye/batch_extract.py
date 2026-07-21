"""
Batch-run RTMPose over THETIS VIDEO_RGB clips and write cleaned, normalized
skeleton JSON (one file per clip) into an output directory that mirrors the
input category structure.

Usage:
    python batch_extract.py --src "G:/competition/THETIS/VIDEO_RGB" \
                             --dst "G:/competition/skeletons" \
                             --limit-per-category 3
"""

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import torch  # noqa: F401  (import before onnxruntime so its CUDA/cuDNN DLLs are on the search path)
import cv2
import numpy as np
from rtmlib import Body
from rtmlib.tools import base as rtmlib_base

# rtmlib only knows cpu/cuda/rocm/mps out of the box; DirectML (Windows' own
# DX12 GPU backend, no separate CUDA/cuDNN install needed) isn't one of them.
rtmlib_base.RTMLIB_SETTINGS["onnxruntime"]["directml"] = "DmlExecutionProvider"

from skeleton_pipeline import clean_clip

FILENAME_RE = re.compile(r"^p(\d+)_([a-zA-Z0-9]+)_s(\d+)$")
BEGINNER_MAX_ID = 31  # p1-p31 beginners, p32-p55 experts (per THETIS README)


def parse_filename(stem):
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    subject_id, action_code, seq = int(m.group(1)), m.group(2), int(m.group(3))
    return {
        "subject_id": subject_id,
        "action_code": action_code,
        "sequence": seq,
        "skill_level": "beginner" if subject_id <= BEGINNER_MAX_ID else "expert",
    }


def run_pose_on_video(body, video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame_kpts, per_frame_scores = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints, scores = body(frame)
        per_frame_kpts.append(np.asarray(keypoints))
        per_frame_scores.append(np.asarray(scores))
    cap.release()
    return per_frame_kpts, per_frame_scores, fps, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="THETIS VIDEO_RGB root")
    ap.add_argument("--dst", required=True, help="output root for skeleton JSON")
    ap.add_argument("--limit-per-category", type=int, default=None,
                     help="cap number of clips processed per action category (for a quick validation pass)")
    ap.add_argument("--mode", default="lightweight", choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "directml", "cpu"],
                     help="onnxruntime execution provider: cuda (NVIDIA CUDA/cuDNN), "
                          "directml (Windows DX12, no separate CUDA install needed), or cpu")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    body = Body(mode=args.mode, to_openpose=False, backend="onnxruntime", device=args.device)

    categories = sorted([p for p in src_root.iterdir() if p.is_dir()])
    print(f"Found {len(categories)} categories under {src_root}")

    summary = {"ok": 0, "dropped_low_confidence": 0, "failed": 0, "skipped_bad_name": 0}
    t_start = time.time()

    for cat_dir in categories:
        out_cat_dir = dst_root / cat_dir.name
        out_cat_dir.mkdir(parents=True, exist_ok=True)

        clips = sorted(cat_dir.glob("*.avi"))
        if args.limit_per_category is not None:
            clips = clips[: args.limit_per_category]

        for clip_path in clips:
            meta = parse_filename(clip_path.stem)
            if meta is None:
                print(f"  [skip] unrecognized filename pattern: {clip_path.name}")
                summary["skipped_bad_name"] += 1
                continue

            out_path = out_cat_dir / f"{clip_path.stem}.json"
            if out_path.exists():
                continue  # resumable: don't redo work already on disk

            try:
                per_frame_kpts, per_frame_scores, fps, w, h = run_pose_on_video(body, clip_path)
                kpts, scores, ok = clean_clip(per_frame_kpts, per_frame_scores)

                if not ok:
                    print(f"  [drop] {clip_path.name}: primary subject visible in too few frames")
                    summary["dropped_low_confidence"] += 1
                    continue

                record = {
                    "source_video": str(clip_path),
                    "category": cat_dir.name,
                    **meta,
                    "fps": fps,
                    "width": w,
                    "height": h,
                    "num_frames": int(kpts.shape[0]),
                    "keypoints_normalized": kpts.tolist(),
                    "keypoint_scores": scores.tolist(),
                }
                with open(out_path, "w") as f:
                    json.dump(record, f)
                summary["ok"] += 1

            except Exception as e:
                print(f"  [FAIL] {clip_path.name}: {e}")
                traceback.print_exc()
                summary["failed"] += 1

        print(f"[{cat_dir.name}] done ({len(clips)} clips attempted)")

    elapsed = time.time() - t_start
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
