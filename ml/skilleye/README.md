# SkillEye — pose extraction pipeline

Setup on the RTX 5060 machine (Windows + conda assumed).

## 1. Create the environment

```
conda create -n torch python=3.10 -y
conda activate torch
```

## 2. Install PyTorch — IMPORTANT, don't skip this note

RTX 5060 is Blackwell architecture. The `cu121` PyTorch build used on the
original machine does **not** support it — it will either fail to see the
GPU or crash at the first real kernel launch with something like
`CUDA error: no kernel image is available for execution on the device`.

Go to https://pytorch.org/get-started/locally/ and pick the newest CUDA
build offered there for Windows/pip (as of writing, cu124 or cu128 — pick
whichever is current and matches the NVIDIA driver already installed on
that machine). Example:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then verify it actually runs on the GPU, not just that it "sees" it:

```
python -c "import torch; x = torch.randn(1000,1000).cuda(); print((x @ x).sum().item())"
```

If that line errors with "no kernel image available", the wheel still
doesn't match the card — try the next newer CUDA build.

## 3. Install the rest

```
pip install -r requirements.txt
```

Do **not** additionally `pip install onnxruntime` (CPU build) — having both
`onnxruntime` and `onnxruntime-gpu` installed at once makes onnxruntime
silently fall back to CPU. Only `onnxruntime-gpu` should be present.

## 4. Get the dataset

Don't unzip a copy of THETIS from the other machine — just clone the RGB
subset directly, it's faster and avoids a 3.6GB transfer:

```
git clone --filter=blob:none --sparse https://github.com/THETIS-dataset/dataset.git THETIS
cd THETIS
git sparse-checkout set VIDEO_RGB papers
```

## 5. Run

```
cd skilleye
python batch_extract.py --src "<path>/THETIS/VIDEO_RGB" --dst "<path>/skeletons" --mode lightweight
```

It's resumable — if it gets interrupted, rerunning the same command skips
clips that already have output JSON on disk.

## What's in here

- `skeleton_pipeline.py` — primary-subject tracking (multi-person frames ->
  one consistent track), confidence-based interpolation, hip-anchored /
  torso-scaled normalization.
- `batch_extract.py` — CLI batch runner over a THETIS-shaped folder tree.
  Writes one JSON per clip: normalized keypoints, per-joint confidence,
  and metadata (subject id, action, beginner/expert, fps/resolution)
  parsed from the filename.
