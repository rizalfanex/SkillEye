"""
Qualitative (not just quantitative) figures for the README: actual example
skeleton poses and an actual quality-scoring comparison on real clips, so the
system's behavior is visible, not just its aggregate accuracy/score numbers.

Usage:
    python generate_qualitative_figures.py --out E:/SkillEye/results/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stroke_dataset import load_records, STROKE_CLASSES
from stgcn_model import COCO17_EDGES
from quality.phases import detect_contact_frame
from quality.score import score_clip

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE_450 = "#2a78d6"
BLUE_600 = "#184f95"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
})


def draw_skeleton(ax, frame, color=BLUE_450, marker_color=BLUE_600):
    for a, b in COCO17_EDGES:
        ax.plot([frame[a, 0], frame[b, 0]], [-frame[a, 1], -frame[b, 1]], color=color, linewidth=2)
    ax.scatter(frame[:, 0], -frame[:, 1], color=marker_color, s=18, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_stroke_gallery(records, out_path):
    """One representative clip's contact-frame pose per stroke class."""
    by_stroke = {}
    for r in records:
        if r["stroke"] not in by_stroke:
            by_stroke[r["stroke"]] = r

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, stroke in zip(axes.flat, STROKE_CLASSES):
        r = by_stroke[stroke]
        contact = detect_contact_frame(r["kpts"])
        draw_skeleton(ax, r["kpts"][contact])
        ax.set_title(stroke.replace("_", " "), fontsize=12)

    fig.suptitle("Example poses at the detected contact frame, one per stroke class",
                 fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_quality_comparison(records, out_path, stroke, low_subject, high_subject):
    """Side-by-side skeleton + score + top suggestions for a low- vs
    high-scoring clip of the same stroke (a real qualitative example of what
    the scoring system actually says, not just an aggregate number)."""
    with open("E:/SkillEye/results/quality_templates/templates.json") as f:
        templates = json.load(f)["templates"]

    low_r = next(r for r in records if r["stroke"] == stroke and r["subject_id"] == low_subject)
    high_r = next(r for r in records if r["stroke"] == stroke and r["subject_id"] == high_subject)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5))
    for ax, r, color in [(axes[0], low_r, CRITICAL), (axes[1], high_r, GOOD)]:
        result = score_clip(r["kpts"], stroke, templates)
        contact = detect_contact_frame(r["kpts"])
        draw_skeleton(ax, r["kpts"][contact])

        lines = [f"{r['skill_level']} (subject {r['subject_id']})",
                 f"score: {result['overall_score']:.0f}/100"]
        lines += [s for s in result["suggestions"][:3]]
        if not result["suggestions"]:
            lines.append("no significant deviations flagged")

        wrapped = []
        for line in lines:
            while len(line) > 46:
                cut = line.rfind(" ", 0, 46)
                wrapped.append(line[:cut])
                line = line[cut + 1:]
            wrapped.append(line)

        ax.set_title("\n".join(wrapped[:2]), fontsize=11, color=color, pad=10)
        ax.text(0.5, -0.05, "\n".join(wrapped[2:]), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color=TEXT_SECONDARY, wrap=True)

    fig.suptitle(f"Quality-scoring example: two real held-out '{stroke.replace('_', ' ')}' clips",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default="E:/SkillEye/skeletons")
    ap.add_argument("--out", default="E:/SkillEye/results/figures")
    args = ap.parse_args()

    with open("E:/SkillEye/results/quality_templates/templates.json") as f:
        val_subjects = set(json.load(f)["val_subjects"])

    records, _ = load_records(args.skeletons)
    val_records = [r for r in records if r["subject_id"] in val_subjects]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_stroke_gallery(val_records, out / "stroke_gallery.png")
    print("wrote", out / "stroke_gallery.png")

    fig_quality_comparison(val_records, out / "quality_comparison_example.png",
                           stroke="forehand_volley", low_subject=18, high_subject=39)
    print("wrote", out / "quality_comparison_example.png")


if __name__ == "__main__":
    main()
