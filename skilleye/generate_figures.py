"""
Generate the README/results figures: accuracy-by-iteration bars, confusion
matrix heatmaps, and cross-validation training curves. Colors follow the
project's dataviz palette (single blue hue for sequential/magnitude encoding,
no rainbow colormaps, direct labels instead of a legend where there's only
one series per panel).

Usage:
    python generate_figures.py --out E:/SkillEye/results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from stroke_dataset import STROKE_CLASSES
from train_beginner_expert_stgcn import SKILL_CLASSES

# --- palette (references/palette.md) ---
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE_300, BLUE_450, BLUE_600 = "#6da7ec", "#2a78d6", "#184f95"
CRITICAL = "#d03b3b"
SEQ_BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"]
BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STEPS)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.edgecolor": TEXT_SECONDARY,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e4e0",
    "grid.linewidth": 0.6,
})


def fig_accuracy_comparison(out_path, stroke_accs, stroke_baseline, skill_accs, skill_baseline):
    """stroke_accs / skill_accs: list of (label, value, std_or_None)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, accs, baseline, title in [
        (axes[0], stroke_accs, stroke_baseline, "Stroke classifier (6-class)"),
        (axes[1], skill_accs, skill_baseline, "Beginner vs. expert"),
    ]:
        labels = [a[0] for a in accs]
        values = [a[1] * 100 for a in accs]
        errs = [a[2] * 100 if a[2] else 0 for a in accs]
        x = np.arange(len(labels))

        ax.bar(x, values, yerr=errs, width=0.55, color=BLUE_450,
               edgecolor="none", capsize=4,
               error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2})
        ax.axhline(baseline * 100, color=CRITICAL, linestyle=(0, (4, 3)), linewidth=1.4)
        ax.text(len(labels) - 0.4, baseline * 100 + 1.5, f"baseline {baseline*100:.0f}%",
                color=CRITICAL, fontsize=9, ha="right")

        for xi, v, e in zip(x, values, errs):
            ax.text(xi, v + e + 1.5, f"{v:.1f}%", ha="center", fontsize=10, color=TEXT_PRIMARY)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("held-out accuracy (%)")
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(axis="x")

    fig.suptitle("Accuracy across iterations (subject-disjoint held-out data)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_confusion_matrix(out_path, cm, class_names, title):
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sums, 1, None)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.2, 0.9 * n + 1.8))
    im = ax.imshow(cm_norm, cmap=BLUE_CMAP, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(False)

    for i in range(n):
        for j in range(n):
            frac = cm_norm[i, j]
            label_color = SURFACE if frac > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{cm[i, j]}\n{frac*100:.0f}%", ha="center", va="center",
                    fontsize=9, color=label_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("recall (row-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_training_curves(out_path, stroke_histories, skill_histories):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, histories, title, baseline in [
        (axes[0], stroke_histories, "Stroke classifier", 1 / len(STROKE_CLASSES)),
        (axes[1], skill_histories, "Beginner vs. expert", None),
    ]:
        arr = np.array(histories) * 100  # (k folds, epochs)
        epochs = np.arange(1, arr.shape[1] + 1)
        mean, std = arr.mean(axis=0), arr.std(axis=0)

        ax.plot(epochs, mean, color=BLUE_600, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=BLUE_300, alpha=0.35, linewidth=0)
        if baseline:
            ax.axhline(baseline * 100, color=CRITICAL, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val accuracy (%)")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylim(0, 100)

    fig.suptitle("Cross-validation training curves (mean ± std across 5 folds)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="E:/SkillEye/results")
    ap.add_argument("--out", default="E:/SkillEye/results/figures")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(results / "stroke_classifier" / "metrics.json") as f:
        stroke_v1 = json.load(f)
    with open(results / "stroke_classifier_v2" / "metrics.json") as f:
        stroke_v2 = json.load(f)
    with open(results / "cross_validation" / "stroke_classifier_cv.json") as f:
        stroke_cv = json.load(f)

    with open(results / "beginner_expert_check.json") as f:
        skill_v1 = json.load(f)
    with open(results / "beginner_expert_stgcn" / "metrics.json") as f:
        skill_v2 = json.load(f)
    with open(results / "cross_validation" / "beginner_expert_cv.json") as f:
        skill_cv = json.load(f)

    # v1's stroke classifier used 5 classes (25% baseline); v2/v3 use 6 classes
    # (16.7% baseline) -- not the same task, so it can't share one baseline line
    # in one chart. This chart instead answers the more important question --
    # does the single-split number hold up under cross-validation -- and v1 is
    # covered separately in the text/tables (stroke_v1 kept for that reason).
    del stroke_v1
    fig_accuracy_comparison(
        out / "accuracy_comparison.png",
        stroke_accs=[
            ("single split (v2)", stroke_v2["best_val_accuracy"], None),
            ("5-fold CV", stroke_cv["mean_accuracy"], stroke_cv["std_accuracy"]),
        ],
        stroke_baseline=1 / len(STROKE_CLASSES),
        skill_accs=[
            ("single split (v2)", skill_v2["best_val_accuracy"], None),
            ("5-fold CV", skill_cv["mean_accuracy"], skill_cv["std_accuracy"]),
        ],
        skill_baseline=skill_v1["logreg"]["baseline_accuracy"],
    )
    print("wrote", out / "accuracy_comparison.png")

    fig_confusion_matrix(
        out / "stroke_confusion_matrix.png",
        stroke_cv["summed_confusion_matrix"], STROKE_CLASSES,
        "Stroke classifier — confusion matrix (5-fold CV, summed)",
    )
    print("wrote", out / "stroke_confusion_matrix.png")

    fig_confusion_matrix(
        out / "skill_confusion_matrix.png",
        skill_cv["summed_confusion_matrix"], SKILL_CLASSES,
        "Beginner vs. expert — confusion matrix (5-fold CV, summed)",
    )
    print("wrote", out / "skill_confusion_matrix.png")

    fig_training_curves(
        out / "training_curves.png",
        stroke_cv["fold_val_acc_histories"],
        skill_cv["fold_val_acc_histories"],
    )
    print("wrote", out / "training_curves.png")


if __name__ == "__main__":
    main()
