"""
CS601T Assignment 5 - Part A, Question 1
One Kaiming-initialized 5x5 convolution filter on one image per class.

Run:
python part_a_q1.py
Dependencies: torch, pillow, matplotlib, numpy
"""
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project paths (fixed to match Assignment_5 structure)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PLOT_DIR = PROJECT_ROOT / "Report" / "Plots" / "Part_A" / "Q1"
MISC_DIR = PROJECT_ROOT / "Misc" / "Part_A" / "Q1"

CLASSES = ["0", "1", "5", "7", "8"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(root, split, cls):
    """Load the first image of a class as a 1x28x28 grayscale tensor."""
    folder = Path(root) / split / cls
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    with Image.open(files[0]) as im:
        im = ImageOps.grayscale(im).resize((28, 28))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0), files[0]


def initialize_filter(seed, device):
    """Create a single 5x5 Kaiming-initialized convolution filter."""
    torch.manual_seed(seed)
    w = torch.empty(1, 1, 5, 5, device=device)
    nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
    return w


def compute_feature_map(x, w):
    """Convolve the input with the filter (stride=1, padding=0)."""
    with torch.no_grad():
        return F.conv2d(x.unsqueeze(0).to(w.device), w, stride=1, padding=0)[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_filter(weights, out_path):
    """Visualize the 5x5 filter with a zero-centered colormap."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    vmax = float(np.abs(weights).max()) or 1.0
    im = ax.imshow(weights, cmap="seismic", vmin=-vmax, vmax=vmax)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Kaiming-initialized 5x5 filter")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_map(fmap, cls, out_path):
    """Save an individual feature map with a perceptually uniform colormap."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(fmap, cmap="viridis")
    ax.set_title(f"Class {cls} feature map")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_input(x, cls, out_path):
    """Save the grayscale input image used for the class."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(x[0].numpy(), cmap="gray")
    ax.set_title(f"Class {cls} input (28x28)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pipeline(x, weights, fmap, cls, out_path):
    """Filter -> Input -> Feature map pipeline for a single class."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))

    vmax = float(np.abs(weights).max()) or 1.0
    axes[0].imshow(weights, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Filter (5x5)")

    axes[1].imshow(x[0].numpy(), cmap="gray")
    axes[1].set_title(f"Class {cls} input")

    axes[2].imshow(fmap, cmap="viridis")
    axes[2].set_title("Feature map (24x24)")

    for a in axes:
        a.axis("off")
    fig.suptitle(f"Convolution pipeline — Class {cls}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_all_classes(inputs, fmaps, out_path):
    """Grid: row 0 = inputs, row 1 = feature maps."""
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(14, 6))
    for j, cls in enumerate(CLASSES):
        axes[0, j].imshow(inputs[cls][0].numpy(), cmap="gray")
        axes[0, j].set_title(f"Class {cls} input")
        axes[1, j].imshow(fmaps[cls], cmap="viridis")
        axes[1, j].set_title(f"Class {cls} feature map")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report / metadata files
# ---------------------------------------------------------------------------
def save_dimension_calculation(out_path, n=28, k=5, p=0, s=1):
    out = (n - k + 2 * p) // s + 1
    text = (
        "Convolution Output Dimension Calculation\n"
        "========================================\n"
        f"Input size (N)   = {n}\n"
        f"Kernel size (K)  = {k}\n"
        f"Stride (S)       = {s}\n"
        f"Padding (P)      = {p}\n\n"
        f"Formula: output = floor((N - K + 2P) / S) + 1\n"
        f"        = floor(({n} - {k} + 2*{p}) / {s}) + 1\n"
        f"        = ({n} - {k} + 2*{p}) / {s} + 1\n"
        f"        = {out}\n\n"
        f"Verified: feature map shape = {out} x {out}\n"
    )
    out_path.write_text(text)
    return out


def save_summary(out_path, weights, records, filter_values_path):
    lines = []
    lines.append("Part A - Q1 Summary\n")
    lines.append("===================\n")
    lines.append(f"Filter values file: {filter_values_path}\n")
    lines.append("Filter (5x5):\n")
    for row in weights:
        lines.append("  " + " ".join(f"{v:+.6f}" for v in row))
    lines.append("\n")
    lines.append("Convolution parameters:\n")
    lines.append("  Kernel = 5x5, Stride = 1, Padding = 0\n")
    lines.append("  Input  = 1x28x28  ->  Output = 1x24x24\n")
    lines.append("\n")
    lines.append("Per-class results:\n")
    lines.append(f"{'Class':<6}{'Image':<40}{'Input':<12}{'Output':<12}{'Verified'}\n")
    for r in records:
        lines.append(
            f"{r['cls']:<6}{r['path'].name:<40}"
            f"{'1x28x28':<12}{'1x24x24':<12}{r['verified']}\n"
        )
    out_path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Seed everything for reproducibility ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # --- Prepare output directories ---
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    MISC_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialize filter (shared across all classes) ---
    w = initialize_filter(args.seed, device)
    weights = w.detach().cpu().numpy().reshape(5, 5)

    # --- 2. Save raw filter values to Misc ---
    filter_values_path = MISC_DIR / "filter_values.txt"
    np.savetxt(filter_values_path, weights, fmt="%.6f")

    # --- 3. Visualize filter in Report ---
    plot_filter(weights, PLOT_DIR / "filter.png")

    # --- 4. Save dimension calculation to Misc ---
    expected = save_dimension_calculation(MISC_DIR / "dimension_calculation.txt")

    # --- 5. Process every class ---
    inputs, fmaps, records = {}, {}, []
    for cls in CLASSES:
        x, path = load_image(DATA_DIR, "train", cls)
        fmap = compute_feature_map(x, w)

        # Verify the expected output shape before continuing
        assert fmap.shape == (expected, expected), (
            f"Unexpected feature map shape {fmap.shape} for class {cls}"
        )

        inputs[cls] = x
        fmaps[cls] = fmap

        # Save raw feature maps
        np.savetxt(MISC_DIR / f"class_{cls}_feature_map.txt", fmap, fmt="%.6f")

        # Save plots to Report
        plot_input(x, cls, PLOT_DIR / f"class_{cls}_input.png")
        plot_feature_map(fmap, cls, PLOT_DIR / f"class_{cls}_feature_map.png")
        plot_pipeline(x, weights, fmap, cls, PLOT_DIR / f"class_{cls}_pipeline.png")

        records.append({"cls": cls, "path": path, "verified": "OK"})
        print(f"Class {cls}: {path.name} -> feature map {fmap.shape} (verified)")

    # --- 6. Combined overview figure ---
    plot_all_classes(inputs, fmaps, PLOT_DIR / "q1_all_classes.png")

    # --- 7. Save summary ---
    save_summary(MISC_DIR / "summary.txt", weights, records, filter_values_path)

    print(f"\nFilter:\n{weights}")
    print(f"\nFormula: output = floor((N - K + 2P)/S) + 1")
    print(f"For N=28, K=5, P=0, S=1 -> output = 24x24 (verified per class).")
    print(f"\nPlots saved to:  {PLOT_DIR.resolve()}")
    print(f"Data saved to:   {MISC_DIR.resolve()}")


if __name__ == "__main__":
    main()
