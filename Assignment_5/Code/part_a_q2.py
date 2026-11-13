"""
CS601T Assignment 5 - Part A, Question 2

Two stacked Conv -> ReLU -> 2x2 MaxPool blocks,
tested on one image per class.

Conv1: 6 filters of 5x5
Conv2: 16 filters of 5x5
Stride: 1, Padding: 0
Pool: 2x2, stride 2

Run:
    python part_a_q2.py

Outputs:
    Report/Plots/Part_A/Q2/
        filters/
        feature_maps/
        summary/
    Misc/Part_A/Q2/
        metadata/
        txt/
        numpy/

Dependencies: torch, pillow, matplotlib, numpy
"""

import argparse
import json
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
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

REPORT_ROOT = PROJECT_ROOT / "Report" / "Plots" / "Part_A" / "Q2"
FILTER_DIR = REPORT_ROOT / "filters"
FMAP_DIR = REPORT_ROOT / "feature_maps"
SUMMARY_DIR = REPORT_ROOT / "summary"

MISC_ROOT = PROJECT_ROOT / "Misc" / "Part_A" / "Q2"
META_DIR = MISC_ROOT / "metadata"
TXT_DIR = MISC_ROOT / "txt"
NPY_DIR = MISC_ROOT / "numpy"

CLASSES = ["0", "1", "5", "7", "8"]


def ensure_dirs():
    for p in (FILTER_DIR, FMAP_DIR, SUMMARY_DIR, META_DIR, TXT_DIR, NPY_DIR):
        p.mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_first_image(root, split, cls):
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvReluPool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        conv = self.conv(x)
        relu = F.relu(conv)
        pool = F.max_pool2d(relu, kernel_size=2, stride=2)
        return conv, relu, pool


class TwoLayerCNNFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvReluPool(1, 6)
        self.layer2 = ConvReluPool(6, 16)

    def forward(self, x):
        c1, r1, p1 = self.layer1(x)
        c2, r2, p2 = self.layer2(p1)
        return {
            "conv1": c1, "relu1": r1, "pool1": p1,
            "conv2": c2, "relu2": r2, "pool2": p2,
        }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _normalize(m):
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi > lo else m


def plot_filter_channels(weights, path, title, n_filters=3):
    """Save a grid: rows = selected filters, columns = input-channel slices.

    For Conv1 there is only one input channel. For Conv2 we show all 6
    channel slices per filter, avoiding the misleading average.
    """
    w = weights.detach().cpu().numpy()  # (out_ch, in_ch, k, k)
    n_filters = min(n_filters, w.shape[0])
    in_ch = w.shape[1]

    cols = in_ch
    rows = n_filters
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows),
                             squeeze=False)

    vmax = float(np.abs(w[:n_filters]).max()) or 1.0

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(w[i, j], cmap="seismic", vmin=-vmax, vmax=vmax)
            axes[i, j].set_title(f"F{i} ch{j}", fontsize=8)
            axes[i, j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_maps(tensor, path, title):
    """Save every channel of a feature map tensor (independently normalized)."""
    arr = tensor.detach().cpu().numpy()
    n = arr.shape[0]
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for i in range(n):
        axes.flat[i].imshow(_normalize(arr[i]), cmap="viridis")
        axes.flat[i].set_title(f"Channel {i + 1}", fontsize=9)
        axes.flat[i].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(x, feats, cls, path):
    """Input -> Conv1 -> ReLU1 -> Pool1 -> Conv2 -> ReLU2 -> Pool2 pipeline."""
    stages = [
        ("Input", x[0].numpy(), "gray"),
        ("Conv1 (ch0)", feats["conv1"][0, 0].numpy(), "viridis"),
        ("ReLU1 (ch0)", feats["relu1"][0, 0].numpy(), "viridis"),
        ("Pool1 (ch0)", feats["pool1"][0, 0].numpy(), "viridis"),
        ("Conv2 (ch0)", feats["conv2"][0, 0].numpy(), "viridis"),
        ("ReLU2 (ch0)", feats["relu2"][0, 0].numpy(), "viridis"),
        ("Pool2 (ch0)", feats["pool2"][0, 0].numpy(), "viridis"),
    ]
    fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 3.2))
    for ax, (name, arr, cm) in zip(axes, stages):
        ax.imshow(_normalize(arr) if cm != "gray" else arr, cmap=cm)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Class {cls}: layer-by-layer pipeline")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report / metadata
# ---------------------------------------------------------------------------
def save_formula_and_dimensions(path):
    text = (
        "Formulas and Expected Dimensions\n"
        "================================\n\n"
        "Convolution formula:\n"
        "    output = floor((N - K + 2P) / S) + 1\n\n"
        "Pooling formula:\n"
        "    output = floor((N - K) / S) + 1\n\n"
        "Trace for a 1x28x28 input:\n"
        "    Conv1 (K=5, S=1, P=0): (28-5)/1 + 1 = 24  -> 6 x 24 x 24\n"
        "    ReLU1:                 unchanged          -> 6 x 24 x 24\n"
        "    Pool1 (K=2, S=2):      (24-2)/2 + 1 = 12  -> 6 x 12 x 12\n"
        "    Conv2 (K=5, S=1, P=0): (12-5)/1 + 1 = 8   -> 16 x 8 x 8\n"
        "    ReLU2:                 unchanged          -> 16 x 8 x 8\n"
        "    Pool2 (K=2, S=2):      (8-2)/2 + 1 = 4    -> 16 x 4 x 4\n"
    )
    path.write_text(text)


def save_dimension_verification(path, records, expected):
    lines = [
        "Dimension Verification\n"
        "======================\n\n"
        "For each stage and each class, the obtained shape is compared\n"
        "against the mathematically expected shape.\n\n"
    ]
    all_pass = True
    for cls, info in records.items():
        lines.append(f"Class {cls} ({info['image']}):\n")
        for stage, exp in expected.items():
            got = info["shapes"][stage]
            ok = tuple([1] + list(exp)) == tuple(got)
            all_pass &= ok
            lines.append(
                f"  {stage:<6}  expected (1,{exp[0]},{exp[1]},{exp[2]})  "
                f"obtained {tuple(got)}  ->  {'PASS' if ok else 'FAIL'}\n"
            )
        lines.append("\n")
    lines.append(f"Overall: {'PASS' if all_pass else 'FAIL'}\n")
    path.write_text("".join(lines))
    return all_pass


def save_feature_map_shapes_json(path, expected):
    data = {"input": [1, 1, 28, 28]}
    for stage in ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2"):
        ch, h, w = expected[stage]
        data[stage] = [1, int(ch), int(h), int(w)]
    path.write_text(json.dumps(data, indent=2))


def save_selected_filters_json(path, conv1_idx, conv2_idx):
    path.write_text(json.dumps({
        "conv1_filter_indices": conv1_idx,
        "conv2_filter_indices": conv2_idx,
        "note": "These exact filter positions are reused in Q3.",
    }, indent=2))


def save_filter_values(path, weights):
    w = weights.detach().cpu().numpy()
    np.savetxt(path, w.reshape(w.shape[0], -1), fmt="%.6f")


def save_summary_json(path, records, expected, selected):
    expected_list = {}
    for stage in ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2"):
        ch, h, w = expected[stage]
        expected_list[stage] = [1, int(ch), int(h), int(w)]
    path.write_text(json.dumps({
        "input": "1x28x28",
        "kernel": 5,
        "stride": 1,
        "padding": 0,
        "pool_kernel": 2,
        "pool_stride": 2,
        "expected_shapes": expected_list,
        "selected_filters": selected,
        "per_class": {
            cls: {"image": info["image"], "shapes": info["shapes"]}
            for cls, info in records.items()
        },
    }, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    ensure_dirs()

    data_dir = DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {data_dir}")
    print(f"Device : {device}")

    model = TwoLayerCNNFeatures().to(device)
    model.eval()

    # Expected shapes (channel, H, W)
    expected = {
        "conv1": (6, 24, 24),
        "relu1": (6, 24, 24),
        "pool1": (6, 12, 12),
        "conv2": (16, 8, 8),
        "relu2": (16, 8, 8),
        "pool2": (16, 4, 4),
    }

    # ---- Formula / dimension file ----
    save_formula_and_dimensions(TXT_DIR / "formula_and_dimensions.txt")
    save_feature_map_shapes_json(META_DIR / "feature_map_shapes.json", expected)

    # ---- Selected filter indices (reused in Q3) ----
    conv1_idx = [0, 1, 2]
    conv2_idx = [0, 1, 2]
    save_selected_filters_json(
        META_DIR / "selected_filters.json", conv1_idx, conv2_idx
    )

    # ---- Filter values + images ----
    save_filter_values(TXT_DIR / "conv1_filter_values.txt", model.layer1.conv.weight)
    save_filter_values(TXT_DIR / "conv2_filter_values.txt", model.layer2.conv.weight)

    plot_filter_channels(
        model.layer1.conv.weight[conv1_idx],
        FILTER_DIR / "conv1_three_filters.png",
        "Conv1 filters (indices 0,1,2) — channel 0 of each",
        n_filters=3,
    )
    plot_filter_channels(
        model.layer2.conv.weight[conv2_idx],
        FILTER_DIR / "conv2_three_filters.png",
        "Conv2 filters (indices 0,1,2) — all 6 input-channel slices",
        n_filters=3,
    )

    # ---- Per-class processing ----
    records = {}
    for cls in CLASSES:
        x, path = load_first_image(data_dir, "train", cls)
        with torch.no_grad():
            feats = model(x.unsqueeze(0).to(device))

        # Save input image
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x[0].numpy(), cmap="gray")
        ax.set_title(f"Class {cls} input")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FMAP_DIR / f"class_{cls}_input.png", dpi=160)
        plt.close(fig)

        shapes = {k: tuple(v.shape) for k, v in feats.items()}
        records[cls] = {"image": path.name, "shapes": shapes}

        # Assert expected shapes (strict verification)
        for stage, exp in expected.items():
            assert shapes[stage] == (1, *exp), (
                f"Shape mismatch for class {cls} {stage}: "
                f"expected {(1, *exp)}, got {shapes[stage]}"
            )

        # Save every stage — including ReLU
        plot_maps(feats["conv1"][0],
                  FMAP_DIR / f"class_{cls}_conv1_maps.png",
                  f"Class {cls}: conv1 (6x24x24)")
        plot_maps(feats["relu1"][0],
                  FMAP_DIR / f"class_{cls}_relu1_maps.png",
                  f"Class {cls}: relu1 (6x24x24)")
        plot_maps(feats["pool1"][0],
                  FMAP_DIR / f"class_{cls}_layer1_pool_maps.png",
                  f"Class {cls}: pool1 (6x12x12)")
        plot_maps(feats["conv2"][0],
                  FMAP_DIR / f"class_{cls}_conv2_maps.png",
                  f"Class {cls}: conv2 (16x8x8)")
        plot_maps(feats["relu2"][0],
                  FMAP_DIR / f"class_{cls}_relu2_maps.png",
                  f"Class {cls}: relu2 (16x8x8)")
        plot_maps(feats["pool2"][0],
                  FMAP_DIR / f"class_{cls}_layer2_pool_maps.png",
                  f"Class {cls}: pool2 (16x4x4)")

        # Combined pipeline figure
        plot_pipeline(x, feats, cls,
                      SUMMARY_DIR / f"class_{cls}_pipeline.png")

        # Save numpy arrays for later reuse (Q3, receptive field, etc.)
        for stage in ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2"):
            np.save(NPY_DIR / f"class_{cls}_{stage}.npy",
                    feats[stage].detach().cpu().numpy())

        print(f"Class {cls}: {path.name} -> {shapes}")

    # ---- Verification report ----
    ok = save_dimension_verification(
        TXT_DIR / "dimension_verification.txt", records, expected
    )
    print(f"\nDimension verification: {'PASS' if ok else 'FAIL'}")

    # ---- Summary ----
    save_summary_json(
        META_DIR / "summary.json", records, expected,
        {"conv1": conv1_idx, "conv2": conv2_idx},
    )

    print(f"\nQ2 complete.")
    print(f"  Report -> {REPORT_ROOT}")
    print(f"  Misc   -> {MISC_ROOT}")


if __name__ == "__main__":
    main()
