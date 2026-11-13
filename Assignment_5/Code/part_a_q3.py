"""
CS601T Assignment 5 - Part A, Question 3

Train the specified CNN using SGD and report split accuracy
and confusion matrices.

Architecture:
28x28 -> Conv(6, 5x5) -> ReLU -> MaxPool(2)
      -> Conv(16, 5x5) -> ReLU -> MaxPool(2)
      -> Flatten(256) -> FC(64, ReLU) -> FC(5 logits)

Run:
    python part_a_q3.py
    python part_a_q3.py --epochs 20

Outputs:
    Report/Plots/Part_A/Q3/
        learning_curves/
        confusion_matrices/
        filters/
        feature_maps/
        activation_boxes/
    Misc/Part_A/Q3/
        model/
        metrics/
        json/
        logs/

Dependencies:
torch, pillow, matplotlib, numpy, scikit-learn
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

REPORT_ROOT = PROJECT_ROOT / "Report" / "Plots" / "Part_A" / "Q3"
CURVE_DIR    = REPORT_ROOT / "learning_curves"
CM_DIR       = REPORT_ROOT / "confusion_matrices"
FILTER_DIR   = REPORT_ROOT / "filters"
FMAP_DIR     = REPORT_ROOT / "feature_maps"
ACTBOX_DIR   = REPORT_ROOT / "activation_boxes"

MISC_ROOT  = PROJECT_ROOT / "Misc" / "Part_A" / "Q3"
MODEL_DIR  = MISC_ROOT / "model"
METRIC_DIR = MISC_ROOT / "metrics"
JSON_DIR   = MISC_ROOT / "json"
LOG_DIR    = MISC_ROOT / "logs"

DPI = 300

CLASSES = ["0", "1", "5", "7", "8"]
TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Distinct colors for the top-5 activation boxes (rank order)
RANK_COLORS = ["red", "green", "blue", "orange", "magenta"]


def ensure_dirs():
    for p in (CURVE_DIR, CM_DIR, FILTER_DIR, FMAP_DIR, ACTBOX_DIR,
              MODEL_DIR, METRIC_DIR, JSON_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Logging helper (tee to console and file)
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DigitDataset(Dataset):
    def __init__(self, root, split):
        self.samples = []
        self.root = Path(root) / split

        for cls in CLASSES:
            folder = self.root / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")
            files = sorted(
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            self.samples.extend((p, TO_IDX[cls]) for p in files)

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        with Image.open(p) as im:
            im = ImageOps.grayscale(im).resize((28, 28))
            a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).unsqueeze(0), y, str(p)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 5)

        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        for fc in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x, return_features=False):
        c1 = self.conv1(x)
        r1 = F.relu(c1)
        p1 = F.max_pool2d(r1, 2, 2)
        c2 = self.conv2(p1)
        r2 = F.relu(c2)
        p2 = F.max_pool2d(r2, 2, 2)
        h = F.relu(self.fc1(torch.flatten(p2, 1)))
        logits = self.fc2(h)
        if return_features:
            return logits, {
                "conv1": c1, "relu1": r1, "pool1": p1,
                "conv2": c2, "relu2": r2, "pool2": p2,
            }
        return logits


# ---------------------------------------------------------------------------
# Exact receptive-field localization for a conv2 feature-map coordinate
# ---------------------------------------------------------------------------
# Layer descriptions in forward execution order, with (kernel, stride, padding).
# We trace from the target activation back through conv1 + pool1 + conv2.
LAYERS_Q3 = [
    {"name": "conv1", "k": 5, "s": 1, "p": 0},
    {"name": "pool1", "k": 2, "s": 2, "p": 0},
    {"name": "conv2", "k": 5, "s": 1, "p": 0},
]


def conv2_receptive_field(row, col):
    """Exact input-space receptive field for a conv2 feature-map coordinate.

    Backward pass over the recorded layers:
        r_in = r_out * s - p,  r_in_hi = r_out * s - p + (k - 1)
    Repeated for each layer from conv2 back to the input. All strides and
    kernel extents are included so the returned bounding box is the *exact*
    set of input pixels that contribute to (row, col).
    """
    r_lo = r_hi = row
    c_lo = c_hi = col

    for layer in reversed(LAYERS_Q3):
        k, s, p = layer["k"], layer["s"], layer["p"]
        r_lo = r_lo * s - p
        r_hi = r_hi * s - p + (k - 1)
        c_lo = c_lo * s - p
        c_hi = c_hi * s - p + (k - 1)

    r_lo = max(0, int(np.floor(r_lo)))
    r_hi = min(27, int(np.ceil(r_hi)))
    c_lo = max(0, int(np.floor(c_lo)))
    c_hi = min(27, int(np.ceil(c_hi)))

    # Effective stride and RF size in input-space (used in metadata).
    eff_stride = 1
    for layer in LAYERS_Q3:
        eff_stride *= layer["s"]
    rf_h = r_hi - r_lo + 1
    rf_w = c_hi - c_lo + 1

    return {
        "row_min": r_lo, "row_max": r_hi,
        "col_min": c_lo, "col_max": c_hi,
        "height": rf_h, "width": rf_w,
        "stride": eff_stride,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    y_all, p_all = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            y_all.extend(y.cpu().tolist())
            p_all.extend(logits.argmax(1).cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(y_all, p_all),
        y_all,
        p_all,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def save_cm(y, pred, path, title):
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), CLASSES, fontsize=9)
    ax.set_yticks(range(5), CLASSES, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(title, fontsize=11)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return cm


def plot_maps(tensor, path, title):
    a = tensor.detach().cpu().numpy()
    n = a.shape[0]
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(n):
        axes.flat[i].imshow(a[i], cmap="viridis")
        axes.flat[i].set_title(f"Filter {i + 1}", fontsize=9)
        axes.flat[i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_filters(weights, filename, title, out):
    n = weights.shape[0]
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(n):
        w = weights[i].detach().cpu().numpy()
        img = w[0] if w.shape[0] == 1 else w.mean(axis=0)
        axes.flat[i].imshow(img, cmap="gray")
        axes.flat[i].set_title(f"Filter {i + 1}", fontsize=9)
        axes.flat[i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    ensure_dirs()

    # Tee stdout to run_log.txt
    log_file = open(LOG_DIR / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        if not DATA_DIR.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Dataset directory: {DATA_DIR}")
        print(f"Report directory:  {REPORT_ROOT}")
        print(f"Misc directory:    {MISC_ROOT}")
        print(f"Device: {device}")

        datasets = {s: DigitDataset(DATA_DIR, s) for s in ("train", "val", "test")}
        loaders = {
            "train": DataLoader(datasets["train"], args.batch_size,
                                shuffle=True, num_workers=0),
            "val": DataLoader(datasets["val"], args.batch_size,
                              shuffle=False, num_workers=0),
            "test": DataLoader(datasets["test"], args.batch_size,
                               shuffle=False, num_workers=0),
        }

        print("\nDimensions:"
              "\nInput: 1x28x28"
              "\nConv1: 6x24x24"
              "\nPool1: 6x12x12"
              "\nConv2: 16x8x8"
              "\nPool2: 16x4x4"
              "\nFlatten: 256"
              "\nFC1: 64"
              "\nOutput: 5")

        for split, ds in datasets.items():
            print(f"{split} images: {len(ds)}")

        model = DigitCNN().to(device)
        loss_fn = nn.CrossEntropyLoss()

        # SGD with momentum for faster, smoother convergence.
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
        )

        # ------------------------------------------------------------------
        # Epoch 0 baseline — evaluate the *initialized* model on both train
        # and val so both curves start from the same model state (and the
        # same x-origin at epoch 0).
        # ------------------------------------------------------------------
        tr_loss0, tr_acc0, _, _ = evaluate(model, loaders["train"], device, loss_fn)
        va_loss0, va_acc0, _, _ = evaluate(model, loaders["val"],   device, loss_fn)

        hist = {
            "train_loss":     [float(tr_loss0)],
            "train_accuracy": [float(tr_acc0)],
            "val_loss":       [float(va_loss0)],
            "val_accuracy":   [float(va_acc0)],
        }

        print(f"Epoch 0/{args.epochs} (baseline): "
              f"train loss={tr_loss0:.4f}, train acc={tr_acc0:.4f}, "
              f"val loss={va_loss0:.4f}, val acc={va_acc0:.4f}")

        for ep in range(args.epochs):
            model.train()
            sum_loss = 0.0
            ys, ps = [], []
            for x, y, _ in loaders["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                sum_loss += loss.item() * y.size(0)
                ys.extend(y.detach().cpu().tolist())
                ps.extend(logits.detach().argmax(1).cpu().tolist())

            tr_loss = sum_loss / len(datasets["train"])
            tr_acc = accuracy_score(ys, ps)
            va_loss, va_acc, _, _ = evaluate(model, loaders["val"], device, loss_fn)

            hist["train_loss"].append(float(tr_loss))
            hist["train_accuracy"].append(float(tr_acc))
            hist["val_loss"].append(float(va_loss))
            hist["val_accuracy"].append(float(va_acc))

            print(f"Epoch {ep + 1}/{args.epochs}: "
                  f"train loss={tr_loss:.4f}, train acc={tr_acc:.4f}, "
                  f"val loss={va_loss:.4f}, val acc={va_acc:.4f}")

        # ---- Save model + history ----
        torch.save(model.state_dict(), MODEL_DIR / "digit_cnn.pt")
        (LOG_DIR / "training_history.json").write_text(
            json.dumps(hist, indent=2), encoding="utf-8"
        )

        # ---- Learning curves (shared x-axis starting at epoch 0) ----
        epochs_axis = list(range(len(hist["train_loss"])))  # 0,1,...,N
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs_axis, hist["train_loss"],     label="Training Loss",   marker="o")
        axes[0].plot(epochs_axis, hist["val_loss"],       label="Validation Loss", marker="s")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_axis, hist["train_accuracy"],     label="Training Accuracy",   marker="o")
        axes[1].plot(epochs_axis, hist["val_accuracy"],       label="Validation Accuracy", marker="s")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(CURVE_DIR / "learning_curves.png", dpi=DPI)
        plt.close(fig)

        # ---- Evaluation on all splits ----
        for split, loader in loaders.items():
            loss, acc, y, pred = evaluate(model, loader, device, loss_fn)
            cm = save_cm(
                y, pred,
                CM_DIR / f"{split}_confusion_matrix.png",
                f"{split.title()} confusion matrix",
            )
            report = classification_report(
                y, pred,
                labels=list(range(5)),
                target_names=CLASSES,
                output_dict=True,
                zero_division=0,
            )
            (METRIC_DIR / f"{split}_metrics.json").write_text(
                json.dumps({
                    "split": split, "loss": loss, "accuracy": acc,
                    "confusion_matrix": cm.tolist(),
                    "classification_report": report,
                }, indent=2),
                encoding="utf-8",
            )
            print(f"\n{split.upper()}:")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print("Confusion matrix (rows=true, columns=predicted):")
            print(cm)
            print(classification_report(
                y, pred, labels=list(range(5)),
                target_names=CLASSES, zero_division=0,
            ))

        # ---- Filters ----
        save_filters(model.conv1.weight, "conv1_all_6_filters.png",
                     "All 6 conv1 filters", FILTER_DIR)
        save_filters(model.conv2.weight, "conv2_all_16_filters.png",
                     "All 16 conv2 filters", FILTER_DIR)

        # ---- Feature maps + top-5 activation boxes per class ----
        ds = datasets["train"]
        for cls in CLASSES:
            idx = next(
                i for i, (_, y) in enumerate(ds.samples)
                if CLASSES[y] == cls
            )
            x, _, source = ds[idx]

            with torch.no_grad():
                _, feats = model(x.unsqueeze(0).to(device),
                                 return_features=True)

            for name in ("conv1", "conv2"):
                plot_maps(
                    feats[name][0],
                    FMAP_DIR / f"class_{cls}_{name}_maps.png",
                    f"Class {cls}: {name} feature maps",
                )

            # Top-5 conv2 ReLU activations
            act = feats["relu2"][0].detach().cpu()
            top = torch.argsort(act.flatten(), descending=True)[:5].tolist()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(x[0].numpy(), cmap="gray")

            boxes = []
            for rank, k in enumerate(top, 1):
                ch, rem = divmod(k, 8 * 8)
                r, c = divmod(rem, 8)

                # Exact receptive-field bounding box in input space.
                rf = conv2_receptive_field(r, c)
                color = RANK_COLORS[(rank - 1) % len(RANK_COLORS)]

                ax.add_patch(plt.Rectangle(
                    (rf["col_min"], rf["row_min"]),
                    max(1, rf["width"]), max(1, rf["height"]),
                    fill=False, edgecolor=color, linewidth=1.8,
                ))
                ax.text(
                    rf["col_min"], rf["row_min"], f"{rank}",
                    color=color, fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, pad=1),
                )

                boxes.append({
                    "rank": rank,
                    "color": color,
                    "channel": int(ch),
                    "feature_row": int(r),
                    "feature_col": int(c),
                    "activation": float(act[ch, r, c]),
                    "receptive_field": rf,
                    "box_xyxy": [
                        rf["col_min"], rf["row_min"],
                        rf["col_max"] + 1, rf["row_max"] + 1,
                    ],
                })

            ax.set_title(f"Class {cls}: top 5 conv2 activations (exact RF)")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(
                ACTBOX_DIR / f"class_{cls}_top5_activation_boxes.png",
                dpi=DPI,
            )
            plt.close(fig)

            (JSON_DIR / f"class_{cls}_top5_activation_boxes.json").write_text(
                json.dumps({
                    "class": cls,
                    "source_image": str(source),
                    "method": "exact receptive-field traceback",
                    "boxes": boxes,
                }, indent=2),
                encoding="utf-8",
            )

        print("\nQ3 completed successfully!")
        print(f"  Report -> {REPORT_ROOT}")
        print(f"  Misc   -> {MISC_ROOT}")

    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    main()
