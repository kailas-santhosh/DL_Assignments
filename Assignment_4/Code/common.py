import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


DEFAULT_CLASSES = [0, 1, 5, 7, 8]
DEFAULT_DATA_ROOT = "Group_24"
OUTPUT_ROOT = Path("outputs")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_split_dir(root, split):
    root = Path(root)
    candidates = [root / split, root / split.lower(), root / split.upper()]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find '{split}' directory under {root.resolve()}. "
        f"Expected something like {root}/{split}/0, {root}/{split}/1, ..."
    )


def load_split(root, split, classes=None):
    split_dir = _find_split_dir(root, split)
    if classes is None:
        classes = DEFAULT_CLASSES

    X, y = [], []
    for cls in classes:
        cls_dir = split_dir / str(cls)
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")

        for file in sorted(cls_dir.iterdir()):
            if not file.is_file():
                continue
            try:
                img = Image.open(file).convert("L").resize((28, 28))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(arr.reshape(-1))
                y.append(int(cls))
            except Exception:
                continue

    if not X:
        raise RuntimeError(f"No images found in {split_dir}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def load_dataset(root, classes=None):
    if classes is None:
        classes = DEFAULT_CLASSES
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = load_split(root, split, classes)
    return data


def make_loader(X, y=None, batch_size=128, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(X_t)
    else:
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def save_sample_grid(X, y, path, title, classes=DEFAULT_CLASSES, n_per_class=1):
    path = Path(path)
    ensure_dir(path.parent)

    selected = []
    for cls in classes:
        idx = np.where(y == cls)[0][:n_per_class]
        selected.extend(idx.tolist())

    rows = len(classes)
    cols = n_per_class

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.2 * cols, 2.2 * rows),
        squeeze=False
    )

    pos = 0

    for r, cls in enumerate(classes):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")

            if pos < len(selected):
                idx = selected[pos]
                ax.imshow(X[idx].reshape(28, 28), cmap="gray")
                ax.set_title(f"Class {cls}")

            pos += 1

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_grid(original_X, reconstructed_X, y, path, title,
                             classes=DEFAULT_CLASSES):
    path = Path(path)
    ensure_dir(path.parent)

    fig, axes = plt.subplots(len(classes), 2, figsize=(5.5, 2.1 * len(classes)))
    axes = np.atleast_2d(axes)

    for r, cls in enumerate(classes):
        idxs = np.where(y == cls)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]

        axes[r, 0].imshow(original_X[idx].reshape(28, 28), cmap="gray")
        axes[r, 0].set_title(f"Original - {cls}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(np.clip(reconstructed_X[idx], 0, 1).reshape(28, 28), cmap="gray")
        axes[r, 1].set_title(f"Reconstructed - {cls}")
        axes[r, 1].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_curve(history, path, title):
    path = Path(path)
    ensure_dir(path.parent)

    fig = plt.figure(figsize=(8, 4.5))
    epochs = range(len(history["train"]))

    # Plot training first
    plt.plot(
        epochs,
        history["train"],
        label="Train",
        linestyle="-",
        marker="o",
        markersize=4,
        markevery=3,
        linewidth=2,
        zorder=2
    )

    # Plot validation second so it remains visible when overlapping
    plt.plot(
        epochs,
        history["val"],
        label="Validation",
        linestyle="--",
        marker="s",
        markersize=4,
        markevery=3,
        linewidth=2,
        zorder=3
    )

    plt.xlabel("Epoch (0 = before training)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_loss_curve(history, path, title):
    path = Path(path)
    ensure_dir(path.parent)

    fig = plt.figure(figsize=(8, 4.5))
    epochs = range(len(history["train"]))

    plt.plot(
        epochs,
        history["train"],
        label="Train",
        linestyle="-",
        marker="o",
        markersize=4,
        markevery=3,
        linewidth=2,
        zorder=2
    )

    plt.plot(
        epochs,
        history["val"],
        label="Validation",
        linestyle="--",
        marker="s",
        markersize=4,
        markevery=3,
        linewidth=2,
        zorder=3
    )

    plt.xlabel("Epoch (0 = before training)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_confusion(y_true, y_pred, classes, path, title):
    path = Path(path)
    ensure_dir(path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return cm.tolist()


def save_csv(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_class_distribution(y, path, title):
    path = Path(path)
    ensure_dir(path.parent)
    vals, counts = np.unique(y, return_counts=True)
    fig = plt.figure(figsize=(6, 4))
    plt.bar([str(v) for v in vals], counts)
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))
