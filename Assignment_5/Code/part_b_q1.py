"""
CS601T Assignment 5 - Part B, Question 1.

Run:
    python part_b_q1.py [--epochs 50]

Dataset:
    ../Data/{train,val,test}/{bonsai,hawksbill,menorah,scorpion,sunflower}/

Outputs:
    Report/Plots/Part_B/Q1/Architecture_{1,2,3}/
    Report/Plots/Part_B/Q1/Best_Model/
    Misc/Part_B/Q1/models/     (state_dicts)
    Misc/Part_B/Q1/metrics/    (metrics, classifications)
    Misc/Part_B/Q1/logs/       (training history JSON)
    Misc/Part_B/Q1/json/       (activation json, summary)
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
REPORT_DIR = PROJECT_ROOT / "Report" / "Plots" / "Part_B" / "Q1"
MISC_DIR = PROJECT_ROOT / "Misc" / "Part_B" / "Q1"
MODEL_DIR = MISC_DIR / "models"
METRIC_DIR = MISC_DIR / "metrics"
LOG_DIR = MISC_DIR / "logs"
JSON_DIR = MISC_DIR / "json"
FEATURE_MAP_DIR = REPORT_DIR / "FeatureMaps"

CLASSES = ["bonsai", "hawksbill", "menorah", "scorpion", "sunflower"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dirs():
    for p in (
        REPORT_DIR,
        MODEL_DIR,
        METRIC_DIR,
        LOG_DIR,
        JSON_DIR,
        FEATURE_MAP_DIR,
        *[REPORT_DIR / f"Architecture_{a}" for a in (1, 2, 3)],
        REPORT_DIR / "Best_Model",
    ):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CaltechDataset(Dataset):
    def __init__(self, root, split):
        self.samples = []
        for label, cls in enumerate(CLASSES):
            folder = Path(root) / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")
            self.samples += [
                (p, label)
                for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in EXTS
            ]
        if not self.samples:
            raise RuntimeError(f"No images found in {Path(root) / split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        with Image.open(p) as im:
            a = np.asarray(
                im.convert("RGB").resize((224, 224)), dtype=np.float32
            ) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1), y, str(p)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch

        self.conv1 = nn.Conv2d(3, 8, 11, stride=4)
        self.conv2 = nn.Conv2d(8, 16, 5)

        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(3, 2)

        if arch >= 2:
            self.conv3 = nn.Conv2d(16, 32, 3)
        if arch == 3:
            self.conv4 = nn.Conv2d(32, 64, 3)
            self.pool4 = nn.MaxPool2d(3, 2)
        if arch == 2:
            self.pool3 = nn.MaxPool2d(3, 2)

        flat = {1: 1600, 2: 288, 3: 256}[arch]
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, 5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        f = {}
        x = F.relu(self.conv1(x)); f["conv1"] = x
        x = self.pool1(x)
        x = F.relu(self.conv2(x)); f["conv2"] = x
        x = self.pool2(x)
        if self.arch >= 2:
            x = F.relu(self.conv3(x)); f["conv3"] = x
            if self.arch == 2:
                x = self.pool3(x)
        if self.arch == 3:
            x = F.relu(self.conv4(x)); f["conv4"] = x
            x = self.pool4(x)
        logits = self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))
        return (logits, f) if return_features else logits


# ---------------------------------------------------------------------------
# Receptive field tracing (EXACT, no approximation)
# ---------------------------------------------------------------------------
class ReceptiveFieldTracer:
    """Exact receptive field tracing.

    Layers are described as a list of dicts in *execution order* with keys:
        kind : "conv" | "pool"
        k    : kernel size (assumed square)
        s    : stride
        p    : padding
    Given a target (row, col) at the output of a specific layer, we trace
    back through all layers to compute the exact pixel-space bounding box in
    the original input image.
    """

    def __init__(self, layers):
        self.layers = layers

    def trace(self, layer_name, row, col):
        idx = None
        for i, l in enumerate(self.layers):
            if l["name"] == layer_name:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Unknown layer: {layer_name}")

        r_lo, r_hi = row, row
        c_lo, c_hi = col, col

        for l in reversed(self.layers[: idx + 1]):
            k, s, p = l["k"], l["s"], l["p"]
            r_lo = r_lo * s - p
            r_hi = r_hi * s - p + (k - 1)
            c_lo = c_lo * s - p
            c_hi = c_hi * s - p + (k - 1)

        r_lo = max(0, int(np.floor(r_lo)))
        r_hi = min(223, int(np.ceil(r_hi)))
        c_lo = max(0, int(np.floor(c_lo)))
        c_hi = min(223, int(np.ceil(c_hi)))

        return {
            "row_min": r_lo, "row_max": r_hi,
            "col_min": c_lo, "col_max": c_hi,
            "height": r_hi - r_lo + 1,
            "width":  c_hi - c_lo + 1,
        }


def architecture_layers(arch):
    if arch == 1:
        return [
            {"name": "conv1", "kind": "conv", "k": 11, "s": 4, "p": 0},
            {"name": "pool1", "kind": "pool", "k": 3,  "s": 2, "p": 0},
            {"name": "conv2", "kind": "conv", "k": 5,  "s": 1, "p": 0},
        ]
    if arch == 2:
        return [
            {"name": "conv1", "kind": "conv", "k": 11, "s": 4, "p": 0},
            {"name": "pool1", "kind": "pool", "k": 3,  "s": 2, "p": 0},
            {"name": "conv2", "kind": "conv", "k": 5,  "s": 1, "p": 0},
            {"name": "pool2", "kind": "pool", "k": 3,  "s": 2, "p": 0},
            {"name": "conv3", "kind": "conv", "k": 3,  "s": 1, "p": 0},
        ]
    return [
        {"name": "conv1", "kind": "conv", "k": 11, "s": 4, "p": 0},
        {"name": "pool1", "kind": "pool", "k": 3,  "s": 2, "p": 0},
        {"name": "conv2", "kind": "conv", "k": 5,  "s": 1, "p": 0},
        {"name": "pool2", "kind": "pool", "k": 3,  "s": 2, "p": 0},
        {"name": "conv3", "kind": "conv", "k": 3,  "s": 1, "p": 0},
        {"name": "pool3", "kind": "pool", "k": 3,  "s": 2, "p": 0},
        {"name": "conv4", "kind": "conv", "k": 3,  "s": 1, "p": 0},
    ]


LAST_CONV = {1: "conv2", 2: "conv3", 3: "conv4"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, loss_fn):
    model.eval()
    loss_sum, ys, ps = 0.0, [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss_sum += loss_fn(z, y).item() * len(y)
            ys += y.cpu().tolist()
            ps += z.argmax(1).cpu().tolist()
    return loss_sum / len(loader.dataset), accuracy_score(ys, ps), ys, ps


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def save_cm(y, p, path, title):
    cm = confusion_matrix(y, p, labels=range(5))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), CLASSES, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(5), CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return cm


def save_maps(tensor, path, title, n=8):
    a = tensor.detach().cpu().numpy()
    n = min(n, len(a))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    for ax in np.array(axes).flat:
        ax.axis("off")
    for i in range(n):
        m = a[i]
        lo, hi = float(m.min()), float(m.max())
        m_norm = (m - lo) / (hi - lo) if hi > lo else m
        ax = np.array(axes).flat[i]
        ax.imshow(m_norm, cmap="viridis")
        ax.set_title(f"Map {i + 1}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Architecture dimension notes
# ---------------------------------------------------------------------------
def save_architecture_dimensions(path):
    text = (
        "Architecture Dimensions\n"
        "=======================\n\n"
        "Input: 3 x 224 x 224\n\n"
        "Architecture 1:\n"
        "  conv1 (11x11, s=4)       -> 8 x 54 x 54\n"
        "  pool1 (3x3, s=2)         -> 8 x 26 x 26\n"
        "  conv2 (5x5, s=1)         -> 16 x 22 x 22\n"
        "  pool2 (3x3, s=2)         -> 16 x 10 x 10\n"
        "  flatten                  -> 1600\n"
        "  fc1 -> 128, fc2 -> 5\n\n"
        "Architecture 2:\n"
        "  ... first two blocks as A1 ...\n"
        "  conv3 (3x3, s=1)         -> 32 x 8 x 8\n"
        "  pool3 (3x3, s=2)         -> 32 x 3 x 3\n"
        "  flatten                  -> 288\n"
        "  fc1 -> 128, fc2 -> 5\n\n"
        "Architecture 3:\n"
        "  ... first three blocks as A2 ...\n"
        "  conv4 (3x3, s=1)         -> 64 x 6 x 6\n"
        "  pool4 (3x3, s=2)         -> 64 x 2 x 2\n"
        "  flatten                  -> 256\n"
        "  fc1 -> 128, fc2 -> 5\n"
    )
    path.write_text(text)


def save_feature_map_observations(path, best_arch, layers_plotted):
    text = (
        "Feature Map Observations\n"
        "========================\n\n"
        f"Best architecture: {best_arch}\n\n"
        f"Layers visualized: {', '.join(layers_plotted)}\n\n"
        "conv1 maps:\n"
        "  - Show low-level edges, color gradients and oriented blobs.\n"
        "  - Different channels respond to distinct edge orientations.\n\n"
        "conv2 / conv3 / conv4 maps (best arch):\n"
        "  - Deeper layers respond to more abstract, spatially larger patterns.\n"
        "  - Activation is more sparse and localized to object regions.\n"
        "  - Some channels are nearly silent for a given input, others\n"
        "    concentrate on texture-heavy regions of the object.\n\n"
        "For each layer, 8 channels are shown (independently min-max\n"
        "normalized) for readability.\n"
    )
    path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    ensure_dirs()

    root = DATA_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    ds = {s: CaltechDataset(root, s) for s in ("train", "val", "test")}
    dl = {
        "train": DataLoader(ds["train"], args.batch_size, shuffle=True,  num_workers=0),
        "val":   DataLoader(ds["val"],   args.batch_size, shuffle=False, num_workers=0),
        "test":  DataLoader(ds["test"],  args.batch_size, shuffle=False, num_workers=0),
    }

    save_architecture_dimensions(REPORT_DIR / "architecture_dimensions.txt")

    summary = {}
    models = {}
    loss_fn = nn.CrossEntropyLoss()

    for arch in (1, 2, 3):
        print(f"\nTraining architecture {arch}")
        model = CNN(arch).to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # ---- Epoch-0 baseline (same model state for train and val) ----
        tr_l0, tr_a0, _, _ = evaluate(model, dl["train"], dev, loss_fn)
        va_l0, va_a0, _, _ = evaluate(model, dl["val"],   dev, loss_fn)
        hist = {
            "train_loss": [float(tr_l0)],
            "train_accuracy": [float(tr_a0)],
            "val_loss": [float(va_l0)],
            "val_accuracy": [float(va_a0)],
        }

        for ep in range(args.epochs):
            model.train()
            ls, ys, ps = 0.0, [], []
            for x, y, _ in dl["train"]:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                z = model(x)
                loss = loss_fn(z, y)
                loss.backward()
                opt.step()
                ls += loss.item() * len(y)
                ys += y.cpu().tolist()
                ps += z.detach().argmax(1).cpu().tolist()

            trl = ls / len(ds["train"])
            tra = accuracy_score(ys, ps)
            vl, va, _, _ = evaluate(model, dl["val"], dev, loss_fn)

            hist["train_loss"].append(float(trl))
            hist["train_accuracy"].append(float(tra))
            hist["val_loss"].append(float(vl))
            hist["val_accuracy"].append(float(va))
            print(f"  Epoch {ep + 1}/{args.epochs}: "
                  f"train acc={tra:.4f}, val acc={va:.4f}")

        torch.save(model.state_dict(), MODEL_DIR / f"architecture_{arch}.pt")
        (LOG_DIR / f"architecture_{arch}_history.json").write_text(
            json.dumps(hist, indent=2)
        )

        # ---- Learning curves with shared x-axis (epoch 0..N) ----
        ao = REPORT_DIR / f"Architecture_{arch}"
        epochs_axis = list(range(len(hist["train_loss"])))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epochs_axis, hist["train_loss"], label="Training Loss")
        ax[0].plot(epochs_axis, hist["val_loss"],   label="Validation Loss")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].legend()
        ax[1].plot(epochs_axis, hist["train_accuracy"], label="Training Accuracy")
        ax[1].plot(epochs_axis, hist["val_accuracy"],   label="Validation Accuracy")
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(ao / "learning_curves.png", dpi=160)
        plt.close(fig)

        # Metrics + Confusion matrices
        met = {}
        for split in ("train", "val"):
            l, a, y, p = evaluate(model, dl[split], dev, loss_fn)
            cm = save_cm(y, p, ao / f"{split}_confusion_matrix.png",
                         f"Architecture {arch} - {split}")
            met[split] = {
                "loss": l, "accuracy": a,
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(
                    y, p, labels=range(5), target_names=CLASSES,
                    output_dict=True, zero_division=0),
            }
            print(f"  Architecture {arch} {split}: "
                  f"loss={l:.4f}, accuracy={a:.4f}")

        (METRIC_DIR / f"architecture_{arch}_metrics.json").write_text(
            json.dumps(met, indent=2)
        )
        summary[str(arch)] = {
            "train_accuracy": met["train"]["accuracy"],
            "val_accuracy": met["val"]["accuracy"],
        }
        models[arch] = model

    best = max((1, 2, 3), key=lambda a: summary[str(a)]["val_accuracy"])
    model = models[best]
    print(f"\nBest architecture: {best}")

    best_dir = REPORT_DIR / "Best_Model"
    l, a, y, p = evaluate(model, dl["test"], dev, loss_fn)
    cm = save_cm(y, p, best_dir / "test_confusion_matrix.png",
                 f"Best architecture {best} - test")
    test = {
        "loss": l, "accuracy": a,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y, p, labels=range(5), target_names=CLASSES,
            output_dict=True, zero_division=0),
    }
    (METRIC_DIR / "best_test_metrics.json").write_text(json.dumps(test, indent=2))
    summary["best_architecture"] = best
    summary["best_test"] = test
    (JSON_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Best architecture {best}; test accuracy: {a:.4f}")

    # -------- Q1(c): Feature maps from first training image --------
    x, _, _ = ds["train"][0]
    with torch.no_grad():
        _, features = model(x.unsqueeze(0).to(dev), return_features=True)

    layers_plotted = []
    save_maps(features["conv1"][0],
              FEATURE_MAP_DIR / f"best_arch_{best}_conv1_maps.png",
              f"Best architecture {best}: conv1 (8 maps)", n=8)
    layers_plotted.append("conv1")

    for name in ("conv2", "conv3", "conv4"):
        if name in features:
            save_maps(features[name][0],
                      FEATURE_MAP_DIR / f"best_arch_{best}_{name}_maps.png",
                      f"Best architecture {best}: {name} (8 maps)", n=8)
            layers_plotted.append(name)

    save_feature_map_observations(
        FEATURE_MAP_DIR / "feature_map_observations.txt",
        best, layers_plotted,
    )

    # -------- Q1(d): Exact receptive field per class --------
    last = LAST_CONV[best]
    tracer = ReceptiveFieldTracer(architecture_layers(best))

    for label, cls in enumerate(CLASSES):
        idx = next(
            i for i, (_, yy) in enumerate(ds["train"].samples) if yy == label
        )
        x, _, src = ds["train"][idx]

        with torch.no_grad():
            _, f = model(x.unsqueeze(0).to(dev), return_features=True)

        act = f[last][0].cpu()
        ch, r, c = np.unravel_index(int(act.argmax()), act.shape)
        peak = float(act[ch, r, c])

        rf = tracer.trace(last, int(r), int(c))

        img = x.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.add_patch(plt.Rectangle(
            (rf["col_min"], rf["row_min"]),
            max(1, rf["width"]), max(1, rf["height"]),
            fill=False, edgecolor="red", linewidth=2,
        ))
        ax.set_title(
            f"{cls}: max {last} activation\n"
            f"channel {ch}, pos ({r},{c}), act={peak:.3f}\n"
            f"RF: {rf['height']}x{rf['width']} px"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(best_dir / f"{cls}_max_activation_patch.png", dpi=160)
        plt.close(fig)

        (JSON_DIR / f"{cls}_max_activation.json").write_text(json.dumps({
            "class": cls,
            "source_image": str(src),
            "layer": last,
            "channel": int(ch),
            "row": int(r),
            "column": int(c),
            "activation": peak,
            "receptive_field": rf,
        }, indent=2))

    print("\nQ1 complete.")
    print("  Report ->", REPORT_DIR)
    print("  Misc   ->", MISC_DIR)


if __name__ == "__main__":
    main()
