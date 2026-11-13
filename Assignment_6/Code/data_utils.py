import re
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLASSES,
    VALIDATION_FRACTION,
    SEED,
    CHECK_DUPLICATES,
    CHECK_FILE_DIMENSIONS,
    VERBOSE,
)


# ---------------------------------------------------------------------------
# Low-level file readers
# ---------------------------------------------------------------------------
def numeric_values(path):
    text = Path(path).read_text(errors="ignore")
    vals = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", text)
    if not vals:
        raise ValueError(f"No numeric values found in {path}")
    return np.asarray([float(x) for x in vals], dtype=np.float32)


def read_handwriting_file(path):
    vals = numeric_values(path)
    n_points = int(round(float(vals[0])))
    coords = vals[1:]
    if n_points <= 0 or coords.size < 2:
        raise ValueError(f"Invalid handwriting sample: {path}")

    needed = 2 * n_points
    # Checklist #4: strict dimension validation — no silent truncation.
    if coords.size != needed:
        raise ValueError(
            f"{path}: expected exactly {needed} coordinate values for "
            f"{n_points} points, got {coords.size} "
            f"(file has {vals.size} total numeric values)"
        )

    seq = coords.reshape(n_points, 2)
    mins = seq.min(axis=0)
    spans = seq.max(axis=0) - mins
    spans[spans == 0] = 1.0
    return ((seq - mins) / spans).astype(np.float32)


def read_mfcc_file(path):
    try:
        arr = np.loadtxt(path, dtype=np.float32)
    except Exception:
        arr = None

    if arr is None or np.asarray(arr).size == 0:
        vals = numeric_values(path)
        if vals.size % 39 != 0:
            raise ValueError(
                f"{path}: MFCC value count {vals.size} is not divisible by 39"
            )
        arr = vals.reshape(-1, 39)

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % 39:
            raise ValueError(
                f"{path}: expected 39 features per frame, got {arr.size} values"
            )
        arr = arr.reshape(-1, 39)

    if arr.shape[1] != 39:
        if arr.shape[0] == 39:
            arr = arr.T
        else:
            raise ValueError(
                f"{path}: expected 39 MFCC columns, found shape {arr.shape}"
            )
    return arr


# ---------------------------------------------------------------------------
# Duplicate detection (checklist #3)
# ---------------------------------------------------------------------------
def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def assert_no_duplicates(train_files, test_files, dataset_name, class_name):
    """Fail loudly if any file content appears in both splits."""
    train_hashes = {_file_sha256(p) for p in train_files}
    test_hashes  = {_file_sha256(p) for p in test_files}
    overlap = train_hashes & test_hashes
    if overlap:
        raise RuntimeError(
            f"[{dataset_name}/{class_name}] {len(overlap)} duplicate "
            f"sample(s) shared between train and test splits."
        )
    if VERBOSE:
        print(
            f"  [dup-check] {dataset_name}/{class_name}: "
            f"train={len(train_files)}, test={len(test_files)}, duplicates=0"
        )


# ---------------------------------------------------------------------------
# Archive extraction & directory lookup
# ---------------------------------------------------------------------------
def extract_archives(dataset_dir, classes):
    dataset_dir = Path(dataset_dir)
    extracted_root = dataset_dir / "_extracted_group24"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        zip_path = dataset_dir / f"{cls}.zip"
        target = extracted_root / cls
        if zip_path.exists() and (not target.exists() or not any(target.rglob("*"))):
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)


def find_split_dir(class_root, split_names):
    """Prefer direct child directory; fall back to recursive search."""
    class_root = Path(class_root)
    wanted = {x.lower() for x in split_names}

    for child in class_root.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return child

    for p in class_root.rglob("*"):
        if p.is_dir() and p.name.lower() in wanted:
            return p
    return None


def locate_class_root(dataset_dir, cls):
    dataset_dir = Path(dataset_dir)
    for opt in (dataset_dir / cls, dataset_dir / "_extracted_group24" / cls):
        if opt.exists():
            return opt
    for p in dataset_dir.rglob("*"):
        if p.is_dir() and p.name == cls and "_extracted_group24" not in p.parts:
            return p
    return None


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------
def collect_dataset(data_root, kind):
    if kind == "handwriting":
        folder       = Path(data_root) / "Handwriting_Data"
        train_splits = ("train",)
        test_splits  = ("dev", "test")
        extension    = ".txt"
        reader       = read_handwriting_file
    else:
        folder       = Path(data_root) / "CV_Data"
        train_splits = ("train",)
        test_splits  = ("test",)
        extension    = ".mfcc"
        reader       = read_mfcc_file

    if not folder.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {folder}")

    classes = CLASSES[kind]
    extract_archives(folder, classes)

    train_x, train_y, test_x, test_y = [], [], [], []

    for label, cls in enumerate(classes):
        root = locate_class_root(folder, cls)
        if root is None:
            raise FileNotFoundError(f"Could not locate class '{cls}' under {folder}")

        train_dir = find_split_dir(root, train_splits)
        test_dir  = find_split_dir(root, test_splits)
        if train_dir is None or test_dir is None:
            raise FileNotFoundError(
                f"Missing train/test-or-dev folders for class '{cls}' under {root}"
            )

        train_files = sorted(
            p for p in train_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == extension
        )
        test_files = sorted(
            p for p in test_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == extension
        )
        if not train_files or not test_files:
            raise RuntimeError(
                f"No {extension} samples for '{cls}': "
                f"train={len(train_files)}, test={len(test_files)}"
            )

        if CHECK_DUPLICATES:
            assert_no_duplicates(train_files, test_files, kind, cls)

        class_train = [reader(p) for p in train_files]
        class_test  = [reader(p) for p in test_files]

        # Checklist #5: validate consistent feature dimensionality per class.
        if CHECK_FILE_DIMENSIONS:
            train_dims = {seq.shape[1] for seq in class_train}
            test_dims  = {seq.shape[1] for seq in class_test}
            all_dims   = train_dims | test_dims
            if len(all_dims) != 1:
                raise ValueError(
                    f"[{kind}/{cls}] inconsistent feature dimensions "
                    f"across files: {sorted(all_dims)}"
                )
            if kind == "cv" and all_dims != {39}:
                raise ValueError(
                    f"[{kind}/{cls}] expected 39 MFCC features, "
                    f"found {sorted(all_dims)}"
                )

        train_x.extend(class_train)
        train_y.extend([label] * len(class_train))
        test_x.extend(class_test)
        test_y.extend([label] * len(class_test))

        if VERBOSE:
            print(
                f"  [{kind}] class '{cls}' (label={label}): "
                f"train={len(class_train)}, test={len(class_test)}"
            )

    return (
        train_x,
        np.asarray(train_y, dtype=np.int64),
        test_x,
        np.asarray(test_y, dtype=np.int64),
        classes,
    )


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(sequences, labels, val_fraction=VALIDATION_FRACTION, seed=SEED):
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            raise ValueError(
                f"Class label {label} has only {len(idx)} sample(s); "
                f"need at least 2 for a stratified train/val split."
            )
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_val = min(n_val, len(idx) - 1)  # always leave ≥1 for training
        val_indices.extend(idx[:n_val])
        train_indices.extend(idx[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# CV standardization (train-only statistics)
# ---------------------------------------------------------------------------
def standardize_cv(train_sequences, val_sequences, test_sequences):
    frames = np.concatenate(train_sequences, axis=0)
    mean = frames.mean(axis=0, keepdims=True)
    std  = frames.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    transform = lambda seqs: [((s - mean) / std).astype(np.float32) for s in seqs]
    return (
        transform(train_sequences),
        transform(val_sequences),
        transform(test_sequences),
        mean,
        std,
    )


# ---------------------------------------------------------------------------
# Normalized handwriting plots
# ---------------------------------------------------------------------------
def save_normalized_handwriting_examples(
    train_sequences, train_labels, class_names, output_dir, per_class=5
):
    """Save trajectory plots for at least five normalized training samples per class."""
    output_dir = Path(output_dir) / "normalized_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, class_name in enumerate(class_names):
        indices = np.where(train_labels == label)[0]

        if len(indices) < per_class:
            print(
                f"[warn] class '{class_name}' has only {len(indices)} "
                f"training sample(s); requested {per_class} plots."
            )
        indices = indices[:per_class]
        if len(indices) == 0:
            continue

        fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
        if len(indices) == 1:
            axes = [axes]

        for sample_no, (ax, idx) in enumerate(zip(axes, indices), start=1):
            seq = train_sequences[idx]
            ax.plot(seq[:, 0], seq[:, 1], linewidth=1.5)
            ax.scatter(seq[0, 0],  seq[0, 1],  marker="o", label="Start")
            ax.scatter(seq[-1, 0], seq[-1, 1], marker="x", label="End")
            ax.set_title(f"Sample {sample_no}")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.grid(alpha=0.25)

        axes[0].legend(fontsize=7)
        fig.suptitle(f"Normalized handwriting samples — {class_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{class_name}_normalized_samples.png", dpi=180)
        plt.close(fig)
