"""
CS601T Assignment 5 - Part B, Question 2: VGG19 transfer learning.

Single-phase training (frozen backbone, no fine-tuning).

Run:
    python part_b_q2.py [--epochs 20] [--batch_size 8]
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import keras
from keras import layers, Model
from keras.applications.vgg19 import VGG19, preprocess_input
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

REPORT_ROOT = PROJECT_ROOT / "Report" / "Plots" / "Part_B" / "Q2"
CURVE_DIR   = REPORT_ROOT / "learning_curves"
CM_DIR      = REPORT_ROOT / "confusion_matrix"
Q2B_DIR     = REPORT_ROOT / "q2b_activation_localization"
GB_DIR      = REPORT_ROOT / "guided_backprop"
CAM_DIR     = REPORT_ROOT / "gradcam"
INPUTS_DIR  = REPORT_ROOT / "inputs"

MISC_ROOT  = PROJECT_ROOT / "Misc" / "Part_B" / "Q2"
MODEL_DIR  = MISC_ROOT / "models"
METRIC_DIR = MISC_ROOT / "metrics"
JSON_DIR   = MISC_ROOT / "json"
LOG_DIR    = MISC_ROOT / "logs"

CLASSES = ["bonsai", "hawksbill", "menorah", "scorpion", "sunflower"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def ensure_dirs():
    for p in (CURVE_DIR, CM_DIR, Q2B_DIR, GB_DIR, CAM_DIR, INPUTS_DIR,
              MODEL_DIR, METRIC_DIR, JSON_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def samples_for(root, split):
    items = []
    for label, cls in enumerate(CLASSES):
        folder = Path(root) / split / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        items += [
            (p, label)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in EXTS
        ]
    if not items:
        raise RuntimeError(f"No images found in {Path(root) / split}")
    return items


def load_img_raw(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((224, 224)), dtype=np.float32)


def preprocess(a):
    return preprocess_input(a.copy())


def deprocess(img_pre):
    rgb = img_pre + IMAGENET_MEAN
    return np.clip(rgb[..., ::-1] / 255.0, 0, 1)


def make_ds(items, batch, shuffle=False, seed=42, augment=False):
    def gen():
        ids = np.arange(len(items))
        if shuffle:
            np.random.default_rng(seed).shuffle(ids)
        for i in ids:
            p, y = items[i]
            yield preprocess(load_img_raw(p)), np.int32(y)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((224, 224, 3), tf.float32),
            tf.TensorSpec((), tf.int32),
        ),
    )
    if augment:
        def aug(x, y):
            x = tf.image.random_flip_left_right(x)
            return x, y
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    """Frozen VGG19 backbone + softmax head (single-phase training)."""
    base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    inp = keras.Input((224, 224, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(5, activation="softmax", name="class_predictions")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


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


def save_input_image(img_pre, path, title):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(deprocess(img_pre))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q2(b) activation localization
# ---------------------------------------------------------------------------
def compute_activation_localization(base, img_pre, target_layer="block5_conv4"):
    """Find strongest unit in `target_layer` and localize input region driving it."""
    feat_model = Model(
        inputs=base.input,
        outputs=base.get_layer(target_layer).output,
    )

    x = tf.convert_to_tensor(img_pre[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        fmap = feat_model(x, training=False)
        flat_idx = tf.argmax(tf.reshape(fmap, [-1]))
        r, c, ch = np.unravel_index(int(flat_idx), fmap.shape[1:])
        r, c, ch = int(r), int(c), int(ch)
        score = fmap[0, r, c, ch]

    grad = tape.gradient(score, x)
    if grad is None:
        raise RuntimeError(f"Gradient is None for {target_layer} activation.")

    sal = np.abs(grad.numpy()[0]).sum(axis=-1)
    sal = sal / (sal.max() + 1e-8)
    sal_smooth = gaussian_filter(sal, sigma=3.0)
    sal_smooth /= (sal_smooth.max() + 1e-8)

    thr = float(np.quantile(sal_smooth, 0.90))
    mask = sal_smooth >= thr
    if not mask.any():
        peak_r, peak_c = np.unravel_index(int(sal_smooth.argmax()),
                                          sal_smooth.shape)
        y0, y1 = max(0, peak_r - 20), min(224, peak_r + 20)
        x0, x1 = max(0, peak_c - 20), min(224, peak_c + 20)
    else:
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

    return {
        "layer": target_layer,
        "channel": ch,
        "row": r,
        "column": c,
        "activation": float(score.numpy()),
        "saliency_map": sal_smooth,
        "bbox_xyxy": [x0, y0, x1, y1],
    }


# ---------------------------------------------------------------------------
# Q2(c) guided backprop
# ---------------------------------------------------------------------------
def build_guided_from_trained(base):
    @tf.custom_gradient
    def guided_relu(x):
        y = tf.nn.relu(x)
        def grad(dy):
            return dy * tf.cast(x > 0, dy.dtype) * tf.cast(dy > 0, dy.dtype)
        return y, grad

    inp = keras.Input((224, 224, 3))
    x = inp
    cfg = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
    for bi, (n, filters) in enumerate(cfg, 1):
        for ci in range(1, n + 1):
            x = layers.Conv2D(
                filters, 3, padding="same", activation=guided_relu,
                name=f"block{bi}_conv{ci}",
            )(x)
        if bi < 5:
            x = layers.MaxPooling2D(2, 2, name=f"block{bi}_pool")(x)
    guided = Model(inp, x)

    for layer in guided.layers:
        if isinstance(layer, layers.Conv2D):
            layer.set_weights(base.get_layer(layer.name).get_weights())
    return guided


def guided_backprop_for_neurons(guided_model, img_pre, neurons):
    inp = tf.convert_to_tensor(img_pre[None, ...], dtype=tf.float32)
    results = []
    for n in neurons:
        ch, r, c = n["channel"], n["row"], n["column"]
        with tf.GradientTape() as tape:
            tape.watch(inp)
            out = guided_model(inp, training=False)
            score = out[0, r, c, ch]
        grad = tape.gradient(score, inp)
        if grad is None:
            raise RuntimeError(f"Guided backprop gradient is None for neuron {n}.")
        g = grad.numpy()[0]
        vis = np.maximum(g, 0).max(axis=-1)
        vis = vis / (vis.max() + 1e-8)
        results.append({
            "channel": ch, "row": r, "column": c,
            "activation": float(score.numpy()),
            "guided_map": vis,
        })
    return results


# ---------------------------------------------------------------------------
# Q2(d) Grad-CAM
# ---------------------------------------------------------------------------
def gradcam_target(model, base, img_pre, target_index,
                   target_layer="block5_conv4"):
    feat_extractor = Model(
        inputs=base.input,
        outputs=base.get_layer(target_layer).output,
    )

    inp = keras.Input((224, 224, 3))
    conv_feat = feat_extractor(inp, training=False)

    layer_names = [l.name for l in base.layers]
    target_idx = layer_names.index(target_layer)
    pool_idx   = layer_names.index("block5_pool")
    x = conv_feat
    for layer in base.layers[target_idx + 1: pool_idx + 1]:
        x = layer(x)
    pooled = x

    gap = layers.GlobalAveragePooling2D()(pooled)

    dense = model.get_layer("class_predictions")

    logits = keras.ops.matmul(gap, dense.kernel)
    if dense.use_bias:
        logits = keras.ops.add(logits, dense.bias)

    cam_model = Model(inp, [conv_feat, logits])

    x_tensor = tf.convert_to_tensor(img_pre[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        conv_features, logits_out = cam_model(x_tensor, training=False)
        target_score = logits_out[:, target_index]

    grads = tape.gradient(target_score, conv_features)
    if grads is None:
        raise RuntimeError(f"Grad-CAM gradient is None for {CLASSES[target_index]}")

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heat = tf.reduce_sum(weights * conv_features, axis=-1)[0]
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    heat = tf.image.resize(heat[..., None], (224, 224)).numpy()[..., 0]
    heat = heat / (heat.max() + 1e-8)

    preds = keras.activations.softmax(logits_out, axis=-1).numpy()[0]
    return heat, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=11)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    ensure_dirs()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    items = {s: samples_for(DATA_DIR, s) for s in ("train", "val", "test")}
    ds = {s: make_ds(items[s], args.batch_size,
                     shuffle=(s == "train"),
                     seed=args.seed,
                     augment=(s == "train"))
          for s in items}
    for s in items:
        print(f"{s}: {len(items[s])} images")

    print("Loading ImageNet-pretrained VGG19; first run downloads weights.")
    model, base = build_model()

    # ---- Epoch-0 baseline: measure train/val on the *freshly initialized* head ----
    def _eval_quick(loader):
        yy, pp, ll = [], [], []
        for x, y in loader:
            probs = model(x, training=False).numpy()
            yy += y.numpy().tolist()
            pp += probs.argmax(1).tolist()
            ll += keras.losses.sparse_categorical_crossentropy(y, probs).numpy().tolist()
        return float(np.mean(ll)), float(accuracy_score(yy, pp))

    tr_l0, tr_a0 = _eval_quick(ds["train"])
    va_l0, va_a0 = _eval_quick(ds["val"])
    print(f"Epoch 0 (init): train loss={tr_l0:.4f}, acc={tr_a0:.4f} | "
          f"val loss={va_l0:.4f}, acc={va_a0:.4f}")

    # ---- Single-phase training (frozen backbone, no fine-tuning) ----
    hist = model.fit(ds["train"], validation_data=ds["val"], epochs=args.epochs)

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    # Prepend epoch-0 so train & val curves start from the same model state.
    history["loss"]         = [tr_l0] + history["loss"]
    history["accuracy"]     = [tr_a0] + history["accuracy"]
    history["val_loss"]     = [va_l0] + history["val_loss"]
    history["val_accuracy"] = [va_a0] + history["val_accuracy"]

    # ---- Save model + history ----
    model.save(MODEL_DIR / "vgg19_part_b_q2.keras")
    (LOG_DIR / "training_history.json").write_text(json.dumps(history, indent=2))

    # ---- Learning curves (shared x-axis starting at epoch 0) ----
    ep = list(range(len(history["loss"])))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["loss"],     label="Training Loss")
    ax[0].plot(ep, history["val_loss"], label="Validation Loss")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, history["accuracy"],     label="Training Accuracy")
    ax[1].plot(ep, history["val_accuracy"], label="Validation Accuracy")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(CURVE_DIR / "learning_curves.png", dpi=160)
    plt.close(fig)

    # ---- Evaluation ----
    metrics = {}
    for split in ("train", "val", "test"):
        yy, pp, ll = [], [], []
        for x, y in ds[split]:
            probs = model(x, training=False).numpy()
            yy += y.numpy().tolist()
            pp += probs.argmax(1).tolist()
            ll += keras.losses.sparse_categorical_crossentropy(y, probs).numpy().tolist()
        cm = save_cm(yy, pp, CM_DIR / f"{split}_confusion_matrix.png",
                     f"VGG19 {split} confusion matrix")
        metrics[split] = {
            "loss": float(np.mean(ll)),
            "accuracy": float(accuracy_score(yy, pp)),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                yy, pp, labels=range(5), target_names=CLASSES,
                output_dict=True, zero_division=0),
        }
        print(f"{split}: loss={np.mean(ll):.4f}, "
              f"accuracy={accuracy_score(yy, pp):.4f}")
    (METRIC_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Pick one training image per class ----
    chosen = {}
    for label, cls in enumerate(CLASSES):
        p = next(p for p, y in items["train"] if y == label)
        raw = load_img_raw(p)
        chosen[cls] = {"path": p, "raw": raw, "pre": preprocess(raw)}
        save_input_image(chosen[cls]["pre"],
                         INPUTS_DIR / f"{cls}_input.png",
                         f"Input: {cls}")

    # ------------------------------------------------------------------
    # Q2(b): activation localization
    # ------------------------------------------------------------------
    activation_info = []
    for cls, rec in chosen.items():
        loc = compute_activation_localization(
            base, rec["pre"], target_layer="block5_conv4"
        )
        rgb = deprocess(rec["pre"])
        x0, y0, x1, y1 = loc["bbox_xyxy"]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].imshow(rgb)
        axes[0].set_title(
            f"{cls}: peak ch={loc['channel']} (r={loc['row']},c={loc['column']})"
        )
        axes[1].imshow(loc["saliency_map"], cmap="inferno")
        axes[1].set_title("Input saliency |dAct/dX|")
        axes[2].imshow(rgb)
        axes[2].add_patch(plt.Rectangle(
            (x0, y0), max(1, x1 - x0), max(1, y1 - y0),
            fill=False, edgecolor="red", linewidth=2,
        ))
        axes[2].set_title("Localized contributing region")
        for a in axes:
            a.axis("off")
        fig.suptitle(f"Q2(b) activation localization — {cls}")
        fig.tight_layout()
        fig.savefig(Q2B_DIR / f"{cls}_activation_localization.png", dpi=160)
        plt.close(fig)

        activation_info.append({
            "class": cls,
            "source_image": str(rec["path"]),
            "layer": loc["layer"],
            "channel": loc["channel"],
            "row": loc["row"],
            "column": loc["column"],
            "activation": loc["activation"],
            "bbox_xyxy": loc["bbox_xyxy"],
            "method": "input-gradient saliency localization",
        })
    (JSON_DIR / "q2b_activation_info.json").write_text(
        json.dumps(activation_info, indent=2)
    )

    # ------------------------------------------------------------------
    # Q2(c): guided backprop for 5 strongest neurons (first class/image)
    # ------------------------------------------------------------------
    guided = build_guided_from_trained(base)

    cam_layer_model = Model(base.input,
                            base.get_layer("block5_conv4").output)
    cls_for_gb = CLASSES[0]
    img_pre = chosen[cls_for_gb]["pre"]
    act = cam_layer_model(img_pre[None, ...], training=False).numpy()[0]

    flat = act.reshape(-1)
    top5_flat = np.argsort(flat)[-5:][::-1]
    top_neurons = []
    for f in top5_flat:
        r, c, ch = np.unravel_index(int(f), act.shape)
        top_neurons.append({
            "channel": int(ch), "row": int(r), "column": int(c),
            "activation": float(act[r, c, ch]),
        })

    gb_results = guided_backprop_for_neurons(guided, img_pre, top_neurons)
    rgb = deprocess(img_pre)

    for i, res in enumerate(gb_results):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(res["guided_map"], cmap="hot")
        ax.set_title(
            f"Neuron ch={res['channel']} (r={res['row']},c={res['column']})\n"
            f"act={res['activation']:.3f}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(GB_DIR / f"{cls_for_gb}_neuron_{i + 1}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flat
    axes[0].imshow(rgb); axes[0].set_title(f"Input: {cls_for_gb}"); axes[0].axis("off")
    for i, res in enumerate(gb_results):
        axes[i + 1].imshow(res["guided_map"], cmap="hot")
        axes[i + 1].set_title(
            f"Neuron {i+1}: ch={res['channel']} (r={res['row']},c={res['column']})",
            fontsize=9)
        axes[i + 1].axis("off")
    fig.suptitle(f"Q2(c) guided backprop — {cls_for_gb}")
    fig.tight_layout()
    fig.savefig(GB_DIR / f"{cls_for_gb}_guided_backprop_top5.png", dpi=160)
    plt.close(fig)

    (JSON_DIR / "q2c_guided_backprop.json").write_text(json.dumps({
        "input_class": cls_for_gb,
        "source_image": str(chosen[cls_for_gb]["path"]),
        "layer": "block5_conv4",
        "neurons": [
            {"channel": r["channel"], "row": r["row"],
             "column": r["column"], "activation": r["activation"]}
            for r in gb_results
        ],
    }, indent=2))

    # ------------------------------------------------------------------
    # Q2(d): Grad-CAM — 5 inputs × 5 targets = 25 heatmaps
    # ------------------------------------------------------------------
    cam_metadata = []
    print("\nGrad-CAM diagnostics (heatmap range / std / predicted):")
    for img_cls, rec in chosen.items():
        rgb = deprocess(rec["pre"])
        combined_fig, combined_axes = plt.subplots(1, 5, figsize=(20, 4.2))
        for target_idx, target_cls in enumerate(CLASSES):
            try:
                heat, pred = gradcam_target(
                    model, base, rec["pre"], target_idx,
                    target_layer="block5_conv4",
                )
            except RuntimeError as e:
                print(f"  [warn] {img_cls}->{target_cls}: {e}")
                heat = np.zeros((224, 224), dtype=np.float32)
                pred = np.zeros(5, dtype=np.float32)

            pred_idx = int(pred.argmax())
            print(f"  {img_cls:10s} → target={target_cls:10s} | "
                  f"heat[{heat.min():.3f},{heat.max():.3f}] "
                  f"std={heat.std():.3f} | "
                  f"pred={CLASSES[pred_idx]} ({pred[pred_idx]:.2f})"
                  f"{' ✓' if pred_idx == target_idx else ''}")

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(rgb)
            ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.set_title(
                f"Input: {img_cls}  |  Target: {target_cls}\n"
                f"Pred: {CLASSES[pred_idx]} ({pred[pred_idx]:.2f})",
                fontsize=10,
            )
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(CAM_DIR / f"{img_cls}_target_{target_cls}.png", dpi=160)
            plt.close(fig)

            combined_axes[target_idx].imshow(rgb)
            combined_axes[target_idx].imshow(heat, cmap="jet", alpha=0.45)
            combined_axes[target_idx].set_title(f"Target: {target_cls}")
            combined_axes[target_idx].axis("off")

            cam_metadata.append({
                "input_class": img_cls,
                "target_class": target_cls,
                "predicted_class": CLASSES[pred_idx],
                "predicted_confidence": float(pred[pred_idx]),
                "heatmap_max": float(heat.max()),
                "heatmap_std": float(heat.std()),
                "correct": bool(pred_idx == target_idx),
            })
        combined_fig.suptitle(f"Grad-CAM sheet — input image class {img_cls}")
        combined_fig.tight_layout()
        combined_fig.savefig(CAM_DIR / f"{img_cls}_all_targets_sheet.png", dpi=160)
        plt.close(combined_fig)

    (JSON_DIR / "q2d_gradcam_metadata.json").write_text(
        json.dumps(cam_metadata, indent=2)
    )

    print("\nPart B Question 2 completed.")
    print(f"  Report -> {REPORT_ROOT}")
    print(f"  Misc   -> {MISC_ROOT}")


if __name__ == "__main__":
    main()
