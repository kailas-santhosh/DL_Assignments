"""
TASK 6 - Weight visualization.

For the best 1-hidden-layer AE and both denoising AEs:
1) visualize input-to-bottleneck weights as 28x28 images;
2) visualize the actual training image that maximally activates each bottleneck neuron.

Run after task3.py and task5.py.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import DEFAULT_CLASSES, DEFAULT_DATA_ROOT, ensure_dir, load_dataset, make_loader, save_json, set_seed
from models import Autoencoder


def load_ae(path, k):
    m = Autoencoder(k, "one_hidden")
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def activations(model, X, batch_size=256):
    out = []
    with torch.no_grad():
        for (xb,) in make_loader(X, None, batch_size, False):
            out.append(model.encode(xb).numpy())
    return np.concatenate(out, axis=0)


def plot_weights(model, path, title):
    W = model.encoder[0].weight.detach().numpy()
    n = W.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(4, rows * 1.9)))
    axes = np.asarray(axes).ravel()
    vmax = max(float(np.max(np.abs(W))), 1e-8)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(W[i].reshape(28, 28), cmap="gray", vmin=-vmax, vmax=vmax)
            ax.set_title(f"N{i}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_max_inputs(model, X, y, path, title):
    Z = activations(model, X)
    idx = np.argmax(Z, axis=0)
    val = np.max(Z, axis=0)
    n = Z.shape[1]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(4, rows * 2.0)))
    axes = np.asarray(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            j = idx[i]
            ax.imshow(X[j].reshape(28, 28), cmap="gray")
            ax.set_title(f"N{i}: class {y[j]}\nact={val[i]:.3f}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(path.with_suffix(".npz"), neuron=np.arange(n), image_index=idx,
                        image_class=y[idx], activation=val)


def process(path, k, X, y, out, name):
    model = load_ae(path, k)
    plot_weights(model, out / f"{name}_input_to_bottleneck_weights.png",
                 f"Task 6: {name} - input-to-bottleneck weights")
    plot_max_inputs(model, X, y, out / f"{name}_maximally_activating_inputs.png",
                    f"Task 6: {name} - maximally activating training inputs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    out = ensure_dir("outputs/task6")

    summary = Path("outputs/task3/best_overall.json")
    if not summary.exists():
        raise FileNotFoundError("Run task3.py first so the best bottleneck is known.")
    with open(summary, encoding="utf-8") as f:
        k = int(json.load(f)["bottleneck"])

    X, y = load_dataset(args.data_root, DEFAULT_CLASSES)["train"]

    ae_path = Path(f"outputs/task3/autoencoders/k{k}/model.pt")
    if not ae_path.exists():
        raise FileNotFoundError(f"Missing Task-3 AE: {ae_path}. Run task3.py first.")
    process(ae_path, k, X, y, out, f"best_1hidden_AE_k{k}")

    for pct in (20, 40):
        dae_path = Path(f"outputs/task5/noise_{pct}/model.pt")
        if not dae_path.exists():
            raise FileNotFoundError(f"Missing Task-5 AE: {dae_path}. Run task5.py first.")
        process(dae_path, k, X, y, out, f"denoising_AE_{pct}percent")

    save_json({"bottleneck": k,
               "visualizations": ["input_to_bottleneck_weights", "maximally_activating_training_inputs"]},
              out / "summary.json")
    print(f"Task 6 complete. Best bottleneck = {k}")


if __name__ == "__main__":
    main()
