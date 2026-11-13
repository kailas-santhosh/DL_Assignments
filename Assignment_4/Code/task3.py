"""
TASK 3 - Classification using compressed representation from 1-hidden-layer AEs.

Run:
    python task3.py --data-root Group_24
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA

from common import (
    DEFAULT_CLASSES, DEFAULT_DATA_ROOT, accuracy, ensure_dir, load_dataset,
    make_loader, plot_confusion, save_accuracy_curve, save_csv, save_json, save_loss_curve, set_seed,
)
from models import Classifier
from task2 import train_autoencoder


FCNN_ARCHITECTURES = {
    "FCNN_1": [128],
    "FCNN_2": [256, 128],
    "FCNN_3": [256, 128, 64],
}


def encode(model, X, batch_size=256):
    device = next(model.parameters()).device
    loader = make_loader(X, None, batch_size, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model.encode(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict(model, X, batch_size=256):
    device = next(model.parameters()).device
    loader = make_loader(X, None, batch_size, shuffle=False)
    model.eval()
    pred = []
    with torch.no_grad():
        for (xb,) in loader:
            pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.asarray(pred)


def train_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes,
                     hidden_layers, epochs=30, batch_size=128, seed=42):
    from task1 import train_classifier as train_pca_classifier
    return train_pca_classifier(
        X_train, y_train, X_val, y_val, input_dim, num_classes,
        hidden_layers, epochs=epochs, batch_size=batch_size, seed=seed
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ae-epochs", type=int, default=40)
    parser.add_argument("--clf-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = ensure_dir("outputs/task3")
    data = load_dataset(args.data_root, DEFAULT_CLASSES)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    label_to_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    ytr = np.array([label_to_index[v] for v in y_train])
    yv = np.array([label_to_index[v] for v in y_val])
    yte = np.array([label_to_index[v] for v in y_test])

    rows = []
    best_overall = None

    for k in [32, 64, 128, 256]:
        model, ae_history = train_autoencoder(
            X_train, X_val, k, "one_hidden",
            epochs=args.ae_epochs, batch_size=args.batch_size, seed=args.seed
        )

        model_dir = ensure_dir(out / "autoencoders" / f"k{k}")
        torch.save(model.state_dict(), model_dir / "model.pt")
        save_json({"architecture": "one_hidden", "bottleneck": k},
                  model_dir / "model_meta.json")
        save_loss_curve(ae_history, model_dir / "ae_loss.png",
                        f"Task 3: 1-hidden AE training, k={k}")

        Ztr = encode(model, X_train)
        Zv = encode(model, X_val)
        Zte = encode(model, X_test)
        np.savez_compressed(
            model_dir / "latent_data.npz",
            Z_train=Ztr, Z_val=Zv, Z_test=Zte,
            y_train=ytr, y_val=yv, y_test=yte
        )

        best_arch = None
        best_val = -1
        best_model = None

        for arch_name, hidden in FCNN_ARCHITECTURES.items():
            clf, history = train_classifier(
                Ztr, ytr, Zv, yv, k, len(DEFAULT_CLASSES),
                hidden, epochs=args.clf_epochs, batch_size=args.batch_size,
                seed=args.seed
            )
            val_pred = predict(clf, Zv)
            val_acc = accuracy(yv, val_pred)

            save_accuracy_curve(
                history, out / f"k{k}_{arch_name}_accuracy.png",
                f"Task 3: k={k}, {arch_name}"
            )
            rows.append({
                "bottleneck": k,
                "architecture": arch_name,
                "validation_accuracy": val_acc,
            })

            if val_acc > best_val:
                best_val = val_acc
                best_arch = arch_name
                best_model = clf

        test_pred = predict(best_model, Zte)
        test_acc = accuracy(yte, test_pred)

        plot_confusion(
            yte, test_pred, list(range(len(DEFAULT_CLASSES))),
            out / f"k{k}_confusion_matrix.png",
            f"Task 3: Test confusion matrix - k={k}, {best_arch}"
        )

        result = {
            "bottleneck": k,
            "best_architecture": best_arch,
            "validation_accuracy": best_val,
            "test_accuracy": test_acc,
        }
        save_json(result, out / f"k{k}_best.json")

        if best_overall is None or test_acc > best_overall["test_accuracy"]:
            best_overall = result

    save_csv(rows, out / "fcnn_validation_results.csv")
    save_json(best_overall, out / "best_overall.json")

    k = best_overall["bottleneck"]
    latent = np.load(out / "autoencoders" / f"k{k}" / "latent_data.npz")
    Z = latent["Z_test"]
    Z2 = SKPCA(n_components=2, random_state=args.seed).fit_transform(Z) if Z.shape[1] > 2 else Z

    fig = plt.figure(figsize=(7, 5.5))
    for i, cls in enumerate(DEFAULT_CLASSES):
        mask = yte == i
        plt.scatter(Z2[mask, 0], Z2[mask, 1], s=10, alpha=0.65, label=str(cls))
    plt.xlabel("Latent dimension 1 / PCA-1")
    plt.ylabel("Latent dimension 2 / PCA-2")
    plt.title(f"Task 3: Best latent representation (k={k})")
    plt.legend(title="Digit")
    plt.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "best_latent_2d.png", dpi=180)
    plt.close(fig)

    print("\nTask 3 complete.")
    print(best_overall)


if __name__ == "__main__":
    main()
