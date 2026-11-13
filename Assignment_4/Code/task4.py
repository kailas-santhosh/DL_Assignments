"""
TASK 4 - Classification using the compressed representation from a
2-hidden-layer autoencoder.

Literal Task-4 architecture:
    784 -> 400 -> bottleneck -> 784

Task 2 separately implements the explicitly specified 3-hidden architecture:
    784 -> 400 -> bottleneck -> 400 -> 784

Run:
    python task4.py --data-root Group_24
"""

import argparse
import numpy as np
import torch

from common import (
    DEFAULT_CLASSES, DEFAULT_DATA_ROOT, accuracy, ensure_dir, load_dataset,
    make_loader, plot_confusion, save_accuracy_curve, save_csv, save_json, save_loss_curve, set_seed,
)
from task2 import train_autoencoder
from task3 import encode
from task1 import train_classifier, predict


FCNN_ARCHITECTURES = {
    "FCNN_1": [128],
    "FCNN_2": [256, 128],
    "FCNN_3": [256, 128, 64],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ae-epochs", type=int, default=40)
    parser.add_argument("--clf-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = ensure_dir("outputs/task4")
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
        ae, ae_history = train_autoencoder(
            X_train, X_val, k, "two_hidden",
            epochs=args.ae_epochs, batch_size=args.batch_size, seed=args.seed
        )

        model_dir = ensure_dir(out / "autoencoders" / f"k{k}")
        torch.save(ae.state_dict(), model_dir / "model.pt")
        save_json({"architecture": "two_hidden", "bottleneck": k},
                  model_dir / "model_meta.json")
        save_loss_curve(ae_history, model_dir / "ae_loss.png",
                        f"Task 4: 2-hidden AE, k={k}")

        Ztr = encode(ae, X_train)
        Zv = encode(ae, X_val)
        Zte = encode(ae, X_test)
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
            save_accuracy_curve(history, out / f"k{k}_{arch_name}_accuracy.png",
                                f"Task 4: k={k}, {arch_name}")
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
            f"Task 4: Test confusion matrix - k={k}, {best_arch}"
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
    print("\nTask 4 complete.")
    print(best_overall)


if __name__ == "__main__":
    main()
