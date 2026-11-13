"""
TASK 1 - PCA dimension reduction + FCNN classification.

Run:
    python task1.py --data-root Group_24
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from common import (
    DEFAULT_CLASSES, DEFAULT_DATA_ROOT, accuracy, ensure_dir, load_dataset,
    make_loader, plot_class_distribution, plot_confusion, save_accuracy_curve, save_csv, save_json,
    save_loss_curve, save_sample_grid, set_seed,
)
from models import Classifier


FCNN_ARCHITECTURES = {
    "FCNN_1": [128],
    "FCNN_2": [256, 128],
    "FCNN_3": [256, 128, 64],
}


def train_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes,
                     hidden_layers, epochs=30, batch_size=128, lr=1e-3, seed=42):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(input_dim, num_classes, hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Shuffle only for optimization. Evaluation loaders retain a fixed order and
    # are used to measure both splits with the same model parameters.
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    train_eval_loader = make_loader(X_train, y_train, batch_size, shuffle=False)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    def evaluate(loader):
        model.eval()
        total = 0
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        if total == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        return total_loss / total, correct / total

    history = {
        "train": [],       # training accuracy, measured after the epoch
        "val": [],         # validation accuracy, measured at the same model state
        "train_loss": [],  # training loss, measured after the epoch
        "val_loss": []     # validation loss, measured at the same model state
    }

    # Epoch 0: measure both datasets before any optimizer updates.
    train_loss, train_acc = evaluate(train_eval_loader)
    val_loss, val_acc = evaluate(val_loader)
    history["train_loss"].append(train_loss)
    history["train"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val"].append(val_acc)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # After each epoch, evaluate train and validation sets using the same
        # updated weights. This makes each pair of plotted values comparable.
        train_loss, train_acc = evaluate(train_eval_loader)
        val_loss, val_acc = evaluate(val_loader)
        history["train_loss"].append(train_loss)
        history["train"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val"].append(val_acc)

    return model, history


def predict(model, X, batch_size=256):
    device = next(model.parameters()).device
    loader = make_loader(X, None, batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.asarray(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = ensure_dir("outputs/task1")
    data = load_dataset(args.data_root, DEFAULT_CLASSES)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    label_to_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    y_train_i = np.array([label_to_index[v] for v in y_train])
    y_val_i = np.array([label_to_index[v] for v in y_val])
    y_test_i = np.array([label_to_index[v] for v in y_test])

    save_sample_grid(X_train, y_train, out / "01_training_samples.png",
                     "Task 1: Training samples")
    plot_class_distribution(y_train, out / "02_class_distribution.png",
                            "Task 1: Training class distribution")

    pca_full = PCA(n_components=256, svd_solver="randomized", random_state=args.seed)
    X_train_256 = pca_full.fit_transform(X_train)
    X_val_256 = pca_full.transform(X_val)
    X_test_256 = pca_full.transform(X_test)

    fig = plt.figure(figsize=(8, 4.5))
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(np.arange(1, 257), cumulative)
    for k in [32, 64, 128, 256]:
        plt.axvline(k, linestyle="--", alpha=0.5)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Task 1: PCA cumulative explained variance")
    plt.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "03_pca_explained_variance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca_full.components_[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"PC {i+1}")
        ax.axis("off")
    fig.suptitle("Task 1: First 16 PCA directions")
    fig.tight_layout()
    fig.savefig(out / "04_pca_directions.png", dpi=180)
    plt.close(fig)

    all_rows = []
    best_by_k = {}

    for k in [32, 64, 128, 256]:
        Xtr = X_train_256[:, :k]
        Xv = X_val_256[:, :k]
        Xte = X_test_256[:, :k]

        best_model = None
        best_arch = None
        best_val = -1

        for arch_name, hidden in FCNN_ARCHITECTURES.items():
            model, history = train_classifier(
                Xtr, y_train_i, Xv, y_val_i, k, len(DEFAULT_CLASSES),
                hidden, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
            )
            val_pred = predict(model, Xv)
            val_acc = accuracy(y_val_i, val_pred)

            save_accuracy_curve(
                {"train": history["train"], "val": history["val"]},
                out / f"k{k}_{arch_name}_accuracy.png",
                f"Task 1: PCA-{k}, {arch_name}"
            )

            save_loss_curve(
                {
                    "train": history["train_loss"],
                    "val": history["val_loss"]
                },
                out / f"k{k}_{arch_name}_loss.png",
                f"Task 1: PCA-{k}, {arch_name} - Loss"
            )
            all_rows.append({
                "dimension": k,
                "architecture": arch_name,
                "validation_accuracy": val_acc,
            })

            if val_acc > best_val:
                best_val = val_acc
                best_model = model
                best_arch = arch_name

        test_pred = predict(best_model, Xte)
        test_acc = accuracy(y_test_i, test_pred)

        best_by_k[str(k)] = {
            "best_architecture": best_arch,
            "validation_accuracy": best_val,
            "test_accuracy": test_acc,
        }

        plot_confusion(
            y_test_i, test_pred, list(range(len(DEFAULT_CLASSES))),
            out / f"k{k}_confusion_matrix.png",
            f"Task 1: Test confusion matrix - PCA {k}, {best_arch}"
        )
        save_json(best_by_k[str(k)], out / f"k{k}_best.json")

    save_csv(all_rows, out / "fcnn_validation_results.csv")
    save_json(best_by_k, out / "best_results_by_dimension.json")

    best_k = max(best_by_k, key=lambda k: best_by_k[k]["test_accuracy"])
    summary = {
        "best_pca_dimension": int(best_k),
        "best_architecture": best_by_k[best_k]["best_architecture"],
        "best_validation_accuracy": best_by_k[best_k]["validation_accuracy"],
        "best_test_accuracy": best_by_k[best_k]["test_accuracy"],
        "assignment_3_best_accuracy": None,
        "assignment_3_note": "Enter the best Assignment-3 test accuracy manually in the report.",
    }
    save_json(summary, out / "summary.json")
    print("\nTask 1 complete.")
    print(summary)


if __name__ == "__main__":
    main()
