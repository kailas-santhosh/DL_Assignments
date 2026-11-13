"""
TASK 2 - Autoencoders for image reconstruction.

Architectures:
  One hidden:    784 -> k -> 784
  Three hidden:  784 -> 400 -> k -> 400 -> 784

Run:
    python task2.py --data-root Group_24
"""

import argparse
import numpy as np
import torch

from common import (
    DEFAULT_CLASSES, DEFAULT_DATA_ROOT, ensure_dir, load_dataset, make_loader,
    save_csv, save_json, save_loss_curve, save_reconstruction_grid, set_seed,
)
from models import Autoencoder


def train_autoencoder(X_train, X_val, bottleneck_dim, architecture,
                      epochs=40, batch_size=128, lr=1e-3, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(bottleneck_dim, architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_loader = make_loader(X_train, None, batch_size, shuffle=True)
    val_loader = make_loader(X_val, None, batch_size, shuffle=False)
    train_eval_loader = make_loader(X_train, None, batch_size, shuffle=False)
    history = {"train": [], "val": []}

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                loss = criterion(model(xb), xb)
                total_loss += loss.item() * xb.size(0)
                total_n += xb.size(0)
        if total_n == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        return total_loss / total_n

    # Epoch 0 records the untrained model on both splits before any updates.
    history["train"].append(evaluate(train_eval_loader))
    history["val"].append(evaluate(val_loader))

    for _ in range(epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

        # Measure both losses after the epoch with the same updated weights.
        history["train"].append(evaluate(train_eval_loader))
        history["val"].append(evaluate(val_loader))

    return model, history


def reconstruct(model, X, batch_size=256):
    device = next(model.parameters()).device
    loader = make_loader(X, None, batch_size, shuffle=False)
    model.eval()
    outputs = []
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def mse(model, X, batch_size=256):
    recon = reconstruct(model, X, batch_size)
    return float(np.mean((X - recon) ** 2)), recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = ensure_dir("outputs/task2")
    data = load_dataset(args.data_root, DEFAULT_CLASSES)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    rows = []
    architectures = ["one_hidden", "three_hidden"]

    for architecture in architectures:
        for k in [32, 64, 128, 256]:
            model, history = train_autoencoder(
                X_train, X_val, k, architecture,
                epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
            )

            train_err, train_recon = mse(model, X_train)
            val_err, val_recon = mse(model, X_val)
            test_err, test_recon = mse(model, X_test)

            folder = ensure_dir(out / architecture / f"k{k}")
            torch.save(model.state_dict(), folder / "model.pt")
            save_json(
                {"architecture": architecture, "bottleneck": k},
                folder / "model_meta.json"
            )

            save_loss_curve(
                history, folder / "loss_curve.png",
                f"Task 2: {architecture}, bottleneck={k}"
            )
            save_reconstruction_grid(
                X_train, train_recon, y_train,
                folder / "train_reconstructions.png",
                f"Task 2: {architecture}, k={k} - Training"
            )
            save_reconstruction_grid(
                X_val, val_recon, y_val,
                folder / "val_reconstructions.png",
                f"Task 2: {architecture}, k={k} - Validation"
            )
            save_reconstruction_grid(
                X_test, test_recon, y_test,
                folder / "test_reconstructions.png",
                f"Task 2: {architecture}, k={k} - Test"
            )

            rows.append({
                "architecture": architecture,
                "bottleneck": k,
                "train_mse": train_err,
                "validation_mse": val_err,
                "test_mse": test_err,
            })
            print(architecture, k, train_err, val_err, test_err)

    save_csv(rows, out / "reconstruction_errors.csv")

    summary = {}
    for arch in architectures:
        candidates = [r for r in rows if r["architecture"] == arch]
        summary[arch] = min(candidates, key=lambda r: r["validation_mse"])
    save_json(summary, out / "summary.json")

    print("\nTask 2 complete.")
    print(summary)


if __name__ == "__main__":
    main()
