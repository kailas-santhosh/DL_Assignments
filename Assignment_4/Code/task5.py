"""TASK 5 - 20% and 40% denoising autoencoders."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch

from common import (DEFAULT_CLASSES, DEFAULT_DATA_ROOT, accuracy, ensure_dir, load_dataset,
                    make_loader, plot_confusion, save_accuracy_curve, save_csv, save_json, save_loss_curve,
                    save_reconstruction_grid, set_seed)
from models import Autoencoder
from task3 import encode
from task1 import train_classifier, predict, FCNN_ARCHITECTURES


def add_noise(X, fraction, seed=42):
    rng = np.random.default_rng(seed)
    out = X.copy()
    mask = rng.random(X.shape) < fraction
    salt = rng.random(X.shape) < .5
    out[mask] = salt[mask].astype(np.float32)
    return out


def train_dae(X, Xv, k, fraction, epochs, batch_size, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(k, "one_hidden").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Create one fixed noisy input for each sample, so epoch-to-epoch loss
    # comparisons measure learning rather than changes in sampled noise.
    xn, xvn = add_noise(X, fraction, seed), add_noise(Xv, fraction, seed + 1)
    train_ds = torch.utils.data.TensorDataset(torch.tensor(xn), torch.tensor(X))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(xvn), torch.tensor(Xv))
    tr = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    tr_eval = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    va = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    hist = {"train": [], "val": []}

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for xb, target in loader:
                xb, target = xb.to(device), target.to(device)
                loss = loss_fn(model(xb), target)
                total_loss += loss.item() * len(xb)
                total_n += len(xb)
        if total_n == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        return total_loss / total_n

    # Epoch 0: evaluate the random initialization before training begins.
    hist["train"].append(evaluate(tr_eval))
    hist["val"].append(evaluate(va))

    for _ in range(epochs):
        model.train()
        for xb, target in tr:
            xb, target = xb.to(device), target.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), target)
            loss.backward()
            opt.step()

        # Evaluate both splits with the same post-epoch model parameters.
        hist["train"].append(evaluate(tr_eval))
        hist["val"].append(evaluate(va))

    return model, hist


def recon(model, X):
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for (xb,) in make_loader(X, None, 256, False):
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--clf-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    out = ensure_dir("outputs/task5")

    best_file = Path("outputs/task3/best_overall.json")
    if not best_file.exists():
        raise FileNotFoundError("Run task3.py first. Task 5 must use its best bottleneck.")
    with open(best_file, encoding="utf-8") as f:
        best_k = int(json.load(f)["bottleneck"])
    with open(f"outputs/task3/k{best_k}_best.json", encoding="utf-8") as f:
        required_arch = json.load(f)["best_architecture"]

    data = load_dataset(args.data_root, DEFAULT_CLASSES)
    X, y = data["train"]; Xv, yv = data["val"]; Xt, yt = data["test"]
    mp = {c:i for i,c in enumerate(DEFAULT_CLASSES)}
    yi = {name: np.array([mp[v] for v in labels]) for name, labels in [("tr",y),("v",yv),("te",yt)]}
    rows = []

    for fraction in (.20, .40):
        pct = int(fraction * 100)
        model, hist = train_dae(X, Xv, best_k, fraction, args.epochs, args.batch_size, args.seed)
        folder = ensure_dir(out / f"noise_{pct}")
        torch.save(model.state_dict(), folder / "model.pt")
        save_json({"bottleneck":best_k, "noise_fraction":fraction, "architecture":"one_hidden_denoising_ae"}, folder / "model_meta.json")
        save_loss_curve(hist, folder / "loss_curve.png", f"Task 5: {pct}% noise")

        rtr, rv, rte = recon(model, X), recon(model, Xv), recon(model, Xt)
        trm, vm, tem = [float(np.mean((a-b)**2)) for a,b in [(X,rtr),(Xv,rv),(Xt,rte)]]
        save_reconstruction_grid(X, rtr, y, folder/"train_original_vs_reconstructed.png", f"Task 5: {pct}% noise - Train")
        save_reconstruction_grid(Xv, rv, yv, folder/"val_original_vs_reconstructed.png", f"Task 5: {pct}% noise - Validation")
        save_reconstruction_grid(Xt, rte, yt, folder/"test_original_vs_reconstructed.png", f"Task 5: {pct}% noise - Test")
        save_reconstruction_grid(add_noise(Xt, fraction, args.seed+10), rte, yt, folder/"test_noisy_vs_reconstructed.png", f"Task 5: {pct}% noisy input vs reconstruction")

        Ztr, Zv, Zte = encode(model, X), encode(model, Xv), encode(model, Xt)
        np.savez_compressed(folder/"latent_data.npz", Z_train=Ztr, Z_val=Zv, Z_test=Zte,
                            y_train=yi["tr"], y_val=yi["v"], y_test=yi["te"])

        for arch, hidden in FCNN_ARCHITECTURES.items():
            clf, ch = train_classifier(Ztr, yi["tr"], Zv, yi["v"], best_k, len(DEFAULT_CLASSES), hidden,
                                       epochs=args.clf_epochs, batch_size=args.batch_size, seed=args.seed)
            vp, tp = predict(clf, Zv), predict(clf, Zte)
            va, ta = accuracy(yi["v"], vp), accuracy(yi["te"], tp)
            selected = arch == required_arch
            save_accuracy_curve(ch, folder/f"{arch}_accuracy.png", f"Task 5: {pct}% noise - {arch}")
            plot_confusion(yi["te"], tp, list(range(5)), folder/f"{arch}_confusion_matrix.png", f"Task 5: {pct}% noise - {arch}")
            rows.append({"noise_percent":pct,"bottleneck":best_k,"fcnn_architecture":arch,
                         "selected_by_task3":selected,"train_mse":trm,"validation_mse":vm,"test_mse":tem,
                         "validation_accuracy":va,"test_accuracy":ta})

    save_csv(rows, out/"denoising_results_all_fcnn.csv")
    save_csv([r for r in rows if r["selected_by_task3"]], out/"denoising_results_required_architecture.csv")
    save_json({"bottleneck":best_k,"required_fcnn_architecture":required_arch}, out/"summary.json")
    print(f"Task 5 complete. Bottleneck={best_k}, Task-3-selected FCNN={required_arch}")


if __name__ == "__main__":
    main()
