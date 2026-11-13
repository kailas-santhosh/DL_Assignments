import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    CONVERGENCE_THRESHOLD,
    SEED,
)
from models import SequenceClassifier


# ---------------------------------------------------------------------------
# Dataset / collate / loaders
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = sequences, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            int(self.labels[idx]),
        )


def collate_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def make_loader(sequences, labels, shuffle=False, seed=None):
    """
    DataLoader factory.

    If shuffle=True and seed is provided, an explicit seeded generator is
    attached so batch ordering is reproducible regardless of prior RNG use
    (checklist item #5). num_workers is fixed to 0 to keep the seeded
    generator behavior deterministic.
    """
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        SequenceDataset(sequences, labels),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_sequences,
        num_workers=0,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    loss = (total_loss / total) if criterion is not None else None
    acc = correct / total if total else 0.0
    return loss, acc, np.asarray(y_true), np.asarray(y_pred)


# ---------------------------------------------------------------------------
# Confusion matrix / plotting
# ---------------------------------------------------------------------------
def make_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1
    return cm


def save_loss_plot(losses, title, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    if losses:
        plt.plot(range(1, len(losses) + 1), losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average training error (cross-entropy loss)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_plot(cm, classes, title, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, labels=classes, rotation=35, ha="right")
    ax.set_yticks(ticks, labels=classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def train_experiment(
    kind,
    config,
    train_x, train_y,
    val_x,   val_y,
    test_x,  test_y,
    classes,
    results_root,
    plots_root,
    device,
    max_epochs=MAX_EPOCHS,
):
    model_type = config["model"]
    hidden     = config["hidden_size"]
    layers     = config["num_layers"]
    run_name   = f"{model_type}_H{hidden}_L{layers}"

    result_dir = Path(results_root) / kind / run_name
    plot_dir   = Path(plots_root)   / kind / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Seeded training loader; other loaders do not shuffle.
    train_loader      = make_loader(train_x, train_y, shuffle=True, seed=SEED)
    train_eval_loader = make_loader(train_x, train_y)   # full-set post-epoch eval
    val_loader        = make_loader(val_x, val_y)
    test_loader       = make_loader(test_x, test_y)

    input_size = train_x[0].shape[1]
    model = SequenceClassifier(
        input_size, hidden, layers, len(classes), model_type
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------
    losses = []

    # Successive-epoch convergence: compare average training losses of two
    # consecutive epochs (post-epoch, full-training-set measurement).
    previous_epoch_loss = None
    converged = False
    reason = "maximum safety epoch limit reached"

    # Best-validation checkpoint tracking.
    best_val_accuracy = -1.0
    best_val_epoch = 0
    best_checkpoint_training_loss = float("inf")
    best_training_loss = float("inf")           # min post-epoch train loss seen
    best_model_state = None                     # snapshot of best-val weights

    for epoch in range(1, max_epochs + 1):
        # --- parameter updates on shuffled minibatches ---
        model.train()
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # --- post-epoch average training loss over the FULL training set ---
        epoch_train_loss, _, _, _ = evaluate(
            model, train_eval_loader, device, criterion
        )
        losses.append(float(epoch_train_loss))
        if epoch_train_loss < best_training_loss:
            best_training_loss = epoch_train_loss

        # --- post-epoch validation accuracy (for best-checkpoint tracking) ---
        _, epoch_val_acc, _, _ = evaluate(
            model, val_loader, device, criterion
        )
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            best_val_epoch = epoch
            best_checkpoint_training_loss = epoch_train_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # --- successive-epoch convergence check ---
        if previous_epoch_loss is not None:
            if abs(previous_epoch_loss - epoch_train_loss) < CONVERGENCE_THRESHOLD:
                converged = True
                reason = (
                    f"successive average training losses differed by less "
                    f"than {CONVERGENCE_THRESHOLD}"
                )
                break
        previous_epoch_loss = epoch_train_loss

    # -------------------------------------------------------------------
    # Restore best-validation checkpoint before final evaluation
    # -------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, train_acc, _, _ = evaluate(model, train_eval_loader, device, criterion)
    _, val_acc,   _, _ = evaluate(model, val_loader,       device, criterion)
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, device, criterion
    )
    cm = make_confusion_matrix(y_true, y_pred, len(classes))

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
    save_loss_plot(
        losses,
        f"{kind}: {run_name} — post-epoch average training loss",
        plot_dir / "training_loss.png",
    )
    # Save restored best-validation model
    torch.save(
        model.state_dict(),
        result_dir / "model_state.pt"
    )

    result = {
        "dataset": kind,
        "model": model_type,
        "hidden_size": hidden,
        "num_layers": layers,
        "epochs": len(losses),
        "converged": converged,
        "stopping_reason": reason,
        "final_training_loss": float(losses[-1]),
        "best_training_loss": float(best_training_loss),
        "best_validation_epoch": int(best_val_epoch),
        "best_validation_accuracy": float(best_val_accuracy),
        "best_checkpoint_training_loss": float(
            best_checkpoint_training_loss
        ),
        "training_accuracy": float(train_acc),
        "validation_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "class_names": list(classes),
        "confusion_matrix": cm.tolist(),
        "losses": [float(v) for v in losses],
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


# ---------------------------------------------------------------------------
# Aggregated per-dataset results
# ---------------------------------------------------------------------------
def save_dataset_results(kind, results, classes, results_root, plots_root):
    dataset_result_dir = Path(results_root) / kind
    dataset_plot_dir   = Path(plots_root)   / kind
    best_dir           = dataset_result_dir / "best_models"
    best_plot_dir      = dataset_plot_dir   / "best_models"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_plot_dir.mkdir(parents=True, exist_ok=True)

    for model_type in ("RNN", "LSTM"):
        candidates = [r for r in results if r["model"] == model_type]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda r: (
                r["validation_accuracy"],
                -r["best_checkpoint_training_loss"],
            ),
        )
        cm = np.asarray(best["confusion_matrix"], dtype=np.int64)
        save_confusion_plot(
            cm,
            classes,
            f"{kind}: Best {model_type} confusion matrix",
            best_plot_dir / f"best_{model_type.lower()}_confusion_matrix.png",
        )
        (best_dir / f"best_{model_type.lower()}_config.json").write_text(
            json.dumps(best, indent=2)
        )

    fields = [
        "dataset", "model", "hidden_size", "num_layers", "epochs", "converged",
        "training_accuracy", "validation_accuracy", "test_accuracy",
        "final_training_loss", "best_training_loss", "best_checkpoint_training_loss",
        "best_validation_epoch", "best_validation_accuracy", "stopping_reason",
    ]
    with (dataset_result_dir / "experiment_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})

    (dataset_result_dir / "all_results.json").write_text(
        json.dumps(results, indent=2)
    )
