"""
generate_results.py

Recreates all results WITHOUT retraining.

Uses

Results/
    logs/training_log.csv
    histories/*.pkl
    checkpoints/*.weights.h5

Then

✓ selects best architecture
✓ loads checkpoint
✓ evaluates
✓ regenerates every plot
"""

import pickle
import pandas as pd

from config import *

from dataset import (
    get_datasets,
)

from model import create_model

from utils import get_optimizer

from evaluator import (
    evaluate_best_model,
    print_evaluation,
)

from plotting import (
    plot_loss_curves,
    plot_epoch_comparison,
    plot_training_accuracy,
    plot_validation_accuracy,
    plot_test_accuracy,
    plot_architecture_comparison,
    plot_confusion_matrix,
)


# =============================================================================
# Load histories
# =============================================================================

def load_all_results():

    log = pd.read_csv(TRAINING_LOG)

    # Remove duplicate rows if any
    log = log.drop_duplicates(
        subset=["Architecture", "Optimizer"],
        keep="first",
    )

    all_results = {}

    for architecture in ARCHITECTURES.keys():

        all_results[architecture] = {
            "results": {}
        }

        rows = log[
            log["Architecture"] == architecture
        ]

        for _, row in rows.iterrows():

            optimizer = row["Optimizer"]

            history_file = (
                HISTORIES_DIR /
                f"{architecture}_{optimizer}.pkl"
            )

            with open(history_file, "rb") as f:
                history = pickle.load(f)

            # --------------------------------------------------
            # Ensure epoch-0 exists for plotting
            # --------------------------------------------------

            if "loss" in history and len(history["loss"]) > 0:
                history["loss"] = [
                    history["loss"][0]
                ] + history["loss"]

            if "accuracy" in history:
                history["accuracy"] = [
                    0.0
                ] + history["accuracy"]

            if "val_accuracy" in history:
                history["val_accuracy"] = [
                    0.0
                ] + history["val_accuracy"]

            train_acc = float(row["Training Accuracy"])
            val_acc   = float(row["Validation Accuracy"])
            test_acc  = float(row["Test Accuracy"])

            # In case CSV accidentally contains percentages
            if train_acc > 1:
                train_acc /= 100.0
            if val_acc > 1:
                val_acc /= 100.0
            if test_acc > 1:
                test_acc /= 100.0

            all_results[architecture]["results"][optimizer] = {

                "history": history,

                "epochs": int(row["Epochs"]),

                "train_accuracy": train_acc,

                "validation_accuracy": val_acc,

                "test_accuracy": test_acc,

                "train_loss":
                    float(row["Training Loss"]),
            }

        # --------------------------------------------------
        # Best optimizer for this architecture
        # --------------------------------------------------

        best_optimizer = max(
            all_results[architecture]["results"],
            key=lambda opt:
                all_results[architecture]["results"][opt]["validation_accuracy"]
        )

        all_results[architecture]["best_optimizer"] = best_optimizer

        all_results[architecture]["best_result"] = (
            all_results[architecture]["results"][best_optimizer]
        )

    return all_results, log


# =============================================================================
# Best model from CSV
# =============================================================================

def get_best_model_from_log(log):

    best = log.loc[
        log["Validation Accuracy"].idxmax()
    ]

    architecture = best["Architecture"]

    optimizer = best["Optimizer"]

    print()

    print(f"Best Architecture : {architecture}")

    print(f"Best Optimizer    : {optimizer}")

    return architecture, optimizer


# =============================================================================
# Load checkpoint
# =============================================================================

def load_best_model(
    architecture,
    optimizer,
    train_size,
):

    optimizer_object = get_optimizer(
        optimizer,
        train_size,
    )[0]

    model = create_model(
        architecture,
        optimizer_object,
    )

    checkpoint = (
        CHECKPOINTS_DIR /
        f"{architecture}_{optimizer}.weights.h5"
    )

    model.load_weights(checkpoint)

    return model


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)

    print("LOADING PREVIOUS RESULTS")

    print("=" * 80)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_size,
    ) = get_datasets()

    #
    # Read histories + CSV
    #

    all_results, log = load_all_results()

    #
    # Best model
    #

    architecture, optimizer = get_best_model_from_log(log)

    #
    # Load checkpoint
    #

    model = load_best_model(

        architecture,

        optimizer,

        train_size,
    )

    #
    # Evaluate
    #

    evaluation_results = evaluate_best_model(

        model,

        train_dataset,

        validation_dataset,

        test_dataset,
    )

    print_evaluation(
        evaluation_results
    )

    #
    # Fake experiment_results
    #

    # ============================================================
    # Generate all figures
    # ============================================================

    plot_loss_curves(all_results)

    plot_epoch_comparison(all_results)

    plot_training_accuracy(all_results)

    plot_validation_accuracy(all_results)

    plot_test_accuracy(all_results)

    plot_architecture_comparison(all_results)

    plot_confusion_matrix(
        evaluation_results["train_confusion_matrix"],
        "Training Confusion Matrix",
        "train_confusion_matrix.png",
    )

    plot_confusion_matrix(
        evaluation_results["validation_confusion_matrix"],
        "Validation Confusion Matrix",
        "validation_confusion_matrix.png",
    )

    plot_confusion_matrix(
        evaluation_results["test_confusion_matrix"],
        "Test Confusion Matrix",
        "test_confusion_matrix.png",
    )

    print("\n✓ All plots generated successfully.")

    print()

    print("=" * 80)

    print("RESULTS GENERATED SUCCESSFULLY")

    print("=" * 80)


# =============================================================================

if __name__ == "__main__":

    main()
