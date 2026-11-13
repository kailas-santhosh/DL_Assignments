"""
plotting.py

Creates all plots required for Assignment 3.

Produces

• Loss curves
• Epoch comparison
• Training accuracy
• Validation accuracy
• Test accuracy
• Architecture comparison
• Confusion matrices

All figures:
    PNG
    400 DPI
"""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    ARCHITECTURES,
    OPTIMIZERS,
    PLOTS_DIR,
    FIGSIZE,
    DPI,
    LINE_WIDTH,
    FONT_SIZE,
    CLASS_NAMES,
)

plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE + 2
plt.rcParams["axes.labelsize"] = FONT_SIZE + 1
plt.rcParams["legend.fontsize"] = FONT_SIZE - 1
plt.rcParams["xtick.labelsize"] = FONT_SIZE - 1
plt.rcParams["ytick.labelsize"] = FONT_SIZE - 1

# ============================================================
# Helper
# ============================================================

def save_figure(filename):

    plt.tight_layout(pad=1.4)

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Helper
# ============================================================

def add_bar_labels(bars):

    ymin, ymax = plt.gca().get_ylim()

    for bar in bars:

        height = bar.get_height()

        # Do not label bars outside the visible axis
        if height < ymin:
            continue

        if height <= 1:
            offset = 0.004
            label = f"{height:.3f}"
        else:
            offset = max(height * 0.01, 1)
            label = str(int(height))

        y = height + offset

        plt.text(
            bar.get_x() + bar.get_width()/2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=True,
        )


def bar_plot(
    labels,
    values,
    xlabel,
    ylabel,
    title,
    filename,
    ylim=None,
    rotation=30,
):

    plt.figure(figsize=(10,7))

    bars = plt.bar(labels, values)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #
    # Automatically choose a nicer y-range
    #
    if ylim is None:

        if max(values) <= 1:
            lower = max(0, min(values) - 0.05)
            margin = max(0.03, (max(values)-min(values))*0.15)
            upper = min(1.05, max(values)+margin)
            plt.ylim(lower, upper)
            plt.margins(y=0.08)

        else:
            plt.ylim(0, max(values) * 1.15)
            plt.margins(y=0.08)

    else:
        plt.ylim(ylim)
        plt.margins(y=0.08)

    plt.title(title)

    plt.xticks(rotation=20, ha="center")

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    add_bar_labels(bars)

    save_figure(filename)


# ============================================================
# Loss Curves
# ============================================================

# ============================================================
# Loss Curves
# ============================================================

def plot_loss_curves(all_results):

    for architecture in ARCHITECTURES:

        # ------------------------------------------------------
        # Find longest history for this architecture
        # ------------------------------------------------------

        max_epochs = max(
            len(
                all_results[architecture]["results"][optimizer]["history"]["loss"]
            )
            for optimizer in OPTIMIZERS
        )

        tick_step = max(1, max_epochs // 10)

        # ======================================================
        # Combined comparison plot
        # ======================================================

        plt.figure(figsize=(10, 6.5))

        for optimizer in OPTIMIZERS:

            history = all_results[architecture]["results"][optimizer]["history"]

            loss = history["loss"]

            epochs = np.arange(len(loss))   # Starts from epoch 0

            plt.plot(
                epochs,
                loss,
                linewidth=LINE_WIDTH,
                label=optimizer,
            )

        plt.xlabel("Epoch")

        plt.ylabel("Average Training Error")

        plt.title(f"{architecture} : Training Error")

        plt.xlim(0, max_epochs - 1)

        plt.xticks(
            np.arange(
                0,
                max_epochs,
                tick_step,
            )
        )

        plt.grid(True, linestyle="--", alpha=0.4)

        plt.legend(
            loc="upper right",
            fontsize=9,
        )

        save_figure(f"{architecture}_loss.png")

        # ======================================================
        # Individual optimizer plots
        # ======================================================

        for optimizer in OPTIMIZERS:

            plt.figure(figsize=(10, 6.5))

            history = all_results[architecture]["results"][optimizer]["history"]

            loss = history["loss"]

            epochs = np.arange(len(loss))

            plt.plot(
                epochs,
                loss,
                linewidth=LINE_WIDTH,
                color="tab:blue",
            )

            tick_step_single = max(1, len(epochs) // 10)

            plt.xlabel("Epoch")

            plt.ylabel("Average Training Error")

            plt.title(
                f"{architecture} : {optimizer} Training Error"
            )

            plt.xlim(0, len(epochs) - 1)

            plt.xticks(
                np.arange(
                    0,
                    len(epochs),
                    tick_step_single,
                )
            )

            plt.grid(True, linestyle="--", alpha=0.4)

            save_figure(
                f"{architecture}_{optimizer}_loss.png"
            )


# ============================================================
# Epoch Comparison
# ============================================================

def plot_epoch_comparison(all_results):

    for architecture in ARCHITECTURES:

        epochs = [
            all_results[architecture]["results"][opt]["epochs"]
            for opt in OPTIMIZERS
        ]

        bar_plot(
            OPTIMIZERS,
            epochs,
            "Optimizer",
            "Epochs",
            f"{architecture} : Convergence Epochs",
            f"{architecture}_epochs.png",
            rotation=25,
        )


# ============================================================
# Training Accuracy
# ============================================================

def plot_training_accuracy(all_results):

    for architecture in ARCHITECTURES:

        values = [
            all_results[architecture]["results"][opt]["train_accuracy"]
            for opt in OPTIMIZERS
        ]

        bar_plot(
            OPTIMIZERS,
            values,
            "Optimizer",
            "Training Accuracy",
            f"{architecture} : Training Accuracy",
            f"{architecture}_training_accuracy.png",
            ylim=(0,1.1),
            rotation=25,
        )

        bar_plot(
            OPTIMIZERS,
            values,
            "Optimizer",
            "Training Accuracy",
            f"{architecture} : Training Accuracy (Zoomed)",
            f"{architecture}_training_accuracy_zoom.png",
            ylim=None,
            rotation=25,
        )


# ============================================================
# Validation Accuracy
# ============================================================

def plot_validation_accuracy(all_results):

    for architecture in ARCHITECTURES:

        values = [
            all_results[architecture]["results"][opt]["validation_accuracy"]
            for opt in OPTIMIZERS
        ]

        bar_plot(
            OPTIMIZERS,
            values,
            "Optimizer",
            "Validation Accuracy",
            f"{architecture} : Validation Accuracy",
            f"{architecture}_validation_accuracy.png",
            ylim=(0,1.1),
            rotation=25,
        )

        bar_plot(
            OPTIMIZERS,
            values,
            "Optimizer",
            "Validation Accuracy",
            f"{architecture} : Validation Accuracy (Zoomed)",
            f"{architecture}_validation_accuracy_zoom.png",
            ylim=None,
            rotation=25,
        )


# ============================================================
# Test Accuracy
# ============================================================

def plot_test_accuracy(all_results):

    for architecture in ARCHITECTURES:

        values = [
            all_results[architecture]["results"][opt]["test_accuracy"]
            for opt in OPTIMIZERS
        ]

        bar_plot(
            OPTIMIZERS,
            values,
            "Optimizer",
            "Test Accuracy",
            f"{architecture} : Test Accuracy",
            f"{architecture}_test_accuracy.png",
            ylim=(0,1.1),
            rotation=25,
        )

        bar_plot(
            OPTIMIZERS,
            values,
            "Optimizer",
            "Test Accuracy",
            f"{architecture} : Test Accuracy (Zoomed)",
            f"{architecture}_test_accuracy_zoom.png",
            ylim=None,
            rotation=25,
        )


# ============================================================
# Architecture Comparison
# ============================================================

def plot_architecture_comparison(all_results):

    architectures = list(ARCHITECTURES.keys())

    accuracies = []
    epochs = []

    for architecture in architectures:

        results = all_results[architecture]["results"]

        best = max(
	    results.values(),
	    key=lambda x: x["validation_accuracy"]
        )

        accuracies.append(best["validation_accuracy"])
        epochs.append(best["epochs"])

    bar_plot(
        architectures,
        accuracies,
        "Architecture",
        "Best Validation Accuracy",
        "Architecture Comparison",
        "architecture_validation_accuracy.png",
        ylim=None,
        rotation=20,
    )

    bar_plot(
        architectures,
        epochs,
        "Architecture",
        "Epochs",
        "Architecture Convergence",
        "architecture_epochs.png",
        rotation=20,
    )


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(confusion,
                          title,
                          filename):

    plt.figure(figsize=(8,7))

    plt.imshow(
        confusion,
        cmap="Blues",
        interpolation="nearest",
    )

    plt.title(title)

    plt.colorbar()

    ticks = np.arange(len(CLASS_NAMES))

    plt.xticks(ticks, CLASS_NAMES, rotation=25)
    plt.yticks(ticks, CLASS_NAMES)

    threshold = confusion.max()/2

    for i in range(confusion.shape[0]):

        for j in range(confusion.shape[1]):

            plt.text(
                j,
                i,
                str(confusion[i,j]),
                ha="center",
                va="center",
                color="white"
                if confusion[i,j] > threshold
                else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    save_figure(filename)


# ============================================================
# Generate Everything
# ============================================================

def generate_all_plots(experiment_results,
                       evaluation_results):

    plot_loss_curves(experiment_results)

    plot_epoch_comparison(experiment_results)

    plot_training_accuracy(experiment_results)

    plot_validation_accuracy(experiment_results)

    plot_test_accuracy(experiment_results)

    plot_architecture_comparison(experiment_results)

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
