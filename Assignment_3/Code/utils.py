"""
utils.py

Utility functions for reproducibility, optimizer creation,
weight handling, CSV logging, and experiment management.
"""

import os
import csv
import random
import numpy as np
import tensorflow as tf

from config import *


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed=RANDOM_SEED):
    """
    Set all random seeds for reproducible experiments.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)

    tf.config.experimental.enable_op_determinism()


# =============================================================================
# Optimizer Factory
# =============================================================================

def get_optimizer(name, full_batch_size=None):
    """
    Returns the optimizer and corresponding batch size.
    """

    name = name.lower()

    if name == "sgd":
        return (
            tf.keras.optimizers.SGD(
                learning_rate=LEARNING_RATE
            ),
            SGD_BATCH_SIZE,
        )

    elif name == "batchgd":
        return (
            tf.keras.optimizers.SGD(
                learning_rate=LEARNING_RATE
            ),
            full_batch_size,
        )

    elif name == "momentum":
        return (
            tf.keras.optimizers.SGD(
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
            ),
            MOMENTUM_BATCH_SIZE,
        )

    elif name == "nag":
        return (
            tf.keras.optimizers.SGD(
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
                nesterov=True,
            ),
            NAG_BATCH_SIZE,
        )

    elif name == "adagrad":
        return (
            tf.keras.optimizers.Adagrad(
                learning_rate=LEARNING_RATE
            ),
            full_batch_size,
        )

    elif name == "rmsprop":
        return (
            tf.keras.optimizers.RMSprop(
                learning_rate=LEARNING_RATE,
                rho=RMSPROP_RHO,
                epsilon=RMSPROP_EPSILON,
            ),
            full_batch_size,
        )

    elif name == "adam":
        return (
            tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE,
                beta_1=ADAM_BETA_1,
                beta_2=ADAM_BETA_2,
                epsilon=ADAM_EPSILON,
            ),
            ADAM_BATCH_SIZE,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# =============================================================================
# Initial Weight Handling
# =============================================================================

def save_initial_weights(model):
    """
    Save identical initial weights for reuse across optimizers.
    """

    return model.get_weights()


def load_initial_weights(model, weights):
    """
    Reload initial weights before training with another optimizer.
    """

    model.set_weights(weights)


# =============================================================================
# Directory Utilities
# =============================================================================

def ensure_directories():
    """
    Ensure all output folders exist.
    """

    for directory in OUTPUT_DIRS:
        os.makedirs(directory, exist_ok=True)


# =============================================================================
# CSV Utilities
# =============================================================================

def save_csv(filepath, rows, header):
    """
    Save rows to CSV.
    """

    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# =============================================================================
# Training Log
# =============================================================================

def append_training_log(
    architecture,
    optimizer,
    epochs,
    train_loss,
    train_acc,
    val_acc,
    test_acc,
):
    """
    Append one experiment result to training_log.csv.
    """

    file_exists = os.path.isfile(TRAINING_LOG)

    with open(TRAINING_LOG, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow(
                [
                    "Architecture",
                    "Optimizer",
                    "Epochs",
                    "Training Loss",
                    "Training Accuracy",
                    "Validation Accuracy",
                    "Test Accuracy",
                ]
            )

        writer.writerow(
            [
                architecture,
                optimizer,
                epochs,
                train_loss,
                train_acc,
                val_acc,
                test_acc,
            ]
        )


# =============================================================================
# Convergence Check
# =============================================================================

def has_converged(previous_loss, current_loss):
    """
    Assignment stopping criterion.

    Stop if

    |loss_i - loss_(i-1)| < 1e-4
    """

    if previous_loss is None:
        return False

    return abs(previous_loss - current_loss) < STOPPING_THRESHOLD


# =============================================================================
# History Utilities
# =============================================================================

def history_to_dict(history):
    """
    Convert Keras History object to dictionary.
    """

    return {
        "loss": history.history["loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
    }


# =============================================================================
# Model Summary
# =============================================================================

def print_experiment_header(architecture, optimizer):
    """
    Nicely formatted experiment header.
    """

    print("\n" + "=" * 70)
    print(f"Architecture : {architecture}")
    print(f"Optimizer    : {optimizer}")
    print("=" * 70)


# =============================================================================
# Time Formatting
# =============================================================================

def format_time(seconds):
    """
    Convert seconds into HH:MM:SS.
    """

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
