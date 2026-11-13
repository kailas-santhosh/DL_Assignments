"""
trainer.py

FINAL VERSION - PART 1

Implements:
- Custom stopping criterion
- Training engine
- Identical initialization
- History collection
- Checkpoint saving
"""

import os
import time
import pickle
import numpy as np
import tensorflow as tf

from config import *

from model import create_model

from dataset import get_dataset_with_batch

from utils import (
    get_optimizer,
    append_training_log,
)


# =============================================================================
# Callback : Assignment Stopping Criterion
# =============================================================================

class ConvergenceCallback(tf.keras.callbacks.Callback):
    """
    Stop when

    | AvgLoss(i)-AvgLoss(i-1) | < threshold
    """

    def __init__(self, threshold=STOPPING_THRESHOLD):

        super().__init__()

        self.threshold = threshold

        self.previous_loss = None

        self.converged_epoch = 0

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        current_loss = float(logs["loss"])

        if self.previous_loss is not None:

            difference = abs(current_loss - self.previous_loss)

            if difference < self.threshold:

                print(
                    f"\nConverged at epoch {epoch + 1} "
                    f"(ΔLoss = {difference:.6f})"
                )

                self.converged_epoch = epoch + 1

                self.model.stop_training = True

        self.previous_loss = current_loss


# =============================================================================
# Callback : Collect History
# =============================================================================

class HistoryCollector(tf.keras.callbacks.Callback):

    def __init__(self):

        super().__init__()

        self.loss = []

        self.train_accuracy = []

        self.validation_accuracy = []

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        self.loss.append(float(logs["loss"]))

        self.train_accuracy.append(float(logs["accuracy"]))

        self.validation_accuracy.append(float(logs["val_accuracy"]))


# =============================================================================
# Callback : Progress Printer
# =============================================================================

class ProgressPrinter(tf.keras.callbacks.Callback):
    """
    Print training progress every N epochs.
    """

    def __init__(self, print_every=50):
        super().__init__()
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        if (epoch + 1) <= 10 or (epoch + 1) % self.print_every == 0:

            print(
                f"Epoch {epoch+1:5d} | "
                f"Loss: {logs['loss']:.6f} | "
                f"Train Acc: {logs['accuracy']:.4f} | "
                f"Val Acc: {logs['val_accuracy']:.4f}"
            )


# =============================================================================
# Trainer
# =============================================================================

class Trainer:

    def __init__(
        self,
        architecture_name,
        optimizer_name,
        train_dataset,
        validation_dataset,
        test_dataset,
        train_size,
        initial_weights,
    ):

        self.architecture_name = architecture_name

        self.optimizer_name = optimizer_name

        self.train_size = train_size

        self.optimizer, self.batch_size = get_optimizer(
            optimizer_name,
            train_size,
        )

        self.train_dataset = get_dataset_with_batch(
            train_dataset,
            self.batch_size,
        )

        self.validation_dataset = get_dataset_with_batch(
            validation_dataset,
            self.batch_size,
        )

        self.test_dataset = get_dataset_with_batch(
            test_dataset,
            self.batch_size,
        )

        self.model = create_model(
            architecture_name,
            self.optimizer,
        )

        #
        # Shared initial weights supplied by run_architecture()
        #

        self.initial_weights = initial_weights

    # =========================================================================

    def reset_model(self):

        self.model.set_weights(self.initial_weights)

    # =========================================================================

    def train(self):

        #
        # Required by assignment
        #

        self.reset_model()

        # ============================================================
        # Evaluate initial model (Epoch 0)
        # Required so every optimizer starts from the same point
        # ============================================================

        initial_loss, initial_accuracy = self.model.evaluate(
            self.train_dataset,
            verbose=0,
        )

        initial_val_loss, initial_val_accuracy = self.model.evaluate(
            self.validation_dataset,
            verbose=0,
        )

        convergence = ConvergenceCallback()

        history_callback = HistoryCollector()

        start = time.time()

        history = self.model.fit(

            self.train_dataset,

            validation_data=self.validation_dataset,

            epochs=MAX_EPOCHS,

            verbose=0,

            shuffle=False,

            callbacks=[
                convergence,
                history_callback,
                ProgressPrinter(print_every=50),
            ],
        )

        training_time = time.time() - start

        #
        # Evaluate
        #

        train_loss = history_callback.loss[-1]

        train_accuracy = history_callback.train_accuracy[-1]

        validation_accuracy = history_callback.validation_accuracy[-1]

        validation_loss = history.history["val_loss"][-1]

        test_loss, test_accuracy = self.model.evaluate(
            self.test_dataset,
            verbose=0,
        )

        #
        # Save history
        #

        history_dict = {

            "loss":
                [float(initial_loss)] +
                history_callback.loss,

            "train_accuracy":
                [float(initial_accuracy)] +
                history_callback.train_accuracy,

            "validation_accuracy":
                [float(initial_val_accuracy)] +
                history_callback.validation_accuracy,

            "validation_loss":
                [float(initial_val_loss)] +
                list(history.history["val_loss"]),
        }

        history_file = (
            HISTORIES_DIR
            /
            f"{self.architecture_name}_{self.optimizer_name}.pkl"
        )

        with open(history_file, "wb") as f:

            pickle.dump(history_dict, f)

        #
        # Save checkpoint
        #

        checkpoint = (
            CHECKPOINTS_DIR
            /
            f"{self.architecture_name}_{self.optimizer_name}.weights.h5"
        )

        self.model.save_weights(checkpoint)

        #
        # CSV log
        #

        append_training_log(

            architecture=self.architecture_name,

            optimizer=self.optimizer_name,

            epochs=len(history_callback.loss),

            train_loss=train_loss,

            train_acc=train_accuracy,

            val_acc=validation_accuracy,

            test_acc=test_accuracy,
        )

        #
        # Result dictionary
        #

        result = {

            "architecture": self.architecture_name,

            "optimizer": self.optimizer_name,

            "model": self.model,

            "epochs": len(history_callback.loss),

            "history": history_dict,

            "train_loss": float(train_loss),

            "validation_loss": float(validation_loss),

            "test_loss": float(test_loss),

            "train_accuracy": float(train_accuracy),

            "validation_accuracy": float(validation_accuracy),

            "test_accuracy": float(test_accuracy),

            "training_time": float(training_time),

            "checkpoint": checkpoint,

            "history_file": history_file,
        }

        return result


# =============================================================================
# Run All Optimizers for One Architecture
# =============================================================================

def run_architecture(
    architecture_name,
    train_dataset,
    validation_dataset,
    test_dataset,
    train_size,
):
    """
    Train one architecture using all optimizers.

    Returns
    -------
    dict
        {
            "architecture": ...,
            "results": {...},
            "best_result": ...
        }
    """

    architecture_results = {}

    # ---------------------------------------------------------
    # Build one reference model and save its initial weights.
    # All optimizers will start from these exact weights.
    # ---------------------------------------------------------

    reference_optimizer = get_optimizer(
        OPTIMIZERS[0],
        train_size,
    )[0]

    reference_model = create_model(
        architecture_name,
        reference_optimizer,
    )

    shared_initial_weights = reference_model.get_weights()

    best_result = None

    best_validation_accuracy = -1.0

    for optimizer_name in OPTIMIZERS:

        print("\n" + "=" * 80)
        print(f"{architecture_name} | {optimizer_name}")
        print("=" * 80)

        trainer = Trainer(

            architecture_name=architecture_name,

            optimizer_name=optimizer_name,

            train_dataset=train_dataset,

            validation_dataset=validation_dataset,

            test_dataset=test_dataset,

            train_size=train_size,

            initial_weights=shared_initial_weights,
        )

        result = trainer.train()

        architecture_results[optimizer_name] = result

        if result["validation_accuracy"] > best_validation_accuracy:

            best_validation_accuracy = result["validation_accuracy"]

            best_result = result

    return {

        "architecture": architecture_name,

        "results": architecture_results,

        "best_result": best_result,
    }


# =============================================================================
# Run Entire Assignment
# =============================================================================

def run_all_experiments(
    train_dataset,
    validation_dataset,
    test_dataset,
    train_size,
):
    """
    Executes

        3 Architectures × 7 Optimizers

    Returns complete experiment dictionary.
    """

    all_results = {}

    overall_best = None

    best_validation_accuracy = -1.0

    for architecture_name in ARCHITECTURES.keys():

        architecture_result = run_architecture(

            architecture_name,

            train_dataset,

            validation_dataset,

            test_dataset,

            train_size,
        )

        all_results[architecture_name] = architecture_result

        if (
            architecture_result["best_result"]["validation_accuracy"]
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                architecture_result["best_result"]["validation_accuracy"]
            )

            overall_best = architecture_result["best_result"]

    return {

        "all_results": all_results,

        "best_model": overall_best,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_best_model(experiment_results):
    """
    Returns trained best model.
    """

    return experiment_results["best_model"]["model"]


def get_best_result(experiment_results):
    """
    Returns best result dictionary.
    """

    return experiment_results["best_model"]


def get_all_histories(experiment_results):
    """
    Collect histories from every experiment.
    """

    histories = {}

    for architecture_name, architecture_data in experiment_results[
        "all_results"
    ].items():

        histories[architecture_name] = {}

        for optimizer_name, result in architecture_data["results"].items():

            histories[architecture_name][optimizer_name] = result["history"]

    return histories


# =============================================================================
# Console Summary
# =============================================================================

def print_summary(experiment_results):
    """
    Prints complete experiment summary.
    """

    print("\n")
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    for architecture_name in ARCHITECTURES.keys():

        print(f"\n{architecture_name}")

        print("-" * 100)

        print(
            "{:<12} {:>10} {:>14} {:>14} {:>14}".format(
                "Optimizer",
                "Epochs",
                "Train Acc",
                "Val Acc",
                "Test Acc",
            )
        )

        print("-" * 100)

        for optimizer_name in OPTIMIZERS:

            result = experiment_results["all_results"][
                architecture_name
            ]["results"][optimizer_name]

            print(
                "{:<12} {:>10} {:>14.4f} {:>14.4f} {:>14.4f}".format(
                    optimizer_name,
                    result["epochs"],
                    result["train_accuracy"],
                    result["validation_accuracy"],
                    result["test_accuracy"],
                )
            )

    print("\n")
    print("=" * 100)
    print("BEST MODEL")
    print("=" * 100)

    best = experiment_results["best_model"]

    print(f"Architecture         : {best['architecture']}")
    print(f"Optimizer            : {best['optimizer']}")
    print(f"Epochs               : {best['epochs']}")
    print(f"Training Accuracy    : {best['train_accuracy']:.4f}")
    print(f"Validation Accuracy  : {best['validation_accuracy']:.4f}")
    print(f"Test Accuracy        : {best['test_accuracy']:.4f}")

    print("=" * 100)
