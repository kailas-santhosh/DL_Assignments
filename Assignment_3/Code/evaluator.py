"""
evaluator.py

Evaluation utilities for Assignment 3.

Generates:
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Confusion Matrix
- Test Confusion Matrix
- Classification Report
- CSV exports
"""

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from config import (
    CLASS_NAMES,
    CONFUSION_DIR,
)


# =============================================================================
# Prediction Helper
# =============================================================================

def predict_dataset(model, dataset):

    dataset = dataset.batch(32)

    y_true = []
    y_pred = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        predicted = np.argmax(predictions, axis=1)
        actual = np.argmax(labels.numpy(), axis=1)

        y_true.extend(actual)
        y_pred.extend(predicted)

    return np.array(y_true), np.array(y_pred)

# =============================================================================
# Evaluate Dataset
# =============================================================================

def evaluate_dataset(
    model,
    dataset,
):
    """
    Compute accuracy and confusion matrix.
    """

    y_true, y_pred = predict_dataset(
        model,
        dataset,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(

        y_true,

        y_pred,

        target_names=CLASS_NAMES,

        digits=4,

        output_dict=True,
    )

    return {

        "accuracy": accuracy,

        "confusion_matrix": cm,

        "classification_report": report,

        "y_true": y_true,

        "y_pred": y_pred,
    }


# =============================================================================
# Save Confusion Matrix CSV
# =============================================================================

def save_confusion_matrix(
    confusion,
    filename,
):
    """
    Save confusion matrix as CSV.
    """

    dataframe = pd.DataFrame(

        confusion,

        index=CLASS_NAMES,

        columns=CLASS_NAMES,
    )

    dataframe.to_csv(

        CONFUSION_DIR / filename,

        index=True,
    )


# =============================================================================
# Save Classification Report
# =============================================================================

def save_classification_report(
    report,
    filename,
):
    """
    Save sklearn classification report.
    """

    dataframe = pd.DataFrame(report).transpose()

    dataframe.to_csv(

        CONFUSION_DIR / filename,

        index=True,
    )


# =============================================================================
# Complete Evaluation
# =============================================================================

def evaluate_best_model(

    model,

    train_dataset,

    validation_dataset,

    test_dataset,
):
    """
    Evaluate the selected best model.

    Returns a dictionary containing
    all metrics.
    """

    train_results = evaluate_dataset(

        model,

        train_dataset,
    )

    validation_results = evaluate_dataset(

        model,

        validation_dataset,
    )

    test_results = evaluate_dataset(

        model,

        test_dataset,
    )

    #
    # Save CSV files
    #

    save_confusion_matrix(

        train_results["confusion_matrix"],

        "train_confusion_matrix.csv",
    )

    save_confusion_matrix(

        test_results["confusion_matrix"],

        "test_confusion_matrix.csv",
    )
    
    save_confusion_matrix(

        validation_results["confusion_matrix"],

        "validation_confusion_matrix.csv",
    )

    save_classification_report(

        train_results["classification_report"],

        "train_classification_report.csv",
    )
    
    save_classification_report(

	validation_results["classification_report"],

	"validation_classification_report.csv",
    )

    save_classification_report(

        test_results["classification_report"],

        "test_classification_report.csv",
    )

    return {

        "train_accuracy":
            train_results["accuracy"],

        "validation_accuracy":
            validation_results["accuracy"],

        "test_accuracy":
            test_results["accuracy"],

        "train_confusion_matrix":
            train_results["confusion_matrix"],

        "validation_confusion_matrix":
            validation_results["confusion_matrix"],

        "test_confusion_matrix":
            test_results["confusion_matrix"],

        "train_report":
            train_results["classification_report"],

        "validation_report":
            validation_results["classification_report"],

        "test_report":
            test_results["classification_report"],
    }


# =============================================================================
# Print Final Results
# =============================================================================

def print_evaluation(results):
    """
    Print evaluation summary.
    """

    print("\n")
    print("=" * 70)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"Training Accuracy   : {results['train_accuracy']:.4f}"
    )

    print(
        f"Validation Accuracy : {results['validation_accuracy']:.4f}"
    )

    print(
        f"Test Accuracy       : {results['test_accuracy']:.4f}"
    )

    print("=" * 70)
