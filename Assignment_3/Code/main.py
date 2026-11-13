"""
main.py

CS601T - Deep Learning
Programming Assignment 3

Runs the complete experiment:

    3 Architectures
            ×
    7 Optimizers

Automatically generates

✓ Training
✓ Validation
✓ Testing

✓ Tables
✓ Checkpoints
✓ Histories
✓ Confusion Matrices
✓ Publication-quality plots (400 DPI)

Run

    python main.py
"""

from config import *

from utils import (
    set_seed,
    ensure_directories,
)

from dataset import (
    get_datasets,
    print_dataset_info,
)

from trainer import (
    run_all_experiments,
    get_best_model,
    print_summary,
)

from evaluator import (
    evaluate_best_model,
    print_evaluation,
)

from plotting import (
    generate_all_plots,
)


# =============================================================================
# Main
# =============================================================================

def main():

    print("\n")
    print("=" * 90)
    print("CS601T : DEEP LEARNING")
    print("PROGRAMMING ASSIGNMENT 3")
    print("=" * 90)

    #
    # Reproducibility
    #

    set_seed()

    #
    # Create output folders
    #

    ensure_directories()

    #
    # Load dataset
    #

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_size,
    ) = get_datasets()

    print_dataset_info(train_size)

    #
    # Run all experiments
    #

    experiment_results = run_all_experiments(

        train_dataset,

        validation_dataset,

        test_dataset,

        train_size,
    )

    #
    # Print experiment summary
    #

    print_summary(experiment_results)

    #
    # Evaluate best model
    #

    best_model = get_best_model(
        experiment_results
    )

    evaluation_results = evaluate_best_model(

        model=best_model,

        train_dataset=train_dataset,

        validation_dataset=validation_dataset,

        test_dataset=test_dataset,
    )

    #
    # Print final evaluation
    #

    print_evaluation(
        evaluation_results
    )

    #
    # Generate all plots
    #

    generate_all_plots(

        experiment_results["all_results"],

        evaluation_results,
    )

    #
    # Completion Message
    #

    print("\n")
    print("=" * 90)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 90)

    print("\nGenerated Outputs")

    print("-------------------------------")

    print("✓ Checkpoints")

    print("✓ Histories")

    print("✓ CSV Tables")

    print("✓ Training Log")

    print("✓ Loss Curves")

    print("✓ Accuracy Plots")

    print("✓ Architecture Comparison")

    print("✓ Confusion Matrices")

    print("\nLocation")

    print("Results/")

    print("=" * 90)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    main()
