import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import (
    CLASSES, CONFIGS, MAX_EPOCHS, SEED,
    RESULTS_HW_ROOT, RESULTS_CV_ROOT,
    PLOTS_HW_ROOT,   PLOTS_CV_ROOT,
)
from data_utils import (
    collect_dataset, stratified_split, standardize_cv,
    save_normalized_handwriting_examples,
)
from train import train_experiment, save_dataset_results


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _roots_for(kind, results_root=None, plots_root=None):
    """
    Resolve the results/plots roots for the given dataset kind.
    CLI overrides, when provided, apply to both datasets but stay separated
    by `kind` subfolder inside train_experiment / save_dataset_results.
    """
    if kind == "handwriting":
        return (
            Path(results_root) if results_root else RESULTS_HW_ROOT,
            Path(plots_root)   if plots_root   else PLOTS_HW_ROOT,
        )
    return (
        Path(results_root) if results_root else RESULTS_CV_ROOT,
        Path(plots_root)   if plots_root   else PLOTS_CV_ROOT,
    )


def run_dataset(data_root, kind, device, max_epochs,
                results_root=None, plots_root=None):
    print(f"\n{'=' * 65}\nLoading {kind} dataset\n{'=' * 65}")
    train_x, train_y, test_x, test_y, classes = collect_dataset(data_root, kind)

    print("Loaded samples:")
    for index, class_name in enumerate(classes):
        print(
            f"  {class_name}: train={int(np.sum(train_y == index))}, "
            f"test/dev={int(np.sum(test_y == index))}"
        )
    print(f"Input feature dimension: {train_x[0].shape[1]}")

    # Stratified validation split — carved from the training pool only.
    train_indices, val_indices = stratified_split(train_x, train_y)
    fit_x = [train_x[i] for i in train_indices]
    fit_y = train_y[train_indices]
    val_x = [train_x[i] for i in val_indices]
    val_y = train_y[val_indices]

    r_root, p_root = _roots_for(kind, results_root, plots_root)
    r_root.mkdir(parents=True, exist_ok=True)
    p_root.mkdir(parents=True, exist_ok=True)

    # Handwriting: normalized trajectory plots go to PLOTS_HW_ROOT.
    # Plotted from the full training pool (before the 15% validation carve-out),
    # which is the intent of the "≥5 normalized training samples per class"
    # report requirement.
    if kind == "handwriting":
        save_normalized_handwriting_examples(
            train_x, train_y, classes, p_root, per_class=5
        )

    # CV: train-only standardization; stats saved under RESULTS_CV_ROOT.
    if kind == "cv":
        fit_x, val_x, test_x, mean, std = standardize_cv(fit_x, val_x, test_x)
        stats_dir = r_root
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.savez(stats_dir / "mfcc_normalization.npz", mean=mean, std=std)

    results = []
    for config in CONFIGS:
        print(f"\nTraining {kind}: {config}")
        result = train_experiment(
            kind, config,
            fit_x, fit_y,
            val_x, val_y,
            test_x, test_y,
            classes,
            results_root=r_root,
            plots_root=p_root,
            device=device,
            max_epochs=max_epochs,
        )
        results.append(result)
        print(
            f"  epochs={result['epochs']} | train={result['training_accuracy']:.4f} "
            f"| validation={result['validation_accuracy']:.4f} "
            f"| test={result['test_accuracy']:.4f} "
            f"| converged={result['converged']}"
        )

    save_dataset_results(kind, results, classes,
                         results_root=r_root, plots_root=p_root)

    print(f"\nBest configurations for {kind}, selected by validation accuracy:")
    for model_type in ("RNN", "LSTM"):
        candidates = [r for r in results if r["model"] == model_type]
        if not candidates:
            print(f"  {model_type}: no runs found — skipped")
            continue
        best = max(
            candidates,
            key=lambda r: (
                r["validation_accuracy"],
                -(r["final_training_loss"] or 0.0),
            ),
        )
        print(
            f"  {model_type}: hidden={best['hidden_size']}, layers={best['num_layers']}, "
            f"validation={best['validation_accuracy']:.4f}, "
            f"test={best['test_accuracy']:.4f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Group 24 CS601T Assignment 6: RNN/LSTM sequence classification"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Folder containing CV_Data and Handwriting_Data"
    )
    parser.add_argument(
        "--only", choices=["all", "handwriting", "cv"], default="all"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
        help=("DEBUG-ONLY safety cap. For real runs leave at MAX_EPOCHS; "
              "normal stopping is governed by the convergence criterion.")
    )
    parser.add_argument(
        "--results-root", default=None,
        help="Override /Results root (default: config.RESULTS_HW_ROOT/CV_ROOT)."
    )
    parser.add_argument(
        "--plots-root", default=None,
        help="Override /Report/Plots root (default: config.PLOTS_HW_ROOT/CV_ROOT)."
    )
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = ["handwriting", "cv"] if args.only == "all" else [args.only]
    all_results = []
    for kind in datasets:
        all_results.extend(
            run_dataset(
                args.data_root, kind, device, args.max_epochs,
                results_root=args.results_root,
                plots_root=args.plots_root,
            )
        )

    # Combined summary goes next to the results roots (into /Results).
    combined_path = Path(args.results_root) if args.results_root else RESULTS_HW_ROOT.parent
    combined_path.mkdir(parents=True, exist_ok=True)
    (combined_path / "combined_results.json").write_text(
        json.dumps(all_results, indent=2)
    )
    print(f"\nAll requested experiments completed.")
    print(f"  Combined summary: {combined_path / 'combined_results.json'}")


if __name__ == "__main__":
    main()
