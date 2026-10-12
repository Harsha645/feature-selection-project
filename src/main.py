"""Main experiment pipeline for Integer Linear Programming (ILP)."""

import os
import numpy as np
import pandas as pd

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    REDUNDANCY_PENALTY,
    EXPERIMENTS,
    ILP_TIME_LIMIT,
    RESULTS_DIR
)

from data_loader import load_dataset
from feature_scoring import (
    compute_feature_scores,
    rank_features
)
from optimization_model import FeatureSelectionProblem
from exact_method import solve_ilp
from evaluation import evaluate_features


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")

    data = load_dataset(
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    # Calculate feature relevance and redundancy
    relevance, redundancy = compute_feature_scores(
        data.X_train,
        data.y_train
    )

    # Create the full optimization problem
    full_problem = FeatureSelectionProblem(
        relevance=relevance,
        redundancy=redundancy,
        max_features=len(relevance),
        penalty=REDUNDANCY_PENALTY
    )

    comparison_rows = []
    selected_rows = []

    # Run ILP for each experiment
    for n_candidates, max_features in EXPERIMENTS:

        print("\n" + "=" * 50)
        print(f"ILP Experiment: {n_candidates} features")
        print(f"Maximum selected: {max_features}")
        print("=" * 50)

        # Rank candidate features
        candidates = rank_features(
            relevance,
            n_candidates
        )

        # Create subset optimization problem
        problem = full_problem.candidate_subset(
            candidates,
            max_features=max_features
        )

        # -------------------------------
        # ILP Optimization
        # -------------------------------

        print("Running ILP...")

        ilp = solve_ilp(
            problem,
            time_limit=ILP_TIME_LIMIT
        )

        # Get selected feature indices
        ilp_indices = candidates[
            np.flatnonzero(ilp.mask)
        ]

        # Evaluate selected features
        ilp_metrics = evaluate_features(
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            ilp_indices
        )

        print("ILP objective:", ilp.objective)
        print("ILP runtime:", ilp.runtime)
        print("ILP selected:", ilp.selected_count)
        print("ILP solver status:", ilp.status)

        comparison_rows.append({
            "candidates": n_candidates,
            "method": "ILP",
            "run": 0,
            "objective": ilp.objective,
            "runtime": ilp.runtime,
            "selected_count": ilp.selected_count,
            "status": ilp.status,
            **ilp_metrics
        })

        selected_rows.append({
            "candidates": n_candidates,
            "method": "ILP",
            "run": 0,
            "features": ", ".join(
                data.feature_names[i]
                for i in ilp_indices
            )
        })

    # -------------------------------
    # Save ILP Results
    # -------------------------------

    comparison = pd.DataFrame(comparison_rows)
    selected = pd.DataFrame(selected_rows)

    comparison.to_csv(
        os.path.join(
            RESULTS_DIR,
            "ilp_comparison.csv"
        ),
        index=False
    )

    selected.to_csv(
        os.path.join(
            RESULTS_DIR,
            "ilp_selected_features.csv"
        ),
        index=False
    )

    print("\nILP Experiment Summary:")
    print(comparison.to_string(index=False))

    print("\nAll ILP experiments completed.")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()