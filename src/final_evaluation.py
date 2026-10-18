"""Penalty sensitivity analysis and final held-out evaluation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    ILP_TIME_LIMIT,
    GA_POPULATION_SIZE,
    GA_NUM_GENERATIONS,
    GA_MUTATION_RATE,
    GA_CROSSOVER_RATE,
    GA_TOURNAMENT_SIZE,
    GA_RUNS,
    RESULTS_DIR,
)

from data_loader import load_dataset
from feature_scoring import compute_feature_scores
from optimization_model import FeatureSelectionProblem
from exact_method import solve_ilp
from heuristic_method import solve_ga
from evaluation import evaluate_features


def evaluate_subset(data, indices, split="validation"):

    if split == "validation":
        X_eval = data.X_val
        y_eval = data.y_val

    elif split == "test":
        X_eval = data.X_test
        y_eval = data.y_test

    else:
        raise ValueError("Invalid evaluation split")

    return evaluate_features(
        data.X_train,
        data.y_train,
        X_eval,
        y_eval,
        indices
    )


def run_ga(problem, seed):

    return solve_ga(
        problem,
        population_size=GA_POPULATION_SIZE,
        generations=GA_NUM_GENERATIONS,
        mutation_rate=GA_MUTATION_RATE,
        crossover_rate=GA_CROSSOVER_RATE,
        tournament_size=GA_TOURNAMENT_SIZE,
        random_state=seed
    )


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    data = load_dataset(
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    relevance, redundancy = compute_feature_scores(
        data.X_train,
        data.y_train
    )

    # Full 30-feature experiment
    max_features = 15

    # Candidate penalties, chosen before test evaluation
    penalties = [0.0, 0.05, 0.10, 0.20, 0.50]

    validation_rows = []
    candidate_results = {}

    print("\nPENALTY SENSITIVITY ANALYSIS")
    print("=" * 60)

    for penalty in penalties:

        problem = FeatureSelectionProblem(
            relevance=relevance,
            redundancy=redundancy,
            max_features=max_features,
            penalty=penalty
        )

        print(f"\nPenalty: {penalty}")

        ilp = solve_ilp(
            problem,
            time_limit=ILP_TIME_LIMIT
        )

        ilp_indices = np.flatnonzero(ilp.mask)

        metrics = evaluate_subset(
            data,
            ilp_indices,
            split="validation"
        )

        validation_rows.append({
            "penalty": penalty,
            "objective": ilp.objective,
            "selected_count": ilp.selected_count,
            "runtime": ilp.runtime,
            **metrics
        })

        candidate_results[penalty] = {
            "problem": problem,
            "ilp": ilp,
            "indices": ilp_indices
        }

        print("Selected features:", ilp.selected_count)
        print("Validation accuracy:", metrics["accuracy"])
        print("Validation F1:", metrics["f1"])

    validation_df = pd.DataFrame(validation_rows)

    validation_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "penalty_sensitivity.csv"
        ),
        index=False
    )

    # Choose penalty by validation F1.
    # Ties: fewer features, then smaller penalty.
    best_row = validation_df.sort_values(
        by=["f1", "selected_count", "penalty"],
        ascending=[False, True, True]
    ).iloc[0]

    best_penalty = float(best_row["penalty"])

    print("\nSELECTED PENALTY:", best_penalty)

    final_problem = candidate_results[
        best_penalty
    ]["problem"]

    final_ilp = candidate_results[
        best_penalty
    ]["ilp"]

    # Final GA runs
    ga_results = []

    for run in range(GA_RUNS):

        ga = run_ga(
            final_problem,
            seed=RANDOM_STATE + run
        )

        ga_results.append(ga)

    # Choose GA run by objective only, not test performance
    best_ga = max(
        ga_results,
        key=lambda result: result.objective
    )

    # Evaluate final selected subsets on held-out test set
    ilp_indices = np.flatnonzero(final_ilp.mask)
    ga_indices = np.flatnonzero(best_ga.mask)

    ilp_test = evaluate_subset(
        data,
        ilp_indices,
        split="test"
    )

    ga_test = evaluate_subset(
        data,
        ga_indices,
        split="test"
    )

    # All-feature baseline
    baseline_test = evaluate_subset(
        data,
        np.arange(data.X_train.shape[1]),
        split="test"
    )

    final_rows = [
        {
            "method": "All Features Baseline",
            "selected_count": data.X_train.shape[1],
            "objective": np.nan,
            **baseline_test
        },
        {
            "method": "ILP",
            "selected_count": final_ilp.selected_count,
            "objective": final_ilp.objective,
            **ilp_test
        },
        {
            "method": "GA",
            "selected_count": best_ga.selected_count,
            "objective": best_ga.objective,
            **ga_test
        }
    ]

    final_df = pd.DataFrame(final_rows)

    final_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_test_results.csv"
        ),
        index=False
    )

    # Save selected feature names
    feature_rows = []

    for method, indices in [
        ("ILP", ilp_indices),
        ("GA", ga_indices)
    ]:
        for index in indices:
            feature_rows.append({
                "method": method,
                "feature_index": int(index),
                "feature_name": data.feature_names[index]
            })

    pd.DataFrame(feature_rows).to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_selected_features.csv"
        ),
        index=False
    )

    # Penalty sensitivity graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        validation_df["penalty"],
        validation_df["f1"],
        marker="o"
    )

    plt.xlabel("Redundancy Penalty")
    plt.ylabel("Validation F1-score")
    plt.title("Penalty Sensitivity Analysis")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "penalty_sensitivity.png"
        )
    )
    plt.close()

    print("\nFINAL HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(final_df.to_string(index=False))

    print("\nFinal evaluation completed.")


if __name__ == "__main__":
    main()