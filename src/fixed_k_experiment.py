"""
Fixed-k scalability experiment.

Compare ILP and GA using:
    n = 10, 20, 30 candidate features
    k = 5 maximum selected features

The same mathematical objective and constraints
are used by both algorithms.
"""

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

from feature_scoring import (
    compute_feature_scores,
    rank_features,
)

from optimization_model import FeatureSelectionProblem

from exact_method import solve_ilp
from heuristic_method import solve_ga
from evaluation import evaluate_features


# Fixed-k experiment settings
CANDIDATE_COUNTS = [10, 20, 30]
MAX_SELECTED_FEATURES = 5
PENALTY = 0.05


def main():

    print("=" * 60)
    print("FIXED-K SCALABILITY EXPERIMENT")
    print("=" * 60)

    print(f"Maximum selected features: {MAX_SELECTED_FEATURES}")
    print(f"Redundancy penalty: {PENALTY}")
    print(f"GA runs per scenario: {GA_RUNS}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # -----------------------------------
    # 1. Load dataset
    # -----------------------------------

    data = load_dataset(
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------
    # 2. Calculate feature scores
    # -----------------------------------

    relevance, redundancy = compute_feature_scores(
        data.X_train,
        data.y_train,
    )

    experiment_rows = []
    selected_feature_rows = []

    # -----------------------------------
    # 3. Run each fixed-k scenario
    # -----------------------------------

    for n_candidates in CANDIDATE_COUNTS:

        print("\n" + "=" * 60)
        print(f"Candidate features: {n_candidates}")
        print(f"Maximum selected: {MAX_SELECTED_FEATURES}")
        print("=" * 60)

        # Rank features using training data only
        candidates = rank_features(
            relevance,
            n_candidates,
        )

        # Create the same problem for ILP and GA
        problem = FeatureSelectionProblem(
            relevance=relevance[candidates],
            redundancy=redundancy[
                np.ix_(candidates, candidates)
            ],
            max_features=MAX_SELECTED_FEATURES,
            penalty=PENALTY,
        )

        # -----------------------------------
        # 4. ILP
        # -----------------------------------

        print("\nRunning ILP...")

        ilp = solve_ilp(
            problem,
            time_limit=ILP_TIME_LIMIT,
        )

        ilp_indices = candidates[
            np.flatnonzero(ilp.mask)
        ]

        ilp_metrics = evaluate_features(
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            ilp_indices,
        )

        experiment_rows.append({
            "candidates": n_candidates,
            "max_features": MAX_SELECTED_FEATURES,
            "penalty": PENALTY,
            "method": "ILP",
            "run": 0,
            "objective": ilp.objective,
            "runtime": ilp.runtime,
            "selected_count": ilp.selected_count,
            "status": ilp.status,
            "optimality_gap_pct": 0.0,
            **ilp_metrics,
        })

        selected_feature_rows.append({
            "candidates": n_candidates,
            "method": "ILP",
            "run": 0,
            "features": ", ".join(
                data.feature_names[i]
                for i in ilp_indices
            ),
        })

        print(f"ILP objective: {ilp.objective:.6f}")
        print(f"ILP runtime: {ilp.runtime:.4f}s")
        print(f"ILP selected: {ilp.selected_count}")

        # -----------------------------------
        # 5. Genetic Algorithm
        # -----------------------------------

        print("\nRunning Genetic Algorithm...")

        for run in range(GA_RUNS):

            ga = solve_ga(
                problem,
                population_size=GA_POPULATION_SIZE,
                generations=GA_NUM_GENERATIONS,
                mutation_rate=GA_MUTATION_RATE,
                crossover_rate=GA_CROSSOVER_RATE,
                tournament_size=GA_TOURNAMENT_SIZE,
                random_state=RANDOM_STATE + run,
            )

            ga_indices = candidates[
                np.flatnonzero(ga.mask)
            ]

            ga_metrics = evaluate_features(
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                ga_indices,
            )

            if abs(ilp.objective) > 1e-12:

                gap = (
                    (ilp.objective - ga.objective)
                    / abs(ilp.objective)
                ) * 100

            else:
                gap = np.nan

            experiment_rows.append({
                "candidates": n_candidates,
                "max_features": MAX_SELECTED_FEATURES,
                "penalty": PENALTY,
                "method": "GA",
                "run": run + 1,
                "objective": ga.objective,
                "runtime": ga.runtime,
                "selected_count": ga.selected_count,
                "status": "Feasible",
                "optimality_gap_pct": max(0.0, gap),
                **ga_metrics,
            })

            selected_feature_rows.append({
                "candidates": n_candidates,
                "method": "GA",
                "run": run + 1,
                "features": ", ".join(
                    data.feature_names[i]
                    for i in ga_indices
                ),
            })

            print(
                f"GA run {run + 1}: "
                f"objective={ga.objective:.6f}, "
                f"runtime={ga.runtime:.4f}s, "
                f"gap={gap:.4f}%"
            )

    # -----------------------------------
    # 6. Save detailed results
    # -----------------------------------

    df = pd.DataFrame(experiment_rows)

    df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "fixed_k_comparison.csv",
        ),
        index=False,
    )

    pd.DataFrame(selected_feature_rows).to_csv(
        os.path.join(
            RESULTS_DIR,
            "fixed_k_selected_features.csv",
        ),
        index=False,
    )

    # -----------------------------------
    # 7. Generate summary statistics
    # -----------------------------------

    summary = df.groupby(
        ["candidates", "method"]
    ).agg(
        mean_objective=("objective", "mean"),
        std_objective=("objective", "std"),
        mean_runtime=("runtime", "mean"),
        mean_accuracy=("accuracy", "mean"),
        mean_f1=("f1", "mean"),
        mean_gap=("optimality_gap_pct", "mean"),
    ).reset_index()

    summary.to_csv(
        os.path.join(
            RESULTS_DIR,
            "fixed_k_summary.csv",
        ),
        index=False,
    )

    # -----------------------------------
    # 8. Calculate GA reliability
    # -----------------------------------

    reliability_rows = []

    for n_candidates in CANDIDATE_COUNTS:

        subset = df[
            df["candidates"] == n_candidates
        ]

        ilp_optimum = subset[
            subset["method"] == "ILP"
        ]["objective"].iloc[0]

        ga_values = subset[
            subset["method"] == "GA"
        ]["objective"].to_numpy()

        successful = np.isclose(
            ga_values,
            ilp_optimum,
            atol=1e-5,
            rtol=0,
        ).sum()

        reliability_rows.append({
            "candidates": n_candidates,
            "ga_runs": len(ga_values),
            "optimal_runs": int(successful),
            "success_rate_pct": (
                successful / len(ga_values)
            ) * 100,
        })

    reliability = pd.DataFrame(reliability_rows)

    reliability.to_csv(
        os.path.join(
            RESULTS_DIR,
            "fixed_k_reliability.csv",
        ),
        index=False,
    )

    # -----------------------------------
    # 9. Runtime graph
    # -----------------------------------

    plt.figure(figsize=(9, 5))

    for method in ["ILP", "GA"]:

        subset = summary[
            summary["method"] == method
        ]

        plt.plot(
            subset["candidates"],
            subset["mean_runtime"],
            marker="o",
            label=method,
        )

    plt.xlabel("Number of Candidate Features")
    plt.ylabel("Mean Runtime (seconds)")
    plt.title("Fixed-k Runtime Comparison (k=5)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "fixed_k_runtime.png",
        )
    )

    plt.close()

    # -----------------------------------
    # 10. Objective graph
    # -----------------------------------

    plt.figure(figsize=(9, 5))

    for method in ["ILP", "GA"]:

        subset = summary[
            summary["method"] == method
        ]

        plt.plot(
            subset["candidates"],
            subset["mean_objective"],
            marker="o",
            label=method,
        )

    plt.xlabel("Number of Candidate Features")
    plt.ylabel("Mean Objective Value")
    plt.title("Fixed-k Objective Comparison (k=5)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "fixed_k_objective.png",
        )
    )

    plt.close()

    # -----------------------------------
    # 11. Reliability graph
    # -----------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        reliability["candidates"].astype(str),
        reliability["success_rate_pct"],
    )

    plt.xlabel("Number of Candidate Features")
    plt.ylabel("GA Optimal-Solution Success Rate (%)")
    plt.title("GA Reliability with Fixed k=5")
    plt.ylim(0, 105)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "fixed_k_reliability.png",
        )
    )

    plt.close()

    # -----------------------------------
    # 12. Print final summary
    # -----------------------------------

    print("\n" + "=" * 60)
    print("FIXED-K EXPERIMENT SUMMARY")
    print("=" * 60)

    print(summary.to_string(index=False))

    print("\nGA RELIABILITY")
    print(reliability.to_string(index=False))

    print("\nAll fixed-k experiments completed.")


if __name__ == "__main__":
    main()