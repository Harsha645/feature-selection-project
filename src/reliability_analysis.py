"""Analyze GA reliability using existing experiment results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR


def main():

    comparison_path = os.path.join(
        RESULTS_DIR,
        "comparison.csv"
    )

    df = pd.read_csv(comparison_path)

    results = []

    for n_features in sorted(df["candidates"].unique()):

        experiment = df[
            df["candidates"] == n_features
        ]

        ilp = experiment[
            experiment["method"] == "ILP"
        ].iloc[0]

        ga = experiment[
            experiment["method"] == "GA"
        ]

        optimal_value = float(ilp["objective"])

        # Allow a small numerical tolerance
        tolerance = 1e-5

        successful_runs = int(
            np.isclose(
                ga["objective"].to_numpy(),
                optimal_value,
                atol=tolerance,
                rtol=0
            ).sum()
        )

        success_rate = (
            successful_runs / len(ga) * 100
        )

        results.append({
            "candidate_features": n_features,
            "ga_runs": len(ga),
            "optimal_runs": successful_runs,
            "success_rate_pct": success_rate,
            "mean_objective": ga["objective"].mean(),
            "std_objective": ga["objective"].std(),
            "mean_runtime": ga["runtime"].mean(),
            "ilp_runtime": float(ilp["runtime"])
        })

    summary = pd.DataFrame(results)

    summary.to_csv(
        os.path.join(
            RESULTS_DIR,
            "ga_reliability.csv"
        ),
        index=False
    )

    print("\nGA RELIABILITY ANALYSIS")
    print("=" * 60)
    print(summary.to_string(index=False))

    # Reliability visualization
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["candidate_features"].astype(str),
        summary["success_rate_pct"]
    )

    plt.xlabel("Number of Candidate Features")
    plt.ylabel("GA Optimal Solution Success Rate (%)")
    plt.title("Genetic Algorithm Reliability")
    plt.ylim(0, 105)
    plt.tight_layout()

    plot_dir = os.path.join(
        RESULTS_DIR,
        "plots"
    )

    os.makedirs(plot_dir, exist_ok=True)

    plt.savefig(
        os.path.join(
            plot_dir,
            "ga_reliability.png"
        )
    )

    plt.close()

    print("\nReliability analysis completed.")


if __name__ == "__main__":
    main()