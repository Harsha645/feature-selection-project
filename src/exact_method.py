"""Exact feature selection using Integer Linear Programming."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pulp


@dataclass
class ILPResult:
    mask: np.ndarray
    objective: float
    runtime: float
    status: str
    selected_count: int


def solve_ilp(problem, time_limit=120):

    n = problem.n_features

    # Create maximization model
    model = pulp.LpProblem(
        "Feature_Selection_ILP",
        pulp.LpMaximize
    )

    # Binary decision variables
    x = pulp.LpVariable.dicts(
        "feature",
        range(n),
        cat=pulp.LpBinary
    )

    # All feature pairs
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]

    # Auxiliary binary variables
    y = pulp.LpVariable.dicts(
        "pair",
        pairs,
        cat=pulp.LpBinary
    )

    # Relevance component
    relevance_term = pulp.lpSum(
        float(problem.relevance[i]) * x[i]
        for i in range(n)
    )

    # Redundancy component
    redundancy_term = pulp.lpSum(
        float(problem.redundancy[i, j]) * y[i, j]
        for i, j in pairs
    )

    # Objective function
    model += (
        relevance_term
        - problem.penalty * redundancy_term
    )

    # Feature count constraints
    model += (
        pulp.lpSum(x[i] for i in range(n)) >= 1,
        "Minimum_Features"
    )

    model += (
        pulp.lpSum(x[i] for i in range(n))
        <= problem.max_features,
        "Maximum_Features"
    )

    # Linearization constraints
    for i, j in pairs:

        model += y[i, j] <= x[i]
        model += y[i, j] <= x[j]

        model += (
            y[i, j] >= x[i] + x[j] - 1
        )

    # Solve and measure runtime
    start = perf_counter()

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=time_limit
    )

    model.solve(solver)

    runtime = perf_counter() - start

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        raise RuntimeError(
            f"ILP did not prove optimality. Status: {status}"
        )

    # Extract selected features
    mask = np.array(
        [
            int(round(pulp.value(x[i])))
            for i in range(n)
        ],
        dtype=int
    )

    if not problem.is_feasible(mask):
        raise RuntimeError("ILP returned infeasible solution")

    objective = problem.score(mask)

    return ILPResult(
        mask=mask,
        objective=objective,
        runtime=runtime,
        status=status,
        selected_count=int(mask.sum())
    )