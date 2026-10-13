"""Binary Genetic Algorithm for feature selection."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class GAResult:
    mask: np.ndarray
    objective: float
    runtime: float
    selected_count: int
    history: list


def solve_ga(
    problem,
    population_size=60,
    generations=100,
    mutation_rate=0.05,
    crossover_rate=0.80,
    tournament_size=3,
    random_state=42
):

    rng = np.random.default_rng(random_state)

    n = problem.n_features
    k = problem.max_features

    if population_size < 2:
        raise ValueError("Population must contain at least 2")

    if not 1 <= tournament_size <= population_size:
        raise ValueError("Invalid tournament size")

    def repair(mask):

        mask = np.asarray(mask, dtype=int).copy()

        selected = np.flatnonzero(mask)

        # Remove excess selected features
        if len(selected) > k:

            remove = rng.choice(
                selected,
                size=len(selected) - k,
                replace=False
            )

            mask[remove] = 0

        # Ensure at least one feature is selected
        if mask.sum() == 0:

            index = rng.integers(0, n)
            mask[index] = 1

        return mask

    def create_individual():

        count = rng.integers(1, k + 1)

        mask = np.zeros(n, dtype=int)

        selected = rng.choice(
            n,
            size=count,
            replace=False
        )

        mask[selected] = 1

        return mask

    def fitness(mask):
        return problem.score(mask)

    def tournament_selection(population, scores):

        candidates = rng.choice(
            len(population),
            size=tournament_size,
            replace=False
        )

        winner = candidates[
            np.argmax(scores[candidates])
        ]

        return population[winner].copy()

    def crossover(parent1, parent2):

        if n < 2 or rng.random() >= crossover_rate:
            return parent1.copy(), parent2.copy()

        point = rng.integers(1, n)

        child1 = np.concatenate([
            parent1[:point],
            parent2[point:]
        ])

        child2 = np.concatenate([
            parent2[:point],
            parent1[point:]
        ])

        return repair(child1), repair(child2)

    def mutate(mask):

        child = mask.copy()

        changes = rng.random(n) < mutation_rate

        child[changes] = 1 - child[changes]

        return repair(child)

    start = perf_counter()

    # Initialize population
    population = np.array([
        create_individual()
        for _ in range(population_size)
    ])

    best_mask = None
    best_score = float("-inf")

    history = []

    # Include generation 0
    for generation in range(generations + 1):

        scores = np.array([
            fitness(individual)
            for individual in population
        ])

        generation_best = int(np.argmax(scores))

        if scores[generation_best] > best_score:

            best_score = float(scores[generation_best])

            best_mask = population[
                generation_best
            ].copy()

        history.append(best_score)

        if generation == generations:
            break

        # Elitism: preserve best individual
        new_population = [best_mask.copy()]

        while len(new_population) < population_size:

            parent1 = tournament_selection(
                population, scores
            )

            parent2 = tournament_selection(
                population, scores
            )

            child1, child2 = crossover(
                parent1, parent2
            )

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)

    runtime = perf_counter() - start

    if not problem.is_feasible(best_mask):
        raise RuntimeError("GA returned infeasible solution")

    return GAResult(
        mask=best_mask,
        objective=best_score,
        runtime=runtime,
        selected_count=int(best_mask.sum()),
        history=history
    )