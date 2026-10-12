# Shared mathematical optimization problem.

from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureSelectionProblem:

    relevance: np.ndarray
    redundancy: np.ndarray
    max_features: int
    penalty: float = 0.10

    def __post_init__(self):

        self.relevance = np.asarray(
            self.relevance,
            dtype=float
        )

        self.redundancy = np.asarray(
            self.redundancy,
            dtype=float
        )

        n = len(self.relevance)

        if (
            self.relevance.ndim != 1
            or self.redundancy.shape != (n, n)
        ):
            raise ValueError("Invalid score dimensions")

        if not 1 <= self.max_features <= n:
            raise ValueError("Invalid max_features")

        if self.penalty < 0:
            raise ValueError("Penalty cannot be negative")

        if not (
            np.isfinite(self.relevance).all()
            and np.isfinite(self.redundancy).all()
        ):
            raise ValueError("Scores must be finite")

    @property
    def n_features(self):
        return len(self.relevance)

    def is_feasible(self, mask):

        mask = np.asarray(mask)

        return (
            mask.shape == (self.n_features,)
            and np.isin(mask, [0, 1]).all()
            and 1 <= int(mask.sum()) <= self.max_features
        )

    def score(self, mask):

        if not self.is_feasible(mask):
            raise ValueError("Infeasible feature subset")

        selected = np.flatnonzero(mask)

        relevance_score = self.relevance[selected].sum()

        pair_matrix = self.redundancy[
            np.ix_(selected, selected)
        ]

        redundancy_score = pair_matrix.sum() / 2

        return float(
            relevance_score
            - self.penalty * redundancy_score
        )

    def candidate_subset(self, indices, max_features):

        indices = np.asarray(indices, dtype=int)

        return FeatureSelectionProblem(
            relevance=self.relevance[indices],
            redundancy=self.redundancy[
                np.ix_(indices, indices)
            ],
            max_features=max_features,
            penalty=self.penalty
        )