"""Feature relevance and redundancy calculations."""

import numpy as np
from sklearn.feature_selection import f_classif


def compute_feature_scores(X_train, y_train):

    # ANOVA F-score: relevance to target
    f_scores, _ = f_classif(X_train, y_train)

    f_scores = np.nan_to_num(
        f_scores,
        nan=0.0,
        posinf=1e12,
        neginf=0.0
    )

    f_scores = np.maximum(f_scores, 0.0)

    # Normalize scores into [0, 1]
    relevance = f_scores / (1.0 + f_scores)

    # Absolute Pearson correlation
    redundancy = np.abs(
        np.corrcoef(X_train, rowvar=False)
    )

    redundancy = np.nan_to_num(
        redundancy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    redundancy = np.clip(redundancy, 0.0, 1.0)

    np.fill_diagonal(redundancy, 0.0)

    return relevance, redundancy


def rank_features(relevance, n_candidates):

    if not 1 <= n_candidates <= len(relevance):
        raise ValueError("Invalid candidate count")

    return np.argsort(
        -np.asarray(relevance),
        kind="stable"
    )[:n_candidates]