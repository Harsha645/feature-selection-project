"""Evaluate selected features using Logistic Regression."""

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_features(
    X_train,
    y_train,
    X_eval,
    y_eval,
    selected_indices
):

    selected_indices = np.asarray(
        selected_indices,
        dtype=int
    )

    if len(selected_indices) == 0:
        raise ValueError("At least one feature is required")

    X_train_selected = X_train[:, selected_indices]
    X_eval_selected = X_eval[:, selected_indices]

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_eval_selected)

    # In this dataset, class 0 = malignant
    return {
        "accuracy": accuracy_score(
            y_eval, predictions
        ),

        "precision": precision_score(
            y_eval,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "recall": recall_score(
            y_eval,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "f1": f1_score(
            y_eval,
            predictions,
            pos_label=0,
            zero_division=0
        )
    }