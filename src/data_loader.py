"""
Loads the Breast Cancer Wisconsin (Diagnostic) dataset from the CSV file in
the project's dataset/ folder and returns train/test splits ready for use by
both the exact method and the heuristic (GA) method.

Dataset file: dataset/breast_cancer.csv
  - 569 samples, 30 numeric features, binary target column "diagnosis"
    (M = malignant, B = benign)
  - Features describe characteristics of cell nuclei from digitized
    images of breast mass biopsies (e.g. mean radius, mean texture,
    worst concavity) - each is human-interpretable.
  - Source: UCI Machine Learning Repository
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dataset():
    """
    Returns (X_train, X_test, y_train, y_test, feature_names)
    for the full 30-feature Breast Cancer dataset. Used as-is by the
    Genetic Algorithm (heuristic method).
    """
    path = PROJECT_ROOT / config.DATASET_DIR / config.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Place the CSV in the '{config.DATASET_DIR}' folder (see README)."
        )

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("id", "Unnamed: 32") if c in df.columns])

    print("Dataset loaded successfully.")



