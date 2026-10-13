"""Load and preprocess the Breast Cancer Wisconsin dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    feature_names: list


def load_dataset(
    test_size=0.20,
    validation_size=0.20,
    random_state=42
):

    # -------------------------------
    # Load CSV dataset
    # -------------------------------

    path = (
        PROJECT_ROOT
        / config.DATASET_DIR
        / config.DATASET_FILE
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(path)

    # Remove unnecessary columns
    df = df.drop(
        columns=[
            c for c in ("id", "Unnamed: 32")
            if c in df.columns
        ]
    )

    # -------------------------------
    # Separate features and target
    # -------------------------------

    if "diagnosis" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'diagnosis' column."
        )

    y = df["diagnosis"].map({
        "M": 0,
        "B": 1
    })

    if y.isna().any():
        raise ValueError(
            "Unexpected diagnosis values in dataset."
        )

    y = y.to_numpy(dtype=int)

    X_df = df.drop(columns=["diagnosis"])
    feature_names = X_df.columns.tolist()

    X = X_df.to_numpy(dtype=float)

    # -------------------------------
    # Train / Validation / Test split
    # -------------------------------

    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    )

    relative_val_size = (
        validation_size / (1 - test_size)
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=y_train_val
        )
    )

    # -------------------------------
    # Feature scaling
    # -------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("Dataset loaded successfully.")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))
    print("Number of features:", len(feature_names))

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names
    )