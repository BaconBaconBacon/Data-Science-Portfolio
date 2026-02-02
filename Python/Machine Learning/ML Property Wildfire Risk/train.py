"""
Standalone training script for wildfire risk ML pipeline.
Handles preprocessing (fit on train only), model training, evaluation, and persistence.
Can run locally or on EC2/SageMaker.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# ─── Preprocessing (fit on train only) ───────────────────────────────────────


def drop_high_nan_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns where NaN fraction exceeds threshold. Compute on train, apply to both."""
    pass


def drop_correlated_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop one of each pair with |r| > threshold. Compute on train, apply to both."""
    pass


def build_imputer(n_neighbors: int = 5) -> KNNImputer:
    """Return a configured KNNImputer. Fit on train only via pipeline."""
    pass


# ─── Train / Test Split ──────────────────────────────────────────────────────


def prepare_split(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 77,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into X_train, X_test, y_train, y_test after dropping non-feature columns."""
    pass


# ─── Model Training ──────────────────────────────────────────────────────────


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 77,
) -> RandomizedSearchCV:
    """RandomizedSearchCV over XGBRegressor hyperparameters."""
    pass


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 77,
) -> RandomizedSearchCV:
    """RandomizedSearchCV over RandomForestRegressor hyperparameters."""
    pass


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return dict with train_rmse and test_rmse."""
    pass


def extract_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.Series:
    """Return top_n features ranked by importance."""
    pass


# ─── Persistence ──────────────────────────────────────────────────────────────


def save_model(model: BaseEstimator, path: Path) -> None:
    """Pickle model to disk."""
    pass


def load_model(path: Path) -> BaseEstimator:
    """Load pickled model from disk."""
    pass


# ─── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pass
