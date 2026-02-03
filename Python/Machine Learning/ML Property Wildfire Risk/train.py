"""
Standalone training script for wildfire risk ML pipeline.
Handles preprocessing (fit on train only), model training, evaluation, and persistence.
Can run locally or on EC2/SageMaker.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ─── Preprocessing (fit on train only) ───────────────────────────────────────


def drop_high_nan_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns where NaN fraction exceeds threshold. Compute on train, apply to both."""
    nan_frac = X_train.isna().mean()
    keep = nan_frac[nan_frac <= threshold].index
    return X_train[keep], X_test[keep]


def drop_correlated_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop second of each pair of columns with |r| > threshold. Compute on train, apply to both."""
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X_train.drop(columns=to_drop), X_test.drop(columns=to_drop)


def build_preprocessing_pipeline(n_neighbors: int = 5) -> Pipeline:
    """Return a Pipeline that imputes then scales. Fit on train only.

    Steps:
        1. KNNImputer  — fill NaN using k-nearest neighbors
        2. StandardScaler — zero-mean, unit-variance
    """
    return Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=n_neighbors)),
            ("scaler", StandardScaler()),
        ]
    )


# ─── Train / Test Split ──────────────────────────────────────────────────────


def prepare_split(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 77,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into X_train, X_test, y_train, y_test after dropping non-feature columns."""
    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ─── Model Training ──────────────────────────────────────────────────────────


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 77,
) -> RandomizedSearchCV:
    """RandomizedSearchCV over XGBRegressor hyperparameters."""
    param_dist = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 12),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "gamma": uniform(0, 0.5),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
    }
    search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 77,
) -> RandomizedSearchCV:
    """RandomizedSearchCV over RandomForestRegressor hyperparameters."""
    param_dist = {
        "n_estimators": randint(100, 1000),
        "max_depth": randint(5, 50),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return dict with train_rmse and test_rmse."""
    return {
        "train_rmse": rmse(model, X_train, y_train),
        "test_rmse": rmse(model, X_test, y_test),
    }


def rmse(
    model: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
):
    return np.sqrt(mean_squared_error(y, model.predict(x)))


def extract_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.Series:
    """Return top_n features ranked by importance."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


# ─── Persistence ──────────────────────────────────────────────────────────────


def save_model(
    model: BaseEstimator,
    path: Path,
    pipeline: Pipeline | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Pickle model and preprocessing artifacts to disk.

    Parameters
    ----------
    model : BaseEstimator
        The trained estimator.
    path : Path
        Destination file path.
    pipeline : Pipeline, optional
        The fitted preprocessing pipeline (imputer + scaler).
    feature_names : list[str], optional
        Ordered feature names after column filtering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "pipeline": pipeline,
        "feature_names": feature_names,
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_model(path: Path) -> dict:
    """Load pickled model artifact from disk.

    Returns a dict with keys: 'model', 'pipeline', 'feature_names'.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # TODO: Implement this once you have AWS up and running.
    pass
