"""
Standalone training script for wildfire risk ML pipeline.
Handles preprocessing (fit on train only), model training, evaluation, and persistence.
Can run locally or on EC2/SageMaker.

Test locally first with the test data:
 python train.py \
    --input data/model_joined.parquet \
    --output Models/test_model.pkl \
    --n-iter 5 \
    --model rf

  Then test with S3:
  # Upload test data to S3
  aws s3 cp data/model_joined.parquet s3://wildfire-risk-ml/data/test_joined.parquet

  # Train from S3, save to S3
  python train.py     --input s3://wildfire-risk-ml/data/test_joined.parquet     --output s3://wildfire-risk-ml/models/test_model.pkl     --n-iter 5

  Full run (after 300k properties are ready):
  python train.py \
    --input s3://wildfire-risk-ml/data/model_joined.parquet \
    --output s3://wildfire-risk-ml/models/best_model.pkl \
    --n-iter 50


"""

import argparse
import os
import pickle
import tempfile

import boto3
import numpy as np
import pandas as pd

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


# ─── AWS ─────────────────────────────────────────────────────────────────────


def upload_to_s3(local_path, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)


def download_from_s3(bucket, key, local_path):
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)


def is_s3_path(path: str) -> bool:
    """Check if path is an S3 URI."""
    return path.startswith("s3://")


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def read_parquet_auto(path: str) -> pd.DataFrame:
    """Read parquet from local path or S3."""
    if is_s3_path(path):
        bucket, key = parse_s3_path(path)
        # Create temp file, close it (Windows needs this), then download
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)  # Close file handle so boto3/pandas can use it
        try:
            print(f"  Downloading from S3: {path}")
            download_from_s3(bucket, key, tmp_path)
            return pd.read_parquet(tmp_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        return pd.read_parquet(path)


def save_model_auto(model, path: str, pipeline=None, feature_names=None) -> None:
    """Save model to local path or S3."""
    if is_s3_path(path):
        bucket, key = parse_s3_path(path)
        # Create temp file, close it (Windows needs this), then write
        fd, tmp_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            save_model(model, Path(tmp_path), pipeline, feature_names)
            upload_to_s3(tmp_path, bucket, key)
            print(f"  Model uploaded to {path}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        save_model(model, Path(path), pipeline, feature_names)
        print(f"  Model saved to {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train wildfire risk model. Supports local and S3 paths."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input parquet (local or s3://...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output model (local or s3://...)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nearest_fire_km",
        help="Target column name (default: nearest_fire_km)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="RandomizedSearchCV iterations (default: 20)",
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["xgboost", "rf", "both"],
        help="Model type: 'xgboost' | 'rf' | 'both' (default: both)",
    )
    parser.add_argument(
        "--nan-thresh",
        type=float,
        default=0.45,
        help="NaN column drop threshold (default: 0.45)",
    )
    parser.add_argument(
        "--corr-thresh",
        type=float,
        default=0.85,
        help="Correlation drop threshold (default: 0.85)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Wildfire Risk Model Training")
    print("=" * 60)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Target:      {args.target}")
    print(f"Model:       {args.model}")
    print(f"n_iter:      {args.n_iter}")
    print(f"cv:          {args.cv}")
    print(f"nan_thresh:  {args.nan_thresh}")
    print(f"corr_thresh: {args.corr_thresh}")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    df = read_parquet_auto(args.input)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Split
    print("\n[2/5] Splitting data...")
    y = df[args.target]
    X = df.drop(columns=[args.target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=77
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Preprocess
    print("\n[3/5] Preprocessing...")
    print(f"  Features before filtering: {X_train.shape[1]}")
    X_train, X_test = drop_high_nan_columns(X_train, X_test, threshold=args.nan_thresh)
    print(f"  After NaN filter: {X_train.shape[1]}")
    X_train, X_test = drop_correlated_features(
        X_train, X_test, threshold=args.corr_thresh
    )
    print(f"  After correlation filter: {X_train.shape[1]}")

    feature_names = X_train.columns.tolist()
    pipeline = build_preprocessing_pipeline(n_neighbors=5)
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    print(f"  Final feature matrix: {X_train.shape}")

    # Train
    print("\n[4/5] Training models...")
    results = {}

    if args.model in ["rf", "both"]:
        print("  Training RandomForest...")
        rf_search = train_random_forest(
            X_train, y_train, n_iter=args.n_iter, cv=args.cv
        )
        rf_metrics = evaluate_model(
            rf_search.best_estimator_, X_train, X_test, y_train, y_test
        )
        results["rf"] = {"search": rf_search, "metrics": rf_metrics}
        print(f"  RF Train RMSE: {rf_metrics['train_rmse']:.4f}")
        print(f"  RF Test  RMSE: {rf_metrics['test_rmse']:.4f}")

    if args.model in ["xgboost", "both"]:
        print("  Training XGBoost...")
        xgb_search = train_xgboost(X_train, y_train, n_iter=args.n_iter, cv=args.cv)
        xgb_metrics = evaluate_model(
            xgb_search.best_estimator_, X_train, X_test, y_train, y_test
        )
        results["xgb"] = {"search": xgb_search, "metrics": xgb_metrics}
        print(f"  XGB Train RMSE: {xgb_metrics['train_rmse']:.4f}")
        print(f"  XGB Test  RMSE: {xgb_metrics['test_rmse']:.4f}")

    # Select best model
    if args.model == "both":
        if (
            results["xgb"]["metrics"]["test_rmse"]
            <= results["rf"]["metrics"]["test_rmse"]
        ):
            best_model = results["xgb"]["search"].best_estimator_
            best_name = "XGBoost"
        else:
            best_model = results["rf"]["search"].best_estimator_
            best_name = "RandomForest"
    elif args.model == "xgboost":
        best_model = results["xgb"]["search"].best_estimator_
        best_name = "XGBoost"
    else:
        best_model = results["rf"]["search"].best_estimator_
        best_name = "RandomForest"

    print(f"\n  Best model: {best_name}")

    # Save
    print("\n[5/5] Saving model...")
    save_model_auto(
        best_model, args.output, pipeline=pipeline, feature_names=feature_names
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
