"""
Training module for wildfire risk ML pipeline.

Provides preprocessing with adaptive imputation based on missingness analysis,
model training with hyperparameter tuning, evaluation, and persistence.

Usage:
    python train.py --input data/model_joined.parquet --output Models/model.pkl
"""

import argparse
import hashlib
import os
import pickle
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from missing_analysis import (
    analyze_missingness,
    get_column_lists,
    print_missingness_report,
)
from settings import PATH_DATA

# Suppress XGBoost device mismatch warning (model on CUDA, inference data on CPU)
warnings.filterwarnings(
    "ignore", message=".*Falling back to prediction using DMatrix.*"
)


def _generate_preprocess_cache_key(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    nan_threshold: float,
    corr_threshold: float,
    mar_corr_threshold: float,
    test_size: float,
    random_state: int,
) -> str:
    """
    Generate MD5 hash key for preprocessing cache.

    Creates a unique identifier based on data shape, column names, and all
    preprocessing parameters to detect when cached results are valid.
    """
    col_hash = hashlib.md5("_".join(sorted(df.columns)).encode()).hexdigest()[:8]
    drop_hash = hashlib.md5("_".join(sorted(drop_cols)).encode()).hexdigest()[:8]

    key_parts = [
        f"rows_{len(df)}",
        f"cols_{len(df.columns)}",
        f"colhash_{col_hash}",
        f"drophash_{drop_hash}",
        f"target_{target_col}",
        f"nan_{nan_threshold}",
        f"corr_{corr_threshold}",
        f"mar_{mar_corr_threshold}",
        f"test_{test_size}",
        f"seed_{random_state}",
    ]
    return hashlib.md5("_".join(key_parts).encode()).hexdigest()[:12]


def _get_preprocess_cache_path(cache_key: str) -> Path:
    """Return the cache file path for a given cache key."""
    cache_dir = PATH_DATA / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"preprocess_{cache_key}.pkl"


def full_metrics(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "Train RMSE": np.sqrt(((y_train - y_pred_train) ** 2).mean()),
        "Test RMSE": np.sqrt(((y_test - y_pred_test) ** 2).mean()),
        "Test MAE": mean_absolute_error(y_test, y_pred_test),
        "Test R²": r2_score(y_test, y_pred_test),
    }


def build_adaptive_pipeline(
    mcar_cols: list[str],
    mar_cols: list[str],
    complete_cols: list[str],
    random_state: int = 77,
) -> Pipeline:
    """
    Build preprocessing pipeline with adaptive imputation per missingness type.

    Parameters
    ----------
    mcar_cols
        Columns with Missing Completely At Random - uses median imputation.
    mar_cols
        Columns with Missing At Random - uses iterative imputation to
        preserve feature correlations.
    complete_cols
        Columns with no missing values - passed through unchanged.
    random_state
        Random seed for IterativeImputer reproducibility.

    Returns
    -------
    Pipeline
        Two-stage pipeline: ColumnTransformer for adaptive imputation,
        then StandardScaler for normalization.
    """
    transformers = []

    if mcar_cols:
        transformers.append(
            ("mcar_imputer", SimpleImputer(strategy="median"), mcar_cols)
        )

    if mar_cols:
        transformers.append(
            (
                "mar_imputer",
                IterativeImputer(max_iter=10, random_state=random_state),
                mar_cols,
            )
        )

    if complete_cols:
        transformers.append(("passthrough", "passthrough", complete_cols))

    return Pipeline(
        [
            ("imputer", ColumnTransformer(transformers, remainder="drop")),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_with_cache(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    nan_threshold: float = 0.45,
    corr_threshold: float = 0.85,
    mar_corr_threshold: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 77,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str], Pipeline]:
    """
    Full preprocessing pipeline with adaptive imputation and disk caching.

    Performs train/test split, drops high-NaN and correlated columns, analyzes
    missingness mechanism per feature, applies appropriate imputation (median
    for MCAR, iterative for MAR), and scales features. All artifacts are
    cached to disk for fast re-runs.

    Parameters
    ----------
    df
        Input DataFrame containing features and target.
    target_col
        Name of the target variable column.
    drop_cols
        Columns to exclude from features (e.g., geometry, identifiers).
    nan_threshold
        Maximum allowed NaN fraction per column (0.0-1.0).
    corr_threshold
        Maximum allowed absolute correlation between feature pairs.
    mar_corr_threshold
        Minimum correlation between missingness indicator and other features
        to classify as MAR (Missing At Random) vs MCAR.
    test_size
        Fraction of data reserved for test set.
    random_state
        Random seed for reproducibility.
    use_cache
        Whether to cache results and load from cache on re-runs.

    Returns
    -------
    X_train
        Preprocessed training features as numpy array.
    X_test
        Preprocessed test features as numpy array.
    y_train
        Training target values.
    y_test
        Test target values.
    feature_names
        Ordered list of feature names matching output columns.
    pipeline
        Fitted preprocessing pipeline for transforming new data.
    """
    cache_key = _generate_preprocess_cache_key(
        df,
        target_col,
        drop_cols,
        nan_threshold,
        corr_threshold,
        mar_corr_threshold,
        test_size,
        random_state,
    )
    cache_file = _get_preprocess_cache_path(cache_key)

    if use_cache and cache_file.exists():
        print(f"Loading cached preprocessing from {cache_file.name}")
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)

            if cached.get("n_rows") == len(df):
                y = df[target_col]
                X = df.drop(columns=[target_col] + drop_cols)
                X = X[cached["nan_keep_cols"]]
                X = X[cached["corr_keep_cols"]]

                train_idx = cached["train_idx"]
                test_idx = cached["test_idx"]
                X_train_df = X.iloc[train_idx]
                X_test_df = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                pipeline = cached["pipeline"]
                X_train = pipeline.transform(X_train_df)
                X_test = pipeline.transform(X_test_df)

                print(f"  Loaded {len(cached['feature_names'])} features from cache")
                return (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    cached["feature_names"],
                    pipeline,
                )
            else:
                print(
                    f"  Cache row count mismatch ({cached.get('n_rows')} vs {len(df)}), recomputing..."
                )
        except Exception as e:
            print(f"  Cache load failed ({e}), recomputing...")

    print("Computing preprocessing (will cache for future runs)...")
    start = time.time()

    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_idx = X_train_df.index.tolist()
    test_idx = X_test_df.index.tolist()

    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"  Features before filtering: {X_train_df.shape[1]}")

    X_train_df, X_test_df = drop_high_nan_columns(
        X_train_df, X_test_df, threshold=nan_threshold
    )
    nan_keep_cols = X_train_df.columns.tolist()
    print(f"  After NaN filter: {X_train_df.shape[1]}")

    X_train_df, X_test_df = drop_correlated_features(
        X_train_df, X_test_df, threshold=corr_threshold
    )
    corr_keep_cols = X_train_df.columns.tolist()
    print(f"  After correlation filter: {X_train_df.shape[1]}")

    print("  Analyzing missing data patterns...")
    t0 = time.time()
    analysis = analyze_missingness(X_train_df, threshold=mar_corr_threshold)
    print_missingness_report(analysis)
    print(f"  Missingness analysis completed in {time.time() - t0:.1f}s")

    mcar_cols, mar_cols, complete_cols = get_column_lists(
        analysis, X_train_df.columns.tolist()
    )

    feature_names = mcar_cols + mar_cols + complete_cols

    print("  Fitting adaptive imputation pipeline...")
    t0 = time.time()
    pipeline = build_adaptive_pipeline(mcar_cols, mar_cols, complete_cols, random_state)
    X_train = pipeline.fit_transform(X_train_df)
    X_test = pipeline.transform(X_test_df)
    print(f"  Pipeline fitting completed in {time.time() - t0:.1f}s")

    print(f"  Final feature matrix: {X_train.shape}")
    print(f"  Preprocessing completed in {time.time() - start:.1f}s")

    if use_cache:
        cache_data = {
            "n_rows": len(df),
            "nan_keep_cols": nan_keep_cols,
            "corr_keep_cols": corr_keep_cols,
            "mcar_cols": mcar_cols,
            "mar_cols": mar_cols,
            "complete_cols": complete_cols,
            "pipeline": pipeline,
            "feature_names": feature_names,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"  Cached preprocessing to {cache_file.name}")

    return X_train, X_test, y_train, y_test, feature_names, pipeline


def drop_high_nan_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove columns with excessive missing values.

    Computes NaN fraction on training data and applies same column selection
    to both train and test sets to prevent data leakage.
    """
    nan_frac = X_train.isna().mean()
    keep = nan_frac[nan_frac <= threshold].index
    return X_train[keep], X_test[keep]


def drop_correlated_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove redundant highly-correlated features.

    For each pair of features with absolute correlation above threshold,
    drops the second feature. Computed on training data only. Also removes
    zero-variance columns to avoid division-by-zero in correlation.
    """
    # Filter out zero-variance columns first (prevents NaN in correlation)
    non_constant = X_train.columns[X_train.std() > 0]
    X_train = X_train[non_constant]
    X_test = X_test[non_constant]

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X_train.drop(columns=to_drop), X_test.drop(columns=to_drop)


def build_simple_pipeline() -> Pipeline:
    """
    Build basic preprocessing pipeline with median imputation.

    Use for quick experiments. For production, prefer build_adaptive_pipeline()
    which selects imputation strategy based on missingness mechanism.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_split(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 77,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split DataFrame into train/test sets after removing non-feature columns.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 77,
    n_jobs: int = 2,
) -> RandomizedSearchCV:
    """
    Train XGBoost regressor with randomized hyperparameter search.

    Uses CUDA GPU acceleration. Searches over tree depth, learning rate,
    regularization, and subsampling parameters.

    Parameters
    ----------
    X_train
        Training features (preprocessed).
    y_train
        Training target values.
    n_iter
        Number of random hyperparameter combinations to try.
    cv
        Number of cross-validation folds.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object with best_estimator_ and cv_results_.
    """
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
        estimator=XGBRegressor(random_state=random_state, device="cuda"),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 77,
    n_jobs: int = 2,
) -> RandomizedSearchCV:
    """
    Train RandomForest regressor with randomized hyperparameter search.

    Parameters
    ----------
    X_train
        Training features (preprocessed).
    y_train
        Training target values.
    n_iter
        Number of random hyperparameter combinations to try.
    cv
        Number of cross-validation folds.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object with best_estimator_ and cv_results_.
    """
    from sklearn.ensemble import RandomForestRegressor

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
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Calculate RMSE on training and test sets.

    Returns
    -------
    dict
        Contains 'train_rmse' and 'test_rmse' values.
    """
    return {
        "train_rmse": rmse(model, X_train, y_train),
        "test_rmse": rmse(model, X_test, y_test),
    }


def rmse(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    """Calculate root mean squared error for model predictions."""
    return np.sqrt(mean_squared_error(y, model.predict(X)))


def extract_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.Series:
    """
    Extract and rank feature importances from tree-based model.

    Returns
    -------
    pd.Series
        Top N features sorted by importance (descending).
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


def save_model(
    model: BaseEstimator,
    path: Path,
    pipeline: Pipeline | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """
    Save trained model and preprocessing artifacts to disk.

    Saves a dictionary containing the model, fitted pipeline, and feature
    names for inference on new data.
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
    """
    Load model artifact from disk.

    Returns
    -------
    dict
        Contains 'model', 'pipeline', and 'feature_names' keys.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wildfire risk model.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to local input parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for local output model",
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
        "--cv",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5)",
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
    parser.add_argument(
        "--mar-thresh",
        type=float,
        default=0.1,
        help="MAR correlation threshold for adaptive imputation (default: 0.1)",
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
    print(f"mar_thresh:  {args.mar_thresh}")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print("\n[2/5] Splitting data...")
    y = df[args.target]
    X = df.drop(columns=[args.target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=77
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n[3/5] Preprocessing...")
    print(f"  Features before filtering: {X_train.shape[1]}")
    X_train, X_test = drop_high_nan_columns(X_train, X_test, threshold=args.nan_thresh)
    print(f"  After NaN filter: {X_train.shape[1]}")
    X_train, X_test = drop_correlated_features(
        X_train, X_test, threshold=args.corr_thresh
    )
    print(f"  After correlation filter: {X_train.shape[1]}")

    print("  Analyzing missing data patterns...")
    analysis = analyze_missingness(X_train, threshold=args.mar_thresh)
    print_missingness_report(analysis)

    mcar_cols, mar_cols, complete_cols = get_column_lists(
        analysis, X_train.columns.tolist()
    )
    feature_names = mcar_cols + mar_cols + complete_cols

    pipeline = build_adaptive_pipeline(
        mcar_cols, mar_cols, complete_cols, random_state=77
    )
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    print(f"  Final feature matrix: {X_train.shape}")

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

    print("\n[5/5] Saving model...")
    joblib.dump(best_model, args.output, pipeline=pipeline, feature_names=feature_names)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
