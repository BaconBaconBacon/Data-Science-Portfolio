"""Training smoke tests: preprocess → XGBoost/RandomForest on reduced feature set.

Uses the test DB (properties_test) with 20 census columns for speed.
Requires PostgreSQL + Census API key.

Run with:
    pytest test_train.py -v
"""

import numpy as np
import pandas as pd
import pytest

e2e = pytest.mark.e2e


@pytest.fixture()
def preprocessed_data(sql_test):
    """Preprocessed training data: 20 properties, 20 census cols, proximity features."""
    from load_properties import Properties
    from load_census import CensusData
    import load_wildfires
    from gis import calc_all_features_parallel
    from train import preprocess_with_cache

    # 1. Get properties (reuse any from prior tests)
    props = Properties(sql_obj=sql_test, verbose=False)
    if props.num_properties < 20:
        props.add_random_properties_geo_first(
            20 - props.num_properties, granularity="county"
        )
    properties_gdf = props.get_properties_gpd()

    # 2. Census merge, reduced to ~20 columns
    census = CensusData(
        sql_obj=sql_test, year=2023, granularity="county", verbose=False
    )
    combined = census.merge_census_info(properties_gdf)
    all_census = [c for c in combined.columns if c.startswith("B")]
    keep_census = all_census[:20]
    non_census = [c for c in combined.columns if not c.startswith("B")]
    combined = combined[non_census + keep_census]

    # 3. Wildfires + proximity
    wildfires = load_wildfires.WildfireData(sql_obj=sql_test)
    proximity = calc_all_features_parallel(
        combined, wildfires.data, n_jobs=2, use_cache=False
    )
    targets = pd.concat([combined, proximity], axis=1)

    # 4. Preprocess
    TARGET_COL = "nearest_fire_km"
    proximity_cols = list(proximity.columns)
    id_cols = ["geometry", "geoid", "block_id", "block_grp", "tract_id", "county_id", "state_id"]
    drop_cols = id_cols + [c for c in proximity_cols if c != TARGET_COL]
    drop_cols = [c for c in drop_cols if c in targets.columns]

    X_train, X_test, y_train, y_test, feature_names, pipeline = preprocess_with_cache(
        targets, TARGET_COL, drop_cols, use_cache=False
    )
    return X_train, X_test, y_train, y_test, feature_names, pipeline


@e2e
class TestTraining:
    def test_preprocessed_shapes(self, preprocessed_data):
        """Verify preprocessing produces valid train/test arrays."""
        X_train, X_test, y_train, y_test, feature_names, _ = preprocessed_data

        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert len(feature_names) == X_train.shape[1]
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)

    def test_train_xgboost(self, preprocessed_data):
        """XGBoost training smoke test (n_iter=2, cv=2)."""
        from train import train_xgboost, evaluate_model, extract_feature_importance

        X_train, X_test, y_train, y_test, feature_names, _ = preprocessed_data

        search = train_xgboost(X_train, y_train, n_iter=2, cv=2)

        assert hasattr(search, "best_estimator_")
        assert hasattr(search.best_estimator_, "feature_importances_")

        metrics = evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)
        assert metrics["train_rmse"] >= 0
        assert metrics["test_rmse"] >= 0

        top_features = extract_feature_importance(
            search.best_estimator_, feature_names, top_n=5
        )
        assert len(top_features) > 0
        assert (top_features >= 0).all()

    def test_train_random_forest(self, preprocessed_data):
        """RandomForest training smoke test (n_iter=2, cv=2)."""
        from train import train_random_forest, evaluate_model, extract_feature_importance

        X_train, X_test, y_train, y_test, feature_names, _ = preprocessed_data

        search = train_random_forest(X_train, y_train, n_iter=2, cv=2)

        assert hasattr(search, "best_estimator_")
        assert hasattr(search.best_estimator_, "feature_importances_")

        metrics = evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)
        assert metrics["train_rmse"] >= 0
        assert metrics["test_rmse"] >= 0

        top_features = extract_feature_importance(
            search.best_estimator_, feature_names, top_n=5
        )
        assert len(top_features) > 0
        assert (top_features >= 0).all()
