"""Automated tests for the wildfire risk ML pipeline.

Run with:
    pytest test_pipeline.py -v                       # Full suite (~2 min)
    pytest test_pipeline.py -m "not db and not e2e"  # Fast unit tests (~10s)
    pytest test_pipeline.py -m "db and not e2e"      # DB tests only (~30s)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from pathlib import Path
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
db = pytest.mark.db
e2e = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sql_test():
    """Shared test DB connection (session-scoped for speed)."""
    from sql_funcs import SQL

    SQL.kill_idle(True)
    obj = SQL(test=True)
    yield obj
    obj.disconnect_and_close()


@pytest.fixture
def small_properties_gdf():
    """10 synthetic property points in EPSG:5070 (CONUS projected CRS)."""
    rng = np.random.default_rng(42)
    xs = rng.uniform(500_000, 2_500_000, 10)
    ys = rng.uniform(200_000, 2_500_000, 10)
    geom = [Point(x, y) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame(
        {"geoid": [f"geo_{i}" for i in range(10)]},
        geometry=geom,
        crs="EPSG:5070",
    )


@pytest.fixture
def small_fires_gdf():
    """5 synthetic fire points in EPSG:5070 with required columns."""
    rng = np.random.default_rng(99)
    xs = rng.uniform(500_000, 2_500_000, 5)
    ys = rng.uniform(200_000, 2_500_000, 5)
    geom = [Point(x, y) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame(
        {
            "FRP": rng.uniform(1.0, 100.0, 5),
            "ACQ_DATE": pd.to_datetime(["2024-01-01"] * 5),
        },
        geometry=geom,
        crs="EPSG:5070",
    )


# ===================================================================
# FAST UNIT TESTS — no DB or API needed
# ===================================================================


class TestSettings:
    def test_census_features_nonempty(self):
        from settings import CENSUS_FEATURES

        assert len(CENSUS_FEATURES) > 10
        assert all(f.startswith("B") for f in CENSUS_FEATURES)

    def test_table_names_defined(self):
        from settings import (
            TABLE_NAME_CENSUS,
            TABLE_NAME_CENSUS_TEST,
            TABLE_NAME_CENSUS_PROPS,
            WILDFIRES_TABLE_NAME,
        )

        assert TABLE_NAME_CENSUS == "census"
        assert TABLE_NAME_CENSUS_TEST == "census_test"
        assert TABLE_NAME_CENSUS_PROPS == "props_census"
        assert WILDFIRES_TABLE_NAME == "wildfires"

    def test_crs_is_projected(self):
        from settings import GIS_DEFAULT_CRS

        assert GIS_DEFAULT_CRS == 5070


class TestMissingnessAnalysis:
    def test_mcar_detection(self):
        """Features with random missingness should be classified as MCAR."""
        from missing_analysis import analyze_missingness

        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(0, 1, n),
            }
        )
        # Add random missingness to feat_a (independent of feat_b)
        mask = rng.random(n) < 0.2
        df.loc[mask, "feat_a"] = np.nan

        result = analyze_missingness(df, threshold=0.1)
        assert "feat_a" in result
        assert result["feat_a"]["mechanism"] == "MCAR"

    def test_mar_detection(self):
        """Features with correlated missingness should be classified as MAR."""
        from missing_analysis import analyze_missingness

        rng = np.random.default_rng(42)
        n = 1000
        feat_b = rng.normal(0, 1, n)
        feat_a = rng.normal(0, 1, n)
        # Make feat_a missing when feat_b is high (correlated missingness)
        df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b})
        df.loc[feat_b > 0.5, "feat_a"] = np.nan

        result = analyze_missingness(df, threshold=0.1)
        assert "feat_a" in result
        assert result["feat_a"]["mechanism"] == "MAR"
        assert result["feat_a"]["corr_with"] == "feat_b"

    def test_no_missing_returns_empty(self):
        from missing_analysis import analyze_missingness

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert analyze_missingness(df) == {}

    def test_zero_variance_indicator_no_warning(self):
        """Near-constant missingness indicator should not cause divide-by-zero."""
        from missing_analysis import analyze_missingness

        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(0, 1, n),
            }
        )
        # Only 1 missing value → indicator has near-zero variance
        df.iloc[0, 0] = np.nan

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = analyze_missingness(df, threshold=0.1)
        # Should complete without RuntimeWarning


class TestPreprocessingHelpers:
    def test_drop_high_nan_columns(self):
        from train import drop_high_nan_columns

        train_df = pd.DataFrame(
            {
                "good": [1, 2, 3, 4, 5],
                "bad": [np.nan, np.nan, np.nan, 4, 5],  # 60% NaN
            }
        )
        test_df = train_df.copy()

        result_train, result_test = drop_high_nan_columns(
            train_df, test_df, threshold=0.45
        )
        assert "good" in result_train.columns
        assert "bad" not in result_train.columns
        assert list(result_train.columns) == list(result_test.columns)

    def test_drop_correlated_features(self):
        from train import drop_correlated_features

        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        train_df = pd.DataFrame(
            {
                "a": x,
                "b": x + rng.normal(0, 0.01, n),  # Near-perfect correlation
                "c": rng.normal(0, 1, n),  # Independent
            }
        )
        test_df = train_df.copy()

        result_train, result_test = drop_correlated_features(
            train_df, test_df, threshold=0.85
        )
        # One of a/b should be dropped
        assert result_train.shape[1] == 2
        assert "c" in result_train.columns

    def test_constant_column_dropped(self):
        """Zero-variance columns should be removed by correlation filter."""
        from train import drop_correlated_features

        train_df = pd.DataFrame(
            {
                "const": [5.0] * 50,
                "vary": np.random.default_rng(0).normal(0, 1, 50),
            }
        )
        test_df = train_df.copy()
        result_train, _ = drop_correlated_features(train_df, test_df)
        assert "const" not in result_train.columns


class TestGIS:
    def test_validate_inputs_crs_mismatch(self, small_properties_gdf, small_fires_gdf):
        from gis import _validate_inputs

        fires_wrong_crs = small_fires_gdf.to_crs("EPSG:4326")
        with pytest.raises(ValueError, match="CRS mismatch"):
            _validate_inputs(small_properties_gdf, fires_wrong_crs)

    def test_validate_inputs_matching_crs(self, small_properties_gdf, small_fires_gdf):
        from gis import _validate_inputs

        _validate_inputs(small_properties_gdf, small_fires_gdf)  # Should not raise

    def test_proximity_features_shape(self, small_properties_gdf, small_fires_gdf):
        from gis import calc_all_features_parallel

        result = calc_all_features_parallel(
            small_properties_gdf,
            small_fires_gdf,
            use_cache=False,
            n_jobs=1,
        )
        assert len(result) == 10
        # Should have: nearest_fire_km, kde_density, idw_score, exp_decay_score,
        # fire_count_* (4 rings), fire_FRP_* (4 rings) = 12 features
        assert result.shape[1] == 12
        assert "nearest_fire_km" in result.columns
        assert "exp_decay_score" in result.columns
        # No NaN in proximity features (all properties should have distances)
        assert result["nearest_fire_km"].isna().sum() == 0

    def test_proximity_features_positive_distances(
        self, small_properties_gdf, small_fires_gdf
    ):
        from gis import calc_all_features_parallel

        result = calc_all_features_parallel(
            small_properties_gdf,
            small_fires_gdf,
            use_cache=False,
            n_jobs=1,
        )
        assert (result["nearest_fire_km"] >= 0).all()


# ===================================================================
# DB INTEGRATION TESTS — needs PostgreSQL running
# ===================================================================


@db
class TestSQLConnection:
    def test_connect(self, sql_test):
        assert sql_test.connection is not None
        assert sql_test.test_mode is True

    def test_check_table_exists_nonexistent(self, sql_test):
        assert sql_test.check_table_exists("nonexistent_table_xyz") is False


@db
class TestSQLDataRoundTrip:
    def test_save_and_read_df(self, sql_test):
        """DataFrame round-trip through SQL."""
        table = "test_roundtrip_df"
        if sql_test.check_table_exists(table):
            sql_test.drop_table(table, confirm=True)

        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4.0, 5.0, 6.0]})
        sql_test.save_df_to_sql(table, df)
        result = sql_test.read_df_from_sql(f"SELECT * FROM {table}")

        assert len(result) == 3
        assert set(result.columns) >= {"col_a", "col_b"}
        sql_test.drop_table(table, confirm=True)

    def test_save_and_read_gpd_preserves_columns(self, sql_test):
        """GeoDataFrame round-trip preserves mixed types and column case."""
        table = "test_roundtrip_gpd"
        if sql_test.check_table_exists(table):
            sql_test.drop_table(table, confirm=True)

        gdf = gpd.GeoDataFrame(
            {
                "ACQ_DATE": pd.to_datetime(["2024-06-15"]),
                "FRP": [42.5],
                "SAT_ID": ["J1V-C2"],
                "COUNT": [7],
            },
            geometry=[Point(1_000_000, 1_000_000)],
            crs="EPSG:5070",
        )
        sql_test.save_gpd_to_sql(table, gdf)
        result = sql_test.read_gpd_from_sql(table)

        assert "SAT_ID" in result.columns, f"Column case lost. Cols: {result.columns.tolist()}"
        assert "ACQ_DATE" in result.columns
        assert "FRP" in result.columns
        assert result["SAT_ID"].iloc[0] == "J1V-C2"
        assert result["FRP"].iloc[0] == pytest.approx(42.5)

        sql_test.drop_table(table, confirm=True)

    def test_table_auto_creation(self, sql_test):
        """save_gpd_to_sql creates table automatically if it doesn't exist."""
        table = "test_auto_create"
        if sql_test.check_table_exists(table):
            sql_test.drop_table(table, confirm=True)

        assert not sql_test.check_table_exists(table)

        gdf = gpd.GeoDataFrame(
            {"val": [1.0]},
            geometry=[Point(0, 0)],
            crs="EPSG:5070",
        )
        sql_test.save_gpd_to_sql(table, gdf)

        assert sql_test.check_table_exists(table)
        sql_test.drop_table(table, confirm=True)

    def test_duplicate_removal(self, sql_test):
        """Dedup uses property_id/geoid/geometry — needs a properties-like table."""
        from load_properties import Properties

        props = Properties(sql_obj=sql_test, verbose=False)
        table = props.table_name

        # Insert duplicate geoid+geometry rows
        gdf = gpd.GeoDataFrame(
            {
                "geoid": ["99999"] * 3,
                "block_id": [0] * 3,
                "block_grp": [0] * 3,
                "tract_id": [0] * 3,
                "county_id": [99] * 3,
                "state_id": [99] * 3,
            },
            geometry=[Point(0, 0)] * 3,
            crs="EPSG:5070",
        )
        sql_test.save_gpd_to_sql(table, gdf)

        count_before = sql_test.read_df_from_sql(
            f"SELECT COUNT(*) as n FROM {table} WHERE geoid = '99999'"
        )["n"].iloc[0]
        assert count_before == 3

        sql_test.drop_duplicates_from_table(table)

        count_after = sql_test.read_df_from_sql(
            f"SELECT COUNT(*) as n FROM {table} WHERE geoid = '99999'"
        )["n"].iloc[0]
        assert count_after == 1

        # Clean up test rows
        sql_test.connection.execute(
            __import__("sqlalchemy").text(f"DELETE FROM {table} WHERE geoid = '99999'")
        )
        sql_test.connection.commit()


# ===================================================================
# MINI END-TO-END TEST — needs PostgreSQL + Census API key
# ===================================================================


@e2e
class TestMiniPipeline:
    def test_full_pipeline_20_properties(self, sql_test):
        """End-to-end: 20 properties → census → wildfires → proximity features."""
        from load_properties import Properties
        from load_census import CensusData
        import load_wildfires
        from gis import calc_all_features_parallel

        # 1. Generate small set of properties
        props = Properties(sql_obj=sql_test, verbose=False)
        if props.num_properties < 20:
            props.add_random_properties_geo_first(
                20 - props.num_properties, granularity="county"
            )
        properties_gdf = props.get_properties_gpd()
        assert len(properties_gdf) >= 20
        assert "geoid" in properties_gdf.columns
        assert properties_gdf.crs is not None

        # 2. Merge census data
        census = CensusData(
            sql_obj=sql_test, year=2023, granularity="county", verbose=False
        )
        combined = census.merge_census_info(properties_gdf)
        census_cols = [c for c in combined.columns if c.startswith("B")]
        assert len(census_cols) > 100, f"Expected 100+ census cols, got {len(census_cols)}"
        assert len(combined) == len(properties_gdf)

        # 3. Load wildfire data
        wildfires = load_wildfires.WildfireData(sql_obj=sql_test)
        assert len(wildfires.data) > 0
        assert "FRP" in wildfires.data.columns

        # 4. Compute proximity features
        proximity = calc_all_features_parallel(
            combined, wildfires.data, n_jobs=2, use_cache=False
        )
        assert proximity.shape[1] == 12
        assert proximity["nearest_fire_km"].isna().sum() == 0

        # 5. Final assembly
        targets = pd.concat([combined, proximity], axis=1)
        assert targets.shape[1] > 850
        assert "nearest_fire_km" in targets.columns
        assert "geometry" in targets.columns
