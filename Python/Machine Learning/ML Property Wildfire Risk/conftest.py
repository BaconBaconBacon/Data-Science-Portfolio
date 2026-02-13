"""Shared pytest fixtures for all test files."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point


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