"""Wildfire proximity scoring functions for property risk assessment.

Standalone utility functions that compute spatial proximity features
between property locations and wildfire detection points. All functions
expect GeoDataFrames in a projected CRS (meters), e.g. EPSG:5070.

Implements caching for calc_all_features() to avoid recomputation on re-runs.
Uses scipy.spatial.cKDTree for fast vectorized distance calculations.
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from settings import (
    GIS_SCORING_DEFAULT_RADIUS_M,
    GIS_SCORING_DEFAULT_POWER,
    GIS_SCORING_DEFAULT_BANDWIDTH_M,
    GIS_SCORING_DEFAULT_RINGS_M,
    PATH_DATA,
)


def _validate_inputs(properties: gpd.GeoDataFrame, fires: gpd.GeoDataFrame) -> None:
    if properties.crs != fires.crs:
        raise ValueError(
            f"CRS mismatch: properties={properties.crs}, fires={fires.crs}. "
            "Reproject to a common CRS before scoring."
        )
    if properties.crs and properties.crs.is_geographic:
        warnings.warn(
            "Both GeoDataFrames use a geographic CRS (degrees). "
            "Distances will be in degrees, not meters. "
            "Consider reprojecting to a projected CRS (e.g. EPSG:5070)."
        )


def _get_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract x,y coordinates as numpy array for KDTree."""
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


def _build_kdtree(fires: gpd.GeoDataFrame) -> tuple[cKDTree, np.ndarray]:
    """Build a KDTree from fire coordinates for fast spatial queries."""
    coords = _get_coords(fires)
    return cKDTree(coords), coords


def _get_nearby_fires(
    point, fires: gpd.GeoDataFrame, radius_m: float, spatial_index
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Return fires within radius_m of point and their distances.

    Uses the spatial index for a fast bounding-box pre-filter,
    then exact distance filtering.
    """
    bounds = (
        point.x - radius_m,
        point.y - radius_m,
        point.x + radius_m,
        point.y + radius_m,
    )
    candidate_idx = list(spatial_index.intersection(bounds))
    if not candidate_idx:
        return fires.iloc[0:0], np.array([])

    candidates = fires.iloc[candidate_idx]
    distances = candidates.geometry.distance(point).values
    mask = distances < radius_m
    return candidates[mask], distances[mask]


def calc_idw(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    radius_m: float = GIS_SCORING_DEFAULT_RADIUS_M,
    power: float = GIS_SCORING_DEFAULT_POWER,
    weight_col: str | None = None,
) -> pd.Series:
    """Inverse distance weighting: sum(w_j / d_j^p) for fires within radius.

    The most common spatial proximity metric:
        score = Σ (1 / d_j^p)   for all fires within radius
        With p=2, this naturally downweights distant fires.

    Uses KDTree for fast vectorized computation.
    """
    _validate_inputs(properties, fires)

    tree, fire_coords = _build_kdtree(fires)
    prop_coords = _get_coords(properties)
    weights = fires[weight_col].values if weight_col else None

    scores = np.zeros(len(properties))

    # Query all neighbors within radius for all properties at once
    neighbors = tree.query_ball_point(prop_coords, radius_m)

    for i, neighbor_idx in enumerate(neighbors):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(prop_coords[i] - fire_coords[neighbor_idx], axis=1)
        dists = np.maximum(dists, 1.0)  # Clamp to avoid division by zero
        w = weights[neighbor_idx] if weights is not None else np.ones(len(dists))
        scores[i] = np.sum(w / dists**power)

    return pd.Series(scores, index=properties.index, name="idw_score")


def calc_kde(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    bandwidth: float = GIS_SCORING_DEFAULT_BANDWIDTH_M,
) -> pd.Series:
    """Kernel density estimation of fire intensity at property locations.

    Builds a 2D Gaussian KDE from fire locations and evaluates it
    at each property point. Returns log1p-transformed density to
    reduce right-skew.

    Steps: build a continuous fire density surface, then sample at
    property locations. This is the textbook GIS approach for point
    pattern risk surfaces, and geopandas/scipy support it.


    Parameters
    ----------
    properties : GeoDataFrame of property points.
    fires : GeoDataFrame of fire points.
    bandwidth : KDE bandwidth in meters (Gaussian sigma).

    Returns
    -------
    pd.Series indexed like properties with log-density values.
    """
    _validate_inputs(properties, fires)

    fire_coords = np.vstack([fires.geometry.x, fires.geometry.y])
    kde = gaussian_kde(
        fire_coords, bw_method=bandwidth / fire_coords.std(axis=1).mean()
    )

    prop_coords = np.vstack([properties.geometry.x, properties.geometry.y])
    density = kde(prop_coords)

    return pd.Series(np.log1p(density), index=properties.index, name="kde_log_density")


def calc_exponential_decay(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    radius_m: float = GIS_SCORING_DEFAULT_RADIUS_M,
    bandwidth: float = GIS_SCORING_DEFAULT_BANDWIDTH_M,
    weight_col: str | None = None,
) -> pd.Series:
    """Exponential decay score: sum(w_j * exp(-d_j / bandwidth)).

    Has a natural physical interpretation (risk decays exponentially
    with distance). Uses KDTree for fast vectorized computation.
    """
    _validate_inputs(properties, fires)

    tree, fire_coords = _build_kdtree(fires)
    prop_coords = _get_coords(properties)
    weights = fires[weight_col].values if weight_col else None

    scores = np.zeros(len(properties))
    neighbors = tree.query_ball_point(prop_coords, radius_m)

    for i, neighbor_idx in enumerate(neighbors):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(prop_coords[i] - fire_coords[neighbor_idx], axis=1)
        w = weights[neighbor_idx] if weights is not None else np.ones(len(dists))
        scores[i] = np.sum(w * np.exp(-dists / bandwidth))

    return pd.Series(scores, index=properties.index, name="exp_decay_score")


def calc_buffer_ring_features(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    rings_m: list[float] | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Count fires (and optionally sum weights) in concentric distance rings.

    Uses KDTree for fast vectorized computation.
    """
    _validate_inputs(properties, fires)
    if rings_m is None:
        rings_m = GIS_SCORING_DEFAULT_RINGS_M
    rings_m = sorted(rings_m)

    tree, fire_coords = _build_kdtree(fires)
    prop_coords = _get_coords(properties)
    weights = fires[weight_col].values if weight_col else None
    outer_radius = rings_m[-1]

    # Build column names
    boundaries = [0] + rings_m
    ring_labels = []
    for j in range(len(rings_m)):
        inner_km = boundaries[j] // 1000
        outer_km = boundaries[j + 1] // 1000
        ring_labels.append(f"{inner_km}_{outer_km}km")

    count_cols = [f"fire_count_{lbl}" for lbl in ring_labels]
    columns = list(count_cols)
    if weight_col:
        weight_cols = [f"fire_{weight_col}_{lbl}" for lbl in ring_labels]
        columns += weight_cols

    result = np.zeros((len(properties), len(columns)))
    neighbors = tree.query_ball_point(prop_coords, outer_radius)

    for i, neighbor_idx in enumerate(neighbors):
        if not neighbor_idx:
            continue
        dists = np.linalg.norm(prop_coords[i] - fire_coords[neighbor_idx], axis=1)

        for j in range(len(rings_m)):
            inner = boundaries[j]
            outer = boundaries[j + 1]
            mask = (dists >= inner) & (dists < outer)
            result[i, j] = mask.sum()
            if weight_col:
                result[i, len(rings_m) + j] = weights[np.array(neighbor_idx)[mask]].sum()

    return pd.DataFrame(result, index=properties.index, columns=columns)


def calc_nearest_fire(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
) -> pd.Series:
    """Distance to the nearest fire point, in kilometers.

    Fully vectorized using KDTree.query() - extremely fast.
    """
    _validate_inputs(properties, fires)

    tree, _ = _build_kdtree(fires)
    prop_coords = _get_coords(properties)

    # Query nearest neighbor for all properties at once
    distances_m, _ = tree.query(prop_coords, k=1)
    distances_km = distances_m / 1000

    return pd.Series(distances_km, index=properties.index, name="nearest_fire_km")


def _generate_cache_key(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    radius_m: float,
    power: float,
    bandwidth: float,
    rings_m: list[float] | None,
    weight_col: str | None,
) -> str:
    """Generate a hash key for caching based on inputs and parameters."""
    # Use property count + bounds + fire count as a fingerprint
    # (Full geometry hashing would be too slow for large datasets)
    prop_bounds = properties.total_bounds
    fire_bounds = fires.total_bounds

    key_parts = [
        f"props_{len(properties)}",
        f"prop_bounds_{prop_bounds[0]:.2f}_{prop_bounds[1]:.2f}_{prop_bounds[2]:.2f}_{prop_bounds[3]:.2f}",
        f"fires_{len(fires)}",
        f"fire_bounds_{fire_bounds[0]:.2f}_{fire_bounds[1]:.2f}_{fire_bounds[2]:.2f}_{fire_bounds[3]:.2f}",
        f"radius_{radius_m}",
        f"power_{power}",
        f"bandwidth_{bandwidth}",
        f"rings_{rings_m}",
        f"weight_{weight_col}",
    ]
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()[:12]


def calc_all_features(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    radius_m: float = GIS_SCORING_DEFAULT_RADIUS_M,
    power: float = GIS_SCORING_DEFAULT_POWER,
    bandwidth: float = GIS_SCORING_DEFAULT_BANDWIDTH_M,
    rings_m: list[float] | None = None,
    weight_col: str | None = "FRP",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compute all proximity features in one call.

    Returns a DataFrame combining IDW, KDE, exponential decay,
    buffer ring counts, and nearest-fire distance.

    Parameters
    ----------
    properties : GeoDataFrame of property points.
    fires : GeoDataFrame of fire points.
    radius_m : Search radius in meters for IDW and decay.
    power : Distance exponent for IDW.
    bandwidth : Bandwidth for KDE and decay.
    rings_m : Ring boundaries for buffer features.
    weight_col : Column in fires to weight by.
    use_cache : If True, cache results to disk and load on re-runs.

    Returns
    -------
    pd.DataFrame with all proximity features.
    """
    _validate_inputs(properties, fires)

    # Check cache
    cache_dir = PATH_DATA / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = _generate_cache_key(
        properties, fires, radius_m, power, bandwidth, rings_m, weight_col
    )
    cache_file = cache_dir / f"gis_features_{cache_key}.parquet"

    if use_cache and cache_file.exists():
        print(f"Loading cached GIS features from {cache_file.name}")
        cached = pd.read_parquet(cache_file)
        # Verify row count matches (sanity check)
        if len(cached) == len(properties):
            return cached.set_index(properties.index)
        else:
            print("Cache size mismatch, recomputing...")

    import time
    print(f"Computing GIS features for {len(properties)} properties...")
    start = time.time()

    print("  [1/5] Nearest fire distance...", end=" ", flush=True)
    t0 = time.time()
    nearest = calc_nearest_fire(properties, fires)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  [2/5] KDE density...", end=" ", flush=True)
    t0 = time.time()
    kde = calc_kde(properties, fires, bandwidth)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  [3/5] IDW scores...", end=" ", flush=True)
    t0 = time.time()
    idw = calc_idw(properties, fires, radius_m, power, weight_col)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  [4/5] Exponential decay...", end=" ", flush=True)
    t0 = time.time()
    decay = calc_exponential_decay(properties, fires, radius_m, bandwidth, weight_col)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  [5/5] Buffer ring counts...", end=" ", flush=True)
    t0 = time.time()
    rings = calc_buffer_ring_features(properties, fires, rings_m, weight_col)
    print(f"done ({time.time() - t0:.1f}s)")

    print(f"  Total: {time.time() - start:.1f}s")

    result = pd.concat([idw, kde, decay, rings, nearest], axis=1)

    # Save to cache
    if use_cache:
        result.reset_index(drop=True).to_parquet(cache_file)
        print(f"Cached GIS features to {cache_file.name}")

    return result
