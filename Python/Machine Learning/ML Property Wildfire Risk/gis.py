"""Wildfire proximity scoring functions for property risk assessment.

Standalone utility functions that compute spatial proximity features
between property locations and wildfire detection points. All functions
expect GeoDataFrames in a projected CRS (meters), e.g. EPSG:5070.

Implements caching for calc_all_features() to avoid recomputation on re-runs.
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
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
        With p=2, this naturally downweights distant fires. This is what the reference's own theoretical formula was going for before they simplified it away.

    Parameters
    ----------
    properties : GeoDataFrame of property points.
    fires : GeoDataFrame of fire points.
    radius_m : Search radius in meters.
    power : Distance exponent (2 = inverse-square).
    weight_col : Optional column in fires to weight by (e.g. 'FRP').

    Returns
    -------
    pd.Series indexed like properties with IDW scores.
    """
    _validate_inputs(properties, fires)
    sindex = fires.sindex
    scores = np.zeros(len(properties))

    for i, (idx, row) in enumerate(properties.iterrows()):
        nearby, dists = _get_nearby_fires(row.geometry, fires, radius_m, sindex)
        if len(dists) == 0:
            continue
        # Clamp minimum distance to 1m to avoid division by zero
        dists = np.maximum(dists, 1.0)
        weights = nearby[weight_col].values if weight_col else np.ones(len(dists))
        scores[i] = np.sum(weights / dists**power)

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
     with distance). The bandwidth parameter controls how quickly.

    Parameters
    ----------
    properties : GeoDataFrame of property points.
    fires : GeoDataFrame of fire points.
    radius_m : Search radius in meters.
    bandwidth : Decay length scale in meters.
    weight_col : Optional column in fires to weight by (e.g. 'FRP').

    Returns
    -------
    pd.Series indexed like properties with decay scores.
    """
    _validate_inputs(properties, fires)
    sindex = fires.sindex
    scores = np.zeros(len(properties))

    for i, (idx, row) in enumerate(properties.iterrows()):
        nearby, dists = _get_nearby_fires(row.geometry, fires, radius_m, sindex)
        if len(dists) == 0:
            continue
        weights = nearby[weight_col].values if weight_col else np.ones(len(dists))
        scores[i] = np.sum(weights * np.exp(-dists / bandwidth))

    return pd.Series(scores, index=properties.index, name="exp_decay_score")


def calc_buffer_ring_features(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    rings_m: list[float] | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Count fires (and optionally sum weights) in concentric distance rings.

    — count fires in 0-10km, 10-25km, 25-50km as separate columns. Let the ML model
    learn the weighting rather than baking it into one formula. This is arguably the
    best approach when the score feeds into an ML model anyway.

    Parameters
    ----------
    properties : GeoDataFrame of property points.
    fires : GeoDataFrame of fire points.
    rings_m : Ring outer boundaries in meters. Default [10k, 25k, 50k, 100k].
    weight_col : Optional column in fires to sum per ring (e.g. 'FRP').

    Returns
    -------
    pd.DataFrame indexed like properties, with columns per ring for
    count and (optionally) weight sum.
    """
    _validate_inputs(properties, fires)
    if rings_m is None:
        rings_m = GIS_SCORING_DEFAULT_RINGS_M
    rings_m = sorted(rings_m)

    sindex = fires.sindex
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
    weight_cols = []
    if weight_col:
        weight_cols = [f"fire_{weight_col}_{lbl}" for lbl in ring_labels]
        columns += weight_cols

    result = np.zeros((len(properties), len(columns)))

    for i, (idx, row) in enumerate(properties.iterrows()):
        nearby, dists = _get_nearby_fires(row.geometry, fires, outer_radius, sindex)
        if len(dists) == 0:
            continue

        for j in range(len(rings_m)):
            inner = boundaries[j]
            outer = boundaries[j + 1]
            mask = (dists >= inner) & (dists < outer)
            result[i, j] = mask.sum()
            if weight_col:
                result[i, len(rings_m) + j] = nearby[weight_col].values[mask].sum()

    return pd.DataFrame(result, index=properties.index, columns=columns)


def calc_nearest_fire(
    properties: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
) -> pd.Series:
    """Distance to the nearest fire point, in kilometers.

    Parameters
    ----------
    properties : GeoDataFrame of property points.
    fires : GeoDataFrame of fire points.

    Returns
    -------
    pd.Series indexed like properties with nearest-fire distance in km.
    Properties with no fires in the dataset get np.inf.
    """
    _validate_inputs(properties, fires)
    sindex = fires.sindex
    distances_km = np.full(len(properties), np.inf)

    for i, (idx, row) in enumerate(properties.iterrows()):
        pt = row.geometry
        # Expanding search: start at 50km, widen if nothing found
        for search_radius in [50_000, 200_000, 500_000, np.inf]:
            if np.isinf(search_radius):
                # Brute force fallback
                all_dists = fires.geometry.distance(pt)
                distances_km[i] = all_dists.min() / 1000
                break
            nearby, dists = _get_nearby_fires(pt, fires, search_radius, sindex)
            if len(dists) > 0:
                distances_km[i] = dists.min() / 1000
                break

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

    print(f"Computing GIS features for {len(properties)} properties...")

    idw = calc_idw(properties, fires, radius_m, power, weight_col)
    kde = calc_kde(properties, fires, bandwidth)
    decay = calc_exponential_decay(properties, fires, radius_m, bandwidth, weight_col)
    rings = calc_buffer_ring_features(properties, fires, rings_m, weight_col)
    nearest = calc_nearest_fire(properties, fires)

    result = pd.concat([idw, kde, decay, rings, nearest], axis=1)

    # Save to cache
    if use_cache:
        result.reset_index(drop=True).to_parquet(cache_file)
        print(f"Cached GIS features to {cache_file.name}")

    return result
