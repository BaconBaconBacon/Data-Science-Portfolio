"""Property data generation and management for wildfire risk modeling.

Generates random US property locations by sampling coordinates within
Continental US (CONUS) bounds, then reverse-geocodes via the Census
Geocoder API to obtain geographic identifiers (state, county, tract,
block group) and persists to PostGIS.

Supports both sequential and parallel property generation for speed.
Test mode generates properties near known wildfire locations for validation.

The code automatically retries failed geocodes (points landing in water,
forests, or unpopulated areas) until exactly N properties are obtained,
with a safety cap of 5*N total attempts.

"""

import censusgeocode as cg
import io
import random
import requests
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from shapely.geometry import Point, Polygon, MultiPolygon
from sql_funcs import SQL
from settings import (
    GIS_DEFAULT_CRS,
    PROP_TABLE_NAME,
    PROP_LABELS_KEYS_MAP,
    PROPERTIES_INIT_COUNT,
    PROP_TABLE_NAME_TEST,
    HEADER_GEOM,
    USA_MIN_LAT,
    USA_MAX_LAT,
    USA_MIN_LON,
    USA_MAX_LON,
    EARTH_RADIUS_KM,
    PROP_ESTIMATE_SAMPLE,
    PROP_PROGRESS_INTERVAL,
    PROP_SAVE_INTERVAL,
    PATH_DATA_CENSUS,
    TIGER_YEAR,
    STATE_FIPS_CODES,
    CONUS_STATE_FIPS,
    CENSUS_VALID_GRANULARITY_LEVELS,
)


def download_tiger_shapefile(granularity: str, year: int = TIGER_YEAR) -> gpd.GeoDataFrame:
    """
    Download Census TIGER shapefile for the specified granularity.

    Downloads shapefiles from Census Bureau, caches locally as parquet.
    County is a single national file; tract/block_group require per-state downloads.

    Parameters
    ----------
    granularity : str
        One of: 'county', 'tract', 'block_group'
    year : int
        TIGER year (default: from settings)

    Returns
    -------
    gpd.GeoDataFrame
        Shapefile with geography boundaries and identifiers
    """
    if granularity not in CENSUS_VALID_GRANULARITY_LEVELS:
        raise ValueError(f"granularity must be one of {CENSUS_VALID_GRANULARITY_LEVELS}")

    cache_path = PATH_DATA_CENSUS / f"tiger_{granularity}_{year}.parquet"
    PATH_DATA_CENSUS.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached TIGER {granularity} shapefile...")
        return gpd.read_parquet(cache_path)

    print(f"Downloading TIGER {granularity} shapefile (year={year})...")

    if granularity == "county":
        # Single national file
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
        gdf = _download_shapefile_from_url(url)
    else:
        # Per-state files for tract and block_group
        folder = "TRACT" if granularity == "tract" else "BG"
        suffix = "tract" if granularity == "tract" else "bg"

        gdfs = []
        for i, fips in enumerate(CONUS_STATE_FIPS):
            url = f"https://www2.census.gov/geo/tiger/TIGER{year}/{folder}/tl_{year}_{fips}_{suffix}.zip"
            print(f"  [{i+1}/{len(CONUS_STATE_FIPS)}] Downloading state {fips}...")
            try:
                state_gdf = _download_shapefile_from_url(url)
                gdfs.append(state_gdf)
            except Exception as e:
                print(f"    Warning: Failed to download state {fips}: {e}")

        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Standardize column names for consistency
    gdf = _standardize_tiger_columns(gdf, granularity)

    # Filter to contiguous US only (exclude AK, HI, PR, territories)
    conus_fips_int = [int(f) for f in CONUS_STATE_FIPS]
    before_count = len(gdf)
    gdf = gdf[gdf["state_id"].isin(conus_fips_int)]
    print(f"  Filtered to CONUS: {before_count} -> {len(gdf)} geographies")

    # Cache to parquet
    print(f"Caching to {cache_path}...")
    gdf.to_parquet(cache_path)

    return gdf


def _download_shapefile_from_url(url: str) -> gpd.GeoDataFrame:
    """Download and extract shapefile from URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Find the .shp file
        shp_name = [n for n in zf.namelist() if n.endswith('.shp')][0]
        # Extract to temp and read
        with zf.open(shp_name.replace('.shp', '.shp')) as shp_file:
            # GeoPandas can read from zip directly
            pass

    # Read directly from zip in memory
    return gpd.read_file(io.BytesIO(response.content))


def _standardize_tiger_columns(gdf: gpd.GeoDataFrame, granularity: str) -> gpd.GeoDataFrame:
    """Standardize TIGER column names to match our schema."""
    # TIGER uses: STATEFP, COUNTYFP, TRACTCE, BLKGRPCE, GEOID
    rename_map = {
        "STATEFP": "state_id",
        "COUNTYFP": "county_id",
        "TRACTCE": "tract_id",
        "BLKGRPCE": "block_grp",
        "GEOID": "geoid",
    }

    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in gdf.columns}
    gdf = gdf.rename(columns=rename_map)

    # Convert ID columns to int where possible
    for col in ["state_id", "county_id", "tract_id", "block_grp"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0).astype(int)

    return gdf


def random_point_in_polygon(geom) -> Point:
    """
    Generate a random point uniformly within a polygon or multipolygon.

    Uses rejection sampling: generates points within bounding box until
    one falls inside the polygon.
    """
    if isinstance(geom, MultiPolygon):
        # Pick a polygon weighted by area
        areas = [p.area for p in geom.geoms]
        total = sum(areas)
        r = random.random() * total
        cumulative = 0
        for p, a in zip(geom.geoms, areas):
            cumulative += a
            if r <= cumulative:
                geom = p
                break

    minx, miny, maxx, maxy = geom.bounds
    max_attempts = 1000

    for _ in range(max_attempts):
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )
        if geom.contains(point):
            return point

    # Fallback: return centroid if rejection sampling fails (very concave shapes)
    return geom.centroid


class Properties:
    """Property location manager with Census geographic identifiers.

    Generates, stores, and retrieves random US property locations with
    associated Census geography codes for later enrichment with ACS5 data.

    Parameters
    ----------
    sql_obj : SQL
        Database connection manager instance.

    Attributes
    ----------
    properties_gpd : gpd.GeoDataFrame
        Property locations with columns: geoid, block_id, block_grp,
        tract_id, county_id, state_id, geometry.
    num_properties : int
        Current count of properties in the database.
    test_mode : bool
        Inherited from sql_obj; uses test table names if True.

    Examples
    --------
    >>> sql = SQL()
    >>> props = Properties(sql)
    >>> props.add_random_properties(100, parallel=True, max_workers=10)
    >>> gdf = props.get_properties_gpd()
    """

    def __init__(
        self,
        sql_obj: SQL,
        verbose: bool = True,
    ):

        self.sql_obj = sql_obj
        self.test_mode = sql_obj.test_mode
        self.verbose = verbose

        if not self.test_mode:
            self.table_name = PROP_TABLE_NAME
        else:
            self.table_name = PROP_TABLE_NAME_TEST
        self._read_from_sql()
        if self.verbose:
            print(f"{self.num_properties} properties loaded.")

    def add_random_properties(
        self, quantity: int, verbose=True, parallel=False, max_workers=10
    ) -> None:
        """
        Add 'quantity' many new random addresses.

        Parameters
        ----------
        quantity : int
            Number of properties to add.
        verbose : bool
            Print progress updates.
        parallel : bool
            If True, use parallel API calls (faster but more aggressive).
        max_workers : int
            Number of parallel threads (only used if parallel=True).
        """
        if self.test_mode:
            # TODO: Check if there arent any in the test property table
            print("Test mode, cannot add more properties.")
            return

        # Route to parallel version if requested
        if parallel:
            return self._add_random_properties_parallel(quantity, verbose, max_workers)

        print(f"Adding {quantity} properties...")

        temp_lst = []
        num_added = 0
        attempts = 0
        max_attempts = quantity * 5  # Safety cap
        last_saved = 0
        start_time = time.time()
        estimate_sample = min(PROP_ESTIMATE_SAMPLE, quantity)
        estimate_shown = False

        while num_added < quantity and attempts < max_attempts:
            attempts += 1

            # Generate random coordinates within Continental US bounds
            lat = random.uniform(USA_MIN_LAT, USA_MAX_LAT)
            long = random.uniform(USA_MIN_LON, USA_MAX_LON)

            try:
                block = cg.coordinates(x=long, y=lat)["2020 Census Blocks"][0]
            except Exception:
                continue  # Retry with new point

            prop = {
                key: (
                    block[PROP_LABELS_KEYS_MAP[key]]
                    if key == "geoid"
                    else int(block[PROP_LABELS_KEYS_MAP[key]])
                )
                for key in PROP_LABELS_KEYS_MAP.keys()
            }
            prop[HEADER_GEOM] = Point(long, lat)
            temp_lst.append(prop)
            num_added += 1

            # Show time estimate after first N properties
            if num_added == estimate_sample and not estimate_shown and verbose:
                elapsed = time.time() - start_time
                per_property = elapsed / estimate_sample
                remaining = quantity - estimate_sample
                est_remaining_sec = remaining * per_property
                success_rate = num_added / attempts * 100 if attempts > 0 else 0
                print(
                    f"  Time estimate: {self._format_duration(est_remaining_sec)} remaining "
                    f"({per_property:.2f}s per property, {success_rate:.1f}% success rate)"
                )
                estimate_shown = True

            if num_added % PROP_PROGRESS_INTERVAL == 0:
                if verbose:
                    elapsed = time.time() - start_time
                    remaining_count = quantity - num_added
                    per_property = elapsed / num_added
                    est_remaining = remaining_count * per_property
                    success_rate = num_added / attempts * 100 if attempts > 0 else 0
                    print(
                        f"  {num_added}/{quantity} properties "
                        f"({attempts} attempts, {success_rate:.1f}% success) "
                        f"- {self._format_duration(est_remaining)} remaining"
                    )
                self._update_gpd_and_sql(temp_lst[last_saved:])
                last_saved = len(temp_lst)

        if last_saved < len(temp_lst):
            self._update_gpd_and_sql(temp_lst[last_saved:])

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        # Reload from SQL to get auto-generated property_ids
        self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
        self._update_property_count()

        total_time = time.time() - start_time
        if verbose:
            success_rate = num_added / attempts * 100 if attempts > 0 else 0
            print(
                f"Completed in {self._format_duration(total_time)} "
                f"({num_added} added from {attempts} attempts, {success_rate:.1f}% success)"
            )

        if num_added < quantity:
            print(
                f"Warning: Only added {num_added}/{quantity} after {max_attempts} attempts."
            )

        return

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def add_random_properties_geo_first(
        self,
        quantity: int,
        granularity: str = "tract",
        verbose: bool = True,
        skip_final_cleanup: bool = False,
    ) -> None:
        """
        Add random properties using geography-first approach.

        Downloads TIGER shapefiles, randomly selects geographies, then generates
        random points within those geometries. Guarantees 100% success rate
        since all points fall within valid census geographies.

        Parameters
        ----------
        quantity : int
            Number of properties to add.
        granularity : str
            Geographic level: 'county', 'tract', or 'block_group'.
        verbose : bool
            Print progress updates.
        skip_final_cleanup : bool
            If True, skip slow cleanup operations (drop_duplicates, reload from SQL).
            Useful when caller needs to signal completion before cleanup.
        """
        if self.test_mode:
            print("Test mode, cannot add more properties.")
            return

        print(f"Adding {quantity} properties using geography-first approach...")
        start_time = time.time()

        # Load TIGER shapefile (downloads if not cached)
        tiger_gdf = download_tiger_shapefile(granularity)
        n_geos = len(tiger_gdf)
        print(f"  Loaded {n_geos} {granularity} geographies")

        results = []
        last_saved = 0
        save_interval = PROP_SAVE_INTERVAL

        for i in range(quantity):
            # 1. Randomly select a geography
            idx = random.randint(0, n_geos - 1)
            geo = tiger_gdf.iloc[idx]

            # 2. Generate random point within that geography
            point = random_point_in_polygon(geo.geometry)

            # 3. Build property dict with census IDs from shapefile
            prop = {
                "geoid": geo["geoid"],
                "state_id": int(geo["state_id"]),
                "county_id": int(geo["county_id"]),
                "tract_id": int(geo.get("tract_id", 0)),
                "block_grp": int(geo.get("block_grp", 0)),
                "block_id": 0,  # Not available at tract/county level
                HEADER_GEOM: point,
            }
            results.append(prop)

            # Progress reporting
            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (quantity - i - 1) / rate if rate > 0 else 0
                print(
                    f"  {i + 1}/{quantity} properties "
                    f"({rate:.1f}/sec, {self._format_duration(remaining)} remaining)"
                )

            # Periodic save to SQL
            if len(results) - last_saved >= save_interval:
                self._update_gpd_and_sql(results[last_saved:])
                last_saved = len(results)

        # Save remaining
        if len(results) > last_saved:
            self._update_gpd_and_sql(results[last_saved:])
            if verbose:
                print(f"  Saved final {len(results) - last_saved} properties to database.")

        if not skip_final_cleanup:
            self.sql_obj.drop_duplicates_from_table(self.table_name)
            self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
            self._update_property_count()

        total_time = time.time() - start_time
        if verbose:
            rate = quantity / total_time if total_time > 0 else 0
            print(
                f"Completed in {self._format_duration(total_time)} "
                f"({quantity} added, {rate:.1f}/sec, 100% success)"
            )

    def add_properties_from_coordinates(
        self,
        coordinates: list[tuple[float, float]],
        verbose: bool = True,
    ) -> None:
        """
        Add properties from a list of (latitude, longitude) tuples.

        Reverse-geocodes each coordinate via Census API to get census
        block/tract/county IDs, then saves to PostGIS.

        Parameters
        ----------
        coordinates : list of (lat, lon) tuples
            GPS coordinates in WGS84 (EPSG:4326). Each tuple is (latitude, longitude).
        verbose : bool
            Print progress updates.
        """
        if self.test_mode:
            print("Test mode, cannot add more properties.")
            return

        quantity = len(coordinates)
        print(f"Adding {quantity} properties from coordinates...")
        start_time = time.time()

        results = []
        failed = 0
        last_saved = 0
        save_interval = PROP_SAVE_INTERVAL

        for i, (lat, lon) in enumerate(coordinates):
            try:
                block = cg.coordinates(x=lon, y=lat)["2020 Census Blocks"][0]

                prop = {
                    key: (
                        block[PROP_LABELS_KEYS_MAP[key]]
                        if key == "geoid"
                        else int(block[PROP_LABELS_KEYS_MAP[key]])
                    )
                    for key in PROP_LABELS_KEYS_MAP.keys()
                }
                prop[HEADER_GEOM] = Point(lon, lat)
                results.append(prop)

            except Exception:
                failed += 1
                if verbose and failed <= 5:
                    print(f"  Warning: Failed to geocode ({lat}, {lon})")

            # Progress reporting
            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (quantity - i - 1) / rate if rate > 0 else 0
                print(
                    f"  {i + 1}/{quantity} processed "
                    f"({len(results)} success, {failed} failed, "
                    f"{self._format_duration(remaining)} remaining)"
                )

            # Periodic save to SQL
            if len(results) - last_saved >= save_interval:
                self._update_gpd_and_sql(results[last_saved:])
                last_saved = len(results)

        # Save remaining
        if len(results) > last_saved:
            self._update_gpd_and_sql(results[last_saved:])
            if verbose:
                print(f"  Saved {len(results)} properties to database.")

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
        self._update_property_count()

        total_time = time.time() - start_time
        if verbose:
            rate = len(results) / total_time if total_time > 0 else 0
            success_rate = len(results) / quantity * 100 if quantity > 0 else 0
            print(
                f"Completed in {self._format_duration(total_time)} "
                f"({len(results)} added, {failed} failed, {success_rate:.1f}% success)"
            )

    @staticmethod
    def _fetch_single_property() -> dict | None:
        """
        Fetch a single random property with census block info.

        Generates random coordinates within CONUS bounds and reverse-geocodes
        via Census API. Returns dict with property data, or None if geocoding
        fails (e.g., point falls in water or unpopulated area).
        """
        try:
            # Generate random coordinates within Continental US bounds
            lat = random.uniform(USA_MIN_LAT, USA_MAX_LAT)
            long = random.uniform(USA_MIN_LON, USA_MAX_LON)

            block = cg.coordinates(x=long, y=lat)["2020 Census Blocks"][0]

            prop = {
                key: (
                    block[PROP_LABELS_KEYS_MAP[key]]
                    if key == "geoid"
                    else int(block[PROP_LABELS_KEYS_MAP[key]])
                )
                for key in PROP_LABELS_KEYS_MAP.keys()
            }
            prop[HEADER_GEOM] = Point(long, lat)
            return prop

        except Exception:
            return None

    @staticmethod
    def _fetch_single_property_with_retry(max_retries: int = 10) -> dict | None:
        """
        Keep generating random points until one successfully geocodes.

        Parameters
        ----------
        max_retries : int
            Maximum attempts before giving up on this slot.

        Returns
        -------
        dict or None
            Property dict if successful, None if all retries exhausted.
        """
        for _ in range(max_retries):
            result = Properties._fetch_single_property()
            if result is not None:
                return result
        return None

    def _add_random_properties_parallel(
        self, quantity: int, verbose: bool, max_workers: int
    ) -> None:
        """
        Parallel version of add_random_properties using ThreadPoolExecutor.

        Retries failed geocodes by submitting new tasks until exactly 'quantity'
        properties are obtained, or max_total_attempts is reached.
        """
        print(f"Adding {quantity} properties with {max_workers} parallel workers...")

        results = []
        total_attempts = 0
        max_total_attempts = quantity * 5  # Safety cap
        start_time = time.time()
        estimate_shown = False
        estimate_sample = min(PROP_ESTIMATE_SAMPLE * max_workers, quantity)
        last_saved = 0
        save_interval = PROP_SAVE_INTERVAL
        last_progress_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start with a pool of tasks (2x workers to keep queue full)
            pending = set()
            initial_batch = min(quantity, max_workers * 2)
            for _ in range(initial_batch):
                pending.add(executor.submit(self._fetch_single_property))
                total_attempts += 1

            while len(results) < quantity and total_attempts < max_total_attempts:
                if not pending:
                    # No pending tasks but still need results - submit more
                    needed = quantity - len(results)
                    batch = min(needed, max_workers * 2)
                    for _ in range(batch):
                        if total_attempts < max_total_attempts:
                            pending.add(executor.submit(self._fetch_single_property))
                            total_attempts += 1
                    if not pending:
                        break

                # Wait for at least one task to complete
                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    try:
                        prop = future.result()
                        if prop is not None:
                            results.append(prop)
                        else:
                            # Failed geocode - submit replacement if we still need more
                            if len(results) < quantity and total_attempts < max_total_attempts:
                                pending.add(executor.submit(self._fetch_single_property))
                                total_attempts += 1
                    except Exception as e:
                        if verbose:
                            print(f"Property fetch exception: {e}")
                        # Submit replacement on exception too
                        if len(results) < quantity and total_attempts < max_total_attempts:
                            pending.add(executor.submit(self._fetch_single_property))
                            total_attempts += 1

                # Time estimate after initial sample
                if len(results) >= estimate_sample and not estimate_shown and verbose:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    remaining = quantity - len(results)
                    est_remaining_sec = remaining / rate if rate > 0 else 0
                    print(
                        f"  Time estimate: {self._format_duration(est_remaining_sec)} remaining "
                        f"({rate:.2f} properties/sec)"
                    )
                    estimate_shown = True

                # Progress every 500 successful properties
                if len(results) >= last_progress_count + 500 and verbose:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    remaining = quantity - len(results)
                    est_remaining = remaining / rate if rate > 0 else 0
                    success_rate = len(results) / total_attempts * 100 if total_attempts > 0 else 0
                    print(
                        f"  {len(results)}/{quantity} properties "
                        f"({total_attempts} attempts, {success_rate:.1f}% success) "
                        f"- {self._format_duration(est_remaining)} remaining"
                    )
                    last_progress_count = (len(results) // 500) * 500

                # Periodic save to SQL
                if len(results) - last_saved >= save_interval:
                    self._update_gpd_and_sql(results[last_saved:])
                    last_saved = len(results)
                    if verbose:
                        print(f"  Saved {last_saved} properties to database...")

        # Save remaining
        if len(results) > last_saved:
            self._update_gpd_and_sql(results[last_saved:])

        # Backup before deduplication for large batches
        if quantity >= 50000:
            self.sql_obj.backup_table_to_parquet(self.table_name)

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        # Reload from SQL to get auto-generated property_ids
        self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
        self._update_property_count()

        total_time = time.time() - start_time
        if verbose:
            rate = len(results) / total_time if total_time > 0 else 0
            success_rate = len(results) / total_attempts * 100 if total_attempts > 0 else 0
            print(
                f"Completed in {self._format_duration(total_time)} "
                f"({len(results)} added from {total_attempts} attempts, "
                f"{success_rate:.1f}% success, {rate:.2f}/sec)"
            )

        if len(results) < quantity:
            print(
                f"Warning: Only added {len(results)}/{quantity} after "
                f"{max_total_attempts} attempts."
            )

    def _add_properties_near_fires(
        self,
        wildfire_gdf: gpd.GeoDataFrame,
        quantity: int,
        max_distance_km: float = 200.0,
        verbose: bool = True,
    ) -> None:
        """
        Add 'quantity' properties at random distances (1 – max_distance_km)
        from known wildfire locations. Uses a haversine offset from a randomly
        selected fire, then reverse-geocodes with censusgeocode.

        Expects wildfire_gdf to have LATITUDE and LONGITUDE columns (WGS84).
        """

        if not self.table_name == PROP_TABLE_NAME_TEST:
            raise RuntimeError("You shouldn't add these outside of test mode.")

        fire_lats = wildfire_gdf["LATITUDE"].values
        fire_lngs = wildfire_gdf["LONGITUDE"].values
        n_fires = len(fire_lats)
        if n_fires == 0:
            print("No wildfire locations available.")
            return

        print(f"Adding {quantity} properties near {n_fires} wildfire locations...")

        temp_lst = [None] * quantity
        last_added = 0
        num_added = 0
        attempts = 0
        max_attempts = quantity * 5

        while num_added < quantity and attempts < max_attempts:
            attempts += 1

            # Pick a random fire
            idx = np.random.randint(n_fires)
            lat1 = np.radians(fire_lats[idx])
            lng1 = np.radians(fire_lngs[idx])

            # Random distance and bearing
            dist_km = np.random.uniform(1.0, max_distance_km)
            bearing = np.random.uniform(0, 2 * np.pi)

            # Haversine offset
            d_r = dist_km / EARTH_RADIUS_KM
            lat2 = np.arcsin(
                np.sin(lat1) * np.cos(d_r)
                + np.cos(lat1) * np.sin(d_r) * np.cos(bearing)
            )
            lng2 = lng1 + np.arctan2(
                np.sin(bearing) * np.sin(d_r) * np.cos(lat1),
                np.cos(d_r) - np.sin(lat1) * np.sin(lat2),
            )

            lat = np.degrees(lat2)
            long = np.degrees(lng2)

            # Skip points outside CONUS
            if not (
                USA_MIN_LAT <= lat <= USA_MAX_LAT and USA_MIN_LON <= long <= USA_MAX_LON
            ):
                continue

            # Reverse-geocode to census block
            try:
                block = cg.coordinates(x=long, y=lat)["2020 Census Blocks"][0]
            except Exception:
                continue

            temp_lst[num_added] = {
                key: (
                    block[PROP_LABELS_KEYS_MAP[key]]
                    if key == "geoid"
                    else int(block[PROP_LABELS_KEYS_MAP[key]])
                )
                for key in PROP_LABELS_KEYS_MAP.keys()
            }
            temp_lst[num_added][HEADER_GEOM] = Point(long, lat)
            num_added += 1

            if verbose and not num_added % 50:
                print(f"Added {num_added}/{quantity} properties...")
                self._update_gpd_and_sql(temp_lst[:num_added])
                last_added = num_added

        if last_added < num_added:
            remaining = [x for x in temp_lst[last_added:num_added] if x is not None]
            if remaining:
                self._update_gpd_and_sql(remaining)

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        self._update_property_count()

        if num_added < quantity:
            print(
                f"Warning: Only added {num_added}/{quantity} after {max_attempts} attempts."
            )
        print(f"Done. Total properties: {self.num_properties}")

    def _update_gpd_and_sql(self, temp_lst: list) -> None:

        tmp = gpd.GeoDataFrame(data=temp_lst, geometry=HEADER_GEOM, crs=4326)
        tmp = tmp.to_crs(GIS_DEFAULT_CRS)
        self.properties_gpd = gpd.GeoDataFrame(
            pd.concat([self.properties_gpd, tmp]).reset_index(drop=True)
        )

        # Save only the new rows, not the entire GeoDataFrame (prevents table bloat)
        self.sql_obj.save_gpd_to_sql(self.table_name, tmp)

        return

    def refresh(self) -> None:
        """Reload properties from the database.

        Use this after external modifications to the properties table
        (e.g., filtering, deletion) to sync the in-memory GeoDataFrame.
        """
        self._read_from_sql()

    def _read_from_sql(self) -> None:

        # check if properties table exists, and connect
        if self.sql_obj.check_table_exists(self.table_name):
            self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
        else:
            if self.verbose:
                print(
                    f"Initializing table '{self.table_name}' with {PROPERTIES_INIT_COUNT} properties."
                )
            self.properties_gpd = self.sql_obj.initialize_properties_table(
                self.table_name
            )
            if self.test_mode:
                if self.verbose:
                    print("Next call _populate_test_props()...")
            else:
                if self.verbose:
                    print(f"Adding {PROPERTIES_INIT_COUNT} new properties to table...")
                self.add_random_properties(PROPERTIES_INIT_COUNT, verbose=False)
            # Reload from SQL to get auto-generated property_ids
            self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
        self._update_property_count()
        return

    def _update_property_count(self):
        self.num_properties = self.properties_gpd.shape[0]
        return

    def get_properties_gpd(self):
        return self.properties_gpd

    @classmethod
    def _populate_test_props(
        cls, sql_obj: SQL, wildfire_gdf, quantity=50, max_distance_km=100.0
    ):
        if not sql_obj.test_mode:
            raise RuntimeError("_populate_test_prop() is only for test mode.")
        sql_obj.drop_table(PROP_TABLE_NAME_TEST, confirm=True)
        instance = cls(sql_obj)
        instance._add_properties_near_fires(wildfire_gdf, quantity, max_distance_km)
        return instance


if __name__ == "__main__":

    # Just in case
    SQL.kill_idle(True)
    print("Starting...")
    sql_obj = SQL()
    print("SQL connected")
    props = Properties(sql_obj=sql_obj)
    print(props.properties_gpd.head())

    try:
        props.add_random_properties(
            int(sys.argv[1]), parallel=True, max_workers=int(sys.argv[2])
        )
        print("Job done.")
        if int(sys.argv[1]) > 10000:
            print("Backing up to parquet...")
            sql_obj.backup_table_to_parquet(PROP_TABLE_NAME)
    finally:
        pro_gdf = props.get_properties_gpd()
        print(f"Successfully added {pro_gdf.shape[0]} properties.")
        sql_obj.disconnect_and_close()
