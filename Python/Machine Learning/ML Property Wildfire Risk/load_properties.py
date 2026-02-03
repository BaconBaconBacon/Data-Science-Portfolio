"""Property data generation and management for wildfire risk modeling.

Generates random US property locations using the random_address library,
reverse-geocodes them via the Census Geocoder API to obtain geographic
identifiers (state, county, tract, block group), and persists to PostGIS.

Supports both sequential and parallel property generation for speed.
Test mode generates properties near known wildfire locations for validation.
"""

import censusgeocode as cg
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd

from concurrent.futures import ThreadPoolExecutor, as_completed
from random_address import real_random_address
from shapely.geometry import Point
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
)


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
    ):

        self.sql_obj = sql_obj
        self.test_mode = sql_obj.test_mode

        if not self.test_mode:
            self.table_name = PROP_TABLE_NAME
        else:
            self.table_name = PROP_TABLE_NAME_TEST
        self._read_from_sql()

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

        temp_lst = [None] * quantity
        last_added = 0
        start_time = time.time()
        estimate_sample = min(10, quantity)
        estimate_shown = False

        for i in range(quantity):

            coords = real_random_address()["coordinates"]

            lat = coords["lat"]
            long = coords["lng"]
            block = cg.coordinates(x=long, y=lat)["2020 Census Blocks"][0]

            temp_lst[i] = {
                key: (
                    block[PROP_LABELS_KEYS_MAP[key]]
                    if key == "geoid"
                    else int(block[PROP_LABELS_KEYS_MAP[key]])
                )
                for key in PROP_LABELS_KEYS_MAP.keys()
            }
            temp_lst[i][HEADER_GEOM] = Point(long, lat)

            # Show time estimate after first N properties
            if (i + 1) == estimate_sample and not estimate_shown and verbose:
                elapsed = time.time() - start_time
                per_property = elapsed / estimate_sample
                remaining = quantity - estimate_sample
                est_remaining_sec = remaining * per_property
                est_total_sec = quantity * per_property
                print(
                    f"  Time estimate: {self._format_duration(est_total_sec)} total "
                    f"({per_property:.2f}s per property, "
                    f"{self._format_duration(est_remaining_sec)} remaining)"
                )
                estimate_shown = True

            if not (i + 1) % 50:
                if verbose:
                    elapsed = time.time() - start_time
                    remaining_count = quantity - (i + 1)
                    per_property = elapsed / (i + 1)
                    est_remaining = remaining_count * per_property
                    print(
                        f"  {i+1}/{quantity} properties... "
                        f"({self._format_duration(est_remaining)} remaining)"
                    )
                self._update_gpd_and_sql(temp_lst[: i + 1])
                last_added = i + 1

        if last_added < quantity:
            remaining = [x for x in temp_lst[last_added:] if x is not None]
            self._update_gpd_and_sql(remaining)

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        self._update_property_count()

        total_time = time.time() - start_time
        if verbose:
            print(f"Completed in {self._format_duration(total_time)}")

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

    @staticmethod
    def _fetch_single_property() -> dict | None:
        """
        Fetch a single random property with census block info.

        Returns dict with property data, or None if geocoding fails.
        """
        try:
            coords = real_random_address()["coordinates"]
            lat = coords["lat"]
            long = coords["lng"]

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

        except (IndexError, KeyError, Exception):
            return None

    def _add_random_properties_parallel(
        self, quantity: int, verbose: bool, max_workers: int
    ) -> None:
        """
        Parallel version of add_random_properties using ThreadPoolExecutor.
        """
        print(f"Adding {quantity} properties with {max_workers} parallel workers...")

        results = []
        failed = 0
        start_time = time.time()
        estimate_shown = False
        estimate_sample = min(10 * max_workers, quantity)
        last_saved = 0
        save_interval = 500

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single_property): i for i in range(quantity)
            }

            for future in as_completed(futures):
                try:
                    prop = future.result()
                    if prop is not None:
                        results.append(prop)
                except Exception:
                    failed += 1

                completed = len(results) + failed

                # Time estimate after initial sample
                if completed == estimate_sample and not estimate_shown and verbose:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    remaining = quantity - completed
                    est_remaining_sec = remaining / rate if rate > 0 else 0
                    print(
                        f"  Time estimate: {self._format_duration(est_remaining_sec)} remaining "
                        f"({rate:.2f} properties/sec)"
                    )
                    estimate_shown = True

                # Progress every 500
                if completed % 500 == 0 and completed > 0 and verbose:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    remaining = quantity - completed
                    est_remaining = remaining / rate if rate > 0 else 0
                    print(
                        f"  {completed}/{quantity} fetched ({len(results)} success, {failed} failed) "
                        f"- {self._format_duration(est_remaining)} remaining"
                    )

                # Periodic save to SQL
                if len(results) - last_saved >= save_interval:
                    self._update_gpd_and_sql(results[last_saved:])
                    last_saved = len(results)
                    if verbose:
                        print(f"  Saved {last_saved} properties to database...")

        # Save remaining
        if len(results) > last_saved:
            self._update_gpd_and_sql(results[last_saved:])

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        self._update_property_count()

        total_time = time.time() - start_time
        if verbose:
            rate = len(results) / total_time if total_time > 0 else 0
            print(
                f"Completed in {self._format_duration(total_time)} "
                f"({len(results)} added, {failed} failed, {rate:.2f}/sec)"
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
            except (IndexError, KeyError, Exception):
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

    def _read_from_sql(self) -> None:

        # check if properties table exists, and connect
        if self.sql_obj.check_table_exists(self.table_name):
            print(f"{self.table_name} found")
            self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
            print(f"'{self.table_name}' table loaded.")
        else:
            print(
                f"Initializing table '{self.table_name}' with {PROPERTIES_INIT_COUNT} properties."
            )
            self.properties_gpd = self.sql_obj.initialize_properties_table(
                self.table_name
            )
            if self.test_mode:
                print("Next call _populate_test_props()...")
            else:
                print(f"Adding {PROPERTIES_INIT_COUNT} new properties to table...")
                self.add_random_properties(PROPERTIES_INIT_COUNT, verbose=False)
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
    try:
        print("SQL connected")
        props = Properties(sql_obj=sql_obj)
        print(props.properties_gpd.head())
        props.add_random_properties(int(sys.argv[1]), parallel=True, max_workers=50)
        print("Job done.")
    finally:
        sql_obj.disconnect_and_close()
