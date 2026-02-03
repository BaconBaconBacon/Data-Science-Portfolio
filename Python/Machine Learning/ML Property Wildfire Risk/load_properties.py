import censusgeocode as cg
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

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
    """
    For each property, we want lat/long, and state/county/tract/block
    group information so that we can later query the ACS5.
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

    def add_random_properties(self, quantity: int, verbose=True) -> None:
        """
        Add 'quantity' many new random addresses
        """
        if self.test_mode:
            # TODO: Check if there arent any in the test property table
            print("Test mode, cannot add more properties.")
            return
        # Could keep a hash of the address, for privacy?
        # self.session = self.Session()
        print(f"Adding {quantity} properties...")

        # TODO: Turn this into a dictionary, should be faster
        temp_lst = [None] * quantity
        last_added = 0
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

            if not (i + 1) % 50:
                if verbose:
                    print(f"Updating with {i+1} properties...")
                self._update_gpd_and_sql(temp_lst[: i + 1])
                last_added = i + 1
        if last_added < quantity:
            remaining = [x for x in temp_lst[last_added:] if x is not None]
            self._update_gpd_and_sql(remaining)

        self.sql_obj.drop_duplicates_from_table(self.table_name)
        self._update_property_count()

        return

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
        props.add_random_properties(int(sys.argv[1]))
        print("Job done.")
    finally:
        sql_obj.disconnect_and_close()
