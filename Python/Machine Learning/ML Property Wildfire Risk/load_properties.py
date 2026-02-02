# import census
import censusgeocode as cg
import sys

# import os
from pathlib import Path
import numpy as np
import pandas as pd

# import sqlalchemy as sql
import geopandas as gpd

from random_address import real_random_address
from shapely.geometry import Point
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from settings import (
    GIS_DEFAULT_CRS,
    PROP_TABLE_NAME,
    PROP_LABELS_KEYS_MAP,
    PATH_DATA_PROPERTIES,
    SQL_ENGINE_STR,
    PROPERTIES_INIT_COUNT,
    PROP_TABLE_NAME_TEST,
    HEADER_GEOM,
)
from sql_funcs import SQL


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

        self.data_path = PATH_DATA_PROPERTIES
        self.test = sql_obj.test_mode
        if not self.test:
            self.table_name = PROP_TABLE_NAME
        else:
            self.table_name = PROP_TABLE_NAME_TEST
        self._connect_to_sql()

    def add_random_properties(self, quantity: int, verbose=True) -> None:
        """
        Add 'quantity' many new random addresses
        """
        if self.test:
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
                key: int(block[PROP_LABELS_KEYS_MAP[key]])
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

        # self._update_gpd_and_sql([x for x in temp_lst if x is not None])
        self.sql_obj.drop_duplicates_from_table(self.table_name)
        self._update_property_count()

        return

    # def property_count(self) ->int:
    #     '''
    #         Returns how many properties are in the list.
    #     '''
    #     return

    def _update_gpd_and_sql(self, temp_lst: list) -> None:

        tmp = gpd.GeoDataFrame(data=temp_lst, geometry=HEADER_GEOM, crs=GIS_DEFAULT_CRS)
        self.properties_gpd = gpd.GeoDataFrame(
            pd.concat([self.properties_gpd, tmp]).reset_index(drop=True)
        )

        self.sql_obj.save_gpd_to_sql(self.table_name, self.properties_gpd)

        return

    def _connect_to_sql(self) -> None:

        # check if properties table exists, and connect
        if self.sql_obj.check_table_exists(self.table_name):

            self.properties_gpd = self.sql_obj.read_gpd_from_sql(self.table_name)
            print(f"'{self.table_name}' table found.")
        else:
            print(
                f"Initializing table '{self.table_name}' with {PROPERTIES_INIT_COUNT} properties."
            )
            self.properties_gpd = self.sql_obj.initialize_properties_table(
                self.table_name
            )
            self.add_random_properties(PROPERTIES_INIT_COUNT, verbose=False)
        self._update_property_count()
        return

    def _update_property_count(self):
        self.num_properties = self.properties_gpd.shape[0]
        return

    def get_properties_gpd(self):
        return self.properties_gpd


if __name__ == "__main__":

    sql_obj = SQL()
    props = Properties(sql_obj=sql_obj)
    print(props.properties_gpd.head())
    props.add_random_properties(int(sys.argv[1]))
    print("Job done.")
