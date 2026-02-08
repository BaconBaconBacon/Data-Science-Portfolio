"""Wildfire data extraction, transformation, and loading (ETL).

Loads VIIRS satellite fire detection data from shapefiles, applies quality
filters (high confidence, presumed vegetation fires), clusters nearby
detections into unique fire events using DBSCAN, and persists to PostGIS.

Data source: NASA FIRMS (Fire Information for Resource Management System)
Satellites: NOAA-21 (N21), Suomi NPP (SNPP)
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from settings import (
    GIS_DESIRED_COLS,
    GIS_DEFAULT_CRS,
    PATH_DATA_WILDFIRES,
    WILDFIRES_CLUSTER_TIME_SCALE,
    WILDFIRES_CLUSTER_SPATIAL_EPS,
    USA_MAX_LAT,
    USA_MIN_LAT,
    USA_MAX_LON,
    USA_MIN_LON,
    WILDFIRES_TABLE_NAME,
    HEADER_GEOM,
    HEADER_SAT_ID,
)
from pathlib import Path
from sklearn.cluster import DBSCAN
from sql_funcs import SQL
from visualize import create_wildfire_map


class WildfireData:
    """ETL handler for NASA FIRMS wildfire detection data.

    Extracts fire detections from VIIRS satellite shapefiles, transforms
    by filtering for CONUS and clustering spatiotemporally similar detections,
    then loads to PostGIS for persistence.

    Parameters
    ----------
    sql_obj : SQL
        Database connection manager instance.

    Attributes
    ----------
    data : gpd.GeoDataFrame
        Processed wildfire data with columns: geometry, ACQ_DATE, FRP,
        SAT_ID, CONFIDENCE, TYPE, LATITUDE, LONGITUDE, fire_radius_m.
    test_mode : bool
        Inherited from sql_obj; affects table naming.
    """

    def __init__(self, sql_obj: SQL):

        self.sql_obj = sql_obj
        self.test_mode = self.sql_obj.test_mode

        self.data_path = PATH_DATA_WILDFIRES

        if self.sql_obj.check_table_exists(WILDFIRES_TABLE_NAME) and not self.test_mode:
            self.data = self._read_from_sql()
            print(
                "Loading previously extracted wildfire data from:",
                self.data[HEADER_SAT_ID].unique(),
            )
        else:
            print("Extracting wildfire data from GIS files.")
            raw_data = self._extract(self.data_path)
            self.data = self._transform(raw_data)

            self._load_to_sql(self.data)

    def _extract(self, data_path: Path) -> gpd.GeoDataFrame:
        """
        Takes as input a string or a list of strings consisting
        of filenames or filepaths to wildfire GIS data.

        """

        filepaths = data_path.rglob("*.shp")

        temp_list = []
        for fp in filepaths:
            if "property" not in fp.name:

                sat_name = fp.name.split(".")[0].split("_")[-2]
                print("Loading satellite: ", sat_name)

                sat_data = gpd.read_file(fp)  # [GIS_DESIRED_COLS]
                sat_data[HEADER_SAT_ID] = sat_name

                # 0 is 'probable wildfire', h is high confidence
                if "TYPE" in sat_data.columns:
                    sat_data = sat_data.query('TYPE==0 & CONFIDENCE=="h"')
                else:
                    sat_data = sat_data.query('CONFIDENCE=="h"')

                # Set coordinate system
                sat_data = sat_data.to_crs(GIS_DEFAULT_CRS)
                temp_list.append(sat_data)

        return gpd.GeoDataFrame(pd.concat(temp_list, ignore_index=True))

    def _transform(self, raw_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        raw_data["ACQ_DATE"] = pd.to_datetime(raw_data["ACQ_DATE"])

        raw_data = raw_data[GIS_DESIRED_COLS]

        usa_data = self._filter_for_usa(raw_data)

        clean_data = self._cluster_duplicate_wildfires(usa_data)

        return clean_data

    def _filter_for_usa(self, raw_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Filters rows for only those geometries existing within the contiguous USA"""

        conds = (
            (USA_MIN_LAT <= raw_data["LATITUDE"])
            & (raw_data["LATITUDE"] <= USA_MAX_LAT)
            & (USA_MIN_LON <= raw_data["LONGITUDE"])
            & (raw_data["LONGITUDE"] <= USA_MAX_LON)
        )
        filtered_data = raw_data[conds]
        return filtered_data

    def _cluster_duplicate_wildfires(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove duplicate wildfire detections using spatiotemporal clustering.

        Currently converting overlapping geometries into a centroid and radius, but may be worth
        including functionatlity to turn it into a wildfire perimeter.

        DBSCAN : https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/
        """

        # Extract coordinates (CRS 5070 is in meters)
        coords = np.column_stack([data.geometry.x, data.geometry.y])

        # Convert dates to numeric (days since min date)
        min_date = data["ACQ_DATE"].min()
        days = (data["ACQ_DATE"] - min_date).dt.days.values

        # Combine spatial + time features
        features = np.column_stack([coords, days * WILDFIRES_CLUSTER_TIME_SCALE])

        # Cluster
        clustering = DBSCAN(eps=WILDFIRES_CLUSTER_SPATIAL_EPS, min_samples=1).fit(
            features
        )
        data = data.copy()
        data["cluster_id"] = clustering.labels_

        aggregated = (
            data.groupby("cluster_id")
            .agg(
                {
                    HEADER_GEOM: lambda g: g.unary_union.centroid,
                    "ACQ_DATE": "min",  # earliest detection date
                    # "LATITUDE": "mean",
                    # "LONGITUDE": "mean",
                    "FRP": "mean",
                    HEADER_SAT_ID: lambda x: ",".join(x.unique()),
                    "CONFIDENCE": "first",
                    "TYPE": "first",
                }
            )
            .reset_index(drop=True)
        )

        centroids_4326 = aggregated.set_geometry(HEADER_GEOM, crs=data.crs).to_crs(4326)
        aggregated["LATITUDE"] = centroids_4326.geometry.y
        aggregated["LONGITUDE"] = centroids_4326.geometry.x

        aggregated["fire_radius_m"] = (
            data.groupby("cluster_id")
            .apply(
                lambda grp: grp.geometry.distance(
                    grp.geometry.unary_union.centroid
                ).max(),
                include_groups=False,
            )
            .values
        )

        return gpd.GeoDataFrame(aggregated, geometry=HEADER_GEOM, crs=data.crs)

    def _load_to_sql(self, clean_data: gpd.GeoDataFrame) -> None:
        """Save the wildfire data to PostgreSQL."""
        self.sql_obj.save_gpd_to_sql(WILDFIRES_TABLE_NAME, clean_data)

    def _read_from_sql(self) -> gpd.GeoDataFrame:
        return self.sql_obj.read_gpd_from_sql(WILDFIRES_TABLE_NAME)

    def visualize_data(self, save_path: Path = None):
        """
        Visualize wildfire locations on a Folium map.

        Parameters
        ----------
        save_path : Path, optional
            If provided, save the map to this HTML file.

        Returns
        -------
        folium.Map
            The generated map object.
        """

        return create_wildfire_map(self.data, save_path=save_path)


if __name__ == "__main__":
    SQL.kill_idle()
    sql_obj = SQL()
    test_obj = WildfireData(sql_obj=sql_obj)
    test_obj.visualize_data()
    sql_obj.disconnect_and_close()
