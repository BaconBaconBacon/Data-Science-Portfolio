"""Census data retrieval and processing for property enrichment.

Fetches American Community Survey 5-year (ACS5) data from the US Census API
at configurable geographic granularity (block group, tract, or county).
Features are normalized to percentages of their universe totals for ML use.

Implements caching to avoid redundant API calls for previously-fetched
geographies. Results are stored in PostGIS in narrow (long) format for
efficient storage and flexible querying.
"""

import census
import geopandas as gpd
import json
import numpy as np
import requests

import os
from pathlib import Path
import pandas as pd

from settings import (
    CENSUS_FEATURES,
    GIS_DEFAULT_CRS,
    PATH_DATA_CENSUS,
    CENSUS_VALID_GRANULARITY_LEVELS,
    CENSUS_CHUNK_SIZE,
    TABLE_NAME_CENSUS,
    TABLE_NAME_CENSUS_TEST,
    TABLE_NAME_CACHE,
    CENSUS_SUMMARY_TABLES,
    TABLE_NAME_CENSUS_PROPS_TEST,
    TABLE_NAME_CENSUS_PROPS,
)

# from sqlalchemy import text
from sql_funcs import SQL


class CensusData:
    """ACS5 Census data handler for property feature enrichment.

    Fetches socioeconomic and housing variables from the Census API,
    caches results to reduce API calls, and merges onto property data.

    Parameters
    ----------
    sql_obj : SQL
        Database connection manager.
    year : int
        ACS5 data year (e.g., 2023).
    granularity : str, default "tract"
        Geographic level: "block_group", "tract", or "county".
    sparse : bool, default True
        If True, fetch only leaf variables (most granular); if False,
        fetch all variables including totals/subtotals.

    Attributes
    ----------
    data : pd.DataFrame or None
        Cached census data in long format from the database.
    granularity : str
        Current geographic granularity level.
    year : int
        ACS5 survey year.

    Notes
    -----
    ACS5 variables are organized by table codes (e.g., B19013 = Median
    Household Income). Some variables are only available at tract level
    or higher. Count variables are normalized to percentages of their
    universe totals (e.g., households, housing units) for ML use.

    See settings.CENSUS_FEATURES for the list of table codes used.
    """

    def __init__(
        self, sql_obj: SQL, year: int, granularity: str = "tract", sparse: bool = True
    ):
        if granularity not in CENSUS_VALID_GRANULARITY_LEVELS:
            raise ValueError(
                f"granularity must be one of {CENSUS_VALID_GRANULARITY_LEVELS}"
            )
        self.granularity = granularity
        self.leafs_only = sparse
        self.year = year

        self.census = census.Census(
            os.environ.get("US_CENSUS_API_KEY"), year=self.year
        ).acs5

        self.metadata_filepath = PATH_DATA_CENSUS / f"acs5_variables_{year}.txt"

        if not self.metadata_filepath.exists():
            self._generate_variable_metadata()
        else:
            # Connect and read metadata file?
            pass

        self.sql_obj = sql_obj
        self.test_mode = self.sql_obj.test_mode

        if self.test_mode:
            self.table_name = TABLE_NAME_CENSUS_TEST
            self.sql_obj.drop_table(TABLE_NAME_CENSUS_TEST, confirm=True)
            # self.data = None
        else:
            self.table_name = TABLE_NAME_CENSUS
        self.data = self._read_from_sql()

        # Chosen variables for model (query these from the SQL table, if they don't exist, generate it
        self.census_variables_list = []

    def merge_census_info(self, properties_gpd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """check existing data, only fetch missing geoids"""

        # TODO: *** Encapsulate this stuff for clarity ***

        if "geoid" not in properties_gpd.columns:
            raise ValueError("No geoid column in input.")

        # Check existing data
        merge_cols = self._get_merge_columns()

        existing_wide = self._get_existing_wide(merge_cols)
        existing_wide = self._handle_missing_geos(
            properties_gpd, existing_wide, merge_cols
        )

        # Merge wide census data onto all properties
        result = properties_gpd.merge(existing_wide, on=merge_cols, how="left")

        # Drop geo ID columns
        drop_cols = [x for x in result.columns if ("_id" in x) or ("_grp" in x)]
        result = result.drop(columns=drop_cols)

        if self.test_mode:
            try:
                _ = self._read_from_sql()
                print("_read_from_sql() Sucessfully tested.")
            except Exception as e:
                print(f"_read_from_sql() failed with {e}")
        return result

    def _add_missing_geos(
        self,
        gdf: gpd.GeoDataFrame,
        wide: pd.DataFrame,
        merge_cols: list,
        missing_geos,
    ) -> pd.DataFrame:
        variables = (
            self._get_leaf_variables() if self.leafs_only else self._get_all_variables()
        )
        self.sql_obj.create_census_cache_table()

        missing_props = gdf.merge(missing_geos, on=merge_cols, how="inner")

        new_census_df = self._extract_from_geoid(missing_props, variables)

        new_cleaned = self._transform_and_clean(new_census_df)
        self._load_to_sql(new_cleaned)

        new_wide = self._wide_from_cleaned(new_cleaned, merge_cols)
        existing_wide = (
            pd.concat([wide, new_wide], ignore_index=True)
            if not wide.empty
            else new_wide
        )
        self.data = self._read_from_sql()

        return existing_wide

    def _handle_missing_geos(
        self, gdf: gpd.GeoDataFrame, wide, merge_cols: list
    ) -> pd.DataFrame:

        required_geos = gdf[merge_cols].drop_duplicates()

        if not wide.empty:
            check = required_geos.merge(
                wide[merge_cols].drop_duplicates(),
                on=merge_cols,
                how="left",
                indicator=True,
            )
            missing_geos = check[check["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
        else:
            missing_geos = required_geos

        print(
            f"{len(required_geos)} geographies needed, "
            f"{len(required_geos) - len(missing_geos)} cached, "
            f"{len(missing_geos)} to fetch."
        )
        if not missing_geos.empty:
            wide = self._add_missing_geos(gdf, wide, merge_cols, missing_geos)

        return wide

    def _get_existing_wide(self, merge_cols: list) -> pd.DataFrame:

        if self.data is None or self.data.empty:
            return pd.DataFrame()
        filtered = self.data[
            (self.data["granularity"] == self.granularity)
            & (self.data["year"] == self.year)
        ]
        if filtered.empty:
            return pd.DataFrame()
        wide = filtered.pivot_table(
            index=merge_cols, columns="variable", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None

        return wide

    def _wide_from_cleaned(self, df: pd.DataFrame, merge_cols: list) -> pd.DataFrame:

        census_cols = [c for c in df.columns if c.startswith("B")]
        return df[merge_cols + census_cols].drop_duplicates(subset=merge_cols)

    def _get_merge_columns(self) -> list[str]:
        if self.granularity == "county":
            merge_cols = ["state_id", "county_id"]
        elif self.granularity == "tract":
            merge_cols = ["state_id", "county_id", "tract_id"]
        else:  # block_group
            merge_cols = ["state_id", "county_id", "tract_id", "block_grp"]
        return merge_cols

    def _get_geoid_query_strings(self, row: pd.Series) -> tuple[str, str, str]:

        if self.granularity == "county":
            state = str(row["state_id"]).zfill(2)
            county = str(row["county_id"]).zfill(3)
            geo_id = f"{state}{county}"
            query_for = f"county:{county}"
            query_in = f"state:{state}"
        elif self.granularity == "tract":
            state = str(row["state_id"]).zfill(2)
            county = str(row["county_id"]).zfill(3)
            tract = str(row["tract_id"]).zfill(6)

            geo_id = f"{state}{county}{tract}"
            query_for = f"tract:{tract}"
            query_in = f"state:{state} county:{county}"
        else:  # block_group
            state = str(row["state_id"]).zfill(2)
            county = str(row["county_id"]).zfill(3)
            tract = str(row["tract_id"]).zfill(6)
            blkgrp = str(row["block_grp"])

            geo_id = f"{state}{county}{tract}{blkgrp}"
            query_for = f"block group:{blkgrp}"
            query_in = f"state:{state} county:{county} tract:{tract}"
        return geo_id, query_for, query_in

    def _extract_from_geoid(self, properties_data, variables):

        merge_cols = self._get_merge_columns()
        unique_geos = properties_data[merge_cols].drop_duplicates()
        print(f"Unique geoids: {len(unique_geos)}.")

        var_chunks = [
            variables[i : i + CENSUS_CHUNK_SIZE]
            for i in range(0, len(variables), CENSUS_CHUNK_SIZE)
        ]

        results = []
        for idx, row in unique_geos.iterrows():

            # Check cache first
            cached = self.sql_obj.get_cached_census_results(
                merge_cols, row, self.granularity, self.year
            )

            if cached is not None:
                print(f"[{idx}] Cache hit")
                cached_dict = cached.to_dict()
                for col in merge_cols:
                    cached_dict[col] = row[col]
                results.append(cached_dict)
                continue

            # Query API
            geo_id, query_for, query_in = self._get_geoid_query_strings(row)

            print(f"[{idx}] Querying id {geo_id} at {self.granularity}...")

            combined_result = {}
            try:
                for chunk in var_chunks:
                    result = self.census.get(chunk, {"for": query_for, "in": query_in})
                    if result:
                        combined_result.update(result[0])

                if combined_result:
                    # Cache immediately after success
                    self._cache_results(row, merge_cols, combined_result)
                    for col in merge_cols:
                        combined_result[col] = row[col]
                    results.append(combined_result)

            except Exception as e:
                print(f"Skipped! Error for {geo_id}: {e}")

        if len(results) > 0:
            return properties_data.merge(
                pd.DataFrame(results), on=merge_cols, how="left"
            )
        else:
            raise RuntimeError("Length of results is zero!")

    def _cache_results(self, row, merge_cols, result_dict):
        """Save results to cache table."""
        records = []
        for var, val in result_dict.items():
            if (var in merge_cols) or (
                var in ["state", "county", "tract", "block group", "GEO_ID", "NAME"]
            ):
                continue  # Skip geo columns returned by API
            records.append(
                {
                    "state_id": row.get("state_id", 0),
                    "county_id": row.get("county_id", 0),
                    "tract_id": row.get("tract_id", 0),
                    "block_grp": row.get("block_grp", 0),
                    "granularity": self.granularity,
                    "year": self.year,
                    "variable": var,
                    "value": val,
                }
            )
        if records:
            self.sql_obj.save_df_to_sql(
                table_name=TABLE_NAME_CACHE, df=pd.DataFrame(records)
            )

    def _transform_and_clean(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        census_cols = [c for c in df.columns if c.startswith("B")]

        # clean  sentinel values
        df[census_cols] = df[census_cols].replace(-666666666, np.nan)

        # normalize count tables to percentages of universe total
        for table in CENSUS_FEATURES:
            if table in CENSUS_SUMMARY_TABLES:
                continue

            table_cols = [c for c in census_cols if c.startswith(table)]
            if not table_cols:
                continue

            # Reconstruct total from leaves (no extra API call needed)
            total = df[table_cols].sum(axis=1).replace(0, np.nan)
            for col in table_cols:
                df[col] = df[col] / total

        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how="all")

        return df

    def _load_to_sql(self, clean_data: pd.DataFrame) -> None:
        """
        Save the data as a SQL db.?
        """
        merge_cols = self._get_merge_columns()
        census_cols = [c for c in clean_data.columns if c.startswith("B")]

        # Melt wide format to narrow: one row per (geography, variable)
        # gets past row size limits
        # Save props joined with census
        props_long = (
            clean_data[["geoid"] + merge_cols + census_cols]
            .copy()
            .melt(
                id_vars=["geoid"] + merge_cols,
                var_name="variable",
                value_name="value",
            )
        )
        props_long["value"] = props_long["value"].astype(float)
        props_long["granularity"] = self.granularity
        props_long["year"] = self.year

        table_name = (
            TABLE_NAME_CENSUS_PROPS_TEST if self.test_mode else TABLE_NAME_CENSUS_PROPS
        )
        self.sql_obj.save_df_to_sql(table_name, props_long)

        # Save just the census information to TABLE_NAME_CENSUS.
        census_df = clean_data.drop_duplicates(merge_cols)
        census_long = (
            census_df[merge_cols + census_cols]
            .copy()
            .melt(
                id_vars=merge_cols,
                var_name="variable",
                value_name="value",
            )
        )
        census_long["value"] = census_long["value"].astype(float)
        census_long["granularity"] = self.granularity
        census_long["year"] = self.year

        table_name = TABLE_NAME_CENSUS_TEST if self.test_mode else TABLE_NAME_CENSUS
        self.sql_obj.save_df_to_sql(table_name, census_long)

        return

    def _read_from_sql(self) -> pd.DataFrame | None:
        if self.sql_obj.check_table_exists(self.table_name):
            df = self.sql_obj.read_df_from_sql(self.table_name)
            if "granularity" not in df.columns or "year" not in df.columns:
                self.sql_obj.drop_table(self.table_name, confirm=True)
                return None
            else:
                return df
        else:
            return None

    def _generate_variable_metadata(self) -> None:

        acs5_dict = {}
        for item in self.census.tables():
            keys = [x for x in item.keys() if "name" not in x and "variables" not in x]
            acs5_dict[item["name"]] = {key: item[key] for key in keys}

        with open(self.metadata_filepath, "w") as f:
            json.dump(acs5_dict, f, indent=4)
        return

    def get_variable_universe(self, variable: str | list) -> str | list:

        with open(self.metadata_filepath, "r") as f:
            loaded_dict = json.load(f)
        if isinstance(variable, str):
            return loaded_dict[variable]["universe"]
        elif isinstance(variable, list):
            return [loaded_dict[var]["universe"] for var in variable]
        else:
            raise TypeError("'variable' must be a string or list of strings.")

    def get_acs5_fields(self, year=2023):
        """
        Cheeky bypass for the census package's fields() method.
        TODO: Figure out what's wrong with fields(), branch the repo, fix it, send PR
        """
        url = f"https://api.census.gov/data/{year}/acs/acs5/variables.json"
        r = requests.get(url)
        return r.json()["variables"].keys()

    def get_acs5_fields_with_labels(self, year=2023):
        """
        Cheeky bypass for the census package's fields() method.

        Gets keys and labels, the labels are used to filter the columns of a given table.
        TODO: Figure out what's wrong with fields(), branch the repo, fix it, send PR
        """
        url = f"https://api.census.gov/data/{year}/acs/acs5/variables.json"
        r = requests.get(url)
        vars = r.json()["variables"]
        return {k: v.get("label", "") for k, v in vars.items()}

    def _get_all_variables(self) -> list:
        """
        Expand table group codes to individual variable codes.
        E.g., "B01001" -> ["B01001_001E", "B01001_002E", ...]
        """
        all_fields = self.get_acs5_fields(year=self.year)

        all_vars = []
        for table in CENSUS_FEATURES:
            table_vars = [
                v for v in all_fields if v.startswith(table) and v.endswith("E")
            ]
            all_vars.extend(table_vars)
        return all_vars

    def _get_leaf_variables(self):
        """Keep only leaf (most granular) variables — drop totals/subtotals."""
        all_fields = self.get_acs5_fields_with_labels(year=self.year)

        all_vars = []
        for table in CENSUS_FEATURES:
            table_vars = {
                k: v
                for k, v in all_fields.items()
                if k.startswith(table) and k.endswith("E")
            }
            # A variable is a leaf if no other variable's label starts with its label
            for var, label in table_vars.items():
                is_leaf = not any(
                    other_label.startswith(label) and other_label != label
                    for other_var, other_label in table_vars.items()
                )
                if is_leaf:
                    all_vars.append(var)
        return all_vars

    def _get_variable_name_from_code(self, code: str) -> str:

        # TODO: Get human readable label from census code.
        return


if __name__ == "__main__":
    SQL.kill_idle()
    sql_obj = SQL()
    test_obj = CensusData(sql_obj=sql_obj, year=2023)
    sql_obj.disconnect_and_close()
