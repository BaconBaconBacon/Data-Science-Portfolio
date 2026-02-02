import census
import geopandas as gpd
import json
import numpy as np
import requests

import os
from pathlib import Path
import pandas as pd

# import pytidycensus as tc
from settings import (
    CENSUS_FEATURES,
    GIS_DEFAULT_CRS,
    PATH_DATA_CENSUS,
    TEST_SQL_ENGINE_STR,
    CENSUS_VALID_GRANULARITY_LEVELS,
    TABLE_NAME_CENSUS,
    TABLE_NAME_CENSUS_TEST,
    TABLE_NAME_CACHE,
    CENSUS_SUMMARY_TABLES,
)

# from sqlalchemy import text
from sql_funcs import SQL


class CensusData:
    """
    Based on the properties of interest, we need to pull ACS5
    features relevant to each property. There are over 60,000
    possible features to choose, so will need to find guess
    relevant ones.

    FOR ACS5, some variables are at block group precision, and
    some are stored at tract and higher.

    To prep for machine learning, it's best to get each feature
    into a percentage of total in their respective census universe.

    Universes (can extract this programmatically from pytidycensus responses):
        Households
        Housing Units
        Civ. employed pop 16 years and over
        Occupied housing units

    Feature choices suggested by LLM (each will need unique cleaning/binning):
        Income :
            Median household income : B19013
            Household income in last year : B19001
            Public Assistance income : B19058

        Population :
            Sex by age: B01001
            Total population : B01003
        Housing :
            Housing unit count : B25001
            Household type : B11001
            Household type by relatives & nonrelatives : B11002
            Tenure by age of householder by occupants per room :B25015
            Tenure by plumbing facilities by occupants per room : B25016
            Single-parent householes : B11005
            Group Quarters Population : B26001
            Multigenerational households : B11017
            Units in structure (detached, MF, mobile homes) : B25024
            Year structure built : B25034
            Occupied vs vacant : B25002
            Owner vs renter occupied : B25003
            Rooms : B25017
            Mobile homes vs. conventional structures : B25024
            Total housing units (derive housing density): B25001
            Vacancy status : B25002
            Seasonal / recreational / occasional-use units : B25004
            House heating fuel : B25040
            Tenure by Occupants per Room: B25014
        Ethnicity :
            Language isolation : B16002
        Economy :
            Poverty rate : B17001
            Rent burden : B25070
            Mortgage burden: B25091
            Energy costs (selected monthly costs) : B25031
            Gini index : B19083

        Employment:
            Sex of workers by place of work: B08008
            Unemployment : B23025
            Sex by Occupation for Civ employed >=16 years old :C24010
            Worker class (pvt/gov/self): C24060
        Education :
            Educational attainment : B15003
        Transport :
            Household Size by Vehicles Available : B08201
            Number of Workers in Household by Vehicles Available : B08203
            Means of Transportation to Work : B08301
            Travel time to work : B08303
            Sex of workers by Means of Transportation to Work : B08006
        Disability :
            Disability by age : C18108



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

        if self.sql_obj.test_mode is True:

            self.table_name = TABLE_NAME_CENSUS_TEST
            if self.sql_obj.check_table_exists(TABLE_NAME_CENSUS_TEST):
                self.sql_obj.drop_table(TABLE_NAME_CENSUS_TEST, confirm=True)
        else:
            self.table_name = TABLE_NAME_CENSUS
        # Chosen variables for model (query these from the SQL table, if they don't exist, generate it
        self.census_variables_list = []

    def merge_census_info(self, properties_gpd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """TODO: This is running super slow. Better to query at the block group level."""
        if "geoid" not in properties_gpd.columns:
            raise ValueError("No geoid column in input.")

        if self.leafs_only:
            variables = self._get_leaf_variables()
        else:
            variables = self._get_all_variables()
        print(
            f"Getting {len(variables)} many features for {properties_gpd.shape[0]} "
            f"many properties at the {self.granularity} level."
        )

        self.sql_obj.create_census_cache_table()
        census_df = self._extract_from_geoid(properties_gpd, variables)

        cleaned_df = self._transform_and_clean(census_df)

        self._load_to_sql(cleaned_df)

        return cleaned_df

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

        chunk_size = 50
        var_chunks = [
            variables[i : i + chunk_size] for i in range(0, len(variables), chunk_size)
        ]

        results = []
        for idx, row in unique_geos.iterrows():

            # Check cache first
            cached = self.sql_obj.get_cached_census_results(
                merge_cols, row, self.granularity, self.year
            )

            if cached is not None:
                print(f"[{idx}] Cache hit")
                for col in merge_cols:
                    cached[col] = row[col]
                results.append(cached)
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

        return properties_data.merge(pd.DataFrame(results), on=merge_cols, how="left")

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

        # 3. Drop columns that are entirely NaN
        df.dropna(axis=1, how="all", inplace=True)

        return df

    def _load_to_sql(self, clean_data: pd.DataFrame) -> None:
        """
        Save the data as a SQL db.?
        """
        merge_cols = self._get_merge_columns()
        census_cols = [c for c in clean_data.columns if c.startswith("B")]

        # Melt wide format to narrow: one row per (geography, variable)
        # gets past row size limits
        long_df = (
            clean_data[merge_cols + census_cols]
            .copy()
            .melt(
                id_vars=merge_cols,
                var_name="variable",
                value_name="value",
            )
        )

        table_name = (
            TABLE_NAME_CENSUS_TEST if self.sql_obj.test_mode else TABLE_NAME_CENSUS
        )
        self.sql_obj.save_df_to_sql(table_name, long_df)
        return

    def _read_from_sql(self) -> pd.DataFrame:

        return

    def initialize_census_db(self) -> None:

        return

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

    def add_census_variables(self, var_list: list | str) -> None:
        """TODO: Implement, should update the saved HDD version CENSUS_FEATURES, and the
        saved HDD version of the SQL DB with the census info. Eventually adding in the
        SQL DB functionality."""
        if isinstance(var_list, list):
            for var in var_list:
                pass
        elif isinstance(var_list, str):
            pass
        else:
            raise TypeError("'var_list' must be a string of list of strings")
        return

    # def visualize_data(self):
    #     return

    # def _store_feature_info(self):
    #     """Stores the descriptions of the features in a table in the SQL DB.
    #     This table can be updated and queried to add additional features to
    #     the combined GeoDataFrame."""
    #     return

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

    def _get_leaf_variables(self, year=2023):
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


if __name__ == "__main__":

    sql_obj = SQL()

    test_obj = CensusData(sql_obj=sql_obj, year=2023)
    test_obj.visualize_data()
