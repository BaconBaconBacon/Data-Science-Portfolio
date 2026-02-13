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
import re
import requests
import time

import os
from pathlib import Path
import pandas as pd
import sqlalchemy as s

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

from sql_funcs import SQL
from functools import lru_cache


@lru_cache(maxsize=4)
def _fetch_census_labels(year: int) -> dict:
    """Fetch census variable labels, with local JSON cache to avoid repeat downloads."""
    cache_dir = Path("data") / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"census_labels_{year}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    print(f"Fetching census variable labels for {year} from API (~10MB, one-time download)...")
    url = f"https://api.census.gov/data/{year}/acs/acs5/variables.json"
    r = requests.get(url)
    vars_data = r.json()["variables"]
    labels = {
        k: {"label": v.get("label", ""), "concept": v.get("concept", "")}
        for k, v in vars_data.items()
    }

    with open(cache_file, "w") as f:
        json.dump(labels, f)
    print(f"Cached labels to {cache_file}")

    return labels


def _shorten_concept(concept: str) -> str:
    """Shorten a Census concept string into a compact topic label.

    Example: 'Place of Birth by Nativity and Citizenship Status' -> 'Birthplace/Citizenship'
    """
    short = concept

    # Remove parenthetical qualifiers (inflation-adjusted, dollars, etc.)
    short = re.sub(r"\s*\([^)]*\)", "", short)

    # Remove common filler phrases
    filler = [
        "in the United States",
        "for the Population",
        "of the Population",
        "of the Total",
        "in the Past 12 Months",
        "for Occupied Housing Units",
        "for Housing Units",
    ]
    for f in filler:
        short = short.replace(f, "")

    # Collapse "X by Y by Z" -> keep first noun phrase from each
    if " by " in short:
        parts = short.split(" by ")
        shortened = []
        for p in parts:
            words = p.strip().split()
            skip = {"of", "the", "and", "in", "for", "a", "an", "or"}
            core = [w for w in words if w.lower() not in skip]
            shortened.append(" ".join(core[:2]) if core else p.strip())
        short = "/".join(shortened)

    # Clean up extra whitespace
    short = re.sub(r"\s{2,}", " ", short).strip()

    if len(short) > 45:
        short = short[:42] + "..."

    return short


def census_code_to_label(code: str, year: int = 2023) -> str:
    """
    Convert census variable code to a concise human-readable label.

    Standalone function that fetches from Census API.
    Results are cached for the session.

    Parameters
    ----------
    code : str
        Census variable code (e.g., 'B25040_004E')
    year : int
        ACS5 year (default: 2023)

    Returns
    -------
    str
        Concise label: 'Topic: last hierarchy level(s)'

    Example
    -------
    >>> census_code_to_label('B25040_004E')
    'House Heating/Fuel: Fuel oil, kerosene, etc.'
    """
    labels = _fetch_census_labels(year)
    info = labels.get(code)
    if not info:
        return f"Unknown: {code}"

    concept = info.get("concept", "")
    label = info.get("label", "")

    # Clean up label — remove "Estimate!!Total:!!" prefix and parentheticals
    clean_label = label.replace("Estimate!!Total:!!", "").replace("Estimate!!", "")
    clean_label = re.sub(r"\s*\([^)]*\)", "", clean_label)

    # Split the !! hierarchy and keep only the last meaningful part
    parts = [p.strip().rstrip(":") for p in clean_label.split("!!") if p.strip()]
    if parts:
        clean_label = parts[-1]

    if concept:
        short_concept = _shorten_concept(concept)
        # Skip label if it's just "Total", empty, or restates the concept
        if clean_label in ("Total", ""):
            return short_concept
        # Check if label is redundant with concept
        concept_lower = concept.lower()
        label_lower = clean_label.lower()
        if label_lower in concept_lower or concept_lower in label_lower:
            return short_concept
        concept_words = set(concept_lower.split())
        label_words = set(label_lower.split())
        if (
            len(label_words) > 3
            and len(concept_words & label_words) / len(label_words) > 0.5
        ):
            return short_concept
        result = f"{short_concept}: {clean_label}"
    else:
        result = clean_label

    if len(result) > 55:
        result = result[:52] + "..."
    return result


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
        self,
        sql_obj: SQL,
        year: int,
        granularity: str = "tract",
        sparse: bool = True,
        verbose: bool = True,
    ):
        if granularity not in CENSUS_VALID_GRANULARITY_LEVELS:
            raise ValueError(
                f"granularity must be one of {CENSUS_VALID_GRANULARITY_LEVELS}"
            )
        self.granularity = granularity
        self.leafs_only = sparse
        self.year = year
        self.verbose = verbose

        self.census = census.Census(
            os.environ.get("US_CENSUS_API_KEY"), year=self.year
        ).acs5

        self.metadata_filepath = PATH_DATA_CENSUS / f"acs5_variables_{year}.txt"

        if not self.metadata_filepath.exists():
            self._generate_variable_metadata()
        else:
            pass

        self.sql_obj = sql_obj
        self.test_mode = self.sql_obj.test_mode

        if self.test_mode:
            if self.verbose:
                print("starting in test mode")
            self.table_name = TABLE_NAME_CENSUS_TEST
            self.sql_obj.drop_table(TABLE_NAME_CENSUS_TEST, confirm=True)
            self.sql_obj.drop_table(TABLE_NAME_CENSUS_PROPS_TEST, confirm=True)
        else:
            self.table_name = TABLE_NAME_CENSUS
        self.data = self._read_from_sql()

        # Chosen variables for model (query these from the SQL table, if they don't exist, generate it
        self.census_variables_list = []

    def merge_census_info(self, properties_gpd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """check existing data, only fetch missing geoids"""

        if "geoid" not in properties_gpd.columns:
            raise ValueError("No geoid column in input.")

        # Check existing data
        merge_cols = self._get_merge_columns()
        initial_count = len(properties_gpd)

        existing_wide = self._get_existing_wide(merge_cols)
        existing_wide, failed_geos = self._handle_missing_geos(
            properties_gpd, existing_wide, merge_cols
        )

        # Drop properties in failed geos (e.g., Virginia independent cities)
        if failed_geos:
            failed_df = pd.DataFrame(failed_geos)
            # Mark rows to drop
            properties_gpd = properties_gpd.merge(
                failed_df, on=merge_cols, how="left", indicator="_failed"
            )
            properties_gpd = properties_gpd[properties_gpd["_failed"] == "left_only"]
            properties_gpd = properties_gpd.drop(columns=["_failed"])
            dropped = initial_count - len(properties_gpd)
            if self.verbose:
                print(
                    f"Dropped {dropped} properties in {len(failed_geos)} geographies "
                    "with no census data (e.g., Virginia independent cities)."
                )

        # Ensure properties have correct dtypes for merge
        for col in merge_cols:
            if col in properties_gpd.columns:
                properties_gpd[col] = properties_gpd[col].astype("int64")
            if col in existing_wide.columns:
                existing_wide[col] = existing_wide[col].astype("int64")

        # Merge wide census data onto all properties
        result = properties_gpd.merge(existing_wide, on=merge_cols, how="left")

        # Drop geo ID columns
        drop_cols = [x for x in result.columns if ("_id" in x) or ("_grp" in x)]
        result = result.drop(columns=drop_cols)

        if self.test_mode:
            try:
                _ = self._read_from_sql()
                print("_read_from_sql() Successfully tested.")
            except Exception as e:
                print(f"_read_from_sql() failed with {e}")
        return result

    def _add_missing_geos(
        self,
        gdf: gpd.GeoDataFrame,
        wide: pd.DataFrame,
        merge_cols: list,
        missing_geos,
        use_bulk: bool = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        variables = (
            self._get_leaf_variables() if self.leafs_only else self._get_all_variables()
        )
        self.sql_obj.create_census_cache_table()

        missing_props = gdf.merge(missing_geos, on=merge_cols, how="inner")

        # Auto-detect bulk vs individual based on dataset sparsity
        if use_bulk is None:
            unique_geos = len(missing_geos)
            num_states = missing_geos["state_id"].nunique()

            # Threshold: use individual if unique_geos < num_states * 50
            threshold = num_states * 10
            use_bulk = unique_geos >= threshold

            if self.verbose:
                print(
                    f"Census fetch: {unique_geos} unique geos across {num_states} states"
                )
                print(
                    f"Using {'BULK' if use_bulk else 'INDIVIDUAL'} fetch (threshold: {threshold})"
                )

        # Use bulk fetch (by state) for speed, or per-geo for individual
        # Note: _bulk_fetch_missing_geos now processes state-by-state and saves directly to SQL
        if use_bulk:
            _, failed_geos = self._bulk_fetch_missing_geos(missing_props, variables)
        else:
            new_census_df, failed_geos = self._extract_from_geoid(
                missing_props, variables
            )
            # Non-bulk path still needs manual transform/save
            if not new_census_df.empty:
                print("Transforming data...")
                new_cleaned = self._transform_and_clean(new_census_df)
                print("Saving to database...")
                self._load_to_sql(new_cleaned)

        # Refresh data from SQL (bulk path saves directly, non-bulk saves above)
        self.data = self._read_from_sql()

        # Build wide format from census table (geography-level, not property-level)
        # CRITICAL FIX: Only read the missing geographies we just fetched, not entire table
        census_table = TABLE_NAME_CENSUS_TEST if self.test_mode else TABLE_NAME_CENSUS
        if self.sql_obj.check_table_exists(census_table) and not missing_geos.empty:
            # Build WHERE clause to filter to only missing geographies
            where_clauses = []
            for _, row in missing_geos.iterrows():
                clause_parts = [f"state_id = {int(row['state_id'])}"]
                if "county_id" in row and pd.notna(row.get("county_id")):
                    clause_parts.append(f"county_id = {int(row['county_id'])}")
                if "tract_id" in row and pd.notna(row.get("tract_id")):
                    clause_parts.append(f"tract_id = {int(row['tract_id'])}")
                if "block_grp" in row and pd.notna(row.get("block_grp")):
                    clause_parts.append(f"block_grp = {int(row['block_grp'])}")
                where_clauses.append("(" + " AND ".join(clause_parts) + ")")

            missing_filter = " OR ".join(where_clauses)

            census_long = self.sql_obj.read_df_from_sql(
                f"""SELECT * FROM {census_table}
                WHERE granularity = '{self.granularity}'
                AND year = {self.year}
                AND ({missing_filter})"""
            )
            if not census_long.empty:
                # Pivot back to wide format
                new_wide = census_long.pivot_table(
                    index=merge_cols,
                    columns="variable",
                    values="value",
                    aggfunc="first",
                ).reset_index()
                new_wide.columns.name = None

                # Ensure merge_cols have consistent dtypes
                for col in merge_cols:
                    new_wide[col] = new_wide[col].astype("int64")

                existing_wide = (
                    pd.concat([wide, new_wide], ignore_index=True).drop_duplicates(
                        merge_cols
                    )
                    if not wide.empty
                    else new_wide
                )
            else:
                existing_wide = wide
        else:
            existing_wide = wide

        return existing_wide, failed_geos

    def _handle_missing_geos(
        self, gdf: gpd.GeoDataFrame, wide, merge_cols: list
    ) -> tuple[pd.DataFrame, list[dict]]:

        required_geos = gdf[merge_cols].drop_duplicates()
        failed_geos = []

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

        if self.verbose:
            print(
                f"{len(required_geos)} geographies needed, "
                f"{len(required_geos) - len(missing_geos)} cached, "
                f"{len(missing_geos)} to fetch."
            )
        if not missing_geos.empty:
            wide, failed_geos = self._add_missing_geos(
                gdf, wide, merge_cols, missing_geos, use_bulk=None  # Auto-detect
            )

        return wide, failed_geos

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

    def _fetch_bulk_by_state(
        self, state_id: str, variables: list, var_chunks: list
    ) -> list[dict]:
        """
        Fetch ALL geographies in a state with one API call per variable chunk.

        Instead of querying each block group individually, this queries all
        block groups in a state at once using the Census API wildcard syntax.
        """
        results = []

        # Build the query based on granularity
        # Block groups require full hierarchy: state -> county -> tract -> block group
        state_str = str(state_id).zfill(2)
        if self.granularity == "block_group":
            query = {
                "for": "block group:*",
                "in": f"state:{state_str} county:* tract:*",
            }
        elif self.granularity == "tract":
            query = {"for": "tract:*", "in": f"state:{state_str}"}
        else:  # county
            query = {"for": "county:*", "in": f"state:{state_str}"}

        # Accumulate results across variable chunks
        combined_by_geo = {}  # geo_key -> {var: value, ...}

        for chunk in var_chunks:
            try:
                chunk_results = self.census.get(chunk, query)
                if chunk_results:
                    for row in chunk_results:
                        # Build geo key based on granularity
                        if self.granularity == "block_group":
                            geo_key = (
                                row.get("state", ""),
                                row.get("county", ""),
                                row.get("tract", ""),
                                row.get("block group", ""),
                            )
                        elif self.granularity == "tract":
                            geo_key = (
                                row.get("state", ""),
                                row.get("county", ""),
                                row.get("tract", ""),
                            )
                        else:  # county
                            geo_key = (
                                row.get("state", ""),
                                row.get("county", ""),
                            )

                        if geo_key not in combined_by_geo:
                            combined_by_geo[geo_key] = {}
                        combined_by_geo[geo_key].update(row)
            except Exception as e:
                print(f"  Error fetching chunk for state {state_str}: {e}")
                continue

        # Convert to list of dicts with proper column names
        for geo_key, data in combined_by_geo.items():
            record = {}
            if self.granularity == "block_group":
                record["state_id"] = int(geo_key[0]) if geo_key[0] else 0
                record["county_id"] = int(geo_key[1]) if geo_key[1] else 0
                record["tract_id"] = int(geo_key[2]) if geo_key[2] else 0
                record["block_grp"] = int(geo_key[3]) if geo_key[3] else 0
            elif self.granularity == "tract":
                record["state_id"] = int(geo_key[0]) if geo_key[0] else 0
                record["county_id"] = int(geo_key[1]) if geo_key[1] else 0
                record["tract_id"] = int(geo_key[2]) if geo_key[2] else 0
            else:  # county
                record["state_id"] = int(geo_key[0]) if geo_key[0] else 0
                record["county_id"] = int(geo_key[1]) if geo_key[1] else 0

            # Add census variables (skip geo columns)
            skip_cols = {"state", "county", "tract", "block group", "GEO_ID", "NAME"}
            for var, val in data.items():
                if var not in skip_cols:
                    record[var] = val
            results.append(record)

        return results

    def _get_cached_states(self) -> set:
        """Get set of state_ids that are already cached for this granularity/year."""
        q = f"""
        SELECT DISTINCT state_id FROM {TABLE_NAME_CACHE}
        WHERE granularity = :granularity AND year = :year AND variable != '_FAILED_'
        """
        result = pd.read_sql(
            s.text(q),
            self.sql_obj.connection,
            params={"granularity": self.granularity, "year": self.year},
        )
        return set(result["state_id"].tolist())

    def _save_state_to_cache(self, state_results: list[dict], merge_cols: list) -> None:
        """Save a state's census data to cache in long format."""
        if not state_results:
            return

        # Get state_id from first result
        state_id = state_results[0].get("state_id", 0)

        # Delete existing cache for this state/granularity/year before inserting
        delete_q = f"""
            DELETE FROM {TABLE_NAME_CACHE}
            WHERE state_id = :state_id
              AND granularity = :granularity
              AND year = :year
        """
        self.sql_obj.connection.execute(
            s.text(delete_q),
            {"state_id": state_id, "granularity": self.granularity, "year": self.year},
        )
        self.sql_obj.connection.commit()

        records = []
        for row in state_results:
            for var, val in row.items():
                if var in merge_cols or not var.startswith("B"):
                    continue
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
            df = pd.DataFrame(records)
            self.sql_obj.save_df_to_sql(TABLE_NAME_CACHE, df)

    def _load_cached_data_wide(self, state_ids: list, merge_cols: list) -> pd.DataFrame:
        """Load cached census data for given states in wide format, chunked by state."""
        if not state_ids:
            return pd.DataFrame()

        results = []
        start_time = time.time()
        for i, state_id in enumerate(state_ids, 1):
            if len(state_ids) > 1 and (i % 10 == 0 or i == len(state_ids)):
                elapsed = time.time() - start_time
                avg_per_state = elapsed / i
                remaining = avg_per_state * (len(state_ids) - i)
                print(
                    f"  Loading state {i}/{len(state_ids)}... ({remaining:.0f}s remaining)"
                )

            q = f"""
            SELECT state_id, county_id, tract_id, block_grp, variable, value
            FROM {TABLE_NAME_CACHE}
            WHERE state_id = :state_id
              AND granularity = :granularity
              AND year = :year
              AND variable != '_FAILED_'
            """
            df = pd.read_sql(
                s.text(q),
                self.sql_obj.connection,
                params={
                    "state_id": int(state_id),
                    "granularity": self.granularity,
                    "year": self.year,
                },
            )
            if df.empty:
                continue

            wide = df.pivot_table(
                index=merge_cols, columns="variable", values="value", aggfunc="first"
            ).reset_index()
            wide.columns.name = None
            results.append(wide)

            del df  # Free memory immediately

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def _bulk_fetch_missing_geos(
        self, properties_data: pd.DataFrame, variables: list
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Bulk fetch census data for all missing geographies by state.

        Fetches and caches each state's data immediately to avoid memory issues.
        On subsequent runs, cached states are loaded from SQL instead of API.
        """
        merge_cols = self._get_merge_columns()
        unique_states = list(properties_data["state_id"].unique())
        total_states = len(unique_states)

        # properties_data is already filtered to MISSING geographies only
        # If a state appears here, its cache is incomplete - must fetch
        states_to_fetch = unique_states

        print(f"Fetching {len(states_to_fetch)} states with missing geographies...")

        # Fetch states with incomplete cache
        if states_to_fetch:
            var_chunks = [
                variables[i : i + CENSUS_CHUNK_SIZE]
                for i in range(0, len(variables), CENSUS_CHUNK_SIZE)
            ]

            start_time = time.time()
            for i, state_id in enumerate(states_to_fetch):
                completed = i + 1
                state_str = str(state_id).zfill(2)
                print(
                    f"[{completed}/{len(states_to_fetch)}] Fetching state {state_str}..."
                )

                state_results = self._fetch_bulk_by_state(
                    state_id, variables, var_chunks
                )

                if state_results:
                    print(
                        f"  Got {len(state_results)} {self.granularity}s, saving to cache..."
                    )
                    self._save_state_to_cache(state_results, merge_cols)
                else:
                    print(f"  No data for state {state_str}")

                # Progress estimate after first 3 states
                if completed == min(3, len(states_to_fetch)):
                    elapsed = time.time() - start_time
                    per_state = elapsed / completed
                    remaining = (len(states_to_fetch) - completed) * per_state
                    print(
                        f"  Time estimate: {self._format_duration(remaining)} remaining "
                        f"({per_state:.1f}s per state)"
                    )

            total_time = time.time() - start_time
            print(f"Fetch complete in {self._format_duration(total_time)}")

        # Check granularity BEFORE processing (only once)
        print("Checking output tables...")
        self._check_and_prepare_output_tables()

        # Process state by state to reduce RAM usage
        print("Processing states...")
        failed_geos = []
        start_time = time.time()

        for i, state_id in enumerate(unique_states, 1):
            # Progress estimate
            if i > 1:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i - 1)) * (len(unique_states) - i + 1)
                print(
                    f"  Processing state {i}/{len(unique_states)}... ({remaining:.0f}s remaining)"
                )
            else:
                print(f"  Processing state {i}/{len(unique_states)}...")

            # Load single state from cache
            state_wide = self._load_cached_data_wide([state_id], merge_cols)
            if state_wide.empty:
                continue

            # Filter properties to this state
            state_props = properties_data[properties_data["state_id"] == state_id]

            # Merge with census data
            merged = state_props.merge(state_wide, on=merge_cols, how="left")

            # Track failed geos
            census_cols = [c for c in state_wide.columns if c.startswith("B")]
            if census_cols:
                missing_mask = merged[census_cols[0]].isna()
                if missing_mask.any():
                    failed_rows = merged[missing_mask][merge_cols].drop_duplicates()
                    failed_geos.extend(failed_rows.to_dict("records"))

            # Transform and save (if has data)
            if not merged.empty and census_cols:
                cleaned = self._transform_and_clean(merged)
                self._save_state_to_output(cleaned, merge_cols)
                del cleaned

            # Free memory
            del state_wide, state_props, merged

        total_time = time.time() - start_time
        print(f"Processing complete in {self._format_duration(total_time)}")

        # Return empty DF (data already saved to SQL), plus failed geos
        return pd.DataFrame(), failed_geos

    def _extract_from_geoid(self, properties_data, variables):

        merge_cols = self._get_merge_columns()
        unique_geos = properties_data[merge_cols].drop_duplicates()
        total_geos = len(unique_geos)
        print(f"Unique geoids: {total_geos}.")

        var_chunks = [
            variables[i : i + CENSUS_CHUNK_SIZE]
            for i in range(0, len(variables), CENSUS_CHUNK_SIZE)
        ]

        # Time tracking setup
        start_time = time.time()
        estimate_sample = min(5, total_geos)
        estimate_shown = False
        api_calls = 0
        cache_hits = 0

        results = []
        failed_geos = []  # Track geos that couldn't be fetched
        for i, (idx, row) in enumerate(unique_geos.iterrows()):
            completed = i + 1

            # Check if this is a known failed geography
            if self.sql_obj.is_cached_failure(
                merge_cols, row, self.granularity, self.year
            ):
                print(f"[{completed}/{total_geos}] Skipping known failed geography")
                failed_geos.append(row[merge_cols].to_dict())
                continue

            # Check cache for successful results
            cached = self.sql_obj.get_cached_census_results(
                merge_cols, row, self.granularity, self.year
            )

            if cached is not None:
                cache_hits += 1
                print(f"[{completed}/{total_geos}] Cache hit")
                cached_dict = cached.to_dict()
                for col in merge_cols:
                    cached_dict[col] = row[col]
                results.append(cached_dict)
            else:
                # Query API
                api_calls += 1
                geo_id, query_for, query_in = self._get_geoid_query_strings(row)

                print(
                    f"[{completed}/{total_geos}] Querying id {geo_id} at {self.granularity}..."
                )

                combined_result = {}
                try:
                    for chunk in var_chunks:
                        result = self.census.get(
                            chunk, {"for": query_for, "in": query_in}
                        )
                        if result:
                            combined_result.update(result[0])

                    if combined_result:
                        # Cache immediately after success
                        self._cache_results(row, merge_cols, combined_result)
                        for col in merge_cols:
                            combined_result[col] = row[col]
                        results.append(combined_result)
                    else:
                        # API returned no data - cache the failure
                        self._cache_failed_geo(row, merge_cols)
                        failed_geos.append(row[merge_cols].to_dict())

                except Exception as e:
                    print(f"Skipped! Error for {geo_id}: {e}")
                    self._cache_failed_geo(row, merge_cols)
                    failed_geos.append(row[merge_cols].to_dict())

            # Show initial time estimate after sample completes
            if completed == estimate_sample and not estimate_shown:
                elapsed = time.time() - start_time
                per_geo = elapsed / completed
                remaining = total_geos - completed
                est_remaining_sec = remaining * per_geo
                est_total_sec = total_geos * per_geo
                print(
                    f"  Time estimate: {self._format_duration(est_total_sec)} total "
                    f"({per_geo:.2f}s per geography, "
                    f"{self._format_duration(est_remaining_sec)} remaining)"
                )
                estimate_shown = True

            # Show progress every 10 geographies (after estimate shown)
            elif completed % 10 == 0 and estimate_shown:
                elapsed = time.time() - start_time
                per_geo = elapsed / completed
                remaining = total_geos - completed
                est_remaining = remaining * per_geo
                print(
                    f"  [{completed}/{total_geos}] {api_calls} API calls, {cache_hits} cache hits "
                    f"- {self._format_duration(est_remaining)} remaining"
                )

        # Final completion message
        total_time = time.time() - start_time
        print(
            f"Completed fetching census data in {self._format_duration(total_time)} "
            f"({api_calls} API calls, {cache_hits} cache hits)"
        )

        if len(results) > 0:
            return (
                properties_data.merge(pd.DataFrame(results), on=merge_cols, how="left"),
                failed_geos,
            )
        else:
            # All geos failed - return empty dataframe with correct columns
            print(
                f"Warning: No census data available for any of the {total_geos} "
                "geographies (e.g., Virginia independent cities)."
            )
            return properties_data.iloc[:0], failed_geos

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

    def _cache_failed_geo(self, row: pd.Series, merge_cols: list) -> None:
        """Cache a failed geography with sentinel marker to prevent re-querying."""
        record = {
            "state_id": row.get("state_id", 0),
            "county_id": row.get("county_id", 0),
            "tract_id": row.get("tract_id", 0),
            "block_grp": row.get("block_grp", 0),
            "granularity": self.granularity,
            "year": self.year,
            "variable": "_FAILED_",
            "value": None,
        }
        self.sql_obj.save_df_to_sql(
            table_name=TABLE_NAME_CACHE, df=pd.DataFrame([record])
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
            total = (
                df[table_cols].sum(axis=1).replace(0, np.nan).infer_objects(copy=False)
            )
            for col in table_cols:
                df[col] = df[col] / total

        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how="all")

        return df

    def _load_to_sql(self, clean_data: pd.DataFrame) -> None:
        """
        Load cleaned census data into the SQL database.
        """
        # Check for granularity mismatch in existing props_census table
        props_table = (
            TABLE_NAME_CENSUS_PROPS_TEST if self.test_mode else TABLE_NAME_CENSUS_PROPS
        )
        if self.sql_obj.check_table_exists(props_table):
            existing = self.sql_obj.read_df_from_sql(
                f"SELECT DISTINCT granularity FROM {props_table} LIMIT 1"
            )
            if not existing.empty:
                existing_gran = existing["granularity"].iloc[0]
                if existing_gran != self.granularity:
                    print(
                        f"\nTable '{props_table}' exists with granularity '{existing_gran}'."
                    )
                    print(f"Current run uses granularity '{self.granularity}'.")
                    response = (
                        input("Drop existing table and continue? [y/N]: ")
                        .strip()
                        .lower()
                    )
                    if response == "y":
                        self.sql_obj.drop_table(props_table, confirm=True)
                    else:
                        raise RuntimeError(
                            f"Cannot insert '{self.granularity}' data into table with '{existing_gran}' schema. "
                            "Drop the table manually or run with matching granularity."
                        )

        merge_cols = self._get_merge_columns()
        census_cols = [c for c in clean_data.columns if c.startswith("B")]

        # Ensure merge_cols exist and have correct dtypes
        for col in merge_cols:
            if col not in clean_data.columns:
                raise ValueError(
                    f"Missing merge column '{col}' in clean_data. "
                    f"Columns: {clean_data.columns.tolist()}"
                )
            clean_data[col] = clean_data[col].astype("int64")

        # Ensure geoid exists
        if "geoid" not in clean_data.columns:
            raise ValueError(
                f"Missing 'geoid' column in clean_data. "
                f"Columns: {clean_data.columns.tolist()}"
            )

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

        # Check for granularity mismatch in existing census table
        census_table = TABLE_NAME_CENSUS_TEST if self.test_mode else TABLE_NAME_CENSUS
        if self.sql_obj.check_table_exists(census_table):
            existing = self.sql_obj.read_df_from_sql(
                f"SELECT DISTINCT granularity FROM {census_table} LIMIT 1"
            )
            if not existing.empty:
                existing_gran = existing["granularity"].iloc[0]
                if existing_gran != self.granularity:
                    print(
                        f"\nTable '{census_table}' exists with granularity '{existing_gran}'."
                    )
                    print(f"Current run uses granularity '{self.granularity}'.")
                    response = (
                        input("Drop existing table and continue? [y/N]: ")
                        .strip()
                        .lower()
                    )
                    if response == "y":
                        self.sql_obj.drop_table(census_table, confirm=True)
                    else:
                        raise RuntimeError(
                            f"Cannot insert '{self.granularity}' data into table with '{existing_gran}' schema."
                        )

        self.sql_obj.save_df_to_sql(census_table, census_long)

    def _check_table_granularity(self, table_name: str) -> None:
        """Check if existing table has matching granularity, prompt to drop if not."""
        if not self.sql_obj.check_table_exists(table_name):
            return

        existing = self.sql_obj.read_df_from_sql(
            f"SELECT DISTINCT granularity FROM {table_name} LIMIT 1"
        )
        if existing.empty:
            return

        existing_gran = existing["granularity"].iloc[0]
        if existing_gran != self.granularity:
            print(f"\nTable '{table_name}' exists with granularity '{existing_gran}'.")
            print(f"Current run uses granularity '{self.granularity}'.")
            response = (
                input("Drop existing table and continue? [y/N]: ").strip().lower()
            )
            if response == "y":
                self.sql_obj.drop_table(table_name, confirm=True)
            else:
                raise RuntimeError(
                    f"Cannot insert '{self.granularity}' data into table with '{existing_gran}' schema."
                )

    def _check_and_prepare_output_tables(self) -> None:
        """Check granularity for both output tables before processing."""
        props_table = (
            TABLE_NAME_CENSUS_PROPS_TEST if self.test_mode else TABLE_NAME_CENSUS_PROPS
        )
        census_table = TABLE_NAME_CENSUS_TEST if self.test_mode else TABLE_NAME_CENSUS

        self._check_table_granularity(props_table)
        self._check_table_granularity(census_table)

    def _save_state_to_output(self, clean_data: pd.DataFrame, merge_cols: list) -> None:
        """Melt and insert one state's data to output tables (no granularity check)."""
        census_cols = [c for c in clean_data.columns if c.startswith("B")]

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

        props_table = (
            TABLE_NAME_CENSUS_PROPS_TEST if self.test_mode else TABLE_NAME_CENSUS_PROPS
        )
        self.sql_obj.save_df_to_sql(props_table, props_long)

        # Save just the census information
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

        census_table = TABLE_NAME_CENSUS_TEST if self.test_mode else TABLE_NAME_CENSUS
        self.sql_obj.save_df_to_sql(census_table, census_long)

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
        """
        Get human-readable label from census variable code.

        Parameters
        ----------
        code : str
            Census variable code (e.g., 'B25031_003E')

        Returns
        -------
        str
            Human-readable label from Census API
        """
        labels = self.get_acs5_fields_with_labels(year=self.year)
        return labels.get(code, f"Unknown: {code}")


if __name__ == "__main__":
    SQL.kill_idle()
    sql_obj = SQL()
    test_obj = CensusData(sql_obj=sql_obj, year=2023)
    sql_obj.disconnect_and_close()
