""" Stores commonly used SQL functions."""

import sqlalchemy as s
from scipy.stats import foldnorm

from settings import (
    SQL_ENGINE_STR,
    TEST_SQL_ENGINE_STR,
    PROP_LABELS_KEYS_MAP,
    GIS_DEFAULT_CRS,
    TABLE_NAME_CACHE,
    HEADER_GEOM,
)
import pandas as pd
import geopandas as gpd


class SQL:

    def __init__(self, test=False) -> None:
        # TODO: Move the SQL stuff into here when other classes are ready

        self.test_mode = test
        self.connection = None
        self.engine = None

        if self.test_mode:
            self.engine_string = TEST_SQL_ENGINE_STR
        else:
            self.engine_string = SQL_ENGINE_STR

        self._connect_to_sql()

        return

    def _connect_to_sql(self):
        self.engine = s.create_engine(self.engine_string)
        self.connection = self.engine.connect()
        return

    def disconnect_and_close(self) -> None:
        return

    def _execute_string(self, string: str) -> None:
        """Execute string without committing."""
        string = self._sanitize_string(string)
        self.connection.execute(string)
        return

    def create_table(self) -> None:
        return

    def check_table_exists(self, table_name: str) -> bool:
        return s.inspect(self.engine).has_table(table_name)

    def drop_table(self, table_name: str, confirm: bool = False) -> None:

        if not confirm:
            raise RuntimeError(
                "Potentially accidental function call. Set confirm variable."
            )
        try:
            q = s.text(f"DROP TABLE IF EXISTS {table_name};")
            self.connection.execute(q)
            self.connection.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to drop table: {table_name}': {e}")

        result = self.connection.execute(
            s.text(
                "SELECT EXISTS ("
                "SELECT FROM information_schema.tables "
                f"WHERE table_name = '{table_name}'"
                ");"
            )
        ).scalar()
        if result:
            raise RuntimeError(f"Table '{table_name}' still exists after DROP.")

        return

    def get_table_names(self) -> list:
        return

    def rename_table(self) -> None:
        return

    def add_rows_to_table(self) -> None:
        return

    def remove_rows_from_table(self) -> None:
        return

    def add_columns_to_table(self) -> None:
        return

    def remove_columns_from_table(self) -> None:
        return

    def execute_and_commit_string(self, query: str):
        self.execute_string(query)
        self.connection.commit()
        return

    def execute_string(self, query: str) -> None:
        query = self._sanitize_string(query)
        self.connection.execute(s.text(query))
        return

    def _sanitize_string(self, string: str) -> str:
        if "DROP TABLE" in string or "CREATE TABLE" in string:
            raise ValueError("Invalid SQL string")
        else:
            return string

    # def query_db(self, query: str):
    #     query = self._sanitize_string(query)
    #     return

    def create_census_cache_table(self) -> None:
        """
        Creates a temporary table to store census information should the API query processes fail part way.
        TODO: Turn these headers/dtypes into a dict for settings.py
        """
        q = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME_CACHE} (
            state_id INTEGER,
            county_id INTEGER,
            tract_id INTEGER,
            block_grp INTEGER,
            granularity VARCHAR(20),
            year INTEGER,
            variable VARCHAR(50),
            value DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (state_id, county_id, tract_id, block_grp, granularity, year, variable)
        );
        """
        self.connection.execute(s.text(q))
        self.connection.commit()
        return

    def get_cached_census_results(
        self, cols, row: pd.Series, granularity: str, year: int
    ) -> pd.Series | None:
        """Return cached census results as a Series (variable names →
        values), or None if not cached."""
        conditions = " AND ".join([f"{col} = {row[col]}" for col in cols])
        q = f"""
            SELECT variable, value FROM {TABLE_NAME_CACHE}
            WHERE {conditions}
            AND granularity = '{granularity}'
            AND year = {year}
            """
        result = self.read_df_from_sql(q)
        if len(result) > 0:
            return pd.Series(result["value"].values, index=result["variable"].values)
        return None

    def save_df_to_sql(self, table_name: str, df: pd.DataFrame) -> None:
        df.to_sql(table_name, self.connection, if_exists="append", index=False)
        self.connection.commit()
        return

    def read_df_from_sql(self, query: str):
        query = self._sanitize_string(query)
        return pd.read_sql(query, self.connection)

    def save_gpd_to_sql(self, table_name: str, gdf: gpd.GeoDataFrame) -> None:
        table_name = self._sanitize_string(table_name)
        gdf.to_postgis(table_name, self.connection, if_exists="append", index=False)
        self.connection.commit()
        return

    def read_gpd_from_sql(self, table_name) -> gpd.GeoDataFrame:
        table_name = self._sanitize_string(table_name)

        q = f"SELECT * FROM {table_name}"

        return gpd.read_postgis(q, con=self.connection, geom_col=HEADER_GEOM)

    def initialize_properties_table(self, prop_name: str):

        prop_name = self._sanitize_string(prop_name)

        print(f"Creating new '{prop_name}' table with 10 entries.")
        q = "CREATE TABLE {} (".format(prop_name)
        for key in PROP_LABELS_KEYS_MAP.keys():
            q += f"{key} BIGINT,"
        q += f"geom geometry(Geometry, {GIS_DEFAULT_CRS}));"  # .format(GIS_DEFAULT_CRS)
        self.connection.execute(s.text(q))

        q = f"SELECT * FROM {prop_name}"
        return gpd.read_postgis(q, con=self.connection, geom_col=HEADER_GEOM)

    def drop_duplicates_from_table(self, table_name) -> None:
        """Drops duplicate addresses without committing changes using internal row identified ctid."""
        # TODO: Floating point rounding errors in the coords may affect this.

        table_name = self._sanitize_string(table_name)

        q = f"DELETE FROM {table_name} "
        q += "WHERE ctid NOT IN ("
        q += "SELECT MIN(ctid) "
        q += "FROM properties "
        q += "GROUP BY state_id, county_id, tract_id, block_id, block_grp, geom);"

        self.connection.execute(s.text(q))
        self.connection.commit()

        return

    def _rename_column(self, table_name: str, old_name: str, new_name: str) -> None:
        self.connection.execute(
            s.text(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name};")
        )
        self.connection.commit()

        return

    def list_table_headers(self, table_name: str) -> list:
        result = self.connection.execute(
            s.text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
            )
        )
        return [row[0] for row in result]
