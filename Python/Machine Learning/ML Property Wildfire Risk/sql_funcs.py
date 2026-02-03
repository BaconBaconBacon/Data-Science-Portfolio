"""PostgreSQL database interface for the wildfire risk ML pipeline.

Provides the SQL class for managing connections to a PostgreSQL/PostGIS database,
with methods for:
    - Table creation, modification, and deletion
    - Reading/writing pandas DataFrames and GeoPandas GeoDataFrames
    - Census data caching to reduce API calls
    - Connection lifecycle management and idle connection cleanup

Requires a running PostgreSQL instance with PostGIS extension enabled.
Connection string is configured in settings.py.
"""

import sqlalchemy as s

from settings import (
    SQL_ENGINE_STR,
    PROP_LABELS_KEYS_MAP,
    GIS_DEFAULT_CRS,
    TABLE_NAME_CACHE,
    HEADER_GEOM,
)
import pandas as pd
import geopandas as gpd


class SQL:
    """Database connection manager for PostgreSQL/PostGIS.

    Handles connection lifecycle, provides convenience methods for common
    SQL operations, and supports both regular DataFrames and spatial
    GeoDataFrames via PostGIS.

    Parameters
    ----------
    test : bool, default False
        If True, operates in test mode (uses test table names).

    Attributes
    ----------
    test_mode : bool
        Whether running in test mode.
    connection : sqlalchemy.Connection
        Active database connection.
    engine : sqlalchemy.Engine
        SQLAlchemy engine instance.

    Examples
    --------
    >>> sql = SQL()
    >>> df = sql.read_df_from_sql("SELECT * FROM properties LIMIT 10")
    >>> sql.disconnect_and_close()
    """

    def __init__(self, test=False) -> None:
        self.test_mode = test
        self.connection = None
        self.engine = None

        self.engine_string = SQL_ENGINE_STR

        self._connect_to_sql()

        return

    def _connect_to_sql(self):
        self.engine = s.create_engine(self.engine_string)
        self.connection = self.engine.connect()
        return

    def disconnect_and_close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    @classmethod
    def kill_idle(cls, test: bool = False) -> int:
        """Terminate all other PostgreSQL connections, then close our own.

        Kills idle, idle-in-transaction, and active connections from other
        sessions. Creates a temporary connection, cleans up, and disposes.
        Call before creating a long-lived SQL instance to clear stale locks.

        Returns the number of connections terminated.
        """
        tmp = cls(test=test)
        result = tmp.connection.execute(
            s.text(
                "SELECT pg_terminate_backend(pid) "
                "FROM pg_stat_activity "
                "WHERE pid <> pg_backend_pid()"
            )
        )
        count = result.rowcount
        print(f"kill_idle: terminated {count} connection(s)")
        tmp.connection.commit()
        tmp.disconnect_and_close()
        return count

    def _execute_string(self, string: str) -> s.engine.Result:
        """Execute string without committing."""
        string = self._sanitize_string(string)
        result = self.connection.execute(s.text(string))
        return result

    def create_table(self, table_name: str, columns: dict[str, str]) -> None:
        """Create a table with the given column name -> SQL type mapping.

        Parameters
        ----------
        table_name : str
            Name of the table to create.
        columns : dict[str, str]
            Mapping of column name to SQL type, e.g. {"id": "SERIAL PRIMARY KEY", "name": "TEXT"}.
        """
        if not columns:
            raise ValueError("columns dict must not be empty.")
        col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
        q = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs});"
        self.connection.execute(s.text(q))
        self.connection.commit()

    def check_table_exists(self, table_name: str) -> bool:
        return s.inspect(self.engine).has_table(table_name)

    def drop_table(self, table_name: str, confirm: bool = False) -> None:
        print(table_name)
        if not confirm:
            raise RuntimeError(
                "Potentially accidental function call. Set confirm variable."
            )
        try:
            q = s.text(f"DROP TABLE IF EXISTS {table_name};")
            self.connection.execute(q)
            self.connection.commit()
            if self.check_table_exists(table_name):
                print("Drop failed!")
            else:
                print("Drop success!")
        except Exception as e:
            raise RuntimeError(f"Failed to drop table: {table_name}': {e}")

        result = self.connection.execute(
            s.text(
                "SELECT EXISTS ("
                "SELECT FROM information_schema.tables "
                "WHERE table_name = :table_name"
                ");"
            ),
            {"table_name": table_name},
        ).scalar()
        if result:
            raise RuntimeError(f"Table '{table_name}' still exists after DROP.")

        return

    def get_table_names(self) -> list[str]:
        """Return a list of all table names in the public schema."""
        return s.inspect(self.engine).get_table_names()

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table from old_name to new_name."""
        q = f"ALTER TABLE {old_name} RENAME TO {new_name};"
        self.connection.execute(s.text(q))
        self.connection.commit()

    def add_rows_to_table(self, table_name: str, rows: list[dict]) -> None:
        """Insert rows into a table. Each dict maps column name -> value.

        Uses parameterized INSERT via SQLAlchemy to avoid injection.
        """
        if not rows:
            return
        columns = list(rows[0].keys())
        col_str = ", ".join(columns)
        param_str = ", ".join(f":{col}" for col in columns)
        q = s.text(f"INSERT INTO {table_name} ({col_str}) VALUES ({param_str})")
        self.connection.execute(q, rows)
        self.connection.commit()

    def remove_rows_from_table(self, table_name: str, conditions: dict) -> None:
        """Delete rows matching all key=value conditions.

        Parameters
        ----------
        table_name : str
            Target table.
        conditions : dict
            Column name -> value pairs joined with AND.
        """
        if not conditions:
            raise ValueError("conditions must not be empty (would delete all rows).")
        where_clause = " AND ".join(f"{col} = :{col}" for col in conditions)
        q = s.text(f"DELETE FROM {table_name} WHERE {where_clause}")
        self.connection.execute(q, conditions)
        self.connection.commit()

    def add_columns_to_table(self, table_name: str, columns: dict[str, str]) -> None:
        """Add columns to an existing table.

        Parameters
        ----------
        table_name : str
            Target table.
        columns : dict[str, str]
            Mapping of column name to SQL type, e.g. {"score": "DOUBLE PRECISION"}.
        """
        if not columns:
            raise ValueError("columns dict must not be empty.")
        for col_name, col_type in columns.items():
            q = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
            self.connection.execute(s.text(q))
        self.connection.commit()

    def remove_columns_from_table(self, table_name: str, columns: list[str]) -> None:
        """Drop columns from an existing table.

        Parameters
        ----------
        table_name : str
            Target table.
        columns : list[str]
            Column names to drop.
        """
        if not columns:
            raise ValueError("columns list must not be empty.")
        for col_name in columns:
            q = f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {col_name};"
            self.connection.execute(s.text(q))
        self.connection.commit()

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
        conditions = " AND ".join([f"{col} = :{col}" for col in cols])
        q = s.text(
            f"""
            SELECT variable, value FROM {TABLE_NAME_CACHE}
            WHERE {conditions}
            AND granularity = :granularity
            AND year = :year
            """
        )
        params: dict[str, int | str] = {col: int(row[col]) for col in cols}
        params["granularity"] = granularity
        params["year"] = year
        result = pd.read_sql(q, self.connection, params=params)
        if len(result) > 0:
            return pd.Series(result["value"].values, index=result["variable"].values)
        return None

    def save_df_to_sql(self, table_name: str, df: pd.DataFrame) -> None:
        df.to_sql(table_name, self.connection, if_exists="append", index=False)
        self.connection.commit()
        return

    def read_df_from_sql(self, query: str):
        query = self._sanitize_string(query)
        return pd.read_sql(query, self.engine)

    def save_gpd_to_sql(self, table_name: str, gdf: gpd.GeoDataFrame) -> None:
        table_name = self._sanitize_string(table_name)
        gdf.to_postgis(table_name, self.connection, if_exists="append", index=False)
        self.connection.commit()
        return

    def read_gpd_from_sql(self, table_name) -> gpd.GeoDataFrame:
        table_name = self._sanitize_string(table_name)

        q = f"SELECT * FROM {table_name}"

        return gpd.read_postgis(q, con=self.engine, geom_col=HEADER_GEOM)

    def initialize_properties_table(self, prop_name: str):

        prop_name = self._sanitize_string(prop_name)

        print(f"Creating empty '{prop_name}' table.")
        q = "CREATE TABLE {} (".format(prop_name)
        for key in PROP_LABELS_KEYS_MAP.keys():
            col_type = "TEXT" if key == "geoid" else "BIGINT"
            q += f"{key} {col_type},"
        q += f"{HEADER_GEOM} geometry(Geometry, {GIS_DEFAULT_CRS}));"  # .format(GIS_DEFAULT_CRS)
        self.connection.execute(s.text(q))
        self.connection.commit()

        q = f"SELECT * FROM {prop_name}"
        return gpd.read_postgis(q, con=self.engine, geom_col=HEADER_GEOM)

    def drop_duplicates_from_table(self, table_name) -> None:
        """Drops duplicate addresses without committing changes using internal row identified ctid."""
        # TODO: Floating point rounding errors in the coords may affect this.

        table_name = self._sanitize_string(table_name)

        q = f"DELETE FROM {table_name} "
        q += "WHERE ctid NOT IN ("
        q += "SELECT MIN(ctid) "
        q += f"FROM {table_name} "
        q += f"GROUP BY state_id, county_id, tract_id, block_id, block_grp, {HEADER_GEOM});"

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
                "SELECT column_name FROM information_schema.columns WHERE table_name = :table_name"
            ),
            {"table_name": table_name},
        )
        return [row[0] for row in result]
