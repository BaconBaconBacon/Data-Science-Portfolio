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
from datetime import datetime
from pathlib import Path

from settings import (
    SQL_ENGINE_STR,
    PROP_LABELS_KEYS_MAP,
    GIS_DEFAULT_CRS,
    TABLE_NAME_CACHE,
    TABLE_NAME_CENSUS_PROPS,
    TABLE_NAME_CENSUS_PROPS_TEST,
    HEADER_GEOM,
    DATA_DIR,
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
        # Base connection string (without database name) for admin operations
        self._base_conn_str = self.engine_string.rsplit("/", 1)[0]

        self._connect_to_sql()

    def _create_database(self, database_name: str) -> bool:
        """Create the target PostgreSQL database if it doesn't exist.

        Handles corrupted databases where catalog entry exists but data directory
        is missing. Automatically drops and recreates in this case.

        Connects to the 'postgres' system database to execute CREATE DATABASE.
        Uses autocommit mode as required by PostgreSQL for CREATE DATABASE.

        Parameters
        ----------
        database_name : str
            Name of the database to create.

        Returns
        -------
        bool
            True if database created or already exists, False on failure.
        """
        # First, try to connect to the database to verify it's actually usable
        test_conn_str = f"{self._base_conn_str}/{database_name}"

        try:
            test_engine = s.create_engine(test_conn_str)
            test_conn = test_engine.connect()

            # Database exists and is usable - check if PostGIS is installed
            result = test_conn.execute(
                s.text("SELECT 1 FROM pg_extension WHERE extname = 'postgis'")
            )
            postgis_installed = result.fetchone() is not None
            test_conn.close()
            test_engine.dispose()

            # If PostGIS is not installed, install it now
            if not postgis_installed:
                print(f"PostGIS not found in '{database_name}', installing...")
                try:
                    db_engine = s.create_engine(test_conn_str)
                    with db_engine.connect() as conn:
                        conn.execute(s.text("CREATE EXTENSION IF NOT EXISTS postgis"))
                        conn.commit()
                    db_engine.dispose()
                    print(f"Installed PostGIS extension in '{database_name}'")
                except Exception as postgis_error:
                    print(
                        f"Error: Failed to install PostGIS extension: {postgis_error}"
                    )
                    return False

            return True
        except s.exc.OperationalError as e:
            error_str = str(e)

            # Check if this is a "missing directory" corruption error
            if "does not exist" in error_str and "subdirectory" in error_str:
                print(
                    f"Detected corrupted database '{database_name}' (missing data directory)"
                )
                print(f"Cleaning up and recreating...")

                # Drop the corrupted database entry
                try:
                    system_engine = s.create_engine(f"{self._base_conn_str}/postgres")
                    with system_engine.connect() as conn:
                        conn.connection.connection.set_isolation_level(0)  # Autocommit
                        # Force disconnect any stale connections
                        conn.execute(
                            s.text(
                                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                                f"WHERE datname = '{database_name}' AND pid <> pg_backend_pid()"
                            )
                        )
                        # Drop the corrupted database
                        conn.execute(s.text(f"DROP DATABASE IF EXISTS {database_name}"))
                    system_engine.dispose()
                    print(f"Dropped corrupted database entry for '{database_name}'")
                except Exception as drop_error:
                    print(f"Error dropping corrupted database: {drop_error}")
                    return False

            elif "does not exist" in error_str:
                # Database simply doesn't exist (normal case)
                pass
            else:
                # Some other connection error
                print(f"Error connecting to database '{database_name}': {e}")
                return False

        # At this point, database either doesn't exist or was just dropped
        # Create it fresh
        try:
            system_engine = s.create_engine(f"{self._base_conn_str}/postgres")
            # Use raw connection with autocommit (CREATE DATABASE cannot run in transaction)
            with system_engine.connect() as conn:
                conn.connection.connection.set_isolation_level(0)  # Autocommit mode
                conn.execute(s.text(f"CREATE DATABASE {database_name}"))
            system_engine.dispose()
            print(f"Created database '{database_name}'")
        except Exception as e:
            print(f"Error: Failed to create database '{database_name}': {e}")
            return False

        # Install PostGIS extension in the new database
        try:
            db_engine = s.create_engine(f"{self._base_conn_str}/{database_name}")
            with db_engine.connect() as conn:
                conn.execute(s.text("CREATE EXTENSION IF NOT EXISTS postgis"))
                conn.commit()
            db_engine.dispose()
            print(f"Installed PostGIS extension in '{database_name}'")
            return True
        except Exception as e:
            print(f"Error: Failed to install PostGIS extension: {e}")
            return False

    def _connect_to_sql(self):
        """Connect to PostgreSQL database, creating it if necessary.

        Extracts database name from connection string, ensures it exists,
        then establishes connection.

        Raises
        ------
        RuntimeError
            If database cannot be created or connection fails.
        """
        # Extract database name from connection string
        # Format: postgresql+psycopg2://user:password@host:port/database
        db_name = self.engine_string.split("/")[-1]

        # Ensure database exists (create if necessary)
        if not self._create_database(db_name):
            raise RuntimeError(
                f"Failed to create or access database '{db_name}'. "
                f"Check PostgreSQL permissions and connection settings."
            )

        # Establish connection to target database
        try:
            self.engine = s.create_engine(self.engine_string)
            self.connection = self.engine.connect()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database '{db_name}': {e}")

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
        col_defs = ", ".join(f'"{name}" {dtype}' for name, dtype in columns.items())
        q = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs});"
        self.connection.execute(s.text(q))
        self.connection.commit()

    def check_table_exists(self, table_name: str) -> bool:
        return s.inspect(self.engine).has_table(table_name)

    def drop_table(self, table_name: str, confirm: bool = False) -> None:
        if not confirm:
            raise RuntimeError(
                "Potentially accidental function call. Set confirm variable."
            )
        if self.check_table_exists(table_name=table_name):
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
        else:
            print(f"{table_name} does not exist, cannot drop it!")

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

    def execute_string(self, query: str) -> None:
        query = self._sanitize_string(query)
        self.connection.execute(s.text(query))

    def _sanitize_string(self, string: str) -> str:
        if "DROP TABLE" in string or "CREATE TABLE" in string:
            raise ValueError("Invalid SQL string")
        else:
            return string

    def create_census_cache_table(self) -> None:
        """Create a cache table for census API results to resume interrupted fetches."""
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

    def is_cached_failure(
        self, cols, row: pd.Series, granularity: str, year: int
    ) -> bool:
        """Check if geography is a known failure (no census data available)."""
        conditions = " AND ".join([f"{col} = :{col}" for col in cols])
        q = s.text(
            f"""
            SELECT 1 FROM {TABLE_NAME_CACHE}
            WHERE {conditions}
            AND granularity = :granularity
            AND year = :year
            AND variable = '_FAILED_'
            LIMIT 1
            """
        )
        params: dict[str, int | str] = {col: int(row[col]) for col in cols}
        params["granularity"] = granularity
        params["year"] = year
        result = self.connection.execute(q, params).fetchone()
        return result is not None

    def filter_properties_to_conus(self, table_name: str) -> int:
        """
        Delete properties outside contiguous US (AK, HI, PR, territories).

        Removes properties with state_id not in the 48 contiguous states + DC.
        CONUS state FIPS: 01-56 excluding 02 (AK) and 15 (HI).

        Parameters
        ----------
        table_name : str
            Name of the properties table to filter.

        Returns
        -------
        int
            Number of rows deleted.
        """
        table_name = self._sanitize_string(table_name)
        q = s.text(
            f"""
            DELETE FROM {table_name}
            WHERE state_id NOT BETWEEN 1 AND 56
               OR state_id IN (2, 15)
            """
        )
        result = self.connection.execute(q)
        self.connection.commit()
        deleted = result.rowcount
        if deleted > 0:
            print(f"Removed {deleted} properties outside contiguous US")
        return deleted

    def delete_properties_in_failed_geos(
        self, properties_table: str, granularity: str, year: int
    ) -> int:
        """
        Delete properties in geographies known to have no census data.

        Uses the census_cache table to find geographies marked as failed,
        then deletes matching properties.

        Parameters
        ----------
        properties_table : str
            Name of the properties table.
        granularity : str
            Geographic level ('county', 'tract', 'block_group').
        year : int
            Census year.

        Returns
        -------
        int
            Number of properties deleted.
        """
        properties_table = self._sanitize_string(properties_table)

        # Check if census_cache table exists
        if not self.check_table_exists(TABLE_NAME_CACHE):
            # Census cache doesn't exist yet - no failed geos to delete
            return 0

        # Build join conditions based on granularity
        if granularity == "county":
            join_cond = "p.state_id = c.state_id AND p.county_id = c.county_id"
        elif granularity == "tract":
            join_cond = "p.state_id = c.state_id AND p.county_id = c.county_id AND p.tract_id = c.tract_id"
        else:  # block_group
            join_cond = "p.state_id = c.state_id AND p.county_id = c.county_id AND p.tract_id = c.tract_id AND p.block_grp = c.block_grp"

        q = s.text(
            f"""
            DELETE FROM {properties_table} p
            USING {TABLE_NAME_CACHE} c
            WHERE {join_cond}
              AND c.granularity = :granularity
              AND c.year = :year
              AND c.variable = '_FAILED_'
            """
        )
        result = self.connection.execute(q, {"granularity": granularity, "year": year})
        self.connection.commit()
        deleted = result.rowcount
        if deleted > 0:
            print(f"Removed {deleted} properties in failed geographies")
        return deleted

    def save_df_to_sql(
        self, table_name: str, df: pd.DataFrame, chunksize: int = 10000
    ) -> None:
        """Save DataFrame to SQL table with chunked inserts to avoid memory issues."""
        df.to_sql(
            table_name,
            self.connection,
            if_exists="append",
            index=False,
            chunksize=chunksize,
        )
        self.connection.commit()

    def read_df_from_sql(self, query: str):
        query = self._sanitize_string(query)
        return pd.read_sql(query, self.engine)

    def save_gpd_to_sql(self, table_name: str, gdf: gpd.GeoDataFrame) -> None:
        table_name = self._sanitize_string(table_name)

        # Drop property_id if present - let PostgreSQL auto-generate it
        if "property_id" in gdf.columns:
            gdf = gdf.drop(columns=["property_id"])

        if gdf.empty:
            return

        # Build column list (excluding property_id which is auto-generated)
        cols = [c for c in gdf.columns if c != "property_id"]
        col_str = ", ".join(f'"{c}"' for c in cols)

        # Auto-create table if it doesn't exist
        if not self.check_table_exists(table_name):
            col_types = {}
            for col in cols:
                if col == HEADER_GEOM:
                    col_types[col] = f"geometry(Geometry, {GIS_DEFAULT_CRS})"
                else:
                    dtype = gdf[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        col_types[col] = "BIGINT"
                    elif pd.api.types.is_float_dtype(dtype):
                        col_types[col] = "DOUBLE PRECISION"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        col_types[col] = "TIMESTAMP"
                    else:
                        col_types[col] = "TEXT"
            self.create_table(table_name, col_types)

        # Insert row by row using raw SQL to ensure DEFAULT/trigger works
        for _, row in gdf.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if col == HEADER_GEOM:
                    # Convert geometry to WKT and wrap in ST_GeomFromText
                    values.append(f"ST_GeomFromText('{val.wkt}', {GIS_DEFAULT_CRS})")
                elif pd.isna(val):
                    values.append("NULL")
                elif isinstance(val, (int, float)):
                    values.append(str(val))
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped = val.replace("'", "''")
                    values.append(f"'{escaped}'")
                else:
                    # Datetime, Timestamp, and other types need quoting
                    escaped = str(val).replace("'", "''")
                    values.append(f"'{escaped}'")

            val_str = ", ".join(values)
            q = f"INSERT INTO {table_name} ({col_str}) VALUES ({val_str});"
            self.connection.execute(s.text(q))

        self.connection.commit()

    def read_gpd_from_sql(self, table_name) -> gpd.GeoDataFrame:
        table_name = self._sanitize_string(table_name)

        q = f"SELECT * FROM {table_name}"

        return gpd.read_postgis(q, con=self.engine, geom_col=HEADER_GEOM)

    def read_props_census(self) -> gpd.GeoDataFrame:
        """Read pre-merged properties+census data from PostGIS.

        Returns the props_census table containing properties enriched with
        census data from a previous pipeline run.

        Returns
        -------
        gpd.GeoDataFrame
            Properties with census features merged.

        Raises
        ------
        ValueError
            If the props_census table does not exist (pipeline not run yet).
        """
        table = (
            TABLE_NAME_CENSUS_PROPS_TEST if self.test_mode else TABLE_NAME_CENSUS_PROPS
        )
        if not self.check_table_exists(table):
            raise ValueError(f"Table '{table}' does not exist. Run the pipeline first.")
        return self.read_gpd_from_sql(table)

    def initialize_properties_table(self, prop_name: str):

        prop_name = self._sanitize_string(prop_name)

        print(f"Creating empty '{prop_name}' table.")

        # Create sequence for property_id
        seq_name = f"{prop_name}_property_id_seq"
        self.connection.execute(s.text(f"CREATE SEQUENCE IF NOT EXISTS {seq_name};"))

        # Create table with property_id that has a default from sequence
        q = f"CREATE TABLE {prop_name} ("
        q += f"property_id INTEGER DEFAULT nextval('{seq_name}') PRIMARY KEY,"
        for key in PROP_LABELS_KEYS_MAP.keys():
            col_type = "TEXT" if key == "geoid" else "BIGINT"
            q += f"{key} {col_type},"
        q += f"{HEADER_GEOM} geometry(Geometry, {GIS_DEFAULT_CRS}));"
        self.connection.execute(s.text(q))

        # Create trigger to replace NULL with sequence value
        # (to_postgis inserts NULL for columns not in the GeoDataFrame)
        trigger_func = f"""
        CREATE OR REPLACE FUNCTION {prop_name}_set_id()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.property_id IS NULL THEN
                NEW.property_id := nextval('{seq_name}');
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
        self.connection.execute(s.text(trigger_func))

        trigger = f"""
        CREATE TRIGGER {prop_name}_auto_id
        BEFORE INSERT ON {prop_name}
        FOR EACH ROW
        EXECUTE FUNCTION {prop_name}_set_id();
        """
        self.connection.execute(s.text(trigger))
        self.connection.commit()

        q = f"SELECT * FROM {prop_name}"
        return gpd.read_postgis(q, con=self.engine, geom_col=HEADER_GEOM)

    def drop_duplicates_from_table(self, table_name) -> int:
        """Remove duplicate properties with the same geoid AND coordinates.

        Uses ROW_NUMBER() window function for O(n log n) performance instead
        of the slower NOT IN pattern. Keeps the row with the lowest property_id
        for each unique (geoid, geometry) combination.

        Returns
        -------
        int
            Number of duplicate rows removed.
        """
        table_name = self._sanitize_string(table_name)

        # Count before
        count_before = self.connection.execute(
            s.text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar()

        # Use ROW_NUMBER() to efficiently identify duplicates
        # Partition by geoid + geometry text for exact coordinate matching
        # Still faster than NOT IN pattern - O(n log n) vs O(n²)
        q = f"""
        DELETE FROM {table_name}
        WHERE property_id IN (
            SELECT property_id FROM (
                SELECT property_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY geoid, ST_AsText({HEADER_GEOM})
                           ORDER BY property_id
                       ) as rn
                FROM {table_name}
            ) sub
            WHERE rn > 1
        );
        """

        self.connection.execute(s.text(q))
        self.connection.commit()

        # Count after and report
        count_after = self.connection.execute(
            s.text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar()
        removed = count_before - count_after
        if removed > 0:
            print(f"Removed {removed} duplicate rows from '{table_name}'.")
        return removed

    def _rename_column(self, table_name: str, old_name: str, new_name: str) -> None:
        self.connection.execute(
            s.text(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name};")
        )
        self.connection.commit()

    def list_table_headers(self, table_name: str) -> list:
        result = self.connection.execute(
            s.text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :table_name"
            ),
            {"table_name": table_name},
        )
        return [row[0] for row in result]

    def backup_table_to_parquet(
        self, table_name: str, backup_dir: Path = None
    ) -> Path | None:
        """Backup a table to a timestamped parquet file.

        Parameters
        ----------
        table_name : str
            Name of table to backup.
        backup_dir : Path, optional
            Directory for backups. Defaults to PATH_DATA/backups.

        Returns
        -------
        Path or None
            Path to backup file, or None if table is empty/doesn't exist.
        """
        if not self.check_table_exists(table_name):
            print(f"Table '{table_name}' does not exist, skipping backup.")
            return None

        backup_dir = backup_dir or (DATA_DIR / "backups")
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{table_name}_{timestamp}.parquet"

        try:
            gdf = self.read_gpd_from_sql(table_name)
            if gdf.empty:
                print(f"Table '{table_name}' is empty, skipping backup.")
                return None

            # Convert geometry to WKT for parquet storage
            gdf_copy = gdf.copy()
            gdf_copy[HEADER_GEOM] = gdf_copy[HEADER_GEOM].apply(
                lambda g: g.wkt if g else None
            )
            gdf_copy.to_parquet(backup_path)
            print(f"Backed up {len(gdf)} rows to {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Backup failed: {e}")
            return None

    def restore_table_from_parquet(
        self, backup_path: Path, table_name: str, crs: int = None
    ) -> int:
        """Restore a table from a parquet backup.

        Parameters
        ----------
        backup_path : Path
            Path to the parquet backup file.
        table_name : str
            Name of table to restore to (will append, not replace).
        crs : int, optional
            Coordinate reference system. Defaults to GIS_DEFAULT_CRS.

        Returns
        -------
        int
            Number of rows restored.
        """
        from shapely import wkt

        crs = crs or GIS_DEFAULT_CRS

        df = pd.read_parquet(backup_path)
        if df.empty:
            print("Backup file is empty.")
            return 0

        # Convert WKT back to geometry
        df[HEADER_GEOM] = df[HEADER_GEOM].apply(lambda w: wkt.loads(w) if w else None)
        gdf = gpd.GeoDataFrame(df, geometry=HEADER_GEOM, crs=f"EPSG:{crs}")

        self.save_gpd_to_sql(table_name, gdf)
        print(f"Restored {len(gdf)} rows to '{table_name}'")
        return len(gdf)
