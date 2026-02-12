"""Pipeline orchestration: properties -> census merge -> PostGIS persistence.

Automatically selects optimal execution mode:
- Full pipeline: Parallel threads (Producer generates properties, Consumer fetches census)
- Census-only: Single-threaded merge for existing properties
- Properties-only: Single-threaded generation without census

Parallel mode significantly reduces runtime for full pipeline runs.

Usage:
    # Full parallel pipeline: add 300k properties + census merge
    python run_etl_pipeline.py --num-properties 300000 --granularity county

    # Load properties from GPS coordinates file
    python run_etl_pipeline.py --coords-file my_properties.csv --granularity county

    # Census merge only (use existing properties)
    python run_etl_pipeline.py --granularity county

    # Properties only (skip census)
    python run_etl_pipeline.py --num-properties 50000 --skip-census

    # Custom poll settings (full pipeline only)
    python run_etl_pipeline.py --num-properties 1000 --poll-interval 10 --max-polls 50
"""

import argparse
import geopandas as gpd
import threading
import time
import pandas as pd
from queue import Queue
from sql_funcs import SQL
from load_properties import Properties
from load_census import CensusData
import load_wildfires
import gis
from settings import TABLE_NAME_CENSUS_PROPS, PROP_TABLE_NAME, PROP_TABLE_NAME_TEST, PATH_DATA

MERGED_PARQUET_PATH = PATH_DATA / "merged_properties_census.parquet"
TARGETS_PARQUET_PATH = PATH_DATA / "targets_features.parquet"


def _save_merged_result(result, label=""):
    """Save merged properties+census GeoDataFrame to geoparquet."""
    if result is not None and len(result) > 0:
        MERGED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(MERGED_PARQUET_PATH)
        print(f"{label}Saved {len(result)} rows to {MERGED_PARQUET_PATH}")


def run_properties_thread(
    args,
    properties_done: threading.Event,
    properties_error: Queue,
):
    """Producer thread: generates properties and saves to PostGIS."""
    sql_obj = None
    try:
        sql_obj = SQL(test=args.test)
        props = Properties(sql_obj=sql_obj, verbose=False)

        # Filter existing properties to CONUS only
        sql_obj.filter_properties_to_conus(props.table_name)
        props.refresh()  # Reload after filtering

        print(f"[PROPERTIES] Starting with {props.num_properties} existing")

        added_properties = False
        if args.coords_file:
            # Load from CSV file
            print(f"[PROPERTIES] Loading coordinates from {args.coords_file}...")
            df = pd.read_csv(args.coords_file)
            coords = list(zip(df["latitude"], df["longitude"]))
            props.add_properties_from_coordinates(coords)
            added_properties = True
        elif args.num_properties > 0:
            print(
                f"[PROPERTIES] Adding {args.num_properties} using geography-first approach..."
            )
            props.add_random_properties_geo_first(
                args.num_properties,
                granularity=args.granularity,
                skip_final_cleanup=True,  # Don't block on slow cleanup
            )
            added_properties = True

        # Do cleanup FIRST (before signaling census thread)
        if added_properties and args.num_properties > 0:
            # Only need extra cleanup for geo_first (coords method does its own)
            print(f"[PROPERTIES] Running deduplication...")
            cleanup_start = time.time()
            sql_obj.drop_duplicates_from_table(props.table_name)
            props.refresh()
            cleanup_time = time.time() - cleanup_start
            print(f"[PROPERTIES] Cleanup complete ({cleanup_time:.1f}s)")

        # Signal census thread AFTER cleanup - properties are deduplicated
        properties_done.set()

        print(f"[PROPERTIES] Complete. Total: {props.num_properties}")

    except Exception as e:
        properties_error.put(str(e))
        print(f"[PROPERTIES] ERROR: {e}")
        properties_done.set()  # Still signal on error so census thread can exit
    finally:
        if sql_obj:
            sql_obj.disconnect_and_close()


def run_census_thread(
    args,
    properties_done: threading.Event,
    census_error: Queue,
    poll_interval: float = 30.0,
    max_polls: int = 100,
):
    """Consumer thread: polls for new geographies and merges census data."""
    sql_obj = None
    try:
        sql_obj = SQL(test=args.test)
        census = CensusData(
            sql_obj=sql_obj,
            year=args.year,
            granularity=args.granularity,
            verbose=False,
        )
        print(
            f"[CENSUS] Initialized (year={args.year}, granularity={args.granularity})"
        )

        # Delete properties in known-failed geographies (e.g., VA independent cities)
        prop_table = PROP_TABLE_NAME_TEST if args.test else PROP_TABLE_NAME
        sql_obj.delete_properties_in_failed_geos(
            prop_table, args.granularity, args.year
        )

        processed_count = 0
        iteration = 0

        while iteration < max_polls:
            iteration += 1

            # Load current properties from PostGIS
            props = Properties(sql_obj=sql_obj, verbose=False)
            properties_gdf = props.get_properties_gpd()

            if len(properties_gdf) == 0:
                if properties_done.is_set():
                    print("[CENSUS] No properties found and producer done. Exiting.")
                    break
                print(f"[CENSUS] Waiting for properties... (poll {iteration})")
                time.sleep(poll_interval)
                continue

            # Run census merge (handles missing geos internally)
            print(f"[CENSUS] Poll {iteration}: {len(properties_gdf)} properties")
            result = census.merge_census_info(properties_gdf)
            new_count = len(result)

            if new_count > processed_count:
                print(
                    f"[CENSUS] Merged {new_count} (+{new_count - processed_count} new)"
                )
                processed_count = new_count

            # Check if producer is done
            if properties_done.is_set():
                # Do one final pass - always run merge to catch any new geographies
                print(f"[CENSUS] Final pass: {len(properties_gdf)} properties...")
                result = census.merge_census_info(properties_gdf)
                print(f"[CENSUS] Final merge complete: {len(result)} properties")
                _save_merged_result(result, "[CENSUS] ")
                break

            # Wait before next poll
            time.sleep(poll_interval)
        else:
            print(f"[CENSUS] WARNING: Reached max polls ({max_polls}). Exiting.")

        print(f"[CENSUS] Complete. Results in '{TABLE_NAME_CENSUS_PROPS}' table.")

    except Exception as e:
        census_error.put(str(e))
        print(f"[CENSUS] ERROR: {e}")
    finally:
        if sql_obj:
            sql_obj.disconnect_and_close()


def run_parallel(args):
    """Run properties and census threads in parallel (or single-mode for special cases)."""
    has_new_properties = args.num_properties > 0 or args.coords_file

    # Special case: Census-only mode (no properties to add)
    if not has_new_properties and not args.skip_census:
        print("\n[CENSUS-ONLY MODE]")
        print("-" * 40)
        print("Enriching existing properties with census data...")

        SQL.kill_idle(args.test)
        sql_obj = SQL(test=args.test)

        try:
            props = Properties(sql_obj=sql_obj)
            properties_gdf = props.get_properties_gpd()

            if len(properties_gdf) == 0:
                print("No properties found in database. Exiting.")
                return True

            print(f"Found {len(properties_gdf)} existing properties")

            # Run census merge once
            census = CensusData(
                sql_obj=sql_obj,
                year=args.year,
                granularity=args.granularity,
            )
            result = census.merge_census_info(properties_gdf)
            print(f"Merged {len(result)} properties with census data.")
            _save_merged_result(result, "[CENSUS-ONLY] ")
            return True
        except Exception as e:
            print(f"[ERROR] Census-only mode failed: {e}")
            return False
        finally:
            sql_obj.disconnect_and_close()

    # Special case: Properties-only mode (skip census)
    if args.skip_census:
        print("\n[PROPERTIES-ONLY MODE]")
        print("-" * 40)
        print("Generating properties without census enrichment...")

        SQL.kill_idle(args.test)
        sql_obj = SQL(test=args.test)

        try:
            props = Properties(sql_obj=sql_obj)
            sql_obj.filter_properties_to_conus(props.table_name)
            props.refresh()

            print(f"Starting with {props.num_properties} existing properties")

            if args.coords_file:
                print(f"Loading coordinates from {args.coords_file}...")
                df = pd.read_csv(args.coords_file)
                coords = list(zip(df["latitude"], df["longitude"]))
                props.add_properties_from_coordinates(coords)
            elif args.num_properties > 0:
                props.add_random_properties_geo_first(
                    args.num_properties,
                    granularity=args.granularity,
                )

            print(f"Total properties: {props.num_properties}")
            return True
        except Exception as e:
            print(f"[ERROR] Properties-only mode failed: {e}")
            return False
        finally:
            sql_obj.disconnect_and_close()

    # Full parallel mode (properties + census)
    print("\n[PARALLEL MODE]")
    print("-" * 40)

    # Coordination events and thread-safe error queues
    properties_done = threading.Event()
    properties_error = Queue()
    census_error = Queue()

    # Start census thread first (it will wait for properties)
    census_thread = threading.Thread(
        target=run_census_thread,
        args=(args, properties_done, census_error, args.poll_interval, args.max_polls),
        name="CensusThread",
    )
    census_thread.start()

    # Small delay to let census thread initialize
    time.sleep(2)

    # Start properties thread
    properties_thread = threading.Thread(
        target=run_properties_thread,
        args=(args, properties_done, properties_error),
        name="PropertiesThread",
    )
    properties_thread.start()

    # Wait for both to complete
    properties_thread.join()
    print("[MAIN] Properties thread finished.")

    census_thread.join()
    print("[MAIN] Census thread finished.")

    # Report errors
    prop_err = properties_error.get() if not properties_error.empty() else None
    census_err = census_error.get() if not census_error.empty() else None

    if prop_err:
        print(f"[ERROR] Properties failed: {prop_err}")
    if census_err:
        print(f"[ERROR] Census failed: {census_err}")

    return not (prop_err or census_err)


def run_wildfires_and_proximity(sql_obj, combined_gdf, n_jobs=5):
    """Load wildfire data and compute proximity features for all properties."""
    print("\n[WILDFIRES + PROXIMITY]")
    print("-" * 40)

    wildfires = load_wildfires.WildfireData(sql_obj=sql_obj)
    print(f"Wildfire detections: {len(wildfires.data)}")

    print(f"Computing proximity features ({n_jobs} jobs)...")
    proximity_features = gis.calc_all_features_parallel(
        combined_gdf, wildfires.data, n_jobs=n_jobs
    )

    targets_features = pd.concat([combined_gdf, proximity_features], axis=1)

    TARGETS_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    targets_features.to_parquet(TARGETS_PARQUET_PATH)
    print(f"Saved {len(targets_features)} rows x {targets_features.shape[1]} cols to {TARGETS_PARQUET_PATH}")

    return targets_features


def main():
    parser = argparse.ArgumentParser(
        description="Run the full data pipeline: properties -> census merge"
    )
    parser.add_argument(
        "--num-properties",
        type=int,
        default=0,
        help="Number of new properties to add (0 = use existing only)",
    )
    parser.add_argument(
        "--coords-file",
        type=str,
        default=None,
        help="CSV file with latitude,longitude columns to load as properties",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ACS5 survey year (default: 2023)",
    )
    parser.add_argument(
        "--granularity",
        choices=["block_group", "tract", "county"],
        default="county",
        help="Census geographic level (default: county)",
    )
    parser.add_argument(
        "--skip-census",
        action="store_true",
        help="Only run property generation, skip census merge",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between census polling iterations (default: 30)",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=100,
        help="Maximum census polling iterations before timeout (default: 100)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test tables",
    )
    parser.add_argument(
        "--skip-wildfires",
        action="store_true",
        help="Skip wildfire loading and proximity feature computation",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=5,
        help="Number of parallel jobs for proximity computation (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPELINE START")
    print("=" * 60)

    # Kill idle connections before starting
    SQL.kill_idle(args.test)

    start_time = time.time()

    # run_parallel() handles all modes internally (parallel, census-only, properties-only)
    success = run_parallel(args)

    # Wildfire + proximity features step
    if success and not args.skip_wildfires and not args.skip_census:
        if MERGED_PARQUET_PATH.exists():
            combined_gdf = gpd.read_parquet(MERGED_PARQUET_PATH)
            sql_obj = SQL(test=args.test)
            try:
                run_wildfires_and_proximity(sql_obj, combined_gdf, n_jobs=args.n_jobs)
            except Exception as e:
                print(f"[ERROR] Wildfires/proximity failed: {e}")
                success = False
            finally:
                sql_obj.disconnect_and_close()
        else:
            print(f"[WARNING] {MERGED_PARQUET_PATH} not found, skipping wildfire step.")

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"

    print("\n" + "=" * 60)
    if success:
        print(f"PIPELINE COMPLETE ({elapsed_str})")
    else:
        print(f"PIPELINE FAILED ({elapsed_str})")
    print("=" * 60)


if __name__ == "__main__":
    main()
