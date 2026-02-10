"""Pipeline orchestration: properties -> census merge -> PostGIS persistence.

Runs property generation and census merging in parallel:
- Thread 1 (Producer): Generates properties, saves to PostGIS in batches
- Thread 2 (Consumer): Polls for new geographies, fetches census data as they appear

This parallelism significantly reduces total runtime compared to sequential execution.

Usage:
    # Full parallel pipeline: add 300k properties + census merge
    python run_etl_pipeline.py --num-properties 300000 --granularity county

    # Census merge only (use existing properties)
    python run_etl_pipeline.py --granularity county

    # Properties only (skip census)
    python run_etl_pipeline.py --num-properties 50000 --skip-census

    # Sequential mode (disable parallelism)
    python run_etl_pipeline.py --num-properties 1000 --sequential

    # Custom poll settings
    python run_etl_pipeline.py --num-properties 1000 --poll-interval 10 --max-polls 50
"""

import argparse
import threading
import time
from queue import Queue
from sql_funcs import SQL
from load_properties import Properties
from load_census import CensusData
from settings import TABLE_NAME_CENSUS_PROPS, PROP_TABLE_NAME, PROP_TABLE_NAME_TEST


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

        if args.num_properties > 0:
            print(f"[PROPERTIES] Adding {args.num_properties} using geography-first approach...")
            props.add_random_properties_geo_first(
                args.num_properties,
                granularity=args.granularity,
                skip_final_cleanup=True,  # Don't block on slow cleanup
            )

        # Signal census thread IMMEDIATELY - properties are saved to DB
        properties_done.set()

        # Now do cleanup (census thread can finish in parallel)
        if args.num_properties > 0:
            print(f"[PROPERTIES] Running deduplication on ~{props.num_properties} rows...")
            cleanup_start = time.time()
            sql_obj.drop_duplicates_from_table(props.table_name)
            props.refresh()
            cleanup_time = time.time() - cleanup_start
            print(f"[PROPERTIES] Cleanup complete ({cleanup_time:.1f}s)")

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
        print(f"[CENSUS] Initialized (year={args.year}, granularity={args.granularity})")

        # Delete properties in known-failed geographies (e.g., VA independent cities)
        prop_table = PROP_TABLE_NAME_TEST if args.test else PROP_TABLE_NAME
        sql_obj.delete_properties_in_failed_geos(prop_table, args.granularity, args.year)

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
                print(f"[CENSUS] Merged {new_count} (+{new_count - processed_count} new)")
                processed_count = new_count

            # Check if producer is done
            if properties_done.is_set():
                # Do one final pass - always run merge to catch any new geographies
                print(f"[CENSUS] Final pass: {len(properties_gdf)} properties...")
                result = census.merge_census_info(properties_gdf)
                print(f"[CENSUS] Final merge complete: {len(result)} properties")
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
    """Run properties and census threads in parallel."""
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


def run_sequential(args):
    """Run properties then census sequentially (original behavior)."""
    print("\n[SEQUENTIAL MODE]")
    print("-" * 40)

    SQL.kill_idle(args.test)
    sql_obj = SQL(test=args.test)
    success = True

    try:
        # Step 1: Properties
        print("\n[1/2] PROPERTIES")
        print("-" * 40)
        props = Properties(sql_obj=sql_obj)

        # Filter existing properties to CONUS only
        sql_obj.filter_properties_to_conus(props.table_name)
        props.refresh()  # Reload after filtering

        if args.num_properties > 0:
            props.add_random_properties_geo_first(
                args.num_properties,
                granularity=args.granularity,
            )

        properties_gdf = props.get_properties_gpd()
        print(f"Total properties: {len(properties_gdf)}")

        # Step 2: Census merge
        if not args.skip_census:
            print("\n[2/2] CENSUS MERGE")
            print("-" * 40)

            # Delete properties in known-failed geographies
            sql_obj.delete_properties_in_failed_geos(
                props.table_name, args.granularity, args.year
            )
            props.refresh()  # Reload after deletion
            properties_gdf = props.get_properties_gpd()

            census = CensusData(
                sql_obj=sql_obj,
                year=args.year,
                granularity=args.granularity,
            )
            result = census.merge_census_info(properties_gdf)
            print(f"Merged {len(result)} properties with census data.")
            print(f"Results saved to '{TABLE_NAME_CENSUS_PROPS}' table.")
        else:
            print("\n[2/2] CENSUS MERGE - SKIPPED")

    except Exception as e:
        print(f"[ERROR] Sequential pipeline failed: {e}")
        success = False

    finally:
        sql_obj.disconnect_and_close()

    return success


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
        "--year",
        type=int,
        default=2023,
        help="ACS5 survey year (default: 2023)",
    )
    parser.add_argument(
        "--granularity",
        choices=["block_group", "tract", "county"],
        default="tract",
        help="Census geographic level (default: tract)",
    )
    parser.add_argument(
        "--skip-census",
        action="store_true",
        help="Only run property generation, skip census merge",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of parallel (original behavior)",
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
    args = parser.parse_args()

    print("=" * 60)
    print("PIPELINE START")
    print("=" * 60)

    # Kill idle connections before starting
    SQL.kill_idle(args.test)

    start_time = time.time()

    # Choose execution mode
    if args.skip_census or args.sequential or args.num_properties == 0:
        # Sequential for: skip-census, explicit sequential, or census-only runs
        success = run_sequential(args)
    else:
        # Parallel for full pipeline with new properties
        success = run_parallel(args)

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
