"""
Pre-processing step to take a data file from a QD USA PPMS ACT Option and prepare its contents
for the fitting pipeline. This allows for the code to be able to integrate data from other
potential AMRO measurement systems in the future.
"""

import argparse
from amro.data import AMROCleaner


def parse_args():
    """Parse command line arguments for the cleaner script.

    Returns:
        Namespace object containing datafile_type and verbose arguments.
    """
    parser = argparse.ArgumentParser(
        description="Clean and anti-symmetrize raw AMRO data from a QD USA PPMS ACT Option."
    )
    parser.add_argument(
        "--datafile-type",
        type=str,
        default=".dat",
        choices=[".dat", ".csv"],
        help="File extension of raw data files (default: .dat)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed processing info"
    )
    return parser.parse_args()


def main():
    """Run the AMRO data cleaning pipeline.

    Initializes the cleaner with command line arguments, processes all raw
    data files, and prints a summary of processed experiments.
    """
    args = parse_args()

    cleaner = AMROCleaner(datafile_type=args.datafile_type, verbose=args.verbose)

    print("Cleaning raw data files...")
    cleaner.clean_data_from_folder()

    exp_labels = cleaner.get_experiment_labels()
    print(f"\nProcessed {len(exp_labels)} experiments: {exp_labels}")
    print("Cleaned files saved to processed data folder.")


if __name__ == "__main__":
    main()
