"""
Preprocessing code to clean, separate, and anti-symmetrize AMR oscillations measured using Quantum
Design USA's PPMS ACT Option.

Should be used before AMROLoader to prep the AMRO well before it's read into project_data

Can read in data as a .csv or .dat, but expects it to be formatted following
the QD USA PPMS ACT Option's data formatting:
    - Channel 2 resistivity was used for measurements,
    - Column 2, row 12 has the material name and experiment label : "Name"-"Label"
    - Column 2, row 13 has information about the AMRO geometry (parallel vs. perpendicular)
    - Column 2, rows 14 and 15 have accurate lead separation and cross-section data, respectively.
    - Assumes minimum temperature of 1.5K
    - Assumes max absolute magnetic field of 90000 Oe (9 T)

Experimental Assumptions:
    - Assumes each oscillation in a given file has a unique |H| and T.
    - Each oscillation has one full rotation at +H, and another at -H
    - Each oscillation has, for a given sample position, one measurement at +H and one at -H
    - The experiment naming scheme has the prefix defined by HEADER_EXPERIMENT_PREFIX
    - The PPMS file naming scheme uses underscores '_' to separate information
    - The step resolution of the magnetic field values matches  RAW_DATA_OE_MIN_RESOLUTION
"""

from io import TextIOWrapper

from .data_structures import OscillationKey
from ..config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    CLEANER_HEADER_LENGTH,
    CLEANER_COL_RENAME_DICT,
    HEADER_MAGNET_RAW_OE,
    HEADER_TEMP_RAW,
    HEADER_MAGNET_RAW_OE_ABS,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_EXPERIMENT_PREFIX,
    HEADER_ANGLE_DEG,
    CLEANER_OPTION_LABEL,
    CLEANER_OPTION_COORD,
    CLEANER_GEOM_COORD,
    CLEANER_LABEL_COORD,
    CLEANER_CROSS_SEC_COORD,
    CLEANER_WIRE_SEP_COORD,
    HEADER_GEO,
    HEADER_EXP_LABEL,
    HEADER_CROSS_SECTION,
    CLEANER_T_MIN_RESOLUTION,
    HEADER_RES_OHM,
    CLEANER_DROP_COLS,
    HEADER_WIRE_SEP,
    CLEANER_SAVE_FN_SUFFIX,
    CLEANER_ANG_CHANGE_THRESH,
    CLEANER_TEMP_STABLE_THRESH,
    CLEANER_MAG_FIELD_STABLE_THRESH,
    CLEANER_OUTLIER_RES_STD,
)
import pandas as pd
from pathlib import Path
import numpy as np
from ..utils import conversions as c
from warnings import warn

# Suppresses annoying warning when np.sign() is called
pd.options.mode.chained_assignment = None


class AMROCleaner:
    """Cleans and preprocesses raw AMRO data from QD USA PPMS ACT Option files."""

    def __init__(
        self, datafile_type: str = ".dat", verbose: bool = False  # project_name: str,
    ):
        """Initialize the AMROCleaner.

        Args:
            datafile_type: File extension of raw data files ('.dat' or '.csv').
            verbose: If True, print detailed processing information.
        """
        self.load_path = RAW_DATA_PATH
        self.save_path = PROCESSED_DATA_PATH
        # self.project_name = project_name
        self.verbose = verbose
        self.datafile_type = datafile_type
        self.experiment_labels = []
        return

    def get_experiment_labels(self) -> list[str]:
        """Return list of experiment labels that were processed.

        Returns:
            List of experiment label strings.
        """
        return self.experiment_labels

    def clean_data_from_folder(self) -> None:
        """Process all raw data files in the RAW_DATA_PATH folder.

        Reads each valid AMRO data file, extracts metadata from headers,
        filters for oscillation data, removes outliers, anti-symmetrizes
        measurements, and saves cleaned data to PROCESSED_DATA_PATH.
        """
        # Checks RAW_DATA_PATH for .csv and .dat files
        filepaths = list(self.load_path.glob("*" + self.datafile_type))
        for filepath in filepaths:
            if HEADER_EXPERIMENT_PREFIX in filepath.name:
                osc_count = 0

                print(f"Reading {filepath.name}")
                exp_label_fn = self._get_experiment_label_from_fn(filepath.name)

                # For each file, reads and parses the header info
                fp = self.load_path / filepath
                with open(fp) as file:
                    header = self._extract_header(file)
                exp_label_head, geom, wire_sep, cross_section = (
                    self._parse_and_verify_header(header)
                )
                exp_label = self._compare_labels(exp_label_fn, exp_label_head)
                self.experiment_labels.append(exp_label)

                # then reads the data into one large df
                data = self._load_file(fp)
                data = self._get_columns_for_calcs(data)
                data = self._filter_for_oscillation_data(data)
                data = self._clean_outliers(data)

                # Identifies the unique H and T pairings
                osc_labels = self._generate_oscillation_keys(data, exp_label)

                # for each unique H and T pairing, it anti-symmetrizes
                cleaned_oscs = []
                for osc_key in osc_labels:
                    q = f"{HEADER_MAGNET}=={osc_key.magnetic_field} & {HEADER_TEMP}=={osc_key.temperature}"
                    subset_df = data.query(q)
                    if subset_df.shape[0] > 1:
                        if self.verbose:
                            print(f"Reading in {osc_key}...")
                        clean_osc = self._anti_symmetrize_oscillation(subset_df)
                        if clean_osc is not None:
                            cleaned_oscs.append(clean_osc)
                            osc_count += 1
                        else:
                            if self.verbose:
                                print(f"Could not clean {osc_key}, skipping...")
                            continue
                    else:
                        print(f"Subset too small: {osc_key}")
                        continue
                if len(cleaned_oscs) == 0:
                    print("Could not find any oscillations!")
                    return
                else:
                    cleaned_df = pd.concat(cleaned_oscs)

                    cleaned_df[HEADER_EXP_LABEL] = exp_label
                    cleaned_df[HEADER_GEO] = geom
                    cleaned_df[HEADER_CROSS_SECTION] = cross_section
                    cleaned_df[HEADER_WIRE_SEP] = wire_sep

                    cleaned_df = cleaned_df.drop(
                        columns=[HEADER_MAGNET_RAW_OE, HEADER_MAGNET_RAW_OE_ABS]
                    )
                    fn = exp_label + CLEANER_SAVE_FN_SUFFIX
                    cleaned_df.to_csv(PROCESSED_DATA_PATH / fn, sep=",", index=False)
                    print(f"Found {osc_count} oscillations. Saved as {fn}")

            else:
                print(
                    f"HEADER_EXPERIMENT_PREFIX not found in filename, skipping: {filepath.name}"
                )

        return

    def _clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove resistivity outliers from the data.

        Identifies and removes data points whose resistivity falls outside
        a specified number of standard deviations from the group mean.

        Args:
            df: DataFrame containing resistivity measurements.

        Returns:
            DataFrame with outliers removed.
        """
        grouped = df.groupby([HEADER_TEMP, HEADER_MAGNET])[HEADER_RES_OHM]
        group_mean = grouped.transform("mean")
        group_std = grouped.transform("std")

        upper_bounds = group_mean + CLEANER_OUTLIER_RES_STD * group_std
        lower_bounds = group_mean - CLEANER_OUTLIER_RES_STD * group_std
        mask = (df[HEADER_RES_OHM] >= lower_bounds) & (
            df[HEADER_RES_OHM] <= upper_bounds
        )

        num_removed = (~mask).sum()
        if num_removed > 0 and self.verbose:
            print(f"Removed {num_removed} resistivity outliers.")

        return df[mask].copy()

    def _extract_header(self, file: TextIOWrapper) -> list[list]:
        """Extract header lines from a raw data file.

        Args:
            file: Open file handle to read header from.

        Returns:
            List of lists, where each inner list contains comma-separated header values.
        """
        header = []
        for _ in range(CLEANER_HEADER_LENGTH):
            line = next(file)
            line = line.rstrip("\n")
            line = line.split(",")
            header.append(line)
        return header

    def _parse_and_verify_header(self, header: list) -> tuple:
        """Parse and validate header information from a raw data file.

        Verifies the file is from a QD USA AC Transport system and extracts
        experiment metadata including label, geometry, wire separation, and cross-section.

        Args:
            header: List of header lines from _extract_header().

        Returns:
            Tuple of (experiment_label, geometry, wire_sep, cross_section).

        Raises:
            FileNotFoundError: If the file is not a valid ACT Option data file.
        """
        # Verifies the header is for a QD USA AC Transport data file
        if (
            self._get_header_element(header, CLEANER_OPTION_COORD)
            != CLEANER_OPTION_LABEL
        ):
            raise FileNotFoundError(
                "Loaded file is not a datafile of the QD USA ACT Option!"
            )
        geom = self._get_header_element(header, CLEANER_GEOM_COORD)
        exp_label = self._get_header_element(header, CLEANER_LABEL_COORD)
        if HEADER_EXPERIMENT_PREFIX not in exp_label:
            exp_label = None
        else:
            for item in exp_label.split(" "):
                if HEADER_EXPERIMENT_PREFIX in item:
                    exp_label = item

        cross_section = float(self._get_header_element(header, CLEANER_CROSS_SEC_COORD))
        wire_sep = float(self._get_header_element(header, CLEANER_WIRE_SEP_COORD))

        # Raises a warning if the cross-section and length are both the default of 1
        if wire_sep == 1 or cross_section == 1:
            warn(
                f"Default value(s) detected for {exp_label}. Wire separation = {wire_sep}; cross section = {cross_section}. Note that this package expects measurements of resistivity, not resistance."
            )
        if not ("para" in geom.lower() or "perp" in geom.lower()):
            warn(f"Unexpected geometry found in header: {geom}")
        return exp_label, geom, wire_sep, cross_section

    def _get_header_element(self, header: list[list], coord: tuple[int, int]) -> str:
        """Extract a specific element from the header by row and column index.

        Args:
            header: List of header lines.
            coord: Tuple of (row, column) indices (zero-indexed).

        Returns:
            String value at the specified header location.
        """
        row = coord[0]
        col = coord[1]
        element = header[row][col]
        return element

    def _generate_oscillation_keys(
        self, data: pd.DataFrame, exp_label: str
    ) -> list[OscillationKey]:
        """Generate unique OscillationKey objects for each T/H combination in the data.

        Args:
            data: DataFrame containing temperature and magnetic field columns.
            exp_label: Experiment label string.

        Returns:
            List of OscillationKey objects identifying each unique oscillation.
        """
        # The data structures will not work if two oscillations have the same keys
        df = data[[HEADER_TEMP, HEADER_MAGNET]].drop_duplicates()
        osc_labels = []
        for _, row in df.iterrows():
            osc_labels.append(
                OscillationKey(
                    experiment_label=exp_label,
                    temperature=row[HEADER_TEMP],
                    magnetic_field=row[HEADER_MAGNET],
                )
            )

        return osc_labels

    def _anti_symmetrize_oscillation(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Anti-symmetrize an oscillation by averaging +H and -H measurements.

        For each sample angle, averages the resistivity measured at positive
        and negative magnetic field to remove Hall Effect contributions.

        Args:
            raw_df: DataFrame containing raw oscillation data with +/- field measurements.

        Returns:
            DataFrame with anti-symmetrized resistivity values, or None if verification fails.
        """
        # For each angle, verify there are two resistivities and the +/- mag field values match
        grouped_df = raw_df.groupby(
            [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
        )

        # There should be only 2 measurements per angle, per T, per H
        counted_df = grouped_df.count()
        avg_count = np.mean(counted_df[HEADER_RES_OHM].values)
        if avg_count < 2:
            raw_df = self._handle_missing_measurements(raw_df, counted_df)

        elif avg_count > 2:
            raw_df = self._handle_extra_measurements(raw_df, counted_df)
            # Re-count after handling extras
            grouped_df = raw_df.groupby(
                [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
            )
            counted_df = grouped_df.count()
            avg_count = np.mean(counted_df[HEADER_RES_OHM].values)
            if avg_count < 2:
                raw_df = self._handle_missing_measurements(raw_df, counted_df)

        grouped_df = raw_df.groupby(
            [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
        )
        averaged_df = grouped_df.mean()

        averaged_df = self._verify_averaged_df(averaged_df, raw_df)

        return averaged_df

    def _handle_missing_measurements(
        self, df: pd.DataFrame, counted_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove angles that have fewer than 2 measurements per angle.

        Args:
            df: DataFrame containing oscillation measurements.
            counted_df: DataFrame with count of measurements per angle.

        Returns:
            DataFrame with incomplete measurement angles removed.
        """
        if self.verbose:
            print("Handling missing measurements...")

        missing_angles = counted_df[counted_df[HEADER_RES_OHM] < 2][
            HEADER_ANGLE_DEG
        ].values

        if self.verbose:
            print(f"Angles with missing measurements: {missing_angles}")

        # Remove those rows
        df = df[~df[HEADER_ANGLE_DEG].isin(missing_angles)]
        return df

    def _handle_extra_measurements(
        self, df: pd.DataFrame, counted_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove duplicate measurements at angles with more than 2 data points.

        Keeps only the first measurement for each unique combination of
        temperature, magnetic field, angle, and field polarity.

        Args:
            df: DataFrame containing oscillation measurements.
            counted_df: DataFrame with count of measurements per angle.

        Returns:
            DataFrame with extra measurements removed.
        """
        if self.verbose:
            print("Handling extra measurements...")

        # ID angles with extra measurements
        extra_angles = counted_df[counted_df[HEADER_RES_OHM] > 2][
            HEADER_ANGLE_DEG
        ].values

        if self.verbose:
            print(f"Angles with extra measurements: {extra_angles}")
        # Could implement more adaptive code, but for now the user should be inputting
        # better data
        df["Field Polarity"] = np.sign(df[HEADER_MAGNET_RAW_OE].values)

        df = df.drop_duplicates(
            subset=[HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG, "Field Polarity"],
            keep="first",
        )
        df = df.drop(columns=["Field Polarity"])
        return df

    def _load_file(self, fp: Path) -> pd.DataFrame:
        """Load raw data file into a DataFrame, skipping header rows.

        Args:
            fp: Path to the raw data file.

        Returns:
            DataFrame containing the raw measurement data.
        """
        data = pd.read_table(fp, skiprows=CLEANER_HEADER_LENGTH, delimiter=",")
        return data

    def _get_columns_for_calcs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame columns for anti-symmetrization calculations.

        Renames columns, drops unused columns, and creates derived columns
        for temperature and magnetic field values.

        Args:
            df: Raw DataFrame from _load_file().

        Returns:
            DataFrame with standardized column names and derived columns.
        """
        df = df.rename(columns=CLEANER_COL_RENAME_DICT)
        df = df.drop(columns=CLEANER_DROP_COLS)

        df[HEADER_TEMP] = df[HEADER_TEMP_RAW].round(1)

        df[HEADER_MAGNET_RAW_OE_ABS] = df[HEADER_MAGNET_RAW_OE].abs()
        df[HEADER_MAGNET] = c.convert_oe_to_teslas(df[HEADER_MAGNET_RAW_OE_ABS]).round(
            CLEANER_T_MIN_RESOLUTION
        )
        return df

    def _verify_averaged_df(
        self, averaged_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """Verify that anti-symmetrization was performed correctly.

        Checks that row counts, mean values, and angle ranges match expected
        values after averaging.

        Args:
            averaged_df: DataFrame after anti-symmetrization.
            raw_df: Original DataFrame before anti-symmetrization.

        Returns:
            The averaged_df if verification passes, None otherwise.
        """
        try:
            # averaged_df has half the number of rows of raw_df
            assert averaged_df.shape[0] * 2 == raw_df.shape[0]

            # average H matches up to 0.1 Oe
            assert np.round(averaged_df[HEADER_MAGNET_RAW_OE].mean(), 1) == (
                np.round(raw_df[HEADER_MAGNET_RAW_OE].mean(), 1)
            )

            assert averaged_df[HEADER_TEMP].mean() == raw_df[HEADER_TEMP].mean()
            assert averaged_df[HEADER_MAGNET].mean() == raw_df[HEADER_MAGNET].mean()

            # Min and max angles match
            assert averaged_df[HEADER_ANGLE_DEG].max() == raw_df[HEADER_ANGLE_DEG].max()
            assert averaged_df[HEADER_ANGLE_DEG].min() == raw_df[HEADER_ANGLE_DEG].min()

            # Mean resistivities match up to nohm-cm
            assert np.round(averaged_df[HEADER_RES_OHM].mean(), 9) == np.round(
                raw_df[HEADER_RES_OHM].mean(), 9
            )
        except AssertionError:
            averaged_df = None
        return averaged_df

    def _get_experiment_label_from_fn(self, filename) -> None | str:
        """Extract experiment label from a filename.

        Args:
            filename: Filename string to parse.

        Returns:
            Experiment label string if found, None otherwise.
        """
        items = filename.split("_")
        label = None
        for item in items:
            if HEADER_EXPERIMENT_PREFIX in item:
                label = item
        return label

    def _compare_labels(self, label_fn: str | None, label_head: str | None) -> str:
        """Compare and reconcile experiment labels from filename and header.

        Args:
            label_fn: Label extracted from filename.
            label_head: Label extracted from file header.

        Returns:
            The reconciled experiment label string.

        Raises:
            ValueError: If both labels are None.
        """
        if label_fn is None and label_head is None:
            raise ValueError("Could not extract experiment name!")
        elif label_fn is None:
            return label_head
        elif label_head is None:
            return label_fn
        elif label_fn == label_head:
            return label_fn
        else:
            print(
                "Experiment label from filename does not match that in the raw data file."
            )
            print(f"Defaulting to the one extracted from the filename: {label_fn}")
            return label_fn

    def _filter_for_oscillation_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Removes measurements which may be present in the file that are not AMR oscillations.

        Identifies and removes constant angle sweeps of T and H.
        Keeps all partial oscillations, which will be filtered out just before anti-symmetrization.

        Assumes angle step size is constant for a given oscillation.
        """
        df = raw_df.copy()
        grouped_df = df.groupby([HEADER_TEMP, HEADER_MAGNET])

        df["angle_diff"] = grouped_df[HEADER_ANGLE_DEG].diff().abs()
        df["temp_diff"] = grouped_df[HEADER_TEMP].diff().abs()
        df["field_diff"] = grouped_df[HEADER_MAGNET].diff().abs()

        is_oscillation = (
            ((df["angle_diff"] > CLEANER_ANG_CHANGE_THRESH) | (df["angle_diff"].isna()))
            & (
                (df["temp_diff"] < CLEANER_TEMP_STABLE_THRESH)
                | (df["temp_diff"].isna())
            )
            & (
                (df["field_diff"] < CLEANER_MAG_FIELD_STABLE_THRESH)
                | (df["field_diff"].isna())
            )
        )

        oscillation_df = df[is_oscillation].copy()
        oscillation_df = oscillation_df.drop(
            columns=["angle_diff", "temp_diff", "field_diff"]
        )

        if self.verbose:
            n_removed = len(df) - len(oscillation_df)
            print(f"Filtered out {n_removed} constant-angle sweep rows")
        return oscillation_df
