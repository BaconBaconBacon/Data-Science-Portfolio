import numpy as np
import pandas as pd

from ..config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    HEADER_ANGLE_DEG,
    HEADER_ANGLE_RAD,
    HEADER_RES_OHM,
    LOADER_DESIRED_COLS,
    CLEANER_COL_RENAME_DICT,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_EXP_LABEL,
    HEADER_GEO,
    HEADER_WIRE_SEP,
    # HEADER_WIDTH,
    # HEADER_HEIGHT,
    HEADER_0DEG,
    HEADER_MEAN,
    HEADER_RES_DEL_MEAN_OHM,
    HEADER_RES_DEF_MEAN_NORM,
    HEADER_RES_DEL_0DEG_NORM_PCT,
    HEADER_RES_DEL_0DEG_OHM,
    HEADER_RES_DEL_MEAN_NORM_PCT,
    # KEY_RES_CONSTANTS,
    # KEY_TEMP_LABELS,
    # KEY_MAGNET_LABELS,
    HEADER_TEMP_RAW,
    HEADER_MAGNET_RAW_OE_ABS,
    CLEANER_SAVE_FN_SUFFIX,
    HEADER_EXPERIMENT_PREFIX,
    HEADER_CROSS_SECTION,
)
from ..plotting.loader import _quick_plot_amro
from ..utils import utils as u
from ..utils import conversions as c

from pathlib import Path
from .data_structures import (
    ProjectData,
    Experiment,
    AMROscillation,
    ExperimentalData,
    OscillationKey,
)


class AMROLoader:
    """
    Here we load the pre-cleaned and symmetrized data into a single DataFrame.
    We have already checked the data for NaNs, and handled them when they appeared.

    We extract experimental information about temperature ($T$) and magnetic field
    strength ($H$) from the filenames, but we must account for an inconsistent
    naming scheme.

    The 'geo' label indicates the experimental geometry that was used. In 'para',
    the rotation of the sample brings the electrical current vector parallel with the
    magnetic field at 90deg. For the 'perp' geometry, the current vector is held
    orthogonal to the magnetic field for the entire rotation of the sample.

    """

    def __init__(
        self,
        project_name: str,
        verbose: bool = False,
    ):
        """Initialize the AMROLoader.

        Args:
            project_name: Name identifier for the project, used for file naming.
            verbose: If True, print detailed processing information.
        """
        self.project_name = project_name
        self.project_data = ProjectData(project_name=project_name)

        self.pickle_fp = self.project_data.pickle_fp
        self.verbose = verbose
        self.project_data.check_for_saved_data()

    def load_amro(self) -> ProjectData:
        """Load AMRO data from pickle cache or run ETL pipeline.

        Checks for existing pickled data first. If found, loads from cache.
        Otherwise, runs the full ETL pipeline to load and process raw data.

        Returns:
            ProjectData object containing all loaded experiments and oscillations.
        """
        if self.pickle_fp.is_file():
            print("Loading : {}".format(self.project_name))
            self.project_data = ProjectData.load_project_from_pickle(self.pickle_fp)
        else:
            print("Running AMRO ETL.")
            self._run_amro_etl()

        return self.project_data

    def get_amro_data(self) -> ProjectData:
        """Return the loaded project data.

        Returns:
            ProjectData object containing all experiments and oscillations.
        """
        return self.project_data

    def _run_amro_etl(self) -> None:
        """Execute the ETL pipeline for AMRO data.

        Reads processed CSV files from PROCESSED_DATA_PATH, extracts experiment
        metadata and oscillation data, transforms into data structures, and loads
        into the project_data container. Saves the result to a pickle file.
        """
        filenames = list(PROCESSED_DATA_PATH.glob("*.csv"))

        valid_data_found = False
        for filename in filenames:
            # Ensure we are selecting only AMRO data
            if self._is_valid_amro_filename(filename):
                print(f"Reading {filename}")
                valid_data_found = True
                # Read experiment
                experiment_df = pd.read_csv(PROCESSED_DATA_PATH / filename, sep=",")
                (
                    exp_label,
                    osc_keys,
                    geometry,
                    wire_sep,
                    cross_section,
                ) = self._parse_experiment_metadata(experiment_df)
                if exp_label not in self.project_data.experiments_dict:
                    exp = Experiment(
                        experiment_label=exp_label,
                        geometry=geometry,
                        wire_sep=wire_sep,
                        cross_section=cross_section,
                    )
                    self.project_data.add_experiment(exp)
                else:
                    exp = self.project_data.get_experiment(exp_label)

                for osc_key in osc_keys:
                    T_label = osc_key.temperature
                    H_label = osc_key.magnetic_field
                    # Parse oscillation
                    osc = experiment_df.query(
                        f"{HEADER_MAGNET}=={H_label} & {HEADER_TEMP}=={T_label}"
                    )
                    # EXTRACT
                    angles = osc[HEADER_ANGLE_DEG].values
                    resistivities = osc[HEADER_RES_OHM].values

                    # TRANSFORM
                    exp_data = ExperimentalData(
                        experiment_key=osc_key,
                        angles_degs=angles,
                        res_ohms=resistivities,
                    )
                    osc = AMROscillation(key=osc_key, osc_data=exp_data)

                    # LOAD
                    exp.add_oscillation(osc)

        if not valid_data_found:
            print("Could not find valid data!")
        else:
            print("AMRO loading complete")
            self.project_data.save_project_to_pickle()
            print("Project saved as: {}".format(self.project_data.pickle_fp.name))
        return None

    def _is_valid_amro_filename(self, filename: Path) -> bool:
        """Check if a filename matches the expected AMRO data file naming pattern.

        Args:
            filename: Path object of the file to validate.

        Returns:
            True if the filename contains both the experiment prefix and cleaner suffix.
        """
        valid_act_label = HEADER_EXPERIMENT_PREFIX in filename.name
        valid_cleaned_label = CLEANER_SAVE_FN_SUFFIX in filename.name
        return valid_cleaned_label and valid_act_label

    def _parse_experiment_metadata(self, temp_df: pd.DataFrame) -> tuple:
        """Extract experiment metadata from a DataFrame.

        Args:
            temp_df: DataFrame containing experiment data with metadata columns.

        Returns:
            Tuple of (experiment_label, oscillation_keys, geometry, wire_sep, cross_section).
        """
        wire_sep = temp_df[HEADER_WIRE_SEP].unique()[0]
        cross_section = temp_df[HEADER_CROSS_SECTION].unique()[0]
        experiment_label = temp_df[HEADER_EXP_LABEL].unique()[0]
        geometry = temp_df[HEADER_GEO].unique()[0]

        temp_df = temp_df[[HEADER_TEMP, HEADER_MAGNET]].drop_duplicates()

        osc_keys = []
        for _, row in temp_df.iterrows():
            osc_key = OscillationKey(
                experiment_label=experiment_label,
                temperature=row[HEADER_TEMP],
                magnetic_field=row[HEADER_MAGNET],
            )
            osc_keys.append(osc_key)

        return experiment_label, osc_keys, geometry, wire_sep, cross_section

    def _convert_degs_to_rads(
        self, degs: np.ndarray | pd.Series
    ) -> np.ndarray | pd.Series:
        """Convert angle values from degrees to radians.

        Args:
            degs: Angle value(s) in degrees.

        Returns:
            Angle value(s) converted to radians.
        """
        return degs * 2 * np.pi / 360

    def _calculate_uohm_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add micro-ohm-cm versions of resistivity columns to the DataFrame.

        Args:
            df: DataFrame containing resistivity columns in ohm-cm.

        Returns:
            DataFrame with additional columns converted to micro-ohm-cm.
        """
        for col in df.columns:
            if "Res" in col:
                new_col = col.replace("ohm", "uohm")
                df[new_col] = df[col] * 10**6

        return df

    def quick_plot_amro(self) -> None:
        """Generate quick visualization plots of the loaded AMRO data."""
        return _quick_plot_amro(self)
