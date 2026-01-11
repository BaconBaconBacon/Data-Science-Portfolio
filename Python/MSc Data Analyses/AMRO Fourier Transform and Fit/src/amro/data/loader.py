# TODO: Switch it over to using pathlib instead of os
# import os
import numpy as np
import pandas as pd
from ..config.settings import (
    RAW_DATA_PATH,
    HEADER_ANGLE_DEG,
    HEADER_ANGLE_RAD,
    HEADER_RES_OHM,
    LOADER_DESIRED_COLS,
    LOADER_COL_RENAME_DICT,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_ACT,
    HEADER_GEO,
    HEADER_LENGTH,
    HEADER_WIDTH,
    HEADER_HEIGHT,
    HEADER_0DEG,
    HEADER_MEAN,
    HEADER_RES_DEL_MEAN_OHM,
    HEADER_RES_DEF_MEAN_NORM,
    HEADER_RES_DEL_0DEG_NORM_PCT,
    HEADER_RES_DEL_0DEG_OHM,
    HEADER_RES_DEL_MEAN_NORM_PCT,
    KEY_RES_CONSTANTS,
    KEY_TEMP_LABELS,
    KEY_MAGNET_LABELS,
    HEADER_TEMP_RAW,
    HEADER_MAGNET_RAW_OE,
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

# TODO : Hacky solution since .csv files don't have the geometries. Will need to fix when you add the code to AS the AMRO
GEO_DICT = {"ACTRot11": "perp", "ACTRot12": "para"}


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

    TODO: Add the cleaning and symmetrization functionality into the ETL pipeline.
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_data = ProjectData(project_name=project_name)
        self.save_folder = RAW_DATA_PATH
        self.pickle_fp = self.project_data.pickle_fp

    def load_amro(self) -> ProjectData:
        """ """
        if self.pickle_fp.is_file():
            print("Loading : {}".format(self.project_name))
            self.project_data.load_project_from_pickle()
        else:
            print("Running AMRO ETL. Save name: {}".format(self.project_name))
            self._run_amro_etl()
        return self.project_data

    def get_amro_data(self):
        return self.project_data

    def _run_amro_etl(self) -> None:
        """

        Args:
            data_dir:

        Returns:

        """

        filenames = list(self.save_folder.glob("*.csv"))
        # TODO: add some flexibility to naming schemes. Maybe just check whether
        # filename has been loaded already, and whether the H and T data are new

        for filename in filenames:
            # Ensure we are selecting only AMRO data
            if self._is_valid_amro_filename(filename):
                # EXTRACT
                (
                    act_label,
                    T_label,
                    H_label,
                    geometry,
                    length,
                    height,
                    width,
                    angles,
                    resistivities,
                ) = self._extract_experiment_data(filename)

                # TRANSFORM
                exp_key = OscillationKey(
                    experiment_label=act_label,
                    temperature=T_label,
                    magnetic_field=H_label,
                )
                exp_data = ExperimentalData(
                    experiment_key=exp_key,
                    angles_degs=angles,
                    res_ohms=resistivities,
                )
                osc = AMROscillation(key=exp_key, osc_data=exp_data)
                # LOAD
                if act_label not in self.project_data.experiments_dict:
                    exp = Experiment(
                        experiment_label=act_label,
                        geometry=geometry,
                        length=length,
                        height=height,
                        width=width,
                    )
                    self.project_data.add_experiment(exp)
                else:
                    exp = self.project_data.get_experiment(act_label)
                exp.add_oscillation(osc)
            else:
                print("Invalid file name found:\t" + str(filename))
        return None

    def _is_valid_amro_filename(self, filename: Path) -> bool:
        # TODO: Needs to generalize this with the filename handling code
        valid_act_label = "ACTRot" in filename.name
        valid_amro_label = "AMRO" in filename.name
        return valid_amro_label and valid_act_label

    def _extract_experiment_data(self, filename: Path) -> tuple:
        """ """

        file_path = self.save_folder / filename
        experiment_df = pd.read_csv(file_path, sep=",")
        experiment_df = experiment_df.rename(columns=LOADER_COL_RENAME_DICT)
        (act_label, T_label, H_label, geometry, length, height, width) = (
            self._parse_experiment_metadata(experiment_df, filename)
        )

        angles = experiment_df[HEADER_ANGLE_DEG].values
        resistivities = experiment_df[HEADER_RES_OHM].values
        return (
            act_label,
            T_label,
            H_label,
            geometry,
            length,
            height,
            width,
            angles,
            resistivities,
        )

    def _parse_experiment_metadata(self, temp_df: pd.DataFrame, filename: str) -> tuple:
        for item in filename.name.split("_"):
            if "ACTRot" in item:
                act_label = item

        length = temp_df[HEADER_LENGTH].unique()[0]
        height = temp_df[HEADER_HEIGHT].unique()[0]
        width = temp_df[HEADER_WIDTH].unique()[0]

        # TODO: These will need to be adjusted when the AMRO cleaning code is added
        H_vals = c.convert_oe_to_teslas(temp_df[HEADER_MAGNET_RAW_OE]).round(1)
        H_label = H_vals.unique()[0]
        geometry = GEO_DICT[act_label]
        T_vals = temp_df[HEADER_TEMP_RAW].round(1)
        T_label = T_vals.unique()[0]

        tup = (act_label, T_label, H_label, geometry, length, height, width)
        return tup

    def _convert_degs_to_rads(
        self, degs: np.ndarray | pd.Series
    ) -> np.ndarray | pd.Series:

        return degs * 2 * np.pi / 360

    def _calculate_uohm_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if "Res" in col:
                new_col = col.replace("ohm", "uohm")
                df[new_col] = df[col] * 10**6

        return df

    def quick_plot_amro(self):
        return _quick_plot_amro(self)
