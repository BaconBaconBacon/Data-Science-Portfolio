# TODO: Switch it over to using pathlib instead of os
import os
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
)
from ..plotting.loader import _quick_plot_amro
from ..utils import utils as u
from pathlib import Path


class AMROLoader:
    """
    Here we load the pre-cleaned and symmetrized data into a single DataFrame.
    We have already checked the data for NaNs, and handled them when they appeared.

    As this is only a demonstration, we are only using a subset of the total
    available data in order to simplify the output.

    Nonetheless, the code is capable of handling all of the experimental data.

    We extract experimental information about temperature ($T$) and magnetic field
    strength ($H$) from the filenames, but we must account for an inconsistent
    naming scheme.

    The 'geo' label indicates the experimental geometry that was used. In 'para',
    the rotation of the sample brings the electrical current vector parallel with the
    magnetic field at 90deg. For the 'perp' geometry, the current vector is held
    orthogonal to the magnetic field for the entire rotation of the sample.

    TODO: Add the cleaning and symmetrization functionality into the ETL pipeline.
    """

    # TODO: Need to address this nested dictionary data storage
    META_DATA = {
        "ACTRot11": {HEADER_GEO: "perp", "ACT": 11},
        "ACTRot12": {HEADER_GEO: "para", "ACT": 12},
    }

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.save_folder = Path(RAW_DATA_PATH)
        self.file_path = os.path.join(self.save_folder, self.file_name)
        self.AMRO = pd.DataFrame()
        self.experiment_count = 0

    def get_amro(self) -> pd.DataFrame:
        """ """
        if self.file_name.endswith(".csv"):
            if os.path.exists(self.file_path):
                print("Loading : {}".format(self.file_name))
                self.AMRO = pd.read_csv(self.file_path)
            else:
                print("Running AMRO ETL. Save name: {}".format(self.file_name))
                self.AMRO = self._run_amro_etl(self.save_folder)
        else:
            raise TypeError("Wrong file type: {}".format(self.file_name))

        self.experiment_count = (
            self.AMRO[[HEADER_ACT, HEADER_TEMP, HEADER_MAGNET]]
            .drop_duplicates()
            .shape[0]
        )
        if self.experiment_count == 0:
            raise Exception("No AMRO files found in {}".format(self.save_folder))
        else:
            print(f"{self.experiment_count} unique experiments were found and loaded.")
            return self.AMRO

    def _run_amro_etl(self, data_dir: str) -> pd.DataFrame:
        """

        Args:
            data_dir:

        Returns:

        """

        filenames = os.listdir(data_dir)
        amro_df = pd.DataFrame()
        # TODO: add some flexibility to naming schemes. Maybe just check whether filename has been loaded already,
        # and whether the H and T data are new
        for filename in filenames:
            # Ensure we are selecting only AMRO data
            if self._is_valid_amro_filename(filename):
                # EXTRACT
                fp = os.path.join(data_dir, filename)
                data_df = self._extract_experiment_data(fp)

                # TRANSFORM
                data_df = self._transform_experiment_data(data_df)

                # LOAD
                amro_df = self._load_to_csv(data_df, amro_df)
            else:
                print("Invalid file name found:\t" + filename)
        return amro_df

    def _is_valid_amro_filename(self, filename: str) -> bool:

        valid_act_label = any(key in filename for key in self.META_DATA.keys())
        valid_amro_label = "AMRO" in filename
        return valid_amro_label and valid_act_label

    def _extract_experiment_data(self, file_path: str) -> pd.DataFrame:
        """ """
        experiment_df = pd.read_csv(file_path, sep=",")

        # Extract experiment's T and H info from file_path
        exp_labels = self._parse_experiment_name(file_path)

        experiment_df = self._parse_experiment_labels(exp_labels, experiment_df)

        # Select desired columns, rename as needed
        experiment_df = experiment_df.rename(columns=LOADER_COL_RENAME_DICT)[
            LOADER_DESIRED_COLS
        ]
        return experiment_df

    def _parse_experiment_name(self, filepath):
        fn = filepath.split(os.path.sep)[-1]
        # TODO: Generalize replace with a regex editor
        temp_name = fn.replace(".csv", "").replace("0_5", "0.5").replace("1p9", "1.9")

        # TODO: Move this to another function once you think of a not-dumb name for it
        conds = lambda x: ("ACT" in x or x.endswith(HEADER_TEMP) or x.endswith("K"))

        return [a for a in temp_name.split("_") if conds(a)]

    def _parse_experiment_labels(self, exp_labels, temp_df):
        for label in exp_labels:

            # TODO: This logic is  a bit bonkers atm
            if label.endswith("K"):
                T_label = float(label.replace("K", ""))
                temp_df[HEADER_TEMP] = T_label
            elif label.endswith(HEADER_TEMP):
                H_label = float(label.replace(HEADER_TEMP, ""))
                temp_df[HEADER_MAGNET] = H_label
            elif "ACT" in label:
                act_label = label

                temp_df[HEADER_ACT] = act_label
                temp_df[HEADER_GEO] = self.META_DATA[label][HEADER_GEO]
                # TODO: Check this is really unneeded
                temp_df["ACT"] = self.META_DATA[label]["ACT"]

                # Update meta data
                # TODO: Need to address this nested dict storage by using a custom data class
                if HEADER_LENGTH not in self.META_DATA[label].keys():
                    self.META_DATA[label][HEADER_LENGTH] = temp_df[
                        HEADER_LENGTH
                    ].values[0]
                    self.META_DATA[label][HEADER_WIDTH] = temp_df[HEADER_WIDTH].values[
                        0
                    ]
                    self.META_DATA[label][HEADER_HEIGHT] = temp_df[
                        HEADER_HEIGHT
                    ].values[0]

                # TODO: What is going on with this? Encapsulate and make more clear
                # Create additional meta data dictionaries
                self.META_DATA[act_label][KEY_TEMP_LABELS] = []
                self.META_DATA[act_label][KEY_MAGNET_LABELS] = []
                self.META_DATA[act_label][H_label] = {}
                self.META_DATA[act_label][H_label][T_label] = {KEY_RES_CONSTANTS: {}}

        return temp_df

    def _load_to_csv(self, new_df, save_df):
        save_df = pd.concat([save_df, new_df], ignore_index=True)
        save_df.to_csv(self.file_path, sep=",")
        return save_df

    def _transform_experiment_data(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        """ """

        self._get_meta_data(temp_df)
        temp_df = self._create_alt_resistance_units(temp_df)
        temp_df[HEADER_ANGLE_RAD] = self._convert_degs_to_rads(
            temp_df[HEADER_ANGLE_DEG].values
        )
        return temp_df

    def _convert_degs_to_rads(
        self, degs: np.ndarray | pd.Series
    ) -> np.ndarray | pd.Series:

        return degs * 2 * np.pi / 360

    def _get_meta_data(self, df: pd.DataFrame) -> None:
        """ """
        # Todo: This may be unnecessary. Should find a way to generalize the labelling away from ACTRot#
        # TODO: Turn it into 'experiment_label' with the dataclass?
        act_label = "ACTRot" + str(df["ACT"].values[0])
        H_label = df[HEADER_MAGNET].values[0]
        T_label = df[HEADER_TEMP].values[0]

        # Calc for additional columns as needed
        mean_res = df[HEADER_RES_OHM].mean()
        zero_deg_res = df.loc[df[HEADER_ANGLE_DEG].idxmin(), HEADER_RES_OHM]

        # Store additional meta data
        # TODO: Fix this, and store the meta data in some json or something somewhere
        this_meta_data = self.META_DATA[act_label][H_label][T_label]
        this_meta_data[KEY_RES_CONSTANTS][HEADER_MEAN] = mean_res
        this_meta_data[KEY_RES_CONSTANTS][HEADER_0DEG] = zero_deg_res
        return

    def _create_alt_resistance_units(
        self, df: pd.DataFrame, HEADER_RES_DEF_0DEG_NORM=None
    ) -> pd.DataFrame:
        """
        Calculates alternative resistivity units based on the new meta data

        todo: This will be replaced by the data class objects
        """
        act_label = "ACTRot" + str(df["ACT"].values[0])
        h_label = df[HEADER_MAGNET].values[0]
        t_label = df[HEADER_TEMP].values[0]

        res_meta_data = self.META_DATA[act_label][h_label][t_label][KEY_RES_CONSTANTS]

        mean_res = res_meta_data[HEADER_MEAN]
        df[HEADER_RES_DEL_MEAN_OHM] = df[HEADER_RES_OHM] - mean_res
        df[HEADER_RES_DEF_MEAN_NORM] = df[HEADER_RES_DEL_MEAN_OHM] / mean_res
        df[HEADER_RES_DEL_MEAN_NORM_PCT] = df[HEADER_RES_DEF_MEAN_NORM] * 100

        res_at_0deg = res_meta_data[HEADER_0DEG]
        df[HEADER_RES_DEL_0DEG_OHM] = df[HEADER_RES_OHM] - res_at_0deg
        df[HEADER_RES_DEF_0DEG_NORM] = df[HEADER_RES_DEL_0DEG_OHM] / res_at_0deg
        df[HEADER_RES_DEL_0DEG_NORM_PCT] = df[HEADER_RES_DEF_MEAN_NORM] * 100

        # uohms
        df = self._calculate_uohm_cols(df)
        return df

    def _calculate_uohm_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if "Res" in col:
                new_col = col.replace("ohm", "uohm")
                df[new_col] = df[col] * 10**6

        return df

    def quick_plot_amro(self):
        return _quick_plot_amro(self)
