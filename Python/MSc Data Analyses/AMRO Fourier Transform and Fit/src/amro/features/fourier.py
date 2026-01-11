import itertools
import numpy as np
import pandas as pd

from ..config.settings import (
    PROCESSED_DATA_PATH,
    HEADER_ANGLE_DEG,
    HEADER_ANGLE_RAD,
    HEADER_RES_OHM,
    H_PALETTE,
    HEADER_ACT,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_MAG_RATIO,
    HEADER_MAG,
    HEADER_FREQ,
    HEADER_PHASE,
    HEADER_PHASE_RAW,
    HEADER_GEO,
    HEADER_RES_DEL_MEAN_OHM,
)
from ..data.data_structures import (
    ProjectData,
    FourierResult,
    OscillationKey,
    ExperimentalData,
)
from ..plotting.fourier import _plot_n_strongest
from scipy.fft import rfft, rfftfreq
from ..utils import utils as u
from pathlib import Path


class Fourier:
    def __init__(self, amro_data: ProjectData, save_name: str, verbose: bool = False):
        self.project_data = amro_data

        self.all_results_df = pd.DataFrame()
        self.save_name = save_name
        self.save_dir = PROCESSED_DATA_PATH
        self.save_fp = self.save_dir / save_name
        self.verbose = verbose
        if self.save_fp.is_file():
            # TODO: Need to check and make sure it's loading the same data as the AMRO
            print("Loading {}".format(save_name))

        return

    def _check_results_against_amro(self):
        """Ensures that the loaded Fourier transform results has the same
        experiment labels as the AMRO data.
        """

        # todo: implement, maybe with a recursive function?
        return

    def fourier_transform_experiments(self):
        results_list = []
        for exp_label in self.project_data.get_experiment_labels():

            experiment = self.project_data.get_experiment(exp_label)

            # TODO: Encapsulate for loop
            for key in experiment.oscillations_dict.keys():
                osc = experiment.get_oscillation_from_key(key)
                print(
                    f"Fourier Transforming {key.experiment_label}, T={key.temperature}K, H={key.magnetic_field}T"
                )

                xf, yf = self._perform_fourier_transform(osc.osc_data)
                osc.add_fourier_result(xf, yf)
        return

    def get_n_strongest_results(
        self,
        n=0,
        act: str | list = None,
        t: float | list = None,
        h: float | list = None,
    ) -> list:
        """
        Queries the n strongest contributions for each experiment in the data set.
        If n=0, then returns all available contributions sorted by magnitude.
        """
        oscillations = self.project_data.filter_oscillations(
            experiments=act, t_vals=t, h_vals=h
        )

        results = []
        for osc in oscillations:
            if osc.fourier_result is not None:
                strongest = osc.fourier_result.get_n_strongest_components(n)
                results.append((osc.key, strongest))
        return results

    def plot_n_strongest(self, n: int, t: list | float, h: list | float):
        return _plot_n_strongest(self, n, t, h)

    def _perform_fourier_transform(
        self, data: ExperimentalData
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a Fourier transform on the AMR oscillation of an experiment,
        where the mean resistivity has been subtracted from the data to centre
        the oscillation about zero.

        Input:
            df: DataFrame storing an AMRO experiment's data
        Return:
            yf: List of complex numbers storing the amplitudes and phases
            xf: List of the rotational symmetries
        """
        fft_data = data.delta_res_mean_ohms

        # Perform the FFT, where
        yf = rfft(fft_data, n=len(fft_data), norm="ortho")
        xf = rfftfreq(len(fft_data), 1 / len(fft_data))

        if xf[0] == 0:
            xf = xf[1:]
            yf = yf[1:]
        return xf, yf
