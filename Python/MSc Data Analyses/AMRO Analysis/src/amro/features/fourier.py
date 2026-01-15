import itertools
import numpy as np
import pandas as pd

from ..config import (
    FINAL_DATA_PATH,
    PROCESSED_DATA_PATH,
    HEADER_ANGLE_DEG,
    HEADER_ANGLE_RAD,
    HEADER_RES_OHM,
    H_PALETTE,
    HEADER_EXP_LABEL,
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
from ..data import (
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
    """Performs Fourier transforms on AMRO data to extract rotational symmetry components."""

    def __init__(
        self,
        amro_data: ProjectData,
        save_name: str,
        overwrite_result=False,
        verbose: bool = False,
    ):
        """Initialize the Fourier transformer.

        Args:
            amro_data: ProjectData object containing AMRO experiments and oscillations.
            save_name: Name for saving results files.
            overwrite_result: If True, overwrite existing Fourier results.
            verbose: If True, print detailed processing information.
        """
        self.project_data = amro_data

        self.all_results_df = pd.DataFrame()
        self.save_name = save_name
        self.save_dir = FINAL_DATA_PATH
        self.save_fp = self.save_dir / save_name
        self.verbose = verbose
        self.overwrite = overwrite_result

        return

    def fourier_transform_experiments(self) -> None:
        """Perform Fourier transforms on all oscillations in the project data.

        Iterates through all experiments and their oscillations, performing
        FFT analysis on each. Results are stored in the oscillation objects
        and saved to CSV and pickle files.
        """
        results_list = []
        for exp_label in self.project_data.get_experiment_labels():

            experiment = self.project_data.get_experiment(exp_label)

            for key in experiment.oscillations_dict.keys():
                osc = experiment.get_oscillation_from_key(key)

                if osc.fourier_result is not None:
                    if not self.overwrite:
                        print(f"{osc} already has a Fourier result. Skipping...")
                        continue
                    elif self.overwrite:
                        osc.clear_fourier_result()
                if self.verbose:
                    print(
                        f"Fourier Transforming {key.experiment_label}, T={key.temperature}K, H={key.magnetic_field}T"
                    )

                xf, yf = self._perform_fourier_transform(osc.osc_data)
                osc.add_fourier_result(xf, yf)

        print("Saving Fourier results.")
        self.project_data.save_fourier_results_to_csv()
        print("Pickling project data.")
        self.project_data.save_project_to_pickle()

        return

    def get_n_strongest_results(
        self,
        n=0,
        act: str | list = None,
        t: float | list = None,
        h: float | list = None,
    ) -> list:
        """Query the n strongest Fourier components for filtered oscillations.

        Args:
            n: Number of strongest components to retrieve. If 0, returns all components.
            act: Experiment label(s) to filter by.
            t: Temperature value(s) to filter by.
            h: Magnetic field value(s) to filter by.

        Returns:
            List of tuples containing (OscillationKey, strongest_components) for each oscillation.
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
        """Plot bar chart of the n strongest Fourier components.

        Args:
            n: Number of strongest components to plot.
            t: Temperature value(s) to filter the data.
            h: Magnetic field value(s) to filter the data.

        Returns:
            Seaborn FacetGrid object containing the generated plot.
        """
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
