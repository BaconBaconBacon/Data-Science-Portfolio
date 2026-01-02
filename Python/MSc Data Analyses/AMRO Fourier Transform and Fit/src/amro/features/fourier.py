import itertools
import numpy as np
import os
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
)
from ..plotting.fourier import _plot_n_strongest
from scipy.fft import rfft, rfftfreq
from ..utils import utils as u


class Fourier:
    def __init__(self, amro_df: pd.DataFrame, save_name: str):
        self.amro_data = amro_df
        self.labels = self.amro_data[
            [HEADER_ACT, HEADER_TEMP, HEADER_MAGNET]
        ].drop_duplicates()

        self.all_results_df = pd.DataFrame()
        self.save_name = save_name
        self.save_dir = PROCESSED_DATA_PATH
        self.save_fp = os.path.join(self.save_dir, save_name)

        if os.path.exists(self.save_fp):
            # TODO: Need to check and make sure it's loading the same data as the AMRO
            print("Loading {}".format(save_name))
            self.all_results_df = pd.read_csv(self.save_fp)

            # TODO: self._check_results_against_amro()
        return

    def _check_results_against_amro(self):
        """Ensures that the loaded Fourier transform results has the same
        experiment labels as the AMRO data.
        """

        # todo: implement, maybe with a recursive function?
        return

    def fourier_transform_experiments(self):
        results_list = []
        for act_label in self.amro_data[HEADER_ACT].unique():

            act_df = u.query_dataframe(self.amro_data, act=act_label)

            # TODO: Encapsulate
            t_vals, h_vals, geo_label = self._get_experiment_labels(act_df)

            # TODO: Encapsulate for loop
            for t, h in itertools.product(t_vals, h_vals):
                print(f"Fourier Transforming{act_label}, T={t}K, H={h}T")
                ft_df = u.query_dataframe(act_df, h=h, t=t)

                xf, yf = self._perform_fourier_transform(ft_df)

                result_df = self._pack_ft_result(xf, yf, act_label, t, h, geo_label)

                results_list.append(result_df)
                self.all_results_df = pd.concat(
                    [self.all_results_df, result_df], ignore_index=True
                )
        self.all_results_df = pd.concat(results_list, ignore_index=True)
        self._save_results_df()
        return

    def _get_experiment_labels(self, act_df: pd.DataFrame):

        return (
            act_df[HEADER_TEMP].unique(),
            act_df[HEADER_MAGNET].unique(),
            act_df[HEADER_GEO].unique()[0],
        )

    def _save_results_df(self) -> None:
        self.all_results_df.to_csv(self.save_fp, sep=",", index=False)
        print("Results saved to: {}".format(self.save_name))
        return

    def _pack_ft_result(self, xf, yf, act_label, t, h, geo_label):
        """
        Performs a Fast Fourier transform on the AMR oscillation of an experiment.

        Extracts the phase and amplitude of each symmetry's Fourier component.

        """
        freq_df = pd.DataFrame(
            {
                HEADER_FREQ: xf,
                HEADER_MAG: np.abs(yf),
                HEADER_PHASE: np.angle(yf),
            }
        )

        # Amplitudes relative to the strongest
        freq_df[HEADER_MAG_RATIO] = freq_df[HEADER_MAG] / freq_df[HEADER_MAG].max()
        freq_df[HEADER_FREQ] = freq_df[HEADER_FREQ].astype(int)

        # Force positive phase values
        freq_df[HEADER_PHASE_RAW] = freq_df[HEADER_PHASE].copy()
        freq_df[HEADER_PHASE] = np.where(
            freq_df[HEADER_PHASE_RAW] < 0,
            freq_df[HEADER_PHASE_RAW] + 2 * np.pi,
            freq_df[HEADER_PHASE_RAW],
        )

        freq_df[HEADER_ACT] = act_label
        freq_df[HEADER_TEMP] = t
        freq_df[HEADER_MAGNET] = h
        freq_df[HEADER_GEO] = geo_label

        return freq_df

    def get_n_strongest_results(self, n: int):
        """
        Queries the n strongest contributions for each experiment in the data set.
        If n=0, then returns all available contributions sorted by magnitude.
        """
        sort_vals = [HEADER_ACT, HEADER_MAGNET, HEADER_TEMP, HEADER_MAG]
        strongest_df = self.all_results_df.sort_values(by=sort_vals, ascending=False)
        if n > 0:
            return (
                strongest_df.groupby([HEADER_ACT, HEADER_MAGNET, HEADER_TEMP])
                .head(n)
                .reset_index(drop=True)
            )
        elif n == 0:
            return strongest_df
        elif n < 0:
            print("Negative values not accepted.")
            return

    def plot_n_strongest(self, n: int, T: list | float, H: list | float):
        return _plot_n_strongest(self, n, T, H)

    def _perform_fourier_transform(self, df: pd.DataFrame):
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
        fftdata = df["Delta Res. Mean (ohm-cm)"].values

        # Perform the FFT, where
        yf = rfft(fftdata, n=len(fftdata), norm="ortho")
        xf = rfftfreq(len(fftdata), 1 / len(fftdata))

        return xf, yf  # freq_df
