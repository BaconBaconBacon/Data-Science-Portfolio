import itertools
import numpy as np
import os
import pandas as pd

from config.settings import PROCESSED_DATA_PATH, H_PALETTE
from plotting.plotting_fourier import _plot_n_strongest
from scipy.fft import rfft, rfftfreq
from utils import utils_misc as u


class Fourier:
    def __init__(self, amro_df: pd.DataFrame, save_name: str):
        self.amro_data = amro_df
        self.labels = self.amro_data[["ACT_str", "T", "H"]].drop_duplicates()

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
        for act_label in self.amro_data["ACT_str"].unique():
            print("FT'ing: " + act_label)

            act_df = u.QueryDataFrame(self.amro_data, act=act_label)

            # TODO: Encapsulate
            t_vals, h_vals, geo_label = self._get_experiment_labels(act_df)

            # TODO: Encapsulate for loop
            for t, h in itertools.product(t_vals, h_vals):

                ft_df = u.QueryDataFrame(act_df, h=h, t=t)

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

        return act_df["T"].unique(), act_df["H"].unique(), act_df["geo"].unique()[0]

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
                "freqs (cycles/rot)": xf,
                "mag (ohm-cm)": np.abs(yf),
                "phase": np.angle(yf),
            }
        )

        # Amplitudes relative to the strongest
        freq_df["amp_ratio"] = freq_df["mag (ohm-cm)"] / freq_df["mag (ohm-cm)"].max()
        freq_df["freqs (cycles/rot)"] = freq_df["freqs (cycles/rot)"].astype(int)

        # Force positive phase values
        freq_df["phase_raw"] = freq_df["phase"].copy()
        freq_df["phase"] = np.where(
            freq_df["phase_raw"] < 0,
            freq_df["phase_raw"] + 2 * np.pi,
            freq_df["phase_raw"],
        )

        freq_df["ACT_str"] = act_label
        freq_df["T"] = t
        freq_df["H"] = h
        freq_df["geo"] = geo_label

        return freq_df

    def get_n_strongest_results(self, n: int):
        """
        Queries the n strongest contributions for each experiment in the data set.
        If n=0, then returns all available contributions sorted by magnitude.
        """
        sort_vals = ["ACT_str", "H", "T", "mag (ohm-cm)"]
        strongest_df = self.all_results_df.sort_values(by=sort_vals, ascending=False)
        if n > 0:
            return (
                strongest_df.groupby(["ACT_str", "H", "T"])
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
