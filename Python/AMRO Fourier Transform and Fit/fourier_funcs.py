import itertools
import os

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import seaborn as sns


class Fourier:
    def __init__(self, amro, save_name: str, save_dir: str):
        # Get the AMRO
        self.amro_data = amro.AMRO
        # self.meta_data = amro.META_DATA
        self.labels = self.amro_data[["ACT", "T", "H"]].drop_duplicates()

        # Iterate through DataFrame entries, appending results to a new DataFrame
        self.FT_results_df = pd.DataFrame()
        self.save_name = save_name
        self.save_dir = save_dir
        self.save_fp = os.path.join(save_dir, save_name)

        if os.path.exists(self.save_fp):
            # TODO: Need to check and make sure it's loading the same data as the AMRO
            print("loading {}".format(save_name))
            self.FT_results_df = pd.read_csv(self.save_fp)

            return
        else:
            for act_label in self.amro_data["ACT_str"].unique():
                print("FT'ing: " + act_label)

                act_df = self.amro_data.query('ACT_str=="{}"'.format(act_label))
                t_vals = act_df["T"].unique()
                h_vals = act_df["H"].unique()
                geo_label = act_df["geo"].unique()[0]

                for t, h in itertools.product(t_vals, h_vals):
                    # Query the correct dataframe using the experiment labels
                    ft_df = self.amro_data.query(
                        'ACT_str=="{}" & T =={} & H == {}'.format(act_label, t, h)
                    )  # 'ACT_str=="{}"'.format(act_label))  #

                    freq_df = self._fourier_transform(ft_df)

                    freq_df["ACT_str"] = act_label
                    freq_df["ACT"] = float(act_label.replace("ACTRot", ""))
                    freq_df["T"] = t
                    freq_df["H"] = h
                    freq_df["geo"] = geo_label  # [0]

                    self.FT_results_df = pd.concat(
                        [self.FT_results_df, freq_df], ignore_index=True
                    )  # .reset_index(drop=True)

            # Save the results of the FT
            self.FT_results_df.to_csv(self.save_fp, sep=",", index=False)
            print("Results saved to: {}".format(self.save_name))
        return

    def GetNStrongest(self, n: int):
        """
        Queries the n strongest contributions for each experiment in the data set.
        """
        sort_vals = ["ACT", "H", "T", "mag (ohm-cm)"]
        strongest_df = self.FT_results_df.sort_values(by=sort_vals, ascending=False)
        strongest_freqs = strongest_df.groupby(["ACT", "H", "T"]).head(n)
        return strongest_freqs.reset_index(drop=True)

    def PlotNStrongest(self, n: int, T: list | float, H: list | float) -> None:
        """
        Plots the n-strongest.

        If n=0, then plots all available contributions.
        """
        if isinstance(T, list):
            q = "T in {}".format(T)
        else:
            q = "T == {}".format(T)

        if isinstance(H, list):
            q += " & H in {}".format(H)
        else:
            q += " & H == {}".format(H)

        plot_df = self.GetNStrongest(n).query(q)
        # Bypass a formatting bug in catplot
        hue_choice = "H"
        plot_df = plot_df.sort_values(hue_choice)
        plot_df[hue_choice] = plot_df[hue_choice].astype(str)
        sns.set_context("poster")
        g = sns.catplot(
            x="freqs (cycles/rot)",
            y="amp_ratio",
            data=plot_df,
            col="T",
            row="ACT",
            kind="bar",
            hue=hue_choice,
        )
        g.set(xlim=(0.1, None))
        return

    def _fourier_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        fftdata = df["Delta Res. Mean (ohm-cm)"].values

        # Perform the FFT, where yf is the amplitudes and xf are the frequencies
        yf = rfft(fftdata, n=len(fftdata), norm="ortho")
        xf = rfftfreq(len(fftdata), 1 / len(fftdata))

        # Package the results
        freq_df = pd.DataFrame(
            {
                "freqs (cycles/rot)": xf,
                "amps": yf,
                "mag (ohm-cm)": np.abs(yf),
                "phase": np.angle(yf),
            }
        )

        # Amplitudes relative to the strongest
        freq_df["amp_ratio"] = freq_df["mag (ohm-cm)"] / freq_df["mag (ohm-cm)"].max()
        freq_df["freqs (cycles/rot)"] = freq_df["freqs (cycles/rot)"].astype(int)

        # Force positive phase values
        freq_df["phase_raw"] = freq_df["phase"].copy()
        freq_df["phase"] = np.select(
            freq_df["phase_raw"] < 0,
            freq_df["phase_raw"] + 2 * np.pi,
            freq_df["phase_raw"],
        )
        return freq_df


# if __name__ == "__main__":
#     import sys
#     load = LoadAMRO(sys.argv[1],sys.argv[2])
#     _ = load.combineAMRO()
#
