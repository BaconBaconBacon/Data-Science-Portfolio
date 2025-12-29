"""
    Plotting functionality for the utils_fourier.py file.
"""

from utils import utils_misc as u
import seaborn as sns


def _plot_n_strongest(fourier, n: int, T: list | float, H: list | float) -> None:
    """
    Plots the n-strongest.

    If n=0, then plots all available contributions.
    """

    df = fourier.get_n_strongest(n)
    plot_df = u.query_dataframe(df=df, t=T, h=H)

    hue_choice = "ACT"
    plot_df = plot_df.sort_values(hue_choice)
    plot_df[hue_choice] = plot_df[hue_choice].astype(str)
    sns.set_context("poster")
    g = sns.catplot(
        x="freqs (cycles/rot)",
        y="amp_ratio",
        data=plot_df,
        col="T",
        row="H",
        kind="bar",
        hue=hue_choice,
        sharex=False,
    )

    return g
