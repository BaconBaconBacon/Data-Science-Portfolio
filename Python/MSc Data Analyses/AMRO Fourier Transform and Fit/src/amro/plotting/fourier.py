"""
    Plotting functionality for the utils_fourier.py file.
"""

from ..config.settings import (
    HEADER_FREQ,
    HEADER_MAG_RATIO,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_ACT,
)
from ..utils import utils as u
import seaborn as sns
import matplotlib.pyplot as plt


def _plot_n_strongest(
    fourier, n: int, t: list | float, h: list | float
) -> sns.FacetGrid:
    """
    Plots the n-strongest.

    If n=0, then plots all available contributions.
    """
    # TODO: Config file?
    sns.set_context("poster")

    df = fourier.get_n_strongest_results(n)
    plot_df = u.query_dataframe(df=df, t=t, h=h)

    hue_choice = HEADER_ACT

    plot_df = plot_df.sort_values(hue_choice)
    plot_df[hue_choice] = plot_df[hue_choice].astype(str)
    g = sns.catplot(
        x=HEADER_FREQ,
        y=HEADER_MAG_RATIO,
        data=plot_df,
        col=HEADER_TEMP,
        row=HEADER_MAGNET,
        kind="bar",
        hue=hue_choice,
        sharex=False,
    )
    plt.show()

    return g
