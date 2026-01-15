"""Plotting functions for visualizing Fourier analysis results.

Provides functions to create bar plots of the strongest Fourier components
extracted from AMRO data.
"""

from ..config import (
    HEADER_FREQ,
    HEADER_MAG_RATIO,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_EXP_LABEL,
)
from ..utils import utils as u
import seaborn as sns
import matplotlib.pyplot as plt


def _plot_n_strongest(
    fourier, n: int, t: list | float, h: list | float
) -> sns.FacetGrid:
    """Plot bar chart of the n strongest Fourier components from AMRO data.

    Creates a faceted bar plot showing the amplitude ratios of the strongest
    Fourier components, organized by temperature and magnetic field.

    Args:
        fourier: Fourier instance containing transformed AMRO data.
        n: Number of strongest components to plot. If n=0, plots all components.
        t: Temperature value(s) to filter the data.
        h: Magnetic field value(s) to filter the data.

    Returns:
        Seaborn FacetGrid object containing the generated plot.
    """
    # TODO: Config file?
    sns.set_context("poster")

    df = fourier.get_n_strongest_results(n)
    plot_df = u.query_dataframe(df=df, t=t, h=h)

    hue_choice = HEADER_EXP_LABEL

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
