"""
    Plotting functionality for the utils_loader.py file.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from ..config.settings import (
    H_PALETTE,
    HEADER_MAGNET,
    HEADER_TEMP,
    HEADER_ACT,
    HEADER_ANGLE_DEG,
    HEADER_RES_DEL_MEAN_UOHM,
)


def _quick_plot_amro(loader) -> None:
    """ """
    _ = sns.relplot(
        x=HEADER_ANGLE_DEG,
        y=HEADER_RES_DEL_MEAN_UOHM,
        hue=HEADER_MAGNET,
        col=HEADER_TEMP,
        row=HEADER_ACT,
        palette=H_PALETTE,
        linewidth=0,
        facet_kws={"sharey": False},
        data=loader.AMRO_df,
    )
    plt.show()
    return
