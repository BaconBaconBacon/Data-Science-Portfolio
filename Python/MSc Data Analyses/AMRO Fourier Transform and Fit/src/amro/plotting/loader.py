"""
    Plotting functionality for the utils_loader.py file.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from ..config.settings import (
    H_PALETTE,
    HEADER_MAGNET,
    HEADER_TEMP,
    HEADER_EXP_LABEL,
    HEADER_ANGLE_DEG,
    HEADER_RES_DEL_MEAN_UOHM,
)


def _quick_plot_amro(loader) -> None:
    """ """
    data = loader.project_data
    for key, exp in data.experiments_dict.items():

        data = exp.get_experiment_as_dataframe()
        _ = sns.relplot(
            x=HEADER_ANGLE_DEG,
            y=HEADER_RES_DEL_MEAN_UOHM,
            hue=HEADER_MAGNET,
            col=HEADER_TEMP,
            row=HEADER_EXP_LABEL,
            palette=H_PALETTE,
            linewidth=0,
            facet_kws={"sharey": False},
            data=data,
        )
        plt.show()
    return
