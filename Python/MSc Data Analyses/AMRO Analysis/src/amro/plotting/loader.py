"""
    Plotting functionality for the utils_loader.py file.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from ..config import (
    H_PALETTE,
    HEADER_MAGNET,
    HEADER_TEMP,
    HEADER_EXP_LABEL,
    HEADER_ANGLE_DEG,
    HEADER_RES_DEL_MEAN_UOHM,
)


def _quick_plot_amro(loader) -> None:
    """Generate quick visualization plots of AMRO data from the loader.

    Creates faceted scatter plots showing resistivity vs sample angle,
    with data organized by magnetic field strength, temperature, and experiment label.

    Args:
        loader: AMROLoader instance containing project data to visualize.
    """
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
