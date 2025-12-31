"""
    Plotting functionality for the utils_loader.py file.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from config.settings import H_PALETTE


def _quick_plot_amro(loader) -> None:
    """ """
    _ = sns.relplot(
        x="Sample Position (rads)",
        y="Delta Res./R0 Mean (ohm-cm)",
        hue="H",
        col="T",
        row="ACT",
        palette=H_PALETTE,
        facet_kws={"sharey": False, "linewidth": 0},
        data=loader.AMRO,
    )
    plt.show()
    return
