"""
    Plotting functionality for the utils_fitter.py file.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..config import (
    PROCESSED_DATA_PATH,
    FINAL_DATA_PATH,
    H_PALETTE,
    PROCESSED_FIGURES_PATH,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_RES_UOHM,
    HEADER_RES_OHM,
    HEADER_ANGLE_RAD,
    HEADER_ANGLE_DEG,
)
from matplotlib.patches import Patch

from ..data import ProjectData, Experiment
from ..utils import utils as u
from pathlib import Path


# TODO: put this into a config file
hspace = 0.05
wspace = 0.3
context_font_scale = 1


def _plot_fits_with_residuals(
    fitter,
    act_choice: str,
    h_choices: list | None | np.ndarray = None,
    t_choices: list | None | np.ndarray = None,
    figsize: tuple | None = None,
    y_scale: float = 1,
    y_label: str = HEADER_RES_OHM,
    x_label: str = HEADER_ANGLE_DEG,
):
    """
    Plotter to display finished fits over AMRO data, with the option
    to show the residuals. Not intended to be for a polished final
    version.

    Assumes each experiment at a given T has the same H values as all the others in the experiment

    TODO: Runs too slow, switch from sns.scatterplot-> ax.scatter and sns.lineplot -> ax.plot
    """
    # Set seaborn style
    project_data = fitter.project_data

    if act_choice not in project_data.experiments_dict.keys():
        print(f"{act_choice} is not a valid experiment choice.")
        return None, None
    experiment = project_data.get_experiment(act_choice)

    exp_keys = experiment.oscillations_dict.keys()
    t_vals, h_vals = _get_plot_labels(exp_keys)

    if t_choices is not None:
        t_vals = [t for t in t_vals if t in t_choices]

    if h_choices is not None:
        h_vals = [h for h in h_vals if h in h_choices]

    t_vals.sort()
    h_vals.sort()

    n_cols = len(t_vals)
    n_rows = len(h_vals)

    # Calculate figure size if not provided
    if figsize is None:
        figsize = _calculate_fig_size(n_cols=n_cols, n_rows=n_rows)

    # Create subplots
    fig, gs, axes = _create_subplots(
        fig_size=figsize,
        n_rows=n_rows,
        n_cols=n_cols,
        hspace=hspace,
        wspace=wspace,
    )

    _plot_grid(
        experiment=experiment,
        t_vals=t_vals,
        h_vals=h_vals,
        fig=fig,
        gs=gs,
        axes=axes,
        y_scale=y_scale,
        x_label=x_label,
        y_label=y_label,
    )

    # Generate legend
    _generate_legend(fig)

    return fig, axes


def _plot_fits_with_residuals_uohm(
    fitter,
    act_choice: str,
    h_choices=None,
    t_choices=None,
    figsize=None,
):
    fig, axes = _plot_fits_with_residuals(
        fitter=fitter,
        act_choice=act_choice,
        h_choices=h_choices,
        t_choices=t_choices,
        figsize=figsize,
        y_scale=10**6,
        y_label=HEADER_RES_UOHM,
    )
    return fig, axes


def _plot_grid(
    experiment: Experiment, h_vals, t_vals, fig, gs, axes, y_scale, x_label, y_label
):
    n_rows = len(h_vals)

    # Iterate over grid
    for i, H in enumerate(h_vals):
        for j, T in enumerate(t_vals):

            ax_fit = fig.add_subplot(gs[i * 2, j])
            ax_resid = fig.add_subplot(gs[i * 2 + 1, j], sharex=ax_fit)
            axes[i, j] = (ax_fit, ax_resid)
            try:
                osc = experiment.get_oscillation(t=T, h=H)
            except KeyError:
                continue
            y_data = osc.osc_data.res_ohms
            x_plot = osc.osc_data.angles_degs

            y_fit = osc.fit_result.model_res_ohms

            y_data = y_data * y_scale
            y_fit = y_fit * y_scale
            residuals = y_data - y_fit

            _plot_fit_over_data(x_plot, y_data, y_fit, ax_fit, H_PALETTE[H])
            _plot_residuals(x_plot, residuals, ax_resid)

            subplot_title = "{}T | {}K".format(H, T)
            _format_data_axis(ax_fit, n_rows, i, j, subplot_title, x_label, y_label)
            _format_residuals_axis(ax_resid, n_rows, i, j, x_label)


def _plot_residuals(x_plot, residuals, ax_resid):
    sns.scatterplot(
        x=x_plot,
        y=residuals,
        ax=ax_resid,
        color="black",
        linewidth=0,
    )
    return


def _plot_fit_over_data(x_plot, y, y_fit, ax, color) -> None:
    sns.scatterplot(
        x=x_plot,
        y=y,
        color=color,
        ax=ax,
        linewidth=0,
    )
    sns.lineplot(x=x_plot, y=y_fit, color="black", ax=ax)
    return


def _plot_bad_fits(fitter, act_choice: str) -> tuple:

    if len(fitter.failed_fits) == 0:
        print("No fits failed.")
        return None, None

    t_labels = []
    h_labels = []
    for osc_key in fitter.failed_fits:
        if osc_key.experiment_label == act_choice:
            t_labels.append(osc_key.temperature)
            h_labels.append(osc_key.magnetic_field)

    if len(t_labels) > 0 and len(h_labels) > 0:
        fig, axes = _plot_fits_with_residuals(
            fitter=fitter, act_choice=act_choice, t_choices=t_labels, h_choices=h_labels
        )

        plt.show()
        return fig, axes
    else:
        print(f"No fits failed for {act_choice}.")
        return None, None


def _format_data_axis(ax_fit, n_rows, i, j, subplot_title, x_label, y_label):
    ax_fit.set_title(subplot_title, fontsize=10)
    ax_fit.set_xticks([0, 90, 180, 270, 360])
    if i == (n_rows - 1):
        ax_fit.set(xlabel=x_label)
    if j == 0:
        ax_fit.set(ylabel=y_label)


def _format_residuals_axis(ax_resid, n_rows, i, j, x_label):
    if i == (n_rows - 1):
        ax_resid.set(xlabel=x_label)
    else:
        ax_resid.set(xlabel="")
        ax_resid.tick_params(labelbottom=False)

    return


def _generate_legend(figure):
    legend_elements = [
        Patch(facecolor=color, label=str(label)) for label, color in H_PALETTE.items()
    ]

    figure.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        title="H (T)",
    )
    return


def _get_plot_labels(exp_keys: list):

    t_vals = []
    h_vals = []
    for key in exp_keys:
        t_vals.append(key.temperature)
        h_vals.append(key.magnetic_field)

    t_vals = list(set(t_vals))
    h_vals = list(set(h_vals))

    return t_vals, h_vals


# def _get_fit_params(
#     project_data: ProjectData, act: str, h: float, t: float
# ) -> dict | None:
#     exp = project_data.get_experiment(act)
#     osc = exp.get_oscillation(t=t, h=h)
#     if osc.fit_result is None:
#         return None
#     else:
#         return osc.fit_result.guesses_dict


def _calculate_fig_size(n_cols, n_rows):
    width = 6 * n_cols
    height = 6 * n_rows
    return width, height


def _create_subplots(fig_size, n_rows, n_cols, hspace, wspace):
    # Each position gets 2 rows: one for fit, one for residuals
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(
        n_rows * 2,
        n_cols,
        hspace=hspace,
        wspace=wspace,
        height_ratios=[3, 1] * n_rows,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)

    return fig, gs, axes


def _save_plot(fig, filename, dpi=300):
    """
    Save the plot
    """

    filepath = PROCESSED_FIGURES_PATH / filename
    fig.savefig(
        filepath,
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
    )
    print("Saved {}".format(filename))
    return
