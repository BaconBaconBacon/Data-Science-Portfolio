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
    exp_chocie: str,
    h_choices: list | None | np.ndarray = None,
    t_choices: list | None | np.ndarray = None,
    figsize: tuple | None = None,
    y_scale: float = 1,
    y_label: str = HEADER_RES_OHM,
    x_label: str = HEADER_ANGLE_DEG,
    save_fig=False,
):
    """Plot fitted curves overlaid on AMRO data with residual subplots.

    Creates a grid of subplots where each cell shows the fit over experimental
    data with a residual plot below. Grid is organized by magnetic field (rows)
    and temperature (columns).

    Args:
        fitter: AMROFitter instance containing fit results.
        exp_chocie: Experiment label to plot.
        h_choices: Magnetic field values to include. If None, includes all.
        t_choices: Temperature values to include. If None, includes all.
        figsize: Figure size tuple (width, height). Auto-calculated if None.
        y_scale: Scale factor for y-axis values.
        y_label: Label for y-axis.
        x_label: Label for x-axis.
        save_fig: If True, save figure to disk.

    Returns:
        Tuple of (figure, axes) matplotlib objects, or (None, None) if invalid experiment.
    """
    # Set seaborn style
    project_data = fitter.project_data

    if exp_chocie not in project_data.experiments_dict.keys():
        print(f"{exp_chocie} is not a valid experiment choice.")
        return None, None
    experiment = project_data.get_experiment(exp_chocie)

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
    if save_fig:
        filename = f"{exp_chocie}_figure_amro_fits_{fitter.filter_str}.pdf"
        _save_plot(fig, filename)
    return fig, axes


def _plot_fits_with_residuals_uohm(
    fitter,
    exp_chocie: str,
    h_choices=None,
    t_choices=None,
    figsize=None,
    save_fig=False,
):
    """Plot fitted curves with residuals using micro-ohm-cm units.

    Wrapper around _plot_fits_with_residuals that sets appropriate scale and labels.

    Args:
        fitter: AMROFitter instance containing fit results.
        exp_chocie: Experiment label to plot.
        h_choices: Magnetic field values to include.
        t_choices: Temperature values to include.
        figsize: Figure size tuple.
        save_fig: If True, save figure to disk.

    Returns:
        Tuple of (figure, axes) matplotlib objects.
    """
    fig, axes = _plot_fits_with_residuals(
        fitter=fitter,
        exp_chocie=exp_chocie,
        h_choices=h_choices,
        t_choices=t_choices,
        figsize=figsize,
        y_scale=10**6,
        y_label=HEADER_RES_UOHM,
        save_fig=save_fig,
    )
    return fig, axes


def _plot_grid(
    experiment: Experiment, h_vals, t_vals, fig, gs, axes, y_scale, x_label, y_label
):
    """Populate the subplot grid with fit and residual plots.

    Args:
        experiment: Experiment object containing oscillation data and fits.
        h_vals: List of magnetic field values for rows.
        t_vals: List of temperature values for columns.
        fig: Matplotlib figure object.
        gs: GridSpec object defining subplot layout.
        axes: 2D array to store axis pairs.
        y_scale: Scale factor for y-axis values.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
    """
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


def _plot_residuals(x_plot, residuals, ax_resid) -> None:
    """Plot fit residuals as a scatter plot.

    Args:
        x_plot: Array of x-axis values (angles).
        residuals: Array of residual values (data - fit).
        ax_resid: Matplotlib axes object to plot on.
    """
    sns.scatterplot(
        x=x_plot,
        y=residuals,
        ax=ax_resid,
        color="black",
        linewidth=0,
    )
    return


def _plot_fit_over_data(x_plot, y, y_fit, ax, color) -> None:
    """Plot experimental data points with fitted curve overlaid.

    Args:
        x_plot: Array of x-axis values (angles).
        y: Array of experimental data values.
        y_fit: Array of fitted model values.
        ax: Matplotlib axes object to plot on.
        color: Color for the data points.
    """
    sns.scatterplot(
        x=x_plot,
        y=y,
        color=color,
        ax=ax,
        linewidth=0,
    )
    sns.lineplot(x=x_plot, y=y_fit, color="black", ax=ax)
    return


def _plot_bad_fits(fitter, exp_chocie: str) -> tuple:
    """Plot only oscillations where fitting failed.

    Args:
        fitter: AMROFitter instance containing failed fit information.
        exp_chocie: Experiment label to check for failures.

    Returns:
        Tuple of (figure, axes) or (None, None) if no failures found.
    """
    if len(fitter.failed_fits) == 0:
        print("No fits failed.")
        return None, None

    t_labels = []
    h_labels = []
    for osc_key in fitter.failed_fits:
        if osc_key.experiment_label == exp_chocie:
            t_labels.append(osc_key.temperature)
            h_labels.append(osc_key.magnetic_field)

    if len(t_labels) > 0 and len(h_labels) > 0:
        fig, axes = _plot_fits_with_residuals(
            fitter=fitter, exp_chocie=exp_chocie, t_choices=t_labels, h_choices=h_labels
        )

        plt.show()
        return fig, axes
    else:
        print(f"No fits failed for {exp_chocie}.")
        return None, None


def _format_data_axis(ax_fit, n_rows, i, j, subplot_title, x_label, y_label) -> None:
    """Format the data subplot axis with title, ticks, and labels.

    Args:
        ax_fit: Matplotlib axes object for the data plot.
        n_rows: Total number of rows in the grid.
        i: Current row index.
        j: Current column index.
        subplot_title: Title string for the subplot.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
    """
    ax_fit.set_title(subplot_title, fontsize=10)
    ax_fit.set_xticks([0, 90, 180, 270, 360])
    if i == (n_rows - 1):
        ax_fit.set(xlabel=x_label)
    if j == 0:
        ax_fit.set(ylabel=y_label)


def _format_residuals_axis(ax_resid, n_rows, i, j, x_label):
    """Format the residuals subplot axis with appropriate labels.

    Args:
        ax_resid: Matplotlib axes object for the residuals plot.
        n_rows: Total number of rows in the grid.
        i: Current row index.
        j: Current column index.
        x_label: Label for x-axis.
    """
    if i == (n_rows - 1):
        ax_resid.set(xlabel=x_label)
    else:
        ax_resid.set(xlabel="")
        ax_resid.tick_params(labelbottom=False)

    return


def _generate_legend(figure):
    """Add a legend showing magnetic field color mapping.

    Args:
        figure: Matplotlib figure object to add legend to.
    """
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
    """Extract unique temperature and magnetic field values from oscillation keys.

    Args:
        exp_keys: List of OscillationKey objects.

    Returns:
        Tuple of (temperature_list, magnetic_field_list) with unique values.
    """
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
    """Calculate figure size based on grid dimensions.

    Args:
        n_cols: Number of columns in the grid.
        n_rows: Number of rows in the grid.

    Returns:
        Tuple of (width, height) in inches.
    """
    width = 6 * n_cols
    height = 6 * n_rows
    return width, height


def _create_subplots(fig_size, n_rows, n_cols, hspace, wspace):
    """Create figure and gridspec for fit plots with residuals.

    Each grid position gets two rows: one for the fit plot and one for residuals.

    Args:
        fig_size: Tuple of (width, height) for the figure.
        n_rows: Number of data rows (magnetic field values).
        n_cols: Number of columns (temperature values).
        hspace: Vertical spacing between subplots.
        wspace: Horizontal spacing between subplots.

    Returns:
        Tuple of (figure, gridspec, axes_array).
    """
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
    """Save figure to the processed figures directory.

    Args:
        fig: Matplotlib figure object to save.
        filename: Filename for the saved figure.
        dpi: Resolution in dots per inch.
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
