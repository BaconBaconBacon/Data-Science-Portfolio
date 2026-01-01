"""
    Plotting functionality for the utils_fitter.py file.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..config.settings import (
    PROCESSED_DATA_PATH,
    FINAL_DATA_PATH,
    H_PALETTE,
    PROCESSED_FIGURES_PATH,
)
from matplotlib.patches import Patch
from ..utils import utils as u


sns_context = "poster"
hspace = 0.05
wspace = 0.3
context_font_scale = 1


def _plot_fits_with_residuals(
    fitter,
    act_choice: str,
    h_choices=None,
    t_choices=None,
    figsize=None,
    y_scale=1,
    y_label="Res. ch (ohm-cm)",
    x_label="Angle (deg)",
):
    """
    Plotter to display finished fits over AMRO data, with the option
    to show the residuals. Not intended to be for a polished final
    version.
    """
    # Set seaborn style
    # TODO: Use set_context or rcParams to set the plotting parameters like hspace and wspace

    sns.set_style("whitegrid")
    sns.set_context(sns_context)  # , font_scale=context_font_scale)

    data_df = u.query_data_frame(
        fitter.amro_df, act=act_choice, h=h_choices, t=t_choices
    )
    t_vals, h_vals = _get_plot_labels(data_df)

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
        data_df,
        fitter,
        act_choice,
        h_vals,
        t_vals,
        fig,
        gs,
        axes,
        y_scale,
        x_label,
        y_label,
    )

    # Generate legend
    _generate_legend(fig)

    return fig, axes


def _plot_fits_with_residuals_uohm(
    fitter, act_choice: str, h_choices=None, t_choices=None, figsize=None
):
    fig, axes = _plot_fits_with_residuals(
        fitter=fitter,
        act_choice=act_choice,
        h_choices=h_choices,
        t_choices=t_choices,
        figsize=figsize,
        y_scale=10**6,
        y_label="Res. (uohm-cm)",
    )
    return fig, axes


def _plot_grid(
    data_df, fitter, act, h_vals, t_vals, fig, gs, axes, y_scale, x_label, y_label
):
    n_rows = len(h_vals)

    # Iterate over grid
    for i, H in enumerate(h_vals):
        for j, T in enumerate(t_vals):

            ax_fit = fig.add_subplot(gs[i * 2, j])
            ax_resid = fig.add_subplot(gs[i * 2 + 1, j], sharex=ax_fit)
            axes[i, j] = (ax_fit, ax_resid)

            x_data, x_plot, y_data = _get_plot_points(data_df, act, H, T)

            params = _get_fit_params(fitter, act, H, T)
            y_fit = _calculate_model_values(x_data, params)
            if y_fit is None:
                print("No lmfit Result found for {} {}K, {}T".format(act, T, H))
                return

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


def _plot_fit_over_data(x_plot, y, y_fit, ax, color):
    sns.scatterplot(
        x=x_plot,
        y=y,
        color=color,
        ax=ax,
        linewidth=0,
    )
    sns.lineplot(x=x_plot, y=y_fit, color="black", ax=ax)


def _plot_bad_fits(fitter, act_choice: str):

    # TODO: Need to fixed the nested dictionary usage
    # Failed fit labels might benefit from using a DataFrame
    try:
        h_labels = fitter.failed_fit_labels[act_choice].keys()
    except KeyError:
        print("No bad fits found for {}".format(act_choice))
        return None, None
    t_labels = []
    for h_label in h_labels:
        t_labels.append(fitter.failed_fit_labels[act_choice][h_label])

    fig, axes = fitter.plot_fits(act_choice, T_choices=t_labels, H_choices=h_labels)

    plt.show()
    return fig, axes


def _calculate_model_values(x, params):

    (
        amps_list,
        freqs_list,
        phase_list,
        offset,
    ) = params

    # Calculate model's values
    y_fit = u.sine_builder(
        x,
        amps_list,
        freqs_list,
        phase_list,
        offset,
    )
    return y_fit


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


def _get_plot_labels(data_df):

    t_vals = data_df["T"].unique()
    t_vals.sort()

    h_vals = data_df["H"].unique()
    h_vals.sort()
    return t_vals, h_vals


def _get_plot_points(data_df, act_choice, H, T):

    plot_df = u.query_data_frame(data_df, act=act_choice, h=H, t=T)
    x = plot_df["Sample Position (rads)"].values
    x_plot = plot_df["Sample Position (deg)"].values
    y_plot = plot_df["Res. (ohm-cm)"].values

    return x, x_plot, y_plot


def _get_fit_params(fitter, act, h, t):

    try:
        result = fitter.lmfit_results_objs[act][t][h]
    except KeyError:
        print(f"Fit parameters for {act}. {t}K. {h}T not found")
        return None
    if result is None:
        return None
    fit_params = result.params
    return fitter.convert_params_to_ndarrays(fit_params)


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
