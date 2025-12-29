"""
    Plotting functionality for the utils_fitter.py file.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from utils import utils_misc as u
import numpy as np
from matplotlib.patches import Patch

from config.settings import (
    PROCESSED_DATA_PATH,
    FINAL_DATA_PATH,
    H_PALETTE,
    PROCESSED_FIGURES_PATH,
)


def _plot_fits(
    fitter,
    act_choice: str,
    figsize=None,
    show_residuals=True,
    y_scale=1,
    y_label="Res. ch (ohm-cm)",
    x_label="Angle (deg)",
    sns_context="poster",
    delta=False,
    marker_size=60,
    hspace=0.05,
    wspace=0.3,
    context_font_scale=1,
    H_choices=None,
    T_choices=None,
    save_fig=False,
):
    """
    Plotter to display finished fits over AMRO data, with the option
    to show the residuals. Not intended to be for a polished final
    version.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context(sns_context, font_scale=context_font_scale)

    data_df = u.QueryDataFrame(fitter.amro_df, act=act_choice, h=H_choices, t=T_choices)

    t_vals = data_df["T"].unique()
    t_vals.sort()
    n_cols = len(t_vals)

    h_vals = data_df["H"].unique()
    h_vals.sort()
    n_rows = len(h_vals)
    # Calculate figure size if not provided
    if figsize is None:
        width = 4 * n_cols
        height = (6 if show_residuals else 4) * n_rows
        figsize = (width, height)

    # Create subplots
    if show_residuals:
        # Each position gets 2 rows: one for fit, one for residuals
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            n_rows * 2,
            n_cols,
            hspace=hspace,
            wspace=wspace,
            height_ratios=[3, 1] * n_rows,
        )
        axes = np.empty((n_rows, n_cols), dtype=object)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

    # Iterate over grid
    for i, H in enumerate(h_vals):
        for j, T in enumerate(t_vals):

            try:
                result = fitter.lmfit_results_objs[act_choice][T][H]
            except KeyError as e:
                print("ACT", act_choice, "T", T, "H", H)
                print("hvals", h_vals)
                print("tvals", t_vals)
                print(fitter.lmfit_results_objs[act_choice].keys())
                print(data_df)
                # print(self.lmfit_results_objs[act_choice][T].keys())
                raise e
            fit_params = result.params

            plot_df = u.QueryDataFrame(data_df, act=act_choice, h=H, t=T)

            x = plot_df["Sample Position (rads)"].values
            x_plot = plot_df["Sample Position (deg)"].values

            y = plot_df["Res. (ohm-cm)"].values

            df = fitter._get_freqs_guesses(act_choice, H, T)
            fitter.current_f_list = df["freqs (cycles/rot)"].unique()

            (
                amps_list,
                freqs_list,
                phase_list,
                offset,
            ) = fitter._convert_params_to_lists(fit_params)
            del fitter.current_f_list

            y_fit = fitter._sine_builder(
                x,
                amps_list,
                freqs_list,
                phase_list,
                offset,
            )

            y = y * y_scale
            y_fit = y_fit * y_scale
            residuals = (y - y_fit) / y.mean() * 100  #  y - y_fit  #

            if delta:
                data_mean = np.mean(y)
                y = y - data_mean
                y_fit = y_fit - data_mean

            if result is None:
                print("No lmfit Result found for {} {}K, {}T".format(act_choice, T, H))
                continue

            if show_residuals:
                # Create axes for fit and residuals
                ax_fit = fig.add_subplot(gs[i * 2, j])
                ax_resid = fig.add_subplot(gs[i * 2 + 1, j], sharex=ax_fit)
                axes[i, j] = (ax_fit, ax_resid)

                ax_fit.set_xticks([0, 90, 180, 270, 360])
                ax_resid.set_xticks([0, 90, 180, 270, 360])

                # Plot data and fit
                sns.scatterplot(
                    x=x_plot,
                    y=y,
                    color=H_PALETTE[H],
                    ax=ax_fit,
                    linewidth=0,
                    s=marker_size,
                )
                sns.lineplot(x=x_plot, y=y_fit, color="black", ax=ax_fit)
                ax_fit.set_xlabel("")
                ax_fit.tick_params(labelbottom=False)

                # Plot residuals
                sns.scatterplot(
                    x=x_plot,
                    y=residuals,
                    ax=ax_resid,
                    color="black",
                    linewidth=0,
                    s=marker_size,
                )

                # x labels
                if i == (n_rows - 1):
                    ax_resid.set(xlabel=x_label)
                else:
                    ax_resid.set(xlabel="")
                    ax_resid.tick_params(labelbottom=False)

                # titles
                if i == 0:
                    ax_fit.set_title(str(T).replace(".0", "") + "K")

                # y labels
                if j == 0:
                    ax_fit.set(ylabel=y_label)
                    ax_resid.set(ylabel="(% wrt Mean)")
                else:
                    ax_fit.set(ylabel=None)
            else:
                ax = axes[i, j]
                sns.scatterplot(
                    x=x_plot,
                    y=y,
                    color=H_PALETTE[H],
                    ax=ax,
                    linewidth=0,
                    s=marker_size,
                )
                sns.lineplot(x=x_plot, y=y_fit, color=H_PALETTE[H], ax=ax)
                ax.set_title(f"Position ({i}, {j})", fontsize=10)
                ax.set_xticks([0, 90, 180, 270, 360])

                # ax.legend(fontsize=8)
                if i == (n_rows - 1):
                    ax.set(xlabel=x_label)
                if j == 0:
                    ax.set(ylabel=y_label)
    # Generate legend
    legend_elements = [
        Patch(facecolor=color, label=str(label)) for label, color in H_PALETTE.items()
    ]

    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        title="H (T)",
    )

    if save_fig:
        fn = act_choice + "_figure" + fitter.save_name + ".pdf"
        fp = PROCESSED_FIGURES_PATH / fn
        fig.savefig(
            fp,
            dpi=300,
            transparent=False,
            bbox_inches="tight",
        )
    return fig, axes


def _plot_bad_fits(fitter):

    # TODO: Fix this nested dictionary stuff. When the data has it's own class, just iterate over experiments,
    # checking the relevant bool attribute.
    act_labels = fitter.failed_fit_labels.keys()
    for act_label in act_labels:
        h_labels = fitter.failed_fit_labels[act_label].keys()
        t_labels = []
        for h_label in h_labels:
            t_labels.append(fitter.failed_fit_labels[act_label][h_label])
        _, _ = fitter.plot_fits(act_label, T_choices=t_labels, H_choices=h_labels)
    plt.show()
    return
