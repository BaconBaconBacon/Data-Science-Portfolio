import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from headers import HEADER_ERR_SUFFIX, FITTER_PARAM_NAMES, HEADER_TEMP
import helpers as he

# from numpy.polynomial.chebyshev import chebfit, chebval
import interpolation as ip


def flip_legend(ax, title, loc="best", ncols=1, prop=None):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=title, loc=loc, ncol=ncols, prop=prop)


def plot_cheb_fits(
    data_df: pd.DataFrame,
    coeffs: dict,
    residuals: dict,
    temps: list,
    # cheb_degrees: int = 15,
    x_label: str = "Magnetic Flux Density (G)",
    y_label: str = "Sym Res. (ohm-cm)",
    mystery_x_scaling=10000,
    mystery_y_scaling=1000,
    save_fig=False,
):
    fig, axs = plt.subplots(2, 4, sharex=True)
    # fig, axs = plt.subplots(2, 4, sharex=True, sharey=False, gridspec_kw={'height_ratios': [1, 0.5]})

    H_ticks = np.linspace(0, 9, 4)

    fig.set_size_inches(15, 5)

    i = 0
    while i < len(temps):
        T = temps[i]

        sns.set_context("talk")
        fig, axs = plt.subplots(
            2, 1, sharex=True, sharey=False, gridspec_kw={"height_ratios": [1, 0.3]}
        )
        fig.set_size_inches(4, 5)

        fit_ax = axs[0]  # [i]
        res_ax = axs[1]  # [i]

        T = temps[i % len(temps)]

        plot_df = data_df.query("T == {}".format(T))

        plot_x = plot_df[x_label].values / mystery_x_scaling
        plot_y = plot_df[y_label].values * mystery_y_scaling

        # Plot resistivity
        # Data
        sns.scatterplot(
            x=plot_x, y=plot_y, marker="x", ax=fit_ax, color="r", label="Data"
        )
        # PLot Cheb
        sns.lineplot(
            x=plot_x,
            y=ip.evaluate_cheb(plot_x * mystery_x_scaling, coeffs[T])
            * mystery_y_scaling,
            ax=fit_ax,
            color="b",
            label="Cheb. Fit",
        )
        # Residuals
        sns.scatterplot(x=plot_x[1:], y=residuals[T][1:], ax=res_ax)

        # Formatting
        # ax.set_xticklabels(xticklabels, ha="right")

        res_ax.set_xticks(H_ticks)  # , rotation = 45,  ha="right")
        fit_ax.set_ylabel(r"$\rho$ (mohm-cm)")
        res_ax.set_ylabel("Res. (%)")

        res_ax.set_xlabel(None)
        res_ax.set_xlabel(r"B (T)")
        # res_ax.set_ylabel(r'$\rho$ (mohm-cm)')

        res_ax.grid()
        fit_ax.grid()
        fit_ax.legend(loc=(0.075, 0.15))

        plot_title = str(T)
        fit_ax.set_title(plot_title + "K", size=20)

        # Delete repeated axes labels
        # if i >0:
        #     fit_ax.set_ylabel(None)
        #     res_ax.set_ylabel(None)
        # Modify labels and legends for visual clarity
        if T == 2:
            fit_ax.legend().remove()
            fit_ax.set_ylabel(None)

        elif T == 5:
            fit_ax.legend().remove()
            res_ax.set_ylabel("Res. (%)")
        elif T == 10:
            fit_ax.set_xlabel(None)
            res_ax.set_xlabel(None)

            fit_ax.set_ylabel(None)
            res_ax.set_ylabel(None)

        elif T == 15:
            fit_ax.legend().remove()
            fit_ax.set_xlabel(None)
            res_ax.set_xlabel(None)

            res_ax.set_ylabel("Res. (%)")

            # if i < 3:
        #     fit_ax.legend().remove()
        if save_fig:
            fig.savefig(
                "MR Cheb Fits {}K.pdf".format(T),
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )
        i += 1

    scale_factor = 0.9
    # fig.set_size_inches(18*scale_factor,7*scale_factor)
    # plt.subplots_adjust(hspace=0.12, wspace=0.39)
    plt.show()


def plot_lmfit_params(
    param_df: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots fit parameters and their error bars."""
    n_cols = 3
    n_rows = 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=False)
    axs = he.flatten_list(axs)
    print(axs)
    # TODO: make this build a grid programmatically.
    i = 0
    for col in param_df.columns:

        if col in FITTER_PARAM_NAMES:
            print(i)
            cur_ax = axs[i]
            x_vals = param_df[HEADER_TEMP].values
            y_vals = param_df[col].values
            y_errs = param_df[col + HEADER_ERR_SUFFIX].values
            cur_ax.errorbar(
                x=x_vals, y=y_vals, yerr=y_errs, linestyle="None", marker="o"
            )
            cur_ax.set(ylabel=col, xlabel="T (K)")
            i += 1

    return fig, axs
