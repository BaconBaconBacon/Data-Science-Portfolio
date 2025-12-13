import os

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns


from matplotlib.patches import Patch

# TODO: Shove this into a config file
H_PALETTE = {0.5: "tab:red", 3: "tab:green", 7: "tab:orange", 9: "tab:blue"}


class AMROFitter:
    # TODO: Consider removing this functionality.
    # FIT_SYMMETRIES = [2, 4, 6, 8]

    def __init__(
        self,
        amro,
        fourier_results: pd.DataFrame,
        save_name: str,
        min_amp_ratio=0.2,
        max_freq=10,
        verbose=False,
    ) -> None:
        """
        min_amp_ratio :Amplitude must be at this fraction or more of the strongest FT guess
        max_freq : Use to avoid fitting noise
        """

        # Fit Param filter values
        self.min_amp_ratio = min_amp_ratio
        self.max_freq = max_freq
        self.amro_df = amro.AMRO

        self.ft_results_df = self._filter_guess_params(fourier_results)
        self.act_choices = self._get_H_T_values()

        # Save name info
        s = "_ratio_{}_maxf_{}_".format(min_amp_ratio, max_freq)
        self.save_name = save_name + s.replace(".", "p")
        self._create_save_paths()

        self._read_or_initialize()
        self.verbose = verbose

        return

    def _create_save_paths(self):
        self.fit_params_fp = os.path.join("Data", self.save_name + "_fit_params.csv")
        self.results_fp = os.path.join("Data", self.save_name + "_results.pkl")
        # may be redundant, could probably get it from the fit_params_df
        self.fit_amps_fp = os.path.join("Data", self.save_name + "_fit_amps.csv")
        return

    def SineBuilder(
        self, rads, amps: np.array, freqs: np.array, phases: np.array, mean: float | int
    ):
        """Returns a Fourier series consisting of sine terms and an offset."""
        summation = np.sum(
            amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
        )

        return mean * (summation + 1)

    def ObjFcn(self, params, angle, res_data):
        """
        The sinebuilder is fitted by minimizing this least squares objective function.
        """

        amps_list, freqs_list, phase_list, offset = self._convert_params_to_lists(
            params
        )

        res_model = self.SineBuilder(angle, amps_list, freqs_list, phase_list, offset)

        # Want to minimize least squares
        # return (res_model - res_data) ** 2
        return res_model - res_data

    def FitACTExperiment(self, label):
        """"""

        for T_label, H_label in self.act_choices[label]:
            if not self._check_if_already_fitted(label, T_label, H_label):
                print("Fitting {}, {}K, {}T.".format(label, T_label, H_label))

                results_obj = self.FitAMROData(label, H_label, T_label)

                # Make use of the class variables
                self._pack_act_fit_results(
                    results_obj,
                    label,
                    H_label,
                    T_label,  # , all_fits_df, all_results_dict, fitted_amps
                )
                self._save_to_disk()
            else:
                print(
                    "Already fitted {}, {}K, {}T.".format(label, T_label, H_label)
                    + " Skipping..."
                )

        return

    def FitAMROData(
        self, ACT: str, H: int | float, T: int | float  # f_list: list,
    ) -> lm.minimizer.MinimizerResult:
        """
        This function prepares the data of the AMR oscillation for fitting,
         then fits it. To improve the fitting, the data and parameters are
         scaled by the maximum resistivity of the AMRO oscillation. After
         fitting, they are scaled back up. This is to avoid the minimzer
         forgoing an error estimation for any given fit parameter.
        """

        # Select the experimental data to be fitted

        fit_df = self.GetAMROData(ACT, H, T)
        guess_df = self._get_freqs_guesses(ACT, H, T)

        # Extract data we are going to fit
        x = fit_df["Sample Position (rads)"].values
        y = fit_df["Res. (ohm-cm)"].values

        y_norm, y_scale = self._normalize_data(y)
        y_mean = y_norm.mean()

        initial_params = self._initialize_parameters(guess_df, y_mean)

        # TODO: some parameters have No errors, causing NaNs later on.
        # Perform the minimization
        minner = lm.Minimizer(self.ObjFcn, initial_params, fcn_args=(x, y_norm))
        results = minner.minimize()

        # Check if the covariant matrix is singular
        for param_name in results.params.keys():
            if results.params[param_name].stderr is None:
                print(results.params)
                print(
                    "Fit's covariance matrix is singular. Removing bounds and re-fitting."
                )
                for name in results.params:
                    # if name.startswith("phase"):
                    results.params[name].min = -np.inf
                    results.params[name].max = np.inf
                results = minner.minimize()

        # Scale the fit parameters back up
        if results.params["mean"].stderr is not None:
            results.params["mean"].stderr *= y_scale
        else:
            raise ValueError("Mean parameter's error could not be calculated.")

        for param_name in results.params.keys():
            if param_name == "mean":
                results.params[param_name].value *= y_scale
                if results.params[param_name].stderr is not None:
                    results.params[param_name].stderr *= y_scale
                else:
                    print(results.params)
                    raise ValueError(
                        f"{param_name} parameter's error could not be estimated."
                    )
            else:
                if results.params[param_name].stderr is None:
                    print(results.params)
                    raise ValueError(
                        f"{param_name} parameter's error could not be estimated."
                    )

        if self.verbose:
            print(lm.fit_report(results, show_correl=False))
            print()
        del self.current_f_list
        del self.current_f_list_str

        return results

    def _add_parameter(
        self,
        frequency: int,
        params: lm.Parameters,
        mean_val: float,
        guesses_df: pd.DataFrame,
        # y_scale: float,
    ) -> None:
        """
        Forcing all amplitudes to be positive, negative values show up as
        a pi-sized phase offset.

        The y_scale scaling is to improve the error estimation of the fitter.

        We divide the magnitude by the mean in order to align with the way
        the SineBuilder function works.
        """
        temp_df = guesses_df.query("`freqs (cycles/rot)` == {}".format(frequency))
        params.add(
            "amp" + str(frequency),
            value=temp_df["mag (ohm-cm)"].values[0] / mean_val,
            min=0,
        )

        params.add(
            "freq" + str(frequency),
            value=temp_df["freqs (cycles/rot)"].values[0],
            vary=False,
        )

        params.add(
            "phase" + str(frequency),
            value=temp_df["phase"].values[0]  # ,
            # min=-2 * np.pi,
            # max=2 * np.pi,
        )
        return

    def _normalize_data(self, y):
        """Normalize y-values to O(1) for better numerical conditioning."""
        y_scale = np.abs(y).max()
        if y_scale < 1e-10:
            y_scale = 1.0
        return y / y_scale, y_scale

    def _initialize_parameters(
        self,
        guesses: pd.DataFrame,
        mean: float,
    ) -> lm.Parameters:
        """Note that the value of the 'mean' value should have been normalized"""

        freqs_list = self.current_f_list  # guesses["freqs (cycles/rot)"].unique()

        # Generate a Parameters ordered dictionary, to which we add Parameter objects
        initial_p_guesses = lm.Parameters()
        initial_p_guesses.add("mean", value=mean, min=0)

        # Append all Parameter objects, except for the last one (must deal with appended 2)
        i = 0
        while i < (len(freqs_list) - 1):  # Extra 2 will always be at the end of f_list
            freq = int(freqs_list[i])
            self._add_parameter(freq, initial_p_guesses, mean, guesses)
            i += 1

        # Deal with the final element
        # TODO: Surely this is unnecessary?
        if len(freqs_list) == len(guesses):
            # If nothing has been appended, just add the information for the final FT guess
            freq = int(freqs_list[i])
            self._add_parameter(freq, initial_p_guesses, mean, guesses)

        return initial_p_guesses

    def _get_freqs_guesses(self, act, h, t):
        q = 'ACT_str == "{}" & H == {} & T== {}'.format(  #' & `freqs (cycles/rot)` in @f_list'.format(
            act, h, t
        )
        guess_df = self.ft_results_df.query(q)
        self.current_f_list = guess_df["freqs (cycles/rot)"].unique()
        self.current_f_list_str = [str(f) for f in self.current_f_list]

        return guess_df

    def GetAMROData(self, act=None, h=None, t=None):
        """
        Retrieve a subset of the AMRO data.

        TODO: May be redundant.
        """
        conds = []
        if act is not None:
            conds.append('ACT_str == "{}"'.format(act))
        if h is not None:
            conds.append("H =={}".format(h))
        if t is not None:
            conds.append("T=={}".format(t))

        q = " & ".join(conds)

        return self.amro_df.query(q) if q else self.amro_df

    def QuickPlotFits(
        self,
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
    ):
        """
        Plotter to display finished fits over AMRO data, with the option
        to show the residuals. Not intended to be for a polished final
        version.
        """
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context(sns_context, font_scale=context_font_scale)

        data_df = self.amro_df.query('ACT_str=="{}"'.format(act_choice))

        T_vals = data_df["T"].unique()
        T_vals.sort()
        n_cols = len(T_vals)

        H_vals = data_df["H"].unique()
        H_vals.sort()
        n_rows = len(H_vals)

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
        for i, H in enumerate(H_vals):
            for j, T in enumerate(T_vals):
                result = self.lmfit_results_objs[act_choice][T][H]
                fit_params = result.params  # .valuesdict()
                plot_df = data_df.query("H=={} & T =={}".format(H, T))

                # Note: Do not use vars x_label and y_label for this
                x = plot_df["Sample Position (rads)"].values
                x_plot = plot_df["Sample Position (deg)"].values

                y = plot_df["Res. (ohm-cm)"].values

                _ = self._get_freqs_guesses(act_choice, H, T)
                (
                    amps_list,
                    freqs_list,
                    phase_list,
                    offset,
                ) = self._convert_params_to_lists(fit_params)

                y_fit = self.SineBuilder(
                    x,
                    amps_list,
                    freqs_list,
                    phase_list,
                    offset,
                )

                y = y * y_scale
                y_fit = y_fit * y_scale
                residuals = y - y_fit  # (y - y_fit) / y * 100

                if delta:
                    data_mean = np.mean(y)
                    y = y - data_mean
                    y_fit = y_fit - data_mean
                if result is None:
                    print(
                        "No lmfit Result found for {} {}K, {}T".format(act_choice, T, H)
                    )
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
            Patch(facecolor=color, label=str(label))
            for label, color in H_PALETTE.items()
        ]

        fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(0.8, 0.5),
            title="H (T)",
        )
        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 0.95, 1])
        return fig, axes

    def _save_to_disk(self):
        """ """
        with open(self.results_fp, "wb") as f:
            pickle.dump(self.lmfit_results_objs, f)

        self.fit_params_df.to_csv(self.fit_params_fp, sep=",")
        self.fit_amps_df.to_csv(self.fit_amps_fp, sep=",")
        return

    def _convert_params_to_lists(
        self, params_obj: lm.Parameters
    ) -> tuple[list, list, list, list]:
        """
        Ensures the parameters are correctly ordered. Aside from the 'mean' parameter, each
        'phase' and 'freq' are paired based on the 'freq' value.

        Maybe keeping track of the freq's involved can speed this up.

        TODO: Would be nice to front-load as much as possible so it's
        to minimize run time.
        """
        params_dict = params_obj.valuesdict()

        amps_phase = [
            (params_dict[f"amp{f_str}"], params_dict[f"phase{f_str}"])
            for f_str in self.current_f_list_str
        ]
        amps_list, phase_list = zip(*amps_phase)

        return (
            np.asarray(amps_list),
            np.asarray(self.current_f_list.copy()),
            np.asarray(phase_list),
            params_dict["mean"],
        )

    def _filter_guess_params(self, guess_params: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the fit parameter initial guesses from the Fourier transform
        using the given parameters.

        The amp_ratio is the ratio of the amplitude of a given frequency divided
        by that of the strongest frequency's amplitude.

        The freq is the number of oscillations per rotation of the sample.
        """
        q = "amp_ratio > {} & `freqs (cycles/rot)`<{}".format(
            self.min_amp_ratio, self.max_freq
        )
        return guess_params.query(q)

    def _get_init_params(self, act: str, T: str, H: str) -> pd.DataFrame:
        q = 'ACT_str == "{}" & H == {} & T == {}'.format(act, H, T)

        # TODO: Test that this works
        # q += "& `freqs (cycles/rot)`in @self.FIT_SYMMETRIES"

        f_info = self.ft_results_df.query(q)[["freqs (cycles/rot)", "amp_ratio"]]

        return f_info

    def _get_init_freqs(self, act: str, T: str, H: str) -> pd.DataFrame:
        f_info = self._get_init_params(act, T, H)
        return f_info["freqs (cycles/rot)"].values

    def _check_if_already_fitted(self, act, t, h) -> bool:
        """
        Check if the experiment has already been fitted
        """
        try:
            _ = self.lmfit_results_objs[act][t][h]
            return True
        except KeyError:
            return False

    def _read_or_initialize(self):
        load_conds = (
            (os.path.exists(self.fit_params_fp))
            & os.path.exists(self.results_fp)
            & os.path.exists(self.fit_amps_fp)
        )
        if load_conds:
            print("Loading previous fit results.")
            with open(self.results_fp, "rb") as f:
                self.lmfit_results_objs = pickle.load(f)

            self.fit_params_df = pd.read_csv(self.fit_params_fp)
            self.fit_amps_df = pd.read_csv(self.fit_amps_fp)
        else:
            print("Initializing.")
            self.lmfit_results_objs = {}
            self.fit_params_df = pd.DataFrame()
            self.fit_amps_df = pd.DataFrame()

        return

    def _pack_act_fit_results(
        self,
        lmfit_result,
        act_label: str,
        t_label: str,
        h_label: str,
    ):
        """"""

        # Pack just amplitude parameters
        var_names = lmfit_result.var_names
        param_results = lmfit_result.params

        f_list = self._get_init_freqs(act_label, t_label, h_label)

        f_info = pd.DataFrame(
            {
                "act": act_label,
                "T (K)": t_label,
                "H (T)": h_label,
                "f_list": f_list,
            }
        )
        self.fit_amps_df = pd.concat([self.fit_amps_df, f_info])

        # Pack all fit parameters
        params_dict = {
            "ACT": act_label,
            "H": h_label,
            "T": t_label,
            "chi squared": lmfit_result.redchi,
        }
        for var in var_names:
            params_dict[var] = param_results[var].value
            params_dict[var + " err"] = param_results[var].stderr

        params_df = pd.DataFrame(params_dict, index=[0])
        if params_df.isna().any().any():
            print(params_df)
            raise ValueError("NaN found in fitted parameters.")

        self.fit_params_df = pd.concat(
            [self.fit_params_df, params_df], ignore_index=True
        )

        # Add lmfit Results object to dictionary
        if act_label not in self.lmfit_results_objs:
            self.lmfit_results_objs[act_label] = {}
        if h_label not in self.lmfit_results_objs[act_label]:
            self.lmfit_results_objs[act_label][h_label] = {}
        self.lmfit_results_objs[act_label][h_label][t_label] = lmfit_result

        # results_obj, all_fits_df, all_results_dict, fitted_amps

        return

    def _get_H_T_values(self):
        """"""
        H_T_dict = {}
        grouped = self.amro_df[
            ["ACT_str", "H", "T"]
        ].drop_duplicates()  # .groupby(, as_index=False)

        for act in grouped["ACT_str"].unique():
            tmp_list = []
            for _, row in grouped.query('ACT_str=="{}"'.format(act)).iterrows():
                tmp_list.append((row["T"], row["H"]))
            H_T_dict[act] = tmp_list
        return H_T_dict

    def _test_plot_sinebuilder(self):
        """ """
        # Test function
        f = [4, 2]
        amp = [2, 1]
        phase = [0, 0]
        offset = 1

        x = np.linspace(0, 2 * np.pi, 1000)
        y = self.SineBuilder(x, amp, f, phase, offset)

        plt.scatter(x, y)
        return
