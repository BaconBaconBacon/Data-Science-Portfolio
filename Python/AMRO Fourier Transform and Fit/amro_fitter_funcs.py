import lmfit as lm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Shove this into a config file
H_PALETTE = {0.5: "tab:red", 3: "tab:green", 7: "tab:orange", 9: "tab:blue"}


class FitAMRO:
    # TODO: Consider removing this functionality.
    FIT_SYMMETRIES = [2, 4, 6, 8]

    def __init__(
        self, amro, fourier_results: pd.DataFrame, save_name: str, verbose=False
    ) -> None:
        self.ft_results_df = fourier_results
        self.amro_df = amro.AMRO
        self.save_name = save_name
        self.save_path = os.path.join("Data", save_name)
        self.act_choices = self._get_H_T_data()

        # TODO: Add Save/Load functionality for fit results
        self.lmfit_results_objs = {}
        self.fit_params_df = pd.DataFrame()
        self.fit_amps_df = pd.DataFrame()
        self.verbose = verbose

        return

    def SineBuilder(
        self, rads, amps: np.array, freqs: np.array, phases: np.array, mean: float | int
    ):
        """ """
        summation = np.sum(
            amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
        )

        return mean * summation + mean

    def ObjFcn(self, params, angle, res_data):
        """
        The sinebuilder is fitted by minimizing this least squares objective function.
        """

        amps_list, freqs_list, phase_list, offset = self._convert_params_to_lists(
            params
        )

        res_model = self.SineBuilder(angle, amps_list, freqs_list, phase_list, offset)

        # Want to minimize least squares
        return (res_model - res_data) ** 2

    def _convert_params_to_lists(
        self, params_obj: lm.Parameters
    ) -> tuple[list, list, list, list]:
        """
        Ensures the parameters are correctly ordered. Aside from the 'mean' parameter, each
        'phase' and 'freq' are paired based on the 'freq' value.

        Maybe keeping track of the freq's involved can speed this up.

        TODO: Would be nice to front-load this so it's not called everytime the objective
        function is called.
        """
        params_dict = params_obj.valuesdict()

        # phase_list = [params_dict[f"phase{f_str}"] for f_str in self.current_f_list_str]
        # amps_list = [params_dict[f"amp{f_str}"] for f_str in self.current_f_list_str]

        amps_phase = [
            (params_dict[f"amp{f_str}"], params_dict[f"phase{f_str}"])
            for f_str in self.current_f_list_str
        ]
        amps_list, phase_list = zip(*amps_phase)
        # for key in params_obj.keys():
        #     if "amp" in key:
        #         amps_list.append(params_obj[key].value)
        #     elif "freq" in key:
        #         freqs_list.append(params_obj[key].value)
        #     elif "phase" in key:
        #         phase_list.append(params_obj[key].value)

        return (
            np.asarray(amps_list),
            np.asarray(self.current_f_list.copy()),
            np.asarray(phase_list),
            params_dict["mean"],
        )

    def FitAMROData(
        self, ACT: str, H: int | float, T: int | float  # f_list: list,
    ) -> lm.minimizer.MinimizerResult:
        """ """

        # Select the experimental data to be fitted

        fit_df = self.GetAMROData(ACT, H, T)
        guess_df = self._get_freqs_guesses(ACT, H, T)

        # Extract data we are going to fit
        x = fit_df["Sample Position (rads)"].values
        y = fit_df["Res. (ohm-cm)"].values
        y_mean = y.mean()

        initial_params = self._initialize_parameters(guess_df, y_mean)

        # Perform the minimization
        minner = lm.Minimizer(self.ObjFcn, initial_params, fcn_args=(x, y))
        results = minner.minimize()

        if self.verbose:
            print(lm.fit_report(results, show_correl=False))
        del self.current_f_list
        del self.current_f_list_str
        return results

    def _initialize_parameters(
        self, guesses: pd.DataFrame, mean: float
    ) -> lm.Parameters:
        """"""

        freqs_list = guesses["freqs (cycles/rot)"].unique()

        # Generate a Parameters ordered dictionary, to which we add Parameter objects
        initial_p_guesses = lm.Parameters()
        initial_p_guesses.add("mean", value=mean)

        # Append all Parameter objects, except for the last one (must deal with appended 2)
        i = 0
        while i < (len(freqs_list) - 1):  # Extra 2 will always be at the end of f_list
            freq = int(freqs_list[i])
            self._add_freq_parameter(freq, initial_p_guesses, mean, guesses)
            i += 1

        # Deal with the final element
        # TODO: Surely this is unnecessary?
        if len(freqs_list) == len(guesses):
            # If nothing has been appended, just add the information for the final FT guess
            freq = int(freqs_list[i])
            self._add_freq_parameter(freq, initial_p_guesses, mean, guesses)

        return initial_p_guesses

    def PlotFits(self, act_choice: str, figsize=None, show_residuals=True):
        """"""
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context("notebook")

        data_df = self.amro_df.query('ACT_str=="{}"'.format(act_choice))

        T_vals = data_df["T"].unique()
        n_cols = len(T_vals)

        H_vals = data_df["H"].unique()
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
                hspace=0.05,
                wspace=0.3,
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

                x = plot_df["Sample Position (rads)"].values
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

                residuals = y - y_fit

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

                    # Plot data and fit
                    sns.scatterplot(x=x, y=y, color=H_PALETTE[H], ax=ax_fit)
                    sns.lineplot(x=x, y=y_fit, color="black", ax=ax_fit)
                    ax_fit.set_xlabel("")
                    ax_fit.tick_params(labelbottom=False)

                    # Plot residuals
                    sns.scatterplot(x=x, y=residuals, ax=ax_resid, color="black")
                    # result.plot_residuals(ax=ax_resid)

                    # Add title to fit plot
                    ax_fit.set_title(f"Position ({i}, {j})", fontsize=10)
                    # ax_fit.legend(fontsize=8)
                    # ax_resid.legend(fontsize=8)

                else:
                    ax = axes[i, j]
                    sns.scatterplot(x=x, y=y, color=H_PALETTE[H], ax=ax)
                    sns.lineplot(x=x, y=y_fit, color=H_PALETTE[H], ax=ax)
                    ax.set_title(f"Position ({i}, {j})", fontsize=10)
                    # ax.legend(fontsize=8)

        plt.tight_layout()
        return fig, axes

    def GetAMROData(self, act=None, h=None, t=None):
        conds = []
        if act is not None:
            conds.append('ACT_str == "{}"'.format(act))
        if h is not None:
            conds.append("H =={}".format(h))
        if t is not None:
            conds.append("T=={}".format(t))

        q = " & ".join(conds)

        return self.amro_df.query(q) if q else self.amro_df

    def _get_freqs_guesses(self, act, h, t):
        q = 'ACT_str == "{}" & H == {} & T== {}'.format(  #' & `freqs (cycles/rot)` in @f_list'.format(
            act, h, t
        )
        guess_df = self.ft_results_df.query(q)
        self.current_f_list = guess_df["freqs (cycles/rot)"].values
        self.current_f_list_str = [str(f) for f in self.current_f_list]

        return guess_df

    def _add_freq_parameter(
        self,
        frequency: int,
        params: lm.Parameters,
        mean_val: float,
        guesses_df: pd.DataFrame,
    ) -> None:
        """
        Forcing all amplitudes to be positive, negative values show up as
        a pi-sized phase offset.
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
            value=temp_df["phase"].values[0],
            min=-2 * np.pi,
            max=2 * np.pi,
        )
        return

    def FitACTExperiment(self, label, f_rank_min, f_ratio_min_ratio, f_max):
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
            else:
                print(
                    "Already fitted {}, {}K, {}T.".format(label, T_label, H_label)
                    + "Skipping..."
                )
        # replace all NaNs as zeros, assuming the problem was a mismatch between requested frequencies and FT guesses for the given experiment

        return

    def _get_init_params(self, act: str, T: str, H: str) -> pd.DataFrame:
        q = 'ACT_str == "{}" & H == {} & T == {}'.format(act, H, T)
        q += "& `freqs (cycles/rot)`in @self.FIT_SYMMETRIES"

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

    def _pack_act_fit_results(
        self,
        lmfit_result,
        # fits: pd.DataFrame,
        # results: dict,
        # fitted_amps: pd.DataFrame,
        act_label: str,
        t_label: str,
        h_label: str,
        # f_list: list,
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
            }  # ,
            # index=[0],
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
            with pd.option_context("future.no_silent_downcasting", True):
                params_df = params_df.fillna(0)
            print("Filling NaN parameters.")

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

    def _get_H_T_data(self):
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
