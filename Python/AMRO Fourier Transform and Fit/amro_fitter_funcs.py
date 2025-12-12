import lmfit as lm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


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

    def ObjFcn(self, params, deg, res_data):
        """
        The sinebuilder is fitted by minimizing this least squares objective function.
        """
        amps_list = []
        freqs_list = []
        phase_list = []

        for key in params.keys():
            if "amp" in key:
                amps_list.append(params[key].value)
            elif "freq" in key:
                freqs_list.append(params[key].value)
            elif "phase" in key:
                phase_list.append(params[key].value)

        offset = params["mean"].value
        amps_list = np.asarray(amps_list)
        freqs_list = np.asarray(freqs_list)
        phase_list = np.asarray(phase_list)

        res_model = self.SineBuilder(deg, amps_list, freqs_list, phase_list, offset)

        # Want to minimize least squares
        return (res_model - res_data) ** 2

    def FitAMROData(
        self, ACT: str, H: int | float, T: int | float  # f_list: list,
    ) -> lm.minimizer.MinimizerResult:
        """ """

        # Select the experimental data to be fitted
        q = 'ACT_str == "{}" & H == {} & T== {}'.format(ACT, H, T)
        fit_df = self.amro_df.query(q)

        q = 'ACT_str == "{}" & H == {} & T== {}'.format(  #' & `freqs (cycles/rot)` in @f_list'.format(
            ACT, H, T
        )
        guess_df = self.ft_results_df.query(q)
        # f_list = guess_df["freqs (cycles/rot)"].unique()

        # # TODO: Move this into _initialize_parameters()
        # f_list = self._get_init_freqs(ACT, T, H)

        # Extract data we are going to fit
        x = fit_df["Sample Position (rads)"].values
        y = fit_df["Res. (ohm-cm)"].values
        y_mean = y.mean()

        initial_params = self._initialize_parameters(guess_df, y_mean)

        # Perform the minimization
        minner = lm.Minimizer(self.ObjFcn, initial_params, fcn_args=(x, y))
        # kws = {'options': {'maxiter':5000}}

        results = minner.minimize()

        if self.verbose:
            print(lm.fit_report(results, show_correl=False))

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
            self._add_freq_guess(freq, initial_p_guesses, mean, guesses)
            i += 1

        # Deal with the final element
        # TODO: Surely this is unnecessary?
        if len(freqs_list) == len(guesses):
            # If nothing has been appended, just add the information for the final FT guess
            freq = int(freqs_list[i])
            self._add_freq_guess(freq, initial_p_guesses, mean, guesses)

        return initial_p_guesses

    def PlotFits(self, act_choice: str):
        data_df = self.amro_df.query('ACT_str=="{}"'.format(act_choice))
        params_df = self.fit_params_df.query('ACT=="{}"'.format(act_choice))

        T_vals = data_df["T"].unique()
        H_vals = data_df["H"].unique()

        return

    def _add_freq_guess(
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
            print("Fitting {}, {}K, {}T.".format(label, T_label, H_label))

            results_obj = self.FitAMROData(label, H_label, T_label)

            # Make use of the class variables
            self._pack_act_fit_results(
                results_obj,
                label,
                H_label,
                T_label,  # , all_fits_df, all_results_dict, fitted_amps
            )

        # replace all NaNs as zeros, assuming the problem was a mismatch between requested frequencies and FT guesses for the given experiment

        return  # all_fits_df, all_results_dict, fitted_amps

    def _get_init_params(self, act: str, T: str, H: str) -> pd.DataFrame:
        q = 'ACT_str == "{}" & H == {} & T == {}'.format(act, H, T)
        q += "& `freqs (cycles/rot)`in @self.FIT_SYMMETRIES"

        f_info = self.ft_results_df.query(q)[["freqs (cycles/rot)", "amp_ratio"]]

        return f_info

    def _get_init_freqs(self, act: str, T: str, H: str) -> pd.DataFrame:
        f_info = self._get_init_params(act, T, H)
        return f_info["freqs (cycles/rot)"].values

    def _check_if_already_fitted(self) -> bool:
        """
        Check if the experiment has already been fitted
        """
        return

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
