import os

import lmfit as lm
import numpy as np
import pandas as pd
import pickle
from . import utils_misc as u
from plotting.plotting_fitter import (
    _plot_fits_with_residuals,
    _plot_fits_with_residuals_uohm,
    _plot_bad_fits,
    _save_plot,
)

from config.settings import (
    PROCESSED_DATA_PATH,
    FINAL_DATA_PATH,
    PROCESSED_FIGURES_PATH,
)


class AMROFitter:
    # FIT_SYMMETRIES = [2, 4]

    def __init__(
        self,
        amro_df: pd.DataFrame,
        fourier_results: pd.DataFrame,
        save_name: str,
        min_amp_ratio=0.2,
        max_freq=10,
        force_four_and_two_sym=False,
        verbose=False,
        if_save_file_exists_overwrite=False,
    ) -> None:
        """
        min_amp_ratio : Amplitude must be at this fraction or more of the strongest FT guess
        max_freq : Use to avoid fitting noise
        """

        # Fit Param filter values
        self.min_amp_ratio = min_amp_ratio
        self.max_freq = max_freq
        self.amro_df = amro_df
        self.force_four_and_two_sym = force_four_and_two_sym
        self.verbose = verbose
        self.overwrite = if_save_file_exists_overwrite

        self.ft_results_df = self._filter_guess_params(fourier_results)
        self.act_choices = self._get_h_t_values()

        # Save name info
        s = "_ratio_{}_maxf_{}_".format(min_amp_ratio, max_freq)
        self.save_name = save_name + s.replace(".", "p")
        self._create_save_paths()
        self._read_or_initialize()

        # TODO: Address nested list storage
        self.failed_fit_labels = {}
        return

    def _create_save_paths(self):
        s = self.save_name + "_fit_params.csv"
        self.fit_params_fp = FINAL_DATA_PATH / s
        s = self.save_name + "_results.pkl"
        self.results_fp = FINAL_DATA_PATH / s
        # may be redundant, could probably get it from the fit_params_df
        s = self.save_name + "_fit_amps.csv"
        self.fit_amps_fp = FINAL_DATA_PATH / s
        return

    def _obj_func(self, params, angle, res_data):
        """
        The sinebuilder is fitted by minimizing this least squares objective function.
        """

        amps_list, freqs_list, phase_list, offset = (
            self._fast_convert_params_to_ndarrays(params)
        )

        res_model = u.SineBuilder(angle, amps_list, freqs_list, phase_list, offset)

        # Want to minimize least squares
        return res_model - res_data

    def fit_act_experiment(self, label):
        """"""

        for T_label, H_label in self.act_choices[label]:
            if self._check_if_already_fitted(label, T_label, H_label):
                print(
                    "Already fitted {}, {}K, {}T.".format(label, T_label, H_label)
                    + " Skipping..."
                )
                continue
            else:
                print("Fitting {}, {}K, {}T.".format(label, T_label, H_label))

                results_obj = self.fit_amro_data(label, H_label, T_label)

                self._pack_act_fit_results(
                    results_obj,
                    act_label=label,
                    h_label=H_label,
                    t_label=T_label,  # , all_fits_df, all_results_dict, fitted_amps
                )
                self._save_to_disk()
        return

    def fit_amro_data(
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
        fit_df = u.QueryDataFrame(self.amro_df, act=ACT, h=H, t=T)
        guess_df = self._get_freqs_guesses(ACT, H, T)

        self.current_f_list = guess_df["freqs (cycles/rot)"].unique()

        # Extract data we are going to fit
        x = fit_df["Sample Position (rads)"].values
        y = fit_df["Res. (ohm-cm)"].values

        y_norm, y_scale = self._normalize_data(y)
        y_mean = y_norm.mean()

        initial_params = self._initialize_parameters(guess_df, y_mean)

        # Perform the minimization
        minner = lm.Minimizer(self._obj_func, initial_params, fcn_args=(x, y_norm))
        results = minner.minimize()

        # Check if the covariant matrix is singular
        if self._is_covar_matrix_singular(results):
            print("Covariance matrix is singular.")
            results = self._refit(initial_params, x, y_norm)
            if self._is_covar_matrix_singular(results):
                print("Covariance matrix is still singular. Setting errors to np.inf")
                bad_fit_params = []
                for param_name in results.params.keys():
                    bad_fit_params.append(param_name)
                    results.params[param_name].stderr = np.inf
                print(f"Errors for {bad_fit_params} could not be calculated.")
                self._record_failed_fit(ACT, H, T)
        results.params["mean"].value *= y_scale
        results.params["mean"].stderr *= y_scale

        if self.verbose:
            print("\n", lm.fit_report(results, show_correl=False), "\n")
        del self.current_f_list

        return results

    def _record_failed_fit(self, act_label, h_label, t_label) -> None:
        # TODO : Address nested dict storage method
        if act_label not in self.failed_fit_labels.keys():
            self.failed_fit_labels[act_label] = {h_label: [t_label]}
        else:
            self.failed_fit_labels[act_label][h_label].append(t_label)
        return

    def _refit(
        self, init_params: lm.Parameters, x_data: list, y_normalized: list
    ) -> lm.minimizer.MinimizerResult:
        print("Removing parameter bounds and re-fitting.")
        for name in init_params:
            init_params[name].min = -np.inf
            init_params[name].max = np.inf
        minner = lm.Minimizer(
            self._obj_func, init_params, fcn_args=(x_data, y_normalized)
        )
        return minner.minimize()

    def _add_parameter(
        self,
        frequency: int,
        params: lm.Parameters,
        mean_val: float,
        guesses_df: pd.DataFrame,
    ) -> None:
        """
        Forcing all amplitudes to be positive, negative values show up as
        a pi-sized phase offset.

        The y_scale scaling is to improve the error estimation of the fitter.

        We divide the magnitude by the mean in order to align with the way
        the _sine_builder function works.
        """
        temp_df = guesses_df.query("`freqs (cycles/rot)` == {}".format(frequency))
        if temp_df.shape[0] == 0:
            print(
                "FT guess for {} not found. Setting initial guesses to zero.".format(
                    frequency
                )
            )
            temp_df = pd.DataFrame(
                {
                    "freqs": frequency,
                    "mag (ohm-cm)": 0,
                    "freqs (cycles/rot)": 0,
                    "phase": 0,
                },
                index=[0],
            )

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

        # Generate a Parameters ordered dictionary, to which we add Parameter objects
        initial_p_guesses = lm.Parameters()
        initial_p_guesses.add("mean", value=mean, min=0)

        # Append all Parameter objects, except for the last one (must deal with appended 2)

        for freq in self.current_f_list:
            self._add_parameter(int(freq), initial_p_guesses, mean, guesses)

        return initial_p_guesses

    def _get_freqs_guesses(self, act, h, t):
        guess_df = u.QueryDataFrame(self.ft_results_df, act=act, h=h, t=t)
        self.current_f_list = guess_df["freqs (cycles/rot)"].unique()
        if self.force_four_and_two_sym:
            self.current_f_list = np.append(self.current_f_list, [2, 4])

            # Drop duplicates
            self.current_f_list = list(set(self.current_f_list))

        return guess_df

    def _is_covar_matrix_singular(
        self, results_obj: lm.minimizer.MinimizerResult
    ) -> bool:
        return any(
            results_obj.params[param_name].stderr is None
            for param_name in results_obj.params.keys()
        )

    def _are_residuals_acceptable(self) -> bool:
        # TODO: Check the mean absolute residual against some value. This lets us better track poor fits.
        return

    def get_fitted_parameters(
        self,
        act: str | list | None = None,
        h: int | float | list | None = None,
        t: int | float | list | None = None,
    ) -> pd.DataFrame:
        return u.QueryDataFrame(self.fit_params_df, act=act, h=h, t=t)

    def plot_fits_with_residuals(self, act_choice, **kwargs):
        return _plot_fits_with_residuals(self, act_choice, **kwargs)

    def plot_fits_with_residuals_uohm(self, act_choice, **kwargs):
        return _plot_fits_with_residuals_uohm(self, act_choice, **kwargs)

    def plot_bad_fits(self, act: str):
        return _plot_bad_fits(self, act)

    def save_plot(self, fig, filename, dpi=300):
        _save_plot(fig, filename, dpi=dpi)

    def _save_to_disk(self):
        """ """
        with open(self.results_fp, "wb") as f:
            pickle.dump(self.lmfit_results_objs, f)

        self.fit_params_df.to_csv(self.fit_params_fp, sep=",")
        self.fit_amps_df.to_csv(self.fit_amps_fp, sep=",")
        return

    def convert_params_to_ndarrays(
        self, params_obj: lm.Parameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensures the parameters are correctly ordered. Aside from the 'mean' parameter, each
        'phase' and 'freq' are paired based on the 'freq' value. This function ensures the
        amps and phases are in the correct order relative to the frequencies

        This gets called in the objective function, so the priority is speed. self.current_f_list is calculated
        beforehand.

        """
        params_dict = params_obj.valuesdict()

        freqs_list = []
        for key in params_dict.keys():
            if "freq" in key:
                freqs_list.append(int(params_dict[key]))
        amps_list = []
        phases_list = []
        for freq in freqs_list:
            amps_list.append(params_dict["amp{}".format(freq)])
            phases_list.append(params_dict["phase{}".format(freq)])

        return (
            np.asarray(amps_list),
            np.asarray(freqs_list),
            np.asarray(phases_list),
            params_dict["mean"],
        )

    def _fast_convert_params_to_ndarrays(
        self, params_obj: lm.Parameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensures the parameters are correctly ordered. Aside from the 'mean' parameter, each
        'phase' and 'freq' are paired based on the 'freq' value.

        This is somewhat slower than _fast_convert_params_to_ndarrays but it can be called outside
        the class object.

        """
        params_dict = params_obj.valuesdict()
        # TODO: Fix this with the data class. The naming format is 'fragile to naming convention changes'
        amps_phase = [
            (params_dict[f"amp{str(f)}"], params_dict[f"phase{str(f)}"])
            for f in self.current_f_list
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

    def _get_init_params(
        self, act: str | list, T: float | int | list, H: float | int | list
    ) -> pd.DataFrame:

        f_info = u.QueryDataFrame(self.ft_results_df, act=act, h=H, t=T)

        return f_info[["freqs (cycles/rot)", "amp_ratio"]]

    def _get_init_freqs(self, act: str, T: str, H: str) -> list:
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
            os.path.exists(self.fit_params_fp)
            & os.path.exists(self.results_fp)
            & os.path.exists(self.fit_amps_fp)
            & (not self.overwrite)
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
                "ACT_str": act_label,
                "T": t_label,
                "H": h_label,
                "f_list": f_list,
            }
        )
        self.fit_amps_df = pd.concat([self.fit_amps_df, f_info])

        # Pack all fit parameters
        # TODO: Fix this with maybe a custom data class
        # - Dynamic keys via string concatenation: var + " err"
        # - Mixes domain fields with fit parameters
        # - Converted to DataFrame (should be a class)

        params_dict = {
            "ACT_str": act_label,
            "H": float(h_label),
            "T": float(t_label),
            "chi squared": float(lmfit_result.redchi),
        }

        for var in var_names:
            params_dict[var] = param_results[var].value
            params_dict[var + " err"] = param_results[var].stderr

        params_df = pd.DataFrame(params_dict, index=[0])

        self.fit_params_df = (
            pd.concat([self.fit_params_df, params_df], ignore_index=True)
            .reset_index(drop=True)
            .fillna(0)
        )

        # Add lmfit Results object to dictionary
        # TODO: This nested dict builder is a good candidate for a utils function
        # OR Just pre-populate this dictionary with the unique ACT/H/T permutations
        if act_label not in self.lmfit_results_objs.keys():
            self.lmfit_results_objs[act_label] = {}
        if t_label not in self.lmfit_results_objs[act_label].keys():
            self.lmfit_results_objs[act_label][t_label] = {}
        self.lmfit_results_objs[act_label][t_label][h_label] = lmfit_result

        return

    def _get_h_t_values(self):
        """"""
        h_t_dict = {}
        grouped = self.amro_df[["ACT_str", "H", "T"]].drop_duplicates()

        for act in grouped["ACT_str"].unique():
            tmp_list = []
            for _, row in grouped.query('ACT_str=="{}"'.format(act)).iterrows():
                tmp_list.append((float(row["T"]), float(row["H"])))
            h_t_dict[act] = tmp_list
        return h_t_dict
