from pathlib import Path

import lmfit as lm
import numpy as np
import pandas as pd
import pickle

from statsmodels.sandbox.distributions.try_pot import mean_residual_life

from amro import FourierResult
from amro.utils import utils as u
from amro.plotting.fitter import (
    _plot_fits_with_residuals,
    _plot_fits_with_residuals_uohm,
    _plot_bad_fits,
    _save_plot,
)

from amro.config.settings import (
    PROCESSED_DATA_PATH,
    FINAL_DATA_PATH,
    PROCESSED_FIGURES_PATH,
    HEADER_ANGLE_DEG,
    HEADER_ANGLE_RAD,
    HEADER_RES_OHM,
    HEADER_MAGNET,
    HEADER_TEMP,
    HEADER_ACT,
    HEADER_FREQ_LIST,
    HEADER_FIT_CHISQ,
    HEADER_FREQ,
    HEADER_MAG_RATIO,
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_PARAM_FREQ_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
    HEADER_PHASE,
    HEADER_MAG,
)
from amro.data import (
    FitResult,
    OscillationKey,
    ProjectData,
    AMROscillation,
)


class AMROFitter:
    # FIT_SYMMETRIES = [2, 4]

    def __init__(
        self,
        amro_data: ProjectData,
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
        self.project_data = amro_data
        self.force_four_and_two_sym = force_four_and_two_sym
        self.verbose = verbose
        self.overwrite = if_save_file_exists_overwrite

        # self.ft_results_df = self._filter_guess_params(fourier_results)
        self.act_choices = self._get_h_t_values()

        self._name_save_paths(save_name, min_amp_ratio, max_freq)
        self._read_or_initialize()

        self.failed_fits = []
        return

    def _name_save_paths(self, name, amp_ratio, freq):

        s = "_ratio_{}_maxf_{}_".format(amp_ratio, freq)
        self.save_name = name + s.replace(".", "p")

        s = self.save_name + "_fit_params.csv"
        self.fit_params_fp = FINAL_DATA_PATH / s

        s = self.save_name + "_results.pkl"
        self.lmfit_results_fp = FINAL_DATA_PATH / s

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

        res_model = u.sine_builder(angle, amps_list, freqs_list, phase_list, offset)

        return res_model - res_data

    def fit_act_experiment(self, act_label: str):
        """"""
        experiment = self.project_data.get_experiment(act_label)
        for osc_key in experiment.oscillations_dict.keys():
            osc = experiment.get_oscillation_from_key(osc_key)

            if osc.fit_result is not None:
                print(f"Already fitted {osc_key}. Skipping...")
                continue
            if osc.fourier_result is None:
                print(f"No Fourier for {osc_key}. Skipping...")

            print(f"Fitting {osc_key}.")

            lmfit_result, refit_bool = self._fit_oscillation(osc)

            osc.add_fit_result(refit_bool, lmfit_result)

            if not lmfit_result.success:
                self.failed_fits.append(osc.key)
        return

    def _fit_oscillation(
        self, osc: AMROscillation
    ) -> tuple[lm.minimizer.MinimizerResult, bool]:
        """
        This function prepares the data of the AMR oscillation for fitting,
         then fits it. To improve the fitting, the data and parameters are
         scaled by the maximum resistivity of the AMRO oscillation. After
         fitting, they are scaled back up. This is to avoid the minimzer
         forgoing an error estimation for any given fit parameter.
        """

        x = osc.osc_data.angles_rads
        y = osc.osc_data.res_ohms

        initial_params = self._initialize_parameters_from_fourier(
            osc.fourier_result, osc.osc_data.mean_res_ohms
        )

        y_norm, norm_scale = self._normalize_data(y)

        # Perform the minimization
        minner = lm.Minimizer(self._obj_func, initial_params, fcn_args=(x, y_norm))
        results = minner.minimize()

        was_refitted = False
        # Check if the covariant matrix is singular
        if results.covar is None:
            results = self._refit(initial_params, x, y_norm)
            was_refitted = True

        if self.verbose:
            print("\n", lm.fit_report(results, show_correl=False), "\n")
        results.params = self._denormalize_parameters(results.params, norm_scale)
        return results, was_refitted

    def _normalize_data(self, y):
        """Normalize y-values to O(1) for better numerical conditioning."""
        y_scale = np.abs(y).max()
        if y_scale < 1e-10:
            y_scale = 1.0
        return y / y_scale, y_scale

    def _denormalize_parameters(self, params, y_scale):
        """Normalize y-values to O(1) for better numerical conditioning."""
        params[HEADER_PARAM_MEAN_PREFIX].value *= y_scale
        params[HEADER_PARAM_MEAN_PREFIX].stderr *= y_scale
        return params

    def _initialize_parameters_from_fourier(
        self,
        fourier_result: FourierResult,
        mean_res: float,
    ) -> lm.Parameters:
        """Note that the value of the 'mean' value should have been normalized
        TODO: Need to implement a way to filter the guesses for a minimum ratio and max freq
        """

        # Generate a Parameters ordered dictionary, to which we add Parameter objects
        initial_p_guesses = lm.Parameters()
        initial_p_guesses.add(HEADER_PARAM_MEAN_PREFIX, value=mean_res, min=0)

        # Append all Parameter objects, except for the last one (must deal with appended 2)

        for freq in fourier_result.fourier_results_dict.keys():
            amp_ratio_guess, phase_guess = fourier_result.fourier_results_dict[freq]

            # Apply filter
            if freq > self.max_freq or amp_ratio_guess < self.min_amp_ratio:
                continue

            self._add_parameter(
                int(freq), initial_p_guesses, amp_ratio_guess, phase_guess
            )

        return initial_p_guesses

    def _add_parameter(
        self,
        frequency: int,
        params: lm.Parameters,
        amp_ratio_guess: float,
        phase_guess: float,
    ) -> None:
        """
        Forcing all amplitudes to be positive, negative values show up as
        a pi-sized phase offset.

        The y_scale scaling is to improve the error estimation of the fitter.

        We divide the magnitude by the mean in order to align with the way
        """

        params.add(
            HEADER_PARAM_FREQ_PREFIX + str(frequency),
            value=frequency,
            vary=False,
        )

        params.add(
            HEADER_PARAM_AMP_PREFIX + str(frequency),
            value=amp_ratio_guess,
            min=0,
        )

        params.add(
            HEADER_PARAM_PHASE_PREFIX + str(frequency),
            value=phase_guess,
            min=-2 * np.pi,
            max=2 * np.pi,
        )

        return

    def _get_freqs_guesses(self, act, h, t):
        guess_df = u.query_dataframe(self.ft_results_df, act=act, h=h, t=t)
        self.current_f_list = guess_df[HEADER_FREQ].unique()
        if self.force_four_and_two_sym:
            self.current_f_list = np.append(self.current_f_list, [2, 4])

            # Drop duplicates
            self.current_f_list = list(set(self.current_f_list))

        return guess_df

    def _are_residuals_acceptable(self) -> bool:
        # TODO: Check the mean absolute residual against some value. This lets us better track poor fits.
        return

    # def get_fitted_parameters(
    #     self,
    #     act: str | list | None = None,
    #     h: int | float | list | None = None,
    #     t: int | float | list | None = None,
    # ) -> pd.DataFrame:
    #     return u.query_dataframe(self.fit_params_df, act=act, h=h, t=t)

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
        with open(self.lmfit_results_fp, "wb") as f:
            pickle.dump(self.lmfit_results_objs, f)

        self.fit_params_df.to_csv(self.fit_params_fp, sep=",")
        self.fit_amps_df.to_csv(self.fit_amps_fp, sep=",")
        return

    def _fast_convert_params_to_ndarrays(
        self, params_obj: lm.Parameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensures the parameters are correctly ordered for sine_builder.
        Aside from the 'mean' parameter, each 'phase' and 'freq' are
        paired based on the 'freq' value.

        A faster version of u.convert_params_to_ndarrays() meant for use
        in the fitter's objective function.

        """
        params_dict = params_obj.valuesdict()
        # TODO: Fix this with the data class. The naming format is 'fragile to naming convention changes'
        amps_phase = [
            (
                params_dict[HEADER_PARAM_AMP_PREFIX + f"{str(f)}"],
                params_dict[HEADER_PARAM_PHASE_PREFIX + f"{str(f)}"],
            )
            for f in self.current_f_list
        ]
        amps_list, phase_list = zip(*amps_phase)
        return (
            np.asarray(amps_list),
            np.asarray(self.current_f_list.copy()),
            np.asarray(phase_list),
            params_dict[HEADER_PARAM_MEAN_PREFIX],
        )

    def _filter_guess_params(self, guess_params: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the fit parameter initial guesses from the Fourier transform
        using the given parameters.

        The amp_ratio is the ratio of the amplitude of a given frequency divided
        by that of the strongest frequency's amplitude.

        The freq is the number of oscillations per rotation of the sample.
        """

        q = HEADER_MAG_RATIO + f" > {self.min_amp_ratio} "
        q += f"& `{HEADER_FREQ}`<{self.max_freq}"
        return guess_params.query(q)

    def _get_init_params(
        self, act: str | list, T: float | int | list, H: float | int | list
    ) -> pd.DataFrame:

        f_info = u.query_dataframe(self.ft_results_df, act=act, h=H, t=T)

        return f_info[[HEADER_FREQ, HEADER_MAG_RATIO]]

    def _get_init_freqs(self, act: str, T: str, H: str) -> np.ndarray:
        f_info = self._get_init_params(act, T, H)
        return f_info[HEADER_FREQ].values

    def _check_if_already_fitted(self, act, t, h) -> bool:
        """
        Check if the experiment has already been fitted
        # TODO: Need to use ProjectData
        """
        try:
            _ = self.lmfit_results_objs[act][t][h]
            return True
        except KeyError:
            return False

    def _read_or_initialize(self):

        # TODO: Checks if saved fitted results exists via the ProjectData object

        return

    def _get_h_t_values(self):
        """"""
        h_t_dict = {}
        grouped = self.project_data[
            [HEADER_ACT, HEADER_MAGNET, HEADER_TEMP]
        ].drop_duplicates()

        for act in grouped[HEADER_ACT].unique():
            tmp_list = []
            for _, row in grouped.query(HEADER_ACT + f'=="{act}"').iterrows():
                tmp_list.append((float(row[HEADER_TEMP]), float(row[HEADER_MAGNET])))
            h_t_dict[act] = tmp_list
        return h_t_dict
