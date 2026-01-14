from pathlib import Path

import lmfit as lm
import numpy as np
import pandas as pd
import pickle

from statsmodels.sandbox.distributions.try_pot import mean_residual_life

from ..utils import utils as u
from ..plotting.fitter import (
    _plot_fits_with_residuals,
    _plot_fits_with_residuals_uohm,
    _plot_bad_fits,
    _save_plot,
)

from ..config import (
    PROCESSED_DATA_PATH,
    FINAL_DATA_PATH,
    PROCESSED_FIGURES_PATH,
    HEADER_ANGLE_DEG,
    HEADER_ANGLE_RAD,
    HEADER_RES_OHM,
    HEADER_MAGNET,
    HEADER_TEMP,
    HEADER_EXP_LABEL,
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
from ..data import (
    FitResult,
    OscillationKey,
    FourierResult,
    ProjectData,
    AMROscillation,
)


class AMROFitter:

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

        self.filter_str = "ratio_{}_maxf_{}_".format(min_amp_ratio, max_freq)

        self.failed_fits = []
        return

    def _obj_func(self, params: lm.Parameters, angle: np.ndarray, res_data: np.ndarray) -> np.ndarray:
        """
        The sinebuilder is fitted by minimizing this least squares objective function.
        """

        amps_list, freqs_list, phase_list, offset = (
            self._fast_convert_params_to_ndarrays(params, f_list=self.current_f_list)
        )

        res_model = u.sine_builder(angle, amps_list, freqs_list, phase_list, offset)

        return res_model - res_data

    def fit_act_experiment(self, act_label: str) -> None:
        """"""
        if act_label not in self.project_data.experiments_dict.keys():
            print(f"{act_label} is not a valid experiment label.")
            return

        if self.project_data.fit_filter_str is None:
            self.project_data.fit_filter_str = self.filter_str

        experiment = self.project_data.get_experiment(act_label)
        i = 0
        for osc_key in experiment.oscillations_dict.keys():
            i += 1
            osc = experiment.get_oscillation_from_key(osc_key)

            if osc.fit_result is not None and not self.overwrite:
                print(f"Already fitted {osc_key}. Skipping...")
                continue
            elif osc.fourier_result is None:
                print(f"No Fourier for {osc_key}. Skipping...")
                continue
            print(f"Fitting {osc_key}.")

            lmfit_result, refit_bool = self._fit_oscillation(osc)

            osc.add_fit_result(
                lmfit_result=lmfit_result,
                refitted=refit_bool,
            )

            if not lmfit_result.success:
                self.failed_fits.append(osc.key)
        print(f"Total fitted: {i}")
        print("Saving to CSV.")
        self.project_data.save_fit_results_to_csv()
        print(f"Pickling project data.")
        self.project_data.save_project_to_pickle()
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

        initial_params, f_list = self._initialize_parameters_from_fourier(
            osc.fourier_result, osc.osc_data.mean_res_ohms
        )
        self.current_f_list = f_list

        y_norm, norm_scale = self._normalize_data(y)

        minner = lm.Minimizer(self._obj_func, initial_params, fcn_args=(x, y_norm))
        results = minner.minimize()

        was_refitted = False
        if results.covar is None:
            print("Attempting re-fit with infinite bounds for phase.")
            results = self._refit(initial_params, x, y_norm)
            was_refitted = True
            if results.covar is None:
                print("Covar matrix is remains singular.")
            else:
                print("Fit was improved.")
            print("Continuing...")
        # if self.verbose:
        #     print("\n", lm.fit_report(results, show_correl=False), "\n")
        results.params = self._denormalize_parameters(results.params, norm_scale)
        del self.current_f_list
        return results, was_refitted

    def _normalize_data(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        y_scale = np.abs(y).max()
        if y_scale < 1e-10:
            y_scale = 1.0
        return y / y_scale, y_scale

    def _denormalize_parameters(self, params: lm.Parameters, y_scale: float) -> lm.Parameters:
        params[HEADER_PARAM_MEAN_PREFIX].value *= y_scale

        if params[HEADER_PARAM_MEAN_PREFIX].stderr is not None:
            params[HEADER_PARAM_MEAN_PREFIX].stderr *= y_scale

        return params

    def _initialize_parameters_from_fourier(
        self,
        fourier_result: FourierResult,
        mean_res: float,
    ) -> tuple[lm.Parameters, list]:

        # Generate a Parameters ordered dictionary, to which we add Parameter objects
        initial_p_guesses = lm.Parameters()
        initial_p_guesses.add(HEADER_PARAM_MEAN_PREFIX, value=mean_res, min=0)

        # Append all Parameter objects, except for the last one (must deal with appended 2)
        current_freqs = []
        for freq in fourier_result.fourier_results_dict.keys():
            amp_ratio_guess, phase_guess = fourier_result.get_fit_guess(freq)

            # Apply filter
            if freq > self.max_freq or amp_ratio_guess < self.min_amp_ratio:
                continue

            self._add_parameter(
                int(freq), initial_p_guesses, amp_ratio_guess, phase_guess
            )
            current_freqs.append(freq)

        if self.force_four_and_two_sym:
            if 2 not in current_freqs:
                self._add_parameter(2, initial_p_guesses, 0, 0)
                current_freqs.append(2)
            if 4 not in current_freqs:
                self._add_parameter(4, initial_p_guesses, 0, 0)
                current_freqs.append(4)
        return initial_p_guesses, current_freqs

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

    def _are_residuals_acceptable(self, residuals) -> bool:
        # TODO: Check the mean absolute residual against some value. This lets us better track poor fits.
        # The average absolute residual is not greater than 1% of the mean?
        return

    def plot_fits_with_residuals(self, exp_choice, save_fig=False, **kwargs):
        return _plot_fits_with_residuals(self, exp_choice, save_fig=save_fig, **kwargs)

    def plot_fits_with_residuals_uohm(self, exp_choice, save_fig=False, **kwargs):
        return _plot_fits_with_residuals_uohm(
            self, exp_choice, save_fig=save_fig, **kwargs
        )

    def plot_bad_fits(self, exp_choice: str):
        return _plot_bad_fits(self, exp_choice)

    # def save_plot(self, fig, filename, dpi=300):
    #     _save_plot(fig, filename, dpi=dpi)
    #
    # def _save_to_disk(self):
    #     """ """
    #     with open(self.lmfit_results_fp, "wb") as f:
    #         pickle.dump(self.lmfit_results_objs, f)
    #
    #     self.fit_params_df.to_csv(self.fit_params_fp, sep=",")
    #     self.fit_amps_df.to_csv(self.fit_amps_fp, sep=",")
    #     return

    def _fast_convert_params_to_ndarrays(
        self, params_obj: lm.Parameters, f_list: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensures the parameters are correctly ordered for sine_builder.
        Aside from the 'mean' parameter, each 'phase' and 'freq' are
        paired based on the 'freq' value.

        A faster version of u.convert_params_to_ndarrays() meant for use
        in the fitter's objective function.

        """
        params_dict = params_obj.valuesdict()
        amps_phase = [
            (
                params_dict[HEADER_PARAM_AMP_PREFIX + f"{str(f)}"],
                params_dict[HEADER_PARAM_PHASE_PREFIX + f"{str(f)}"],
            )
            for f in f_list
        ]
        amps_list, phase_list = zip(*amps_phase)
        return (
            np.asarray(amps_list),
            np.asarray(f_list),
            np.asarray(phase_list),
            params_dict[HEADER_PARAM_MEAN_PREFIX],
        )

    def _refit(self, params: lm.Parameters, x: np.ndarray, y_norm: np.ndarray) -> lm.minimizer.MinimizerResult:

        for name, param in params.items():
            if HEADER_PARAM_PHASE_PREFIX in name:
                param.set(min=-np.inf, max=np.inf)
        minner = lm.Minimizer(self._obj_func, params, fcn_args=(x, y_norm))
        results = minner.minimize()

        return results
