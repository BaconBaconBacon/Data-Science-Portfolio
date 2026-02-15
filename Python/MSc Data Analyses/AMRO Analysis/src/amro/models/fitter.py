import lmfit as lm
import numpy as np


from ..utils import utils as u
from ..plotting.fitter import (
    _plot_fits_with_residuals,
    _plot_fits_with_residuals_uohm,
    _plot_bad_fits,
)

from ..config import (
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_PARAM_FREQ_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
)
from ..data import (
    FourierResult,
    ProjectData,
    AMROscillation,
)


class AMROFitter:
    """Fits AMRO oscillations using Fourier-based initial guesses and least squares optimization."""

    def __init__(
        self,
        amro_data: ProjectData,
        min_amp_ratio=0.2,
        max_freq=10,
        force_four_and_two_sym=False,
        verbose=False,
        if_save_file_exists_overwrite=False,
    ) -> None:
        """Initialize the AMROFitter.

        Args:
            amro_data: ProjectData object containing AMRO experiments with Fourier results.
            min_amp_ratio: Minimum amplitude ratio threshold relative to strongest component.
            max_freq: Maximum frequency to include in fitting (filters noise).
            force_four_and_two_sym: If True, always include 2-fold and 4-fold symmetry terms.
            verbose: If True, print detailed processing information.
            if_save_file_exists_overwrite: If True, overwrite existing fit results.
        """

        # Fit Param filter values
        self.min_amp_ratio = min_amp_ratio
        self.max_freq = max_freq
        self.project_data = amro_data
        self.force_four_and_two_sym = force_four_and_two_sym
        self.verbose = verbose
        self.overwrite = if_save_file_exists_overwrite

        self.filter_str = "ratio_{}_maxf_{}".format(min_amp_ratio, max_freq)

        self.failed_fits = []
        return

    def _obj_func(
        self, params: lm.Parameters, angle: np.ndarray, res_data: np.ndarray
    ) -> np.ndarray:
        """Compute residuals for least squares minimization.

        Args:
            params: lmfit Parameters object containing amplitude, frequency, and phase values.
            angle: Array of angle values in radians.
            res_data: Array of measured resistivity values (normalized).

        Returns:
            Array of residuals (model - data) for least squares fitting.
        """

        amps_list, freqs_list, phase_list, offset = (
            self._fast_convert_params_to_ndarrays(params, f_list=self.current_f_list)
        )

        res_model = u.sine_builder(angle, amps_list, freqs_list, phase_list, offset)

        return res_model - res_data

    def fit_act_experiment(self, act_label: str) -> None:
        """Fit all oscillations in a specified experiment.

        Args:
            act_label: Experiment label identifying which experiment to fit.
        """
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
        self.project_data.save_fit_results_to_csv()

        fn = (
            self.project_data.project_name
            + "_fit_results_"
            + self.project_data.fit_filter_str
            + ".csv"
        )
        print("Fit results saved to: " + fn)

        self.project_data.save_project_to_pickle()
        print("Project state pickled.")

        return

    def _fit_oscillation(
        self, osc: AMROscillation
    ) -> tuple[lm.minimizer.MinimizerResult, bool]:
        """Fit a single AMRO oscillation using least squares optimization.

        Prepares data by normalizing, initializes parameters from Fourier results,
        performs the fit, and denormalizes the results. Attempts refit with
        relaxed phase bounds if initial fit fails to produce covariance matrix.

        Args:
            osc: AMROscillation object containing experimental data and Fourier results.

        Returns:
            Tuple of (MinimizerResult, was_refitted) indicating fit results and
            whether a refit with relaxed bounds was necessary.
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

        results.params = self._denormalize_parameters(results.params, norm_scale)
        del self.current_f_list
        return results, was_refitted

    def _normalize_data(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Normalize resistivity data by its maximum absolute value.

        Args:
            y: Array of resistivity values to normalize.

        Returns:
            Tuple of (normalized_data, scale_factor) for later denormalization.
        """
        y_scale = np.abs(y).max()
        if y_scale < 1e-10:
            y_scale = 1.0
        return y / y_scale, y_scale

    def _denormalize_parameters(
        self, params: lm.Parameters, y_scale: float
    ) -> lm.Parameters:
        """Scale mean parameter back to original units after fitting.

        Args:
            params: lmfit Parameters object with normalized mean value.
            y_scale: Scale factor used during normalization.

        Returns:
            Parameters object with denormalized mean value and error.
        """
        params[HEADER_PARAM_MEAN_PREFIX].value *= y_scale

        if params[HEADER_PARAM_MEAN_PREFIX].stderr is not None:
            params[HEADER_PARAM_MEAN_PREFIX].stderr *= y_scale

        return params

    def _initialize_parameters_from_fourier(
        self,
        fourier_result: FourierResult,
        mean_res: float,
    ) -> tuple[lm.Parameters, list]:
        """Create initial parameter guesses from Fourier transform results.

        Args:
            fourier_result: FourierResult object containing frequency components.
            mean_res: Mean resistivity value for the oscillation.

        Returns:
            Tuple of (Parameters object, list of frequencies) for fitting.
        """
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
        """Add frequency, amplitude, and phase parameters for a single symmetry component.

        Amplitudes are constrained to be positive (negative values appear as phase offsets).
        Phase values are bounded to [-2*pi, 2*pi].

        Args:
            frequency: Rotational symmetry frequency (cycles per rotation).
            params: lmfit Parameters object to add parameters to.
            amp_ratio_guess: Initial amplitude ratio guess from Fourier transform.
            phase_guess: Initial phase guess from Fourier transform.
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

    def _are_residuals_acceptable(
        self, residuals: np.ndarray, threshold: float = 0.01
    ) -> bool:
        """Check if fit residuals are within acceptable bounds.

        The mean absolute residual is compared against a threshold value.
        This helps identify poor fits that may need manual inspection.

        Args:
            residuals: Array of fit residuals.
            threshold: Maximum acceptable mean absolute residual (default 0.01).

        Returns:
            True if mean absolute residual is below threshold, False otherwise.
        """
        mean_abs_residual = np.mean(np.abs(residuals))
        return mean_abs_residual < threshold

    def plot_fits_with_residuals(self, exp_choice, save_fig=False, **kwargs):
        """Plot fitted curves overlaid on experimental data with residuals.

        Args:
            exp_choice: Experiment label to plot.
            save_fig: If True, save the figure to disk.
            **kwargs: Additional keyword arguments passed to plotting function.

        Returns:
            Tuple of (figure, axes) matplotlib objects.
        """
        return _plot_fits_with_residuals(self, exp_choice, save_fig=save_fig, **kwargs)

    def plot_fits_with_residuals_uohm(self, exp_choice, save_fig=False, **kwargs):
        """Plot fitted curves with residuals using micro-ohm-cm units.

        Args:
            exp_choice: Experiment label to plot.
            save_fig: If True, save the figure to disk.
            **kwargs: Additional keyword arguments passed to plotting function.

        Returns:
            Tuple of (figure, axes) matplotlib objects.
        """
        return _plot_fits_with_residuals_uohm(
            self, exp_choice, save_fig=save_fig, **kwargs
        )

    def plot_bad_fits(self, exp_choice: str):
        """Plot only the oscillations that failed to fit properly.

        Args:
            exp_choice: Experiment label to check for failed fits.

        Returns:
            Tuple of (figure, axes) matplotlib objects, or (None, None) if no failures.
        """
        return _plot_bad_fits(self, exp_choice)

    def _fast_convert_params_to_ndarrays(
        self, params_obj: lm.Parameters, f_list: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert lmfit Parameters to numpy arrays for sine_builder.

        Optimized version for use in the objective function during fitting.
        Extracts amplitude, frequency, phase arrays and mean value in correct order.

        Args:
            params_obj: lmfit Parameters object containing fit parameters.
            f_list: List of frequencies to extract parameters for.

        Returns:
            Tuple of (amplitudes, frequencies, phases, mean) as numpy arrays.
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

    def _refit(
        self, params: lm.Parameters, x: np.ndarray, y_norm: np.ndarray
    ) -> lm.minimizer.MinimizerResult:
        """Attempt refit with relaxed phase parameter bounds.

        Called when initial fit fails to produce a covariance matrix.
        Removes bounds on phase parameters to allow broader exploration.

        Args:
            params: lmfit Parameters object from initial fit attempt.
            x: Array of angle values in radians.
            y_norm: Array of normalized resistivity values.

        Returns:
            MinimizerResult from the refit attempt.
        """
        for name, param in params.items():
            if HEADER_PARAM_PHASE_PREFIX in name:
                param.set(min=-np.inf, max=np.inf)
        minner = lm.Minimizer(self._obj_func, params, fcn_args=(x, y_norm))
        results = minner.minimize()

        return results
