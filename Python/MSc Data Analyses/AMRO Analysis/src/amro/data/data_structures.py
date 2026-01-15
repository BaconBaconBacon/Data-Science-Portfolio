from dataclasses import dataclass, field, fields
import numpy as np
import pandas as pd
import lmfit as lm
import pickle
from pathlib import Path
from ..config import (
    FINAL_DATA_PATH,
    HEADER_EXP_LABEL,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_GEO,
    HEADER_ANGLE_RAD,
    HEADER_ANGLE_DEG,
    HEADER_RES_OHM,
    HEADER_FREQ,
    HEADER_MAG,
    HEADER_PHASE,
    HEADER_MEAN,
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_FREQ_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_FIT_RED_CHISQ,
    HEADER_FIT_CHISQ,
    HEADER_PHASE_RAW,
    HEADER_RES_DEL_0DEG_NORM_PCT,
    HEADER_RES_UOHM,
    HEADER_RES_DEL_MEAN_OHM,
    HEADER_RES_DEL_MEAN_UOHM,
    HEADER_RES_DEF_MEAN_NORM,
    HEADER_RES_DEL_MEAN_NORM_PCT,
    HEADER_RES_DEL_0DEG_OHM,
    HEADER_RES_DEL_0DEG_UOHM,
    HEADER_RES_DEL_0DEG_NORM,
)
from ..utils import conversions as c
from ..utils import utils as u

"""
    Classes for storing and accessing experiments and results.
    
    Intended loading process is :
    1) Read in all AMRO data such that each geometry gets an Experiment class instantiation
        1.1) Each oscillation of which an AMROscillation class instantiation 
        1.2) Each oscillation's data is stored in an ExperimentalData object
        1.3) Each AMROscillation has a unique OscillationKey class instantiation to identify it
    2) Iterate over each experiment's AMROscillation objects
        2.1) Perform Fourier transforms of AMROscillation data
        2.2) Store the results in a FourierResult object in the respective AMROscillation object
    3) Iterate again over each experiment's AMROscillation objects
        3.1) Perform a best fit using the FourierResult info as an initial guess
        3.2) Store the results in a FitResult object in the respective AMROscillation object
    
"""


@dataclass(frozen=True)
class OscillationKey:
    """Identifies an AMR Oscillation's unique experiment key."""

    experiment_label: str
    temperature: float
    magnetic_field: float

    def __str__(self) -> str:
        """Return formatted string representation of the key."""
        return u.format_oscillation_key(
            self.experiment_label, self.temperature, self.magnetic_field
        )

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return str(self)

    def compare_exp_label(self, other_act: str) -> bool:
        """Check if experiment label matches the given value."""
        return self.experiment_label == other_act

    def compare_temperature(self, other_temperature: float) -> bool:
        """Check if temperature matches the given value."""
        return self.temperature == other_temperature

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        """Check if magnetic field matches the given value."""
        return self.magnetic_field == other_magnetic_field

    def compare_keys(self, other_key: "OscillationKey") -> bool:
        """Check if all of this key's attributes match another OscillationKey."""
        same_act = self.compare_exp_label(other_key.experiment_label)
        same_temp = self.compare_temperature(other_key.temperature)
        same_field = self.compare_magnetic_field(other_key.magnetic_field)
        return same_act and same_temp and same_field and same_field

    def get_experiment_label(self) -> str:
        """Return the experiment label."""
        return self.experiment_label

    def get_temperature(self) -> float:
        """Return the temperature in Kelvin."""
        return self.temperature

    def get_magnetic_field(self) -> float:
        """Return the magnetic field in Tesla."""
        return self.magnetic_field


@dataclass
class FitResult:
    """Stores a single AMR Oscillation's best fit results."""

    experiment_key: OscillationKey
    lmfit_result: lm.minimizer.MinimizerResult
    lmfit_params: lm.parameter.Parameters = field(init=False)

    fit_succeeded: bool

    symmetries: list | np.ndarray = field(init=False)
    phases: list | np.ndarray = field(init=False)
    amplitudes: list | np.ndarray = field(init=False)
    mean: float = field(init=False)

    phases_errs: list | np.ndarray = field(init=False)
    amplitudes_errs: list | np.ndarray = field(init=False)
    mean_err: float = field(init=False)

    chi_squared: float = field(init=False)
    red_chi_squared: float = field(init=False)
    covar_matrix: np.ndarray | None = field(init=False)

    model_res_ohms: list | np.ndarray
    model_res_uohms: list | np.ndarray = field(init=False)

    model_residuals_ohms: list | np.ndarray = field(init=False)
    model_residuals_uohms: list | np.ndarray = field(init=False)

    required_refit: bool
    fit_report: str = field(init=False)

    fitted_params_dict: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Return string representation of the fit result."""
        return f"Fit_Result_Object_{self.experiment_key}"

    def __post_init__(self) -> None:
        """Initialize derived attributes from lmfit result."""

        # Get the relevant info from lmfit_result
        self.chi_squared = self.lmfit_result.chisqr
        self.red_chi_squared = self.lmfit_result.redchi
        self.covar_matrix = self.lmfit_result.covar
        self.fit_succeeded = self.lmfit_result.success

        self._parse_params(self.lmfit_result.params)
        self._build_params_dict()

        self.model_res_uohms = c.convert_ohms_to_uohms(self.model_res_ohms)
        self.model_residuals_ohms = self.lmfit_result.residual
        self.model_residuals_uohms = c.convert_ohms_to_uohms(self.model_residuals_ohms)
        return

    def compare_act(self, other_act: str) -> bool:
        """Check if experiment label matches the given value."""
        return self.experiment_key.compare_exp_label(other_act)

    def _build_params_dict(self) -> None:
        """Build dictionary of fitted parameters with their errors."""
        self.fitted_params_dict[HEADER_MEAN] = (self.mean, self.mean_err)
        for i, freq in enumerate(self.symmetries):
            self.fitted_params_dict[freq] = (
                (self.amplitudes[i], self.amplitudes_errs[i]),
                (self.phases[i], self.phases_errs[i]),
            )
        return

    def compare_temperature(self, other_temperature: float) -> bool:
        """Check if temperature matches the given value."""
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        """Check if magnetic field matches the given value."""
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)

    def get_fitted_params(self) -> tuple:
        """Return fitted parameters as numpy arrays without errors."""
        return u.convert_params_to_ndarrays(self.lmfit_params, include_errs=False)

    def get_fitted_params_with_errs(self) -> tuple:
        """Return fitted parameters as numpy arrays with error estimates."""
        return u.convert_params_to_ndarrays(self.lmfit_params, include_errs=True)

    def _parse_params(self, params: lm.Parameters) -> None:
        """Extract parameter values and errors from lmfit Parameters object."""
        self.lmfit_params = params
        self.fit_report = lm.fit_report(params)

        (
            amps_list,
            amps_err_list,
            freqs_list,
            phases_list,
            phases_err_list,
            mean,
            mean_err,
        ) = u.convert_params_to_ndarrays(params, include_errs=True)

        self.symmetries = np.asarray(freqs_list)

        self.phases = np.asarray(phases_list)
        self.phases_errs = np.asarray(phases_err_list)

        self.amplitudes = np.asarray(amps_list)
        self.amplitudes_errs = np.asarray(amps_err_list)

        self.mean = mean
        self.mean_err = mean_err
        return

    def get_experiment_label(self) -> str:
        """Return the experiment label from the key."""
        return self.experiment_key.get_experiment_label()

    def get_temperature(self) -> float:
        """Return the temperature from the key."""
        return self.experiment_key.get_temperature()

    def get_magnetic_field(self) -> float:
        """Return the magnetic field from the key."""
        return self.experiment_key.get_magnetic_field()


@dataclass
class ExperimentalData:
    """Stores a single AMR oscillation's experimental data, i.e. resistivity and sample angle."""

    experiment_key: OscillationKey
    angles_degs: list | np.ndarray
    res_ohms: list | np.ndarray

    # Values for calculations
    angles_rads: np.ndarray = field(init=False)

    # res_{mean}
    mean_res_ohms: float = field(init=False)
    mean_res_uohms: float = field(init=False)

    # res_{\theta=0}
    deg0_res_ohms: float = field(init=False)
    deg0_res_uohms: float = field(init=False)

    # Values for plotting
    # val = (res-res_{mean})
    delta_res_mean_ohms: np.ndarray = field(init=False)
    delta_res_mean_uohms: np.ndarray = field(init=False)

    # val = (res-res_{\theta=0})
    delta_res_0deg_ohms: np.ndarray = field(init=False)
    delta_res_0deg_uohms: np.ndarray = field(init=False)

    # val = (res-res_{constant})/res_{constant}
    delta_res_mean_norm: np.ndarray = field(init=False)
    delta_res_0deg_norm: np.ndarray = field(init=False)

    # val = (res-res_{constant})/res_{constant}*100
    delta_res_mean_norm_pct: np.ndarray = field(init=False)
    delta_res_0deg_norm_pct: np.ndarray = field(init=False)

    def __str__(self) -> str:
        """Return string representation of the data object."""
        return f"AMRO_Data_Object_{self.experiment_key}"

    def __post_init__(self) -> None:
        """Initialize derived values and perform unit conversions."""
        self._validate_inputs()

        self.angles_degs = np.asarray(self.angles_degs)
        self.res_ohms = np.asarray(self.res_ohms)

        self.mean_res_ohms = np.mean(self.res_ohms, dtype=float)
        self._get_angle_zero_res()

        self.angles_rads = c.convert_degs_to_rads(self.angles_degs)

        self._calc_delta_res()
        self._calc_res_uohm_values()
        self._calc_normed_res()

    def compare_act(self, other_act: str) -> bool:
        """Check if experiment label matches the given value."""
        return self.experiment_key.compare_exp_label(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        """Check if temperature matches the given value."""
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        """Check if magnetic field matches the given value."""
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)

    def _calc_normed_res(self) -> None:
        """Calculate normalized resistivity values as fractions and percentages."""
        self.delta_res_mean_norm = self.delta_res_mean_ohms / self.mean_res_ohms
        self.delta_res_0deg_norm = self.delta_res_0deg_ohms / self.deg0_res_ohms

        self.delta_res_mean_norm_pct = self.delta_res_mean_norm * 100
        self.delta_res_0deg_norm_pct = self.delta_res_0deg_norm * 100

    def _calc_delta_res(self) -> None:
        """Calculates various values using different units for clearer plotting."""

        self.delta_res_mean_ohms = self.res_ohms - self.mean_res_ohms
        self.delta_res_0deg_ohms = self.res_ohms - self.deg0_res_ohms
        return

    def _validate_inputs(self) -> None:
        """Validate that angle and resistivity values are non-negative."""
        if min(self.angles_degs) < 0:
            raise ValueError("Angles must be non-negative")
        elif min(self.res_ohms) < 0:
            raise ValueError("Resistivity must be non-negative")

    def _calc_res_uohm_values(self) -> None:
        """Call this last at initialization. Iterates over the class object's attributes."""
        for attribute in fields(self):
            if attribute.name.endswith("_ohms"):
                vals = getattr(self, attribute.name)
                new_name = attribute.name.replace("_ohms", "_uohms")
                new_vals = c.convert_ohms_to_uohms(vals)
                setattr(self, new_name, new_vals)

        return

    def _get_angle_zero_res(self) -> None:
        """Get the resistivity measurement at the very start of the oscillation, i.e. when the sample angle equals 0."""
        id_min = np.argmin(self.angles_degs)
        self.deg0_res_ohms = self.res_ohms[id_min]
        return

    def get_experiment_label(self) -> str:
        """Return the experiment label from the key."""
        return self.experiment_key.get_experiment_label()

    def get_temperature(self) -> float:
        """Return the temperature from the key."""
        return self.experiment_key.get_temperature()

    def get_magnetic_field(self) -> float:
        """Return the magnetic field from the key."""
        return self.experiment_key.get_magnetic_field()


@dataclass
class FourierResult:
    """
    Stores the results of a Fourier Transform. yf is a list of complex numbers outputted by
    rfft()
    """

    key: OscillationKey

    xf: list | np.ndarray
    yf: list | np.ndarray

    phases: np.ndarray = field(init=False)
    amplitudes: np.ndarray = field(init=False)

    amplitudes_ratio: np.ndarray = field(init=False)
    phases_pos: np.ndarray = field(init=False)

    fourier_results_dict: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Process FFT output into amplitudes, phases, and amplitude ratios."""

        self.xf = np.asarray(self.xf).astype(int)
        self.yf = np.asarray(self.yf)

        self.phases = np.angle(self.yf)
        self.amplitudes = np.abs(self.yf)
        self.amplitudes_ratio = np.divide(self.amplitudes, self.amplitudes.max())
        self.phases_pos = np.where(
            self.phases < 0,
            self.phases + 2 * np.pi,
            self.phases,
        )
        for i, f in enumerate(self.xf):

            if f > 0:
                self.fourier_results_dict[f] = (
                    self.amplitudes[i],
                    self.amplitudes_ratio[i],
                    self.phases_pos[i],
                )
            else:
                raise ValueError(f"Invalid symmetry! {f} in {self.key}")

    def get_fit_guess(self, freq: int) -> tuple[float, float]:
        """Get amplitude ratio and phase for a given frequency as fit initial guess.

        Args:
            freq: Frequency component to retrieve.

        Returns:
            Tuple of (amplitude_ratio, phase) for the frequency.
        """
        item = self.fourier_results_dict[freq]
        return item[1], item[2]

    def __str__(self) -> str:
        """Return string representation of the Fourier result."""
        return f"Fourier_Result_Object_{self.key}"

    def get_n_strongest_components(self, n: int = 0) -> zip:
        """Get the n strongest frequency components by amplitude ratio.

        Args:
            n: Number of components to return. If 0, returns all.

        Returns:
            Zip iterator of (frequency, amplitude_ratio) pairs.
        """
        idx_largest = np.argpartition(self.amplitudes_ratio, -n)[-n:]
        n_syms = self.xf[idx_largest]
        n_ratios = self.amplitudes_ratio[idx_largest]
        return zip(n_syms, n_ratios)

    def compare_act(self, other_act: str) -> bool:
        """Check if experiment label matches the given value."""
        return self.key.compare_exp_label(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        """Check if temperature matches the given value."""
        return self.key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        """Check if magnetic field matches the given value."""
        return self.key.compare_magnetic_field(other_magnetic_field)

    def get_experiment_label(self) -> str:
        """Return the experiment label from the key."""
        return self.key.get_experiment_label()

    def get_temperature(self) -> float:
        """Return the temperature from the key."""
        return self.key.get_temperature()

    def get_magnetic_field(self) -> float:
        """Return the magnetic field from the key."""
        return self.key.get_magnetic_field()


@dataclass
class AMROscillation:
    """Stores the data and analysis results of a single AMR oscillation, at a given T and H.

    fit_result and fourier_result are to be added after the AMROscillation object is instantiated.
    """

    key: OscillationKey
    osc_data: ExperimentalData

    fit_result: FitResult | None = None
    fourier_result: FourierResult | None = None

    def __str__(self) -> str:
        """Return string representation of the oscillation."""
        return f"Experiment_Object_{self.key}"

    def compare_act(self, other_act: str) -> bool:
        """Check if experiment label matches the given value."""
        return self.key.compare_exp_label(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        """Check if temperature matches the given value."""
        return self.key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        """Check if magnetic field matches the given value."""
        return self.key.compare_magnetic_field(other_magnetic_field)

    def add_fit_result(
        self,
        lmfit_result: lm.minimizer.MinimizerResult,
        refitted: bool,
    ) -> None:
        """Add fitting results to this oscillation.

        Args:
            lmfit_result: MinimizerResult from lmfit optimization.
            refitted: Whether the fit required relaxed bounds.
        """
        model_vals = self._calc_model_resistivities(lmfit_result.params)
        self.fit_result = FitResult(
            lmfit_result=lmfit_result,
            experiment_key=self.key,
            model_res_ohms=model_vals,
            required_refit=refitted,
            fit_succeeded=lmfit_result.success,
        )
        return

    def add_fourier_result(self, xf: np.ndarray, yf: np.ndarray) -> None:
        """Add Fourier transform results to this oscillation.

        Args:
            xf: Array of frequency values from rfft.
            yf: Array of complex amplitudes from rfft.
        """
        self.fourier_result = FourierResult(key=self.key, xf=xf, yf=yf)
        return

    def get_experiment_label(self) -> str:
        """Return the experiment label from the key."""
        return self.key.get_experiment_label()

    def get_temperature(self) -> float:
        """Return the temperature from the key."""
        return self.key.get_temperature()

    def get_magnetic_field(self) -> float:
        """Return the magnetic field from the key."""
        return self.key.get_magnetic_field()

    def _calc_model_resistivities(self, params: lm.parameter.Parameters) -> np.ndarray:
        """Calculate model resistivity values from fitted parameters.

        Args:
            params: lmfit Parameters object with fitted values.

        Returns:
            Array of model resistivity values.
        """
        params = u.convert_params_to_ndarrays(params)
        model_res = u.calculate_model_resistivities(self.osc_data.angles_rads, params)
        return model_res

    def get_n_strongest_fourier(self, n: int = 0) -> zip | None:
        """Get the n strongest Fourier components for this oscillation.

        Args:
            n: Number of components to return.

        Returns:
            Zip iterator of (frequency, amplitude_ratio) pairs, or None if no Fourier result.
        """
        if self.fourier_result is not None:
            return self.fourier_result.get_n_strongest_components(n)
        else:
            print("Fourier result not present.")
            return None

    def get_oscillation_as_dataframe(self) -> pd.DataFrame:
        """Convert oscillation data to a DataFrame.

        Returns:
            DataFrame with one row per angle measurement, including all derived values.
        """
        data = self.osc_data
        rows = []
        for i in range(len(data.angles_degs)):
            row = {
                HEADER_EXP_LABEL: self.key.experiment_label,
                HEADER_TEMP: self.key.temperature,
                HEADER_MAGNET: self.key.magnetic_field,
                HEADER_ANGLE_DEG: data.angles_degs[i],
                HEADER_ANGLE_RAD: data.angles_rads[i],
                HEADER_RES_OHM: data.res_ohms[i],
                HEADER_RES_UOHM: c.convert_ohms_to_uohms(data.res_ohms[i]),
                HEADER_RES_DEL_MEAN_OHM: data.delta_res_mean_ohms[i],
                HEADER_RES_DEL_MEAN_UOHM: data.delta_res_mean_uohms[i],
                HEADER_RES_DEF_MEAN_NORM: data.delta_res_mean_norm[i],
                HEADER_RES_DEL_MEAN_NORM_PCT: data.delta_res_mean_norm_pct[i],
                HEADER_RES_DEL_0DEG_OHM: data.delta_res_0deg_ohms[i],
                HEADER_RES_DEL_0DEG_UOHM: data.delta_res_0deg_uohms[i],
                HEADER_RES_DEL_0DEG_NORM: data.delta_res_0deg_norm[i],
                HEADER_RES_DEL_0DEG_NORM_PCT: data.delta_res_0deg_norm_pct[i],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def clear_fourier_result(self) -> None:
        """Remove the Fourier result from this oscillation."""
        self.fourier_result = None
        return

    def clear_fit_result(self) -> None:
        """Remove the fit result from this oscillation."""
        self.fit_result = None
        return


@dataclass
class Experiment:
    """Handles dataclasses for all oscillations for a given experimental set up."""

    experiment_label: str
    geometry: str

    wire_sep: float
    cross_section: float
    oscillations_dict: dict = field(default_factory=dict)
    oscillations_count: float = 0
    material: str = None

    def add_oscillation(self, oscillation: AMROscillation) -> None:
        """Add an oscillation to this experiment.

        Args:
            oscillation: AMROscillation object to add.
        """
        new_key = oscillation.key
        if new_key not in self.oscillations_dict.keys():
            self.oscillations_dict[new_key] = oscillation
            self.oscillations_count += 1
        else:
            print(f"Key {new_key} already exists! Use replace_oscillation() instead.")
        return

    def replace_oscillation(self, oscillation: AMROscillation) -> None:
        """Replace an existing oscillation with a new one.

        Args:
            oscillation: AMROscillation object to replace with.
        """
        new_key = oscillation.key
        self.oscillations_dict[new_key] = oscillation
        return

    def get_oscillation(self, t: float, h: float) -> AMROscillation:
        """Retrieve an oscillation by temperature and magnetic field.

        Args:
            t: Temperature in Kelvin.
            h: Magnetic field in Tesla.

        Returns:
            AMROscillation matching the specified conditions.
        """
        request_key = OscillationKey(self.experiment_label, t, h)

        return self.oscillations_dict[request_key]

    def get_oscillation_from_key(self, request_key: OscillationKey) -> AMROscillation:
        """Retrieve an oscillation using an OscillationKey.

        Args:
            request_key: OscillationKey identifying the oscillation.

        Returns:
            AMROscillation matching the key.
        """
        return self.oscillations_dict[request_key]

    def get_multiple_oscillations(
        self,
        t: float | list | None = None,
        h: float | list | None = None,
    ) -> list | None:
        """Filter oscillations by temperature and/or magnetic field.

        Args:
            t: Temperature value(s) to filter by. None returns all temperatures.
            h: Magnetic field value(s) to filter by. None returns all fields.

        Returns:
            List of AMROscillation objects matching the criteria.
        """

        if t is None and h is None:
            return list(self.oscillations_dict.values())

        if t is not None:
            t = np.atleast_1d(t).flatten()

        if h is not None:
            h = np.atleast_1d(h).flatten()

        oscillations = []
        for osc in self.oscillations_dict.values():
            t_matches = (t is None) or (osc.compare_temperature(t))
            h_matches = (h is None) or (osc.compare_magnetic_field(h))
            if t_matches and h_matches:
                oscillations.append(osc)
        return oscillations

    def get_experiment_as_dataframe(self) -> pd.DataFrame:
        """Convert all oscillation data in this experiment to a DataFrame.

        Returns:
            DataFrame with one row per angle measurement, including all derived values.
        """
        rows = []
        for osc_key, osc in self.oscillations_dict.items():
            data = osc.osc_data
            # Each oscillation has arrays of angles and resistivities with derived values   │
            for i in range(len(data.angles_degs)):
                row = {
                    HEADER_EXP_LABEL: self.experiment_label,
                    HEADER_TEMP: osc_key.temperature,
                    HEADER_MAGNET: osc_key.magnetic_field,
                    HEADER_GEO: self.geometry,
                    HEADER_ANGLE_DEG: data.angles_degs[i],
                    HEADER_ANGLE_RAD: data.angles_rads[i],
                    HEADER_RES_OHM: data.res_ohms[i],
                    HEADER_RES_UOHM: c.convert_ohms_to_uohms(data.res_ohms[i]),
                    HEADER_RES_DEL_MEAN_OHM: data.delta_res_mean_ohms[i],
                    HEADER_RES_DEL_MEAN_UOHM: data.delta_res_mean_uohms[i],
                    HEADER_RES_DEF_MEAN_NORM: data.delta_res_mean_norm[i],
                    HEADER_RES_DEL_MEAN_NORM_PCT: data.delta_res_mean_norm_pct[i],
                    HEADER_RES_DEL_0DEG_OHM: data.delta_res_0deg_ohms[i],
                    HEADER_RES_DEL_0DEG_UOHM: data.delta_res_0deg_uohms[i],
                    HEADER_RES_DEL_0DEG_NORM: data.delta_res_0deg_norm[i],
                    HEADER_RES_DEL_0DEG_NORM_PCT: data.delta_res_0deg_norm_pct[i],
                }
                rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class ProjectData:
    """Handles dataclasses for all experiments for a given project"""

    project_name: str
    experiments_dict: dict = field(default_factory=dict)
    experiments_count: float = 0

    fit_filter_str: str | None = None

    def __post_init__(self) -> None:
        """Initialize the pickle file path for this project."""
        self.pickle_fp = FINAL_DATA_PATH / (self.project_name + ".pkl")

    def add_experiment(self, exp: Experiment) -> None:
        """Add an experiment to this project.

        Args:
            exp: Experiment object to add.
        """
        if exp.experiment_label not in self.experiments_dict.keys():
            self.experiments_dict[exp.experiment_label] = exp
            self.experiments_count += 1
        else:
            print("Cannot add experiment twice.")
            print(exp.experiment_label)

    def replace_experiment(self, exp: Experiment) -> None:
        """Replace an existing experiment with a new one.

        Args:
            exp: Experiment object to replace with.
        """
        if exp.experiment_label in self.experiments_dict.keys():
            self.experiments_dict[exp.experiment_label] = exp
        else:
            print("Experiment not found. Cannot replace non-existent experiment.")
            print(exp.experiment_label)

    def get_experiment(self, act_label: str) -> Experiment:
        """Retrieve an experiment by its label.

        Args:
            act_label: Experiment label string.

        Returns:
            Experiment object matching the label.
        """
        return self.experiments_dict[act_label]

    def get_experiment_labels(self) -> list[str]:
        """Return list of all experiment labels in this project."""
        return list(self.experiments_dict.keys())

    def filter_oscillations(
        self,
        experiments: str | list | None = None,
        t_vals: float | list | None = None,
        h_vals: float | list | None = None,
    ) -> list:
        """Filter all oscillations across experiments by label, temperature, and/or field.

        Args:
            experiments: Experiment label(s) to filter by.
            t_vals: Temperature value(s) to filter by.
            h_vals: Magnetic field value(s) to filter by.

        Returns:
            Flattened list of matching AMROscillation objects.
        """
        if experiments is None:
            experiments = self.experiments_dict.keys()
        elif isinstance(experiments, str):
            experiments = [experiments]

        osc_list = []
        for exp_label in experiments:
            exp = self.experiments_dict[exp_label]
            oscs = exp.get_multiple_oscillations(t_vals, h_vals)
            osc_list.append(oscs)
        osc_list = u.flatten_list(osc_list)
        return osc_list

    def check_for_saved_data(self) -> bool:
        """Check for existing pickled project data.

        Returns:
            True if a pickle file exists at the configured path, False otherwise.
        """
        return self.pickle_fp.is_file()

    def read_amro_data_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load AMRO data from a pandas DataFrame into the project structure.

        Args:
            df: DataFrame containing AMRO oscillation data.
        """

        for act in df[HEADER_EXP_LABEL].unique():
            sub_df = u.query_dataframe(df, act=act)
            geom = sub_df[HEADER_GEO].unique()[0]

            try:
                exper = self.get_experiment(act)
            except KeyError:
                exper = Experiment(experiment_label=act, geometry=geom)
                self.add_experiment(exper)

            for t in sub_df[HEADER_TEMP].unique():
                sub_sub_df = u.query_dataframe(sub_df, t=t)
                for h in sub_sub_df[HEADER_MAGNET].unique():
                    experiment_df = u.query_dataframe(sub_sub_df, h=h)

                    key = OscillationKey(
                        experiment_label=act, temperature=t, magnetic_field=h
                    )
                    angles = experiment_df[HEADER_ANGLE_DEG].values
                    res = experiment_df[HEADER_RES_OHM].values

                    exp_data = ExperimentalData(
                        experiment_key=key, angles_degs=angles, res_ohms=res
                    )
                    osc = AMROscillation(key=key, osc_data=exp_data)
                    exper.add_oscillation(osc)
        return

    def read_fit_results_from_dataframe(
        self, df: pd.DataFrame, lmfit_results_dict: dict
    ) -> None:
        """Load fit results from a DataFrame and lmfit objects dictionary.

        Args:
            df: DataFrame containing fit parameter data.
            lmfit_results_dict: Nested dictionary of lmfit MinimizerResult objects.
        """

        for act in df[HEADER_EXP_LABEL].unique():
            sub_df = u.query_dataframe(df, act=act)
            try:
                exper = self.get_experiment(act)
            except KeyError:
                print(
                    f"No Experiment found for {act}. Create Experiment before adding Fourier Results."
                )
                return

            for t in sub_df[HEADER_TEMP].unique():
                sub_sub_df = u.query_dataframe(sub_df, t=t)
                for h in sub_sub_df[HEADER_MAGNET].unique():

                    lmfit_obj, refitted = lmfit_results_dict[act][t][h]

                    osc = exper.get_oscillation(t=t, h=h)

                    osc.add_fit_result(
                        lmfit_result=lmfit_obj,
                        refitted=refitted,
                    )

        return

    def read_fourier_results_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load Fourier results from a pandas DataFrame.

        Args:
            df: DataFrame containing Fourier transform results.
        """
        for act in df[HEADER_EXP_LABEL].unique():
            sub_df = u.query_dataframe(df, act=act)

            try:
                exper = self.get_experiment(act)
            except KeyError:
                print(
                    f"No Experiment found for {act}. Create Experiment before adding Fourier Results."
                )
                continue

            for t in sub_df[HEADER_TEMP].unique():
                sub_sub_df = u.query_dataframe(sub_df, t=t)
                for h in sub_sub_df[HEADER_MAGNET].unique():
                    fourier_result_df = u.query_dataframe(sub_sub_df, h=h)
                    osc = exper.get_oscillation(t=t, h=h)

                    freqs = fourier_result_df[HEADER_FREQ].values

                    mags = fourier_result_df[HEADER_MAG].values
                    phases = fourier_result_df[HEADER_PHASE].values
                    yf = mags[:, 0] + phases[:, 0] * 1j

                    osc.add_fourier_result(xf=freqs, yf=yf)

        return

    def save_project_to_pickle(self, fp: Path = None) -> None:
        """Save this project to a pickle file.

        Args:
            fp: File path for saving. Uses default if None.
        """
        if fp is None:
            fp = self.pickle_fp
        with open(fp, "wb") as f:
            pickle.dump(self, f)
        return

    @classmethod
    def load_project_from_pickle(cls, fp: Path) -> "ProjectData":
        """Load a ProjectData instance from a pickle file.

        Args:
            fp: Path to the pickle file.

        Returns:
            ProjectData instance loaded from file.
        """
        with open(fp, "rb") as f:
            return pickle.load(f)

    def get_fit_results_as_df(self, filepath: Path | str | None = None) -> pd.DataFrame:
        """Convert all fit results to a DataFrame with one row per oscillation.

        Args:
            filepath: Optional file path (not currently used).

        Returns:
            DataFrame containing fit parameters and statistics.
        """
        rows = []
        for act_label in self.experiments_dict.keys():
            experiment = self.experiments_dict[act_label]
            for osc_key in experiment.oscillations_dict.keys():
                osc = experiment.oscillations_dict[osc_key]

                fit_result = osc.fit_result
                if fit_result is None:
                    print(f"No fit result found for {osc_key}")
                    continue

                row = {
                    HEADER_EXP_LABEL: osc_key.experiment_label,
                    HEADER_TEMP: osc_key.temperature,
                    HEADER_MAGNET: osc_key.magnetic_field,
                    HEADER_GEO: experiment.geometry,
                    HEADER_PARAM_MEAN_PREFIX: fit_result.mean,
                    HEADER_PARAM_MEAN_PREFIX + "_err": fit_result.mean_err,
                    HEADER_FIT_CHISQ: fit_result.chi_squared,
                    HEADER_FIT_RED_CHISQ: fit_result.red_chi_squared,
                    "fit_succeeded": fit_result.fit_succeeded,
                    "required_refit": fit_result.required_refit,
                }
                # for freq in fit_result.fitted_params_dict.keys():
                for freq in fit_result.symmetries:
                    params = fit_result.fitted_params_dict[freq]
                    row[HEADER_PARAM_FREQ_PREFIX + str(freq)] = freq
                    row[HEADER_PARAM_AMP_PREFIX + str(freq)] = params[0][0]
                    row[HEADER_PARAM_AMP_PREFIX + str(freq) + "_err"] = params[0][1]
                    row[HEADER_PARAM_PHASE_PREFIX + str(freq)] = params[1][0]
                    row[HEADER_PARAM_PHASE_PREFIX + str(freq) + "_err"] = params[1][1]
                rows.append(row)

        return pd.DataFrame(rows)

    def save_fit_results_to_csv(self, filepath: Path | str | None = None) -> None:
        """Save fit results to a CSV file.

        Args:
            filepath: Path for saving. Uses default naming if None.
        """
        if filepath is None:
            filepath = FINAL_DATA_PATH / (
                f"{self.project_name}_fit_results_" + self.fit_filter_str + ".csv"
            )

        df = self.get_fit_results_as_df(filepath=filepath)
        df.to_csv(filepath, index=False)
        return

    def load_fit_results_from_csv(self, filepath: Path | str | None = None) -> None:
        """Load fit results from a CSV file into the project.

        Args:
            filepath: Path to CSV file. Uses default naming if None.
        """
        if filepath is None:
            filepath = FINAL_DATA_PATH / (
                f"{self.project_name}_fit_results_" + self.fit_filter_str + ".csv"
            )
        df = pd.read_csv(filepath)
        self.read_fit_results_from_dataframe(df=df)
        return

    def get_fourier_results_as_df(
        self, filepath: Path | str | None = None
    ) -> pd.DataFrame:
        """Convert all Fourier results to a DataFrame with one row per frequency component.

        Args:
            filepath: Optional file path (not currently used).

        Returns:
            DataFrame containing frequency, amplitude, and phase data.
        """
        rows = []
        for act_label in self.experiments_dict.keys():
            experiment = self.experiments_dict[act_label]
            for osc_key in experiment.oscillations_dict.keys():
                osc = experiment.oscillations_dict[osc_key]
                fourier_dict = osc.fourier_result.fourier_results_dict
                for freq in fourier_dict:
                    ft = fourier_dict[freq]
                    row = {
                        HEADER_EXP_LABEL: osc_key.experiment_label,
                        HEADER_TEMP: osc_key.temperature,
                        HEADER_MAGNET: osc_key.magnetic_field,
                        HEADER_GEO: experiment.geometry,
                        HEADER_PARAM_FREQ_PREFIX: freq,
                        HEADER_PARAM_AMP_PREFIX: ft[0],
                        HEADER_PARAM_AMP_PREFIX + "_ratio": ft[1],
                        HEADER_PHASE + "_rads": ft[2],
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def save_fourier_results_to_csv(self, filepath: Path | str | None = None) -> None:
        """Save Fourier results to a CSV file.

        Args:
            filepath: Path for saving. Uses default naming if None.
        """
        if filepath is None:
            filepath = FINAL_DATA_PATH / f"{self.project_name}_fourier_results.csv"
        df = self.get_fourier_results_as_df(filepath=filepath)
        df.to_csv(filepath, index=False)
        return

    def load_fourier_results_from_csv(self, filepath: Path | str | None = None) -> None:
        """Load Fourier results from a CSV file into the project.

        Args:
            filepath: Path to CSV file. Uses default naming if None.
        """
        if filepath is None:
            filepath = FINAL_DATA_PATH / f"{self.project_name}_fourier_results.csv"
        df = pd.read_csv(filepath)
        self.read_fourier_results_from_dataframe(df=df)
        return

    def change_project_name(self, new_name: str) -> None:
        """Update the project name.

        Args:
            new_name: New name for the project.
        """
        self.project_name = new_name
        return

    def get_summary_statistics(self) -> dict:
        """Calculate summary statistics for this project.

        Returns:
            Dictionary containing counts of experiments, oscillations, and completed analyses.
        """
        n_oscillations = sum(
            len(exp.oscillations_dict) for exp in self.experiments_dict.values()
        )
        n_fourier = sum(
            1
            for exp in self.experiments_dict.values()
            for osc in exp.oscillations_dict.values()
            if osc.fourier_result is not None
        )
        n_fits = sum(
            1
            for exp in self.experiments_dict.values()
            for osc in exp.oscillations_dict.values()
            if osc.fit_result is not None
        )

        return {
            "n_experiments": len(self.experiments_dict),
            "n_oscillations": n_oscillations,
            "n_fourier_completed": n_fourier,
            "n_fits_completed": n_fits,
        }

    def correct_geometry_scaling(
        self,
        experiment_label: str,
        wire_sep: float,
        width: float = None,
        height: float = None,
        cross_section: float = None,
        force_rescale: bool = False,
    ) -> None:
        """Rescale resistance data to resistivity using correct sample geometry.

        Used when measurements were taken with default geometry values (wire_sep=1, cross_section=1).
        Applies formula: rho = R * (A / L), where A is cross-section and L is wire separation.

        Args:
            experiment_label: Label of experiment to rescale.
            wire_sep: Wire separation distance in cm.
            width: Sample width in cm (optional if cross_section given).
            height: Sample height in cm (optional if cross_section given).
            cross_section: Cross-sectional area in cm^2.
            force_rescale: If True, rescale even if geometry values aren't default.
        """
        old_exp = self.experiments_dict[experiment_label]

        if (old_exp.cross_section != 1 or old_exp.wire_sep != 1) and not force_rescale:
            print(
                f"Experiment cross-section ({old_exp.cross_section}) and wire sep ({old_exp.wire_sep}) do not appear to require scaling."
            )
            print("Set force_rescale to True to force rescaling.")
            return
        elif self._is_valid_scaling_input(
            wire_sep,
            width,
            height,
            cross_section,
        ):
            if cross_section is None:
                cross_section = width * height

            scale_factor = cross_section / wire_sep
            new_exp = Experiment(
                experiment_label=old_exp.experiment_label,
                geometry=old_exp.geometry,
                wire_sep=wire_sep,
                cross_section=cross_section,
            )
            for osc_key, oscillation in old_exp.oscillations_dict.items():
                new_res = oscillation.osc_data.res_ohms * scale_factor
                data = ExperimentalData(
                    experiment_key=osc_key,
                    angles_degs=oscillation.osc_data.angles_degs,
                    res_ohms=new_res,
                )
                new_osc = AMROscillation(
                    key=osc_key,
                    osc_data=data,
                )
                new_exp.add_oscillation(new_osc)
            self.replace_experiment(new_exp)
            self.save_project_to_pickle()
        return

    def _is_valid_scaling_input(
        self,
        wire_sep: float,
        width: float | None,
        height: float | None,
        cross_section: float | None,
    ) -> bool:
        """Validate geometry inputs for rescaling.

        Args:
            wire_sep: Wire separation distance.
            width: Sample width.
            height: Sample height.
            cross_section: Cross-sectional area.

        Returns:
            True if inputs are valid.

        Raises:
            ValueError: If inputs are invalid or negative.
        """
        if cross_section is None:
            if height is None or width is None:
                raise ValueError("Invalid dimension inputs for re-scaling.")
            elif height <= 0 or width <= 0:
                raise ValueError("Negative values not allowed for re-scaling.")
        elif cross_section <= 0:
            raise ValueError("Negative cross-section not allowed for re-scaling.")

        if wire_sep is None:
            raise ValueError("Wire separation must be inputted for re-scaling.")
        elif wire_sep <= 0:
            raise ValueError("Wire separation must be positive.")

        return True
