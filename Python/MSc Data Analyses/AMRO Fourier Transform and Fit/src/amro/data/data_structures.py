from dataclasses import dataclass, field, fields
from typing import Optional
import numpy as np
import pandas as pd
import lmfit as lm
from ..utils import conversions as c
from ..utils import utils as u

"""
    Classes for storing and accessing experiments and results.
"""


@dataclass(frozen=True)
class ExperimentKey:
    """Identifies an AMRO experiment key."""

    act: str
    temperature: float
    magnetic_field: float

    def __str__(self):
        return f"{self.act}_T{self.temperature}K_H{self.magnetic_field}T"

    def compare_act(self, other_act: str) -> bool:
        return self.act == other_act

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.temperature == other_temperature

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.magnetic_field == other_magnetic_field


@dataclass
class FitResult:
    """Store fit and Fourier transform information"""

    experiment_key: ExperimentKey
    lmfit_result: lm.minimizer.MinimizerResult
    lmfit_params: lm.parameter.Parameters = field(init=False)

    fit_succeeded: bool

    symmetries: list | np.ndarray = field(init=False)
    phases: list | np.ndarray = field(init=False)
    amplitudes: list | np.ndarray = field(init=False)

    symmetries_errs: list | np.ndarray = field(init=False)
    phases_errs: list | np.ndarray = field(init=False)
    amplitudes_errs: list | np.ndarray = field(init=False)

    uvars_dict: dict = field(init=False)

    chi_squared: float = field(init=False)
    red_chi_squared: float = field(init=False)
    covar_matrix: np.ndarray | None = field(init=False)

    model_res_ohms: list | np.ndarray
    model_res_uohms: list | np.ndarray = field(init=False)

    model_residuals_ohms: list | np.ndarray = field(init=False)
    model_residuals_uohms: list | np.ndarray = field(init=False)

    required_refit: bool = field(init=False)
    fit_report: str = field(init=False)

    def __str__(self):
        return f"Fit_Result_Object_{self.experiment_key}"

    def __post_init__(self):

        # Get the relevant info from lmfit_result
        self.chi_squared = self.lmfit_result.chisqr
        self.red_chi_squared = self.lmfit_result.redchi
        self.covar_matrix = self.lmfit_result.covar
        self.fit_succeeded = self.lmfit_result.success  # self._check_fit_success()

        self._parse_params(self.lmfit_result.params)

        self.uvars_dict = self.lmfit_result.uvar

        self.model_res_uohms = c.convert_ohms_to_uohms(self.model_res_ohms)
        self.model_residuals_ohms = self.lmfit_result.residual
        self.model_residuals_uohms = c.convert_ohms_to_uohms(self.model_residuals_ohms)
        return

    def compare_act(self, other_act: str) -> bool:
        return self.experiment_key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)

    def get_fitted_params(self):
        return

    def get_fitted_param_errs(self):
        return

    def _parse_params(self, params: lm.Parameters) -> None:
        """Must parse the keys and values of the Parameters objects"""
        self.lmfit_params = params
        self.fit_report = lm.report_fit(params)
        self.params_dict = params.valuesdict()
        return

    # def _check_fit_success(self):
    #     if self.covar_matrix is None:
    #         return False
    #     else:
    #         return True


@dataclass
class AMROData:
    """Stores an experiment's AMRO data, i.e. resistivity and sample angle."""

    experiment_key: ExperimentKey
    angles_degs: list | np.ndarray
    res_ohms: list | np.ndarray

    # Values for calculations
    angles_rads: list | np.ndarray = field(init=False)

    # res_{mean}
    mean_res_ohms: float = field(init=False)
    mean_res_uohms: float = field(init=False)

    # res_{\theta=0}
    deg0_res_ohms: float = field(init=False)
    deg0_res_uohms: float = field(init=False)

    # Values for plotting
    # val = (res-res_{mean})
    delta_res_mean_ohms: list | np.ndarray = field(init=False)
    delta_res_mean_uohms: list | np.ndarray = field(init=False)

    # val = (res-res_{\theta=0})
    delta_res_0deg_ohms: list | np.ndarray = field(init=False)
    delta_rest_0deg_uohms: list | np.ndarray = field(init=False)

    # val = (res-res_{constant})/res_{constant}
    delta_res_mean_norm: list | np.ndarray = field(init=False)
    delta_res_0deg_norm: list | np.ndarray = field(init=False)

    def __str__(self):
        return f"AMRO_Data_Object_{self.experiment_key}"

    def __post_init__(self):
        self.mean_res_ohms = np.mean(self.res_ohms, dtype=float)
        self._get_angle_zero_res()

        self.angles_rads = c.convert_degs_to_rads(self.angles_degs)

        self._calc_plotting_values()
        self._calc_res_uohm_values()
        self._calc_res_normed_values()

    def compare_act(self, other_act: str) -> bool:
        return self.experiment_key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)

    def _calc_res_normed_values(self):
        self.delta_res_0deg_norm = self.delta_res_mean_ohms / self.mean_res_ohms
        self.delta_res_mean_norm = self.delta_res_0deg_ohms / self.deg0_res_ohms

    def _calc_plotting_values(self):
        """Calculates various values using different units for clearer plotting."""

        self.delta_res_mean_ohms = self.res_ohms - self.mean_res_ohms

        return

    def _calc_res_uohm_values(self) -> None:

        for attribute in fields(self):
            if attribute.name.endswith("_ohm"):
                vals = getattr(self, attribute.name)
                new_name = attribute.name.replace("_ohm", "_uohm")
                new_vals = c.convert_ohms_to_uohms(vals)
                setattr(self, new_name, new_vals)

        return

    def _get_angle_zero_res(self):
        """Get the resistivity measurement at the very start of the oscillation, i.e. when the sample angle equals 0."""
        return


@dataclass
class FourierResult:
    """Stores the results of a Fourier Transform."""

    experiment_key: ExperimentKey
    symmetries: list | np.ndarray
    phases: list | np.ndarray
    amplitudes: list | np.ndarray

    def __str__(self):
        return f"Fourier_Result_Object_{self.experiment_key}"

    def get_n_strongest_components(self):
        return

    def compare_act(self, other_act: str) -> bool:
        return self.experiment_key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)


@dataclass
class Experiment:
    """Stores the data and results of an AMRO experiment. fit_result and fourier_result are to be added after the
    dataclass object is instantiated."""

    experiment_key: ExperimentKey
    amro_data: AMROData

    fit_result: Optional[FitResult]
    fourier_result: Optional[FourierResult]

    def __str__(self):
        return f"Experiment_Object_{self.experiment_key}"

    def compare_act(self, other_act: str) -> bool:
        return self.experiment_key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)

    def add_fit_result(self, lmfit_result: lm.minimizer.MinimizerResult):
        self.fit_result = FitResult(lmfit_result)
        return

    def add_fourier_result(self, fourier_result):
        self.fourier_result = FourierResult(
            experiment_key=self.experiment_key,
        )
        return
