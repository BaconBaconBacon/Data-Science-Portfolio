from dataclasses import dataclass, field, fields
import numpy as np
import pandas as pd
import lmfit as lm
import pickle
from pathlib import Path
from amro.config import FINAL_DATA_PATH
from amro.config.settings import (
    HEADER_ACT,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_GEO,
    HEADER_ANGLE_RAD,
    HEADER_ANGLE_DEG,
    HEADER_RES_OHM,
    HEADER_FREQ,
    HEADER_MAG,
    HEADER_PHASE,
)
from amro.utils import conversions as c
from amro.utils import utils as u

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

    def __str__(self):
        return u.format_oscillation_key(
            self.experiment_label, self.temperature, self.magnetic_field
        )

    def __repr__(self):
        return str(self)

    def compare_act(self, other_act: str) -> bool:
        return self.experiment_label == other_act

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.temperature == other_temperature

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.magnetic_field == other_magnetic_field

    def get_experiment_label(self) -> str:
        return self.experiment_label

    def get_temperature(self) -> float:
        return self.temperature

    def get_magnetic_field(self) -> float:
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

    symmetries_errs: list | np.ndarray = field(init=False)
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

    def __str__(self):
        return f"Fit_Result_Object_{self.experiment_key}"

    def __post_init__(self):

        # Get the relevant info from lmfit_result
        self.chi_squared = self.lmfit_result.chisqr
        self.red_chi_squared = self.lmfit_result.redchi
        self.covar_matrix = self.lmfit_result.covar
        self.fit_succeeded = self.lmfit_result.success  # self._check_fit_success()

        self._parse_params(self.lmfit_result.params)

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
        return u.convert_params_to_ndarrays(self.lmfit_params, include_errs=False)

    def get_fitted_params_with_errs(self):
        return u.convert_params_to_ndarrays(self.lmfit_params, include_errs=True)

    def _parse_params(self, params: lm.Parameters) -> None:
        """Must parse the keys and values of the Parameters objects"""
        self.lmfit_params = params
        self.fit_report = lm.report_fit(params)

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
        return self.experiment_key.get_experiment_label()

    def get_temperature(self) -> float:
        return self.experiment_key.get_temperature()

    def get_magnetic_field(self) -> float:
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

    def __str__(self):
        return f"AMRO_Data_Object_{self.experiment_key}"

    def __post_init__(self) -> None:
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
        return self.experiment_key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.experiment_key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.experiment_key.compare_magnetic_field(other_magnetic_field)

    def _calc_normed_res(self) -> None:
        self.delta_res_mean_norm = self.delta_res_mean_ohms / self.mean_res_ohms
        self.delta_res_0deg_norm = self.delta_res_0deg_ohms / self.deg0_res_ohms

        self.delta_res_mean_norm_pct = self.delta_res_mean_norm * 100
        self.delta_res_0deg_norm_pct = self.delta_res_0deg_norm * 100

    def _calc_delta_res(self) -> None:
        """Calculates various values using different units for clearer plotting."""

        self.delta_res_mean_ohms = self.res_ohms - self.mean_res_ohms
        self.delta_res_0deg_ohms = self.res_ohms - self.deg0_res_ohms
        return

    def _validate_inputs(self):
        if min(self.angles_degs) < 0:
            raise ValueError("Angles must be non-negative")
        elif min(self.res_ohms) < 0:
            raise ValueError("Resistivity must be non-negative")

    def _calc_res_uohm_values(self) -> None:
        """Call this last at initialization. Iterates over the class object's attributes."""
        for attribute in fields(self):
            if attribute.name.endswith("_ohm"):
                vals = getattr(self, attribute.name)
                new_name = attribute.name.replace("_ohm", "_uohm")
                new_vals = c.convert_ohms_to_uohms(vals)
                setattr(self, new_name, new_vals)

        return

    def _get_angle_zero_res(self) -> None:
        """Get the resistivity measurement at the very start of the oscillation, i.e. when the sample angle equals 0."""
        id_min = np.argmin(self.angles_degs)
        self.deg0_res_ohms = self.res_ohms[id_min]
        return

    def get_experiment_label(self) -> str:
        return self.experiment_key.get_experiment_label()

    def get_temperature(self) -> float:
        return self.experiment_key.get_temperature()

    def get_magnetic_field(self) -> float:
        return self.experiment_key.get_magnetic_field()


@dataclass
class FourierResult:
    """
    Stores the results of a Fourier Transform. yf is a list of complex numbers outputted by
    rfft()
    """

    key: OscillationKey

    symmetries: list | np.ndarray
    yf: list | np.ndarray

    phases: np.ndarray = field(init=False)
    amplitudes: np.ndarray = field(init=False)

    amplitudes_ratio: np.ndarray = field(init=False)
    phases_pos: np.ndarray = field(init=False)

    guesses_dict: dict = field(default_factory=dict)

    def __post_init__(self) -> None:

        self.symmetries = np.asarray(self.symmetries).astype(int)
        self.yf = np.asarray(self.yf)

        self.phases = np.angle(self.yf)
        self.amplitudes = np.abs(self.yf)
        self.amplitudes_ratio = np.divide(self.amplitudes, self.amplitudes.max())
        self.phases_pos = np.where(
            self.phases < 0,
            self.phases + 2 * np.pi,
            self.phases,
        )
        for i, f in enumerate(self.symmetries):
            self.guesses_dict[f] = (self.amplitudes_ratio[i], self.phases_pos[i])

    def __str__(self):
        return f"Fourier_Result_Object_{self.key}"

    def get_n_strongest_components(self, n=0):
        idx_largest = np.argpartition(self.amplitudes_ratio, -n)[-n:]
        n_syms = self.symmetries[idx_largest]
        n_ratios = self.amplitudes_ratio[idx_largest]
        return zip(n_syms, n_ratios)

    def compare_act(self, other_act: str) -> bool:
        return self.key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.key.compare_magnetic_field(other_magnetic_field)

    def get_experiment_label(self) -> str:
        return self.key.get_experiment_label()

    def get_temperature(self) -> float:
        return self.key.get_temperature()

    def get_magnetic_field(self) -> float:
        return self.key.get_magnetic_field()

    def get_init_params(self):
        """returns fourier component symmetries, their amplitudes and phases, and the mean"""
        return


@dataclass
class AMROscillation:
    """Stores the data and analysis results of a single AMR oscillation, at a given T and H.

    fit_result and fourier_result are to be added after the AMROscillation object is instantiated.
    """

    key: OscillationKey
    osc_data: ExperimentalData

    fit_result: FitResult = field(init=False)
    fourier_result: FourierResult = field(init=False)

    def __str__(self):
        return f"Experiment_Object_{self.key}"

    def compare_act(self, other_act: str) -> bool:
        return self.key.compare_act(other_act)

    def compare_temperature(self, other_temperature: float) -> bool:
        return self.key.compare_temperature(other_temperature)

    def compare_magnetic_field(self, other_magnetic_field: float) -> bool:
        return self.key.compare_magnetic_field(other_magnetic_field)

    def add_fit_result(
        self,
        lmfit_result: lm.minimizer.MinimizerResult,
        refitted: bool,
    ) -> None:
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
        """Input are the results of rfft in fourier.py, runs calculates to read into
        FourierResult. Copy over the fourier._pack_ft_result() function, act, h, and t will
        have been determined previously when accessing the AMROscillation object."""

        self.fourier_result = FourierResult(key=self.key, symmetries=xf, yf=yf)
        return

    def get_experiment_label(self) -> str:
        return self.key.get_experiment_label()

    def get_temperature(self) -> float:
        return self.key.get_temperature()

    def get_magnetic_field(self) -> float:
        return self.key.get_magnetic_field()

    def _calc_model_resistivities(self, params: lm.parameter.Parameters):
        params = u.convert_params_to_ndarrays(params)
        model_res = u.calculate_model_resistivities(self.osc_data.angles_rads, params)
        return model_res

    def get_n_strongest_fourier(self, n=0):
        if self.fourier_result is not None:
            return self.fourier_result.get_n_strongest_components(n)
        else:
            print("Fourier result not present.")
            return None

    def get_fourier_init_params(self):
        return self.fit_result.get_init_params()


@dataclass
class Experiment:
    """Handles dataclasses for all oscillations for a given experimental set up."""

    experiment_label: str  # i.e. ACTRot11
    geometry: str  # i.e. para/perp

    length: float
    width: float
    height: float
    oscillations_dict: dict = field(default_factory=dict)
    oscillations_count: float = 0

    def add_oscillation(self, oscillation: AMROscillation) -> None:
        new_key = oscillation.key
        if new_key not in self.oscillations_dict.keys():
            self.oscillations_dict[new_key] = oscillation
            self.oscillations_count += 1
        else:
            print(f"Key {new_key} already exists! Use replace_oscillation() instead.")
        return

    def replace_oscillation(self, oscillation: AMROscillation) -> None:
        new_key = oscillation.key
        self.oscillations_dict[new_key] = oscillation
        return

    def get_oscillation(self, t: float, h: float) -> AMROscillation:
        request_key = OscillationKey(self.experiment_label, t, h)
        return self.oscillations_dict[request_key]

    def get_oscillation_from_key(self, request_key: OscillationKey) -> AMROscillation:
        return self.oscillations_dict[request_key]

    def get_multiple_oscillations(
        self,
        t: float | list | None = None,
        h: float | list | None = None,
    ) -> list | None:
        """If inputs are not None, returns all oscillations whose T and H
        values exactly match the requested values.  If requested values are
        None, returns all existing entries for that variable.

        If for whatever reason this ever gets slow, try usings sets() for t and h."""

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


@dataclass
class ProjectData:
    """Handles dataclasses for all experiments for a given project"""

    project_name: str
    experiments_dict: dict = field(default_factory=dict)
    experiments_count: float = 0

    def __post_init__(self):

        self.pickle_fp = FINAL_DATA_PATH / self.project_name + ".pkl"

    def add_experiment(self, exp: Experiment) -> None:
        self.experiments_dict[exp.experiment_label] = exp
        self.experiments_count += 1

    def get_experiment(self, act_label: str) -> Experiment:
        return self.experiments_dict[act_label]

    def get_experiment_labels(self):
        return self.experiments_dict.keys()

    def filter_oscillations(
        self,
        experiments: str | list | None = None,
        t_vals: float | list | None = None,
        h_vals: float | list | None = None,
    ) -> list:
        if experiments is None:
            experiments = self.experiments_dict.keys()
        elif isinstance(experiments, str):
            experiments = [experiments]

        osc_list = []
        for exp_label in experiments:
            exp = self.experiments_dict[exp_label]
            oscs = exp.get_multiple_oscillations(t_vals, h_vals)
            osc_list.append(oscs)
        return osc_list

    def read_amro_data_from_dataframe(self, df: pd.DataFrame) -> None:
        """Temporary stop-gap. Want to remove pandas entirely, eventually."""

        for act in df[HEADER_ACT].unique():
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
                    angles = experiment_df[HEADER_ANGLE_DEG]
                    res = experiment_df[HEADER_RES_OHM]

                    exp_data = ExperimentalData(
                        experiment_key=key, angles_degs=angles, res_ohms=res
                    )
                    osc = AMROscillation(key=key, osc_data=exp_data)
                    exper.add_oscillation(osc)
        return

    def read_fit_results_from_dataframe(
        self, df: pd.DataFrame, lmfit_results_dict: dict
    ) -> None:
        """Temporary stop-gap. Want to remove pandas entirely, eventually.

        df: fit_params_df
        lmfit_results_dict: lmfit_results_objs
        """

        for act in df[HEADER_ACT].unique():
            sub_df = u.query_dataframe(df, act=act)
            try:
                exper = self.get_experiment(act)
            except KeyError:
                print(
                    "No Experiment found for "
                    + act
                    + ". Create Experiment before adding Fourier Results."
                )
                return

            for t in sub_df[HEADER_TEMP].unique():
                sub_sub_df = u.query_dataframe(sub_df, t=t)
                for h in sub_sub_df[HEADER_MAGNET].unique():

                    fit_result_df = u.query_dataframe(sub_sub_df, h=h)
                    lmfit_obj, refitted = lmfit_results_dict[act][t][h]

                    osc = exper.get_oscillation(t=t, h=h)

                    osc.add_fit_result(
                        lmfit_result=lmfit_obj,
                        # successful_fit=lmfit_obj.success,
                        refitted=refitted,
                    )

        return

    def read_fourier_results_from_dataframe(self, df: pd.DataFrame) -> None:
        """Temporary stop-gap. Want to remove pandas entirely, eventually."""
        for act in df[HEADER_ACT].unique():
            sub_df = u.query_dataframe(df, act=act)

            try:
                exper = self.get_experiment(act)
            except KeyError:
                print(
                    "No Experiment found for "
                    + act
                    + ". Create Experiment before adding Fourier Results."
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

    def save_to_pickle(self, fp: Path = None) -> None:
        if fp is None:
            fp = self.pickle_fp
        with open(fp, "wb") as f:
            pickle.dump(self.experiments_dict, f)
        return

    def load_from_pickle(self) -> None:
        with open(self.pickle_fp, "rb") as f:
            self.experiments_dict = pickle.load(f)
        return

    def export_fit_results_to(self, suffix=".csv"):
        """Nice-to-have once the dataclasses are implemented in code base."""
        return

    def import_fit_results_from(self, suffix=".csv"):
        """Nice-to-have once the dataclasses are implemented in code base."""
        return

    def export_fourier_results_to(self, suffix=".csv"):
        """Nice-to-have once the dataclasses are implemented in code base."""
        return

    def import_fourier_results_from(self, suffix=".csv"):
        """Nice-to-have once the dataclasses are implemented in code base."""
        return
