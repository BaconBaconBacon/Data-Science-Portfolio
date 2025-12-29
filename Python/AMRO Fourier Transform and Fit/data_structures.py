from dataclasses import dataclass
from typing import Optional

import lmfit as lm

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
        return f"{self.act}_T{self.temperature}_H{self.magnetic_field}"


@dataclass
class FitResult:
    """Store fit and Fourier transform information"""

    experiment_key: ExperimentKey
    lmfit_result: lm.minimizer.MinimizerResult
    chi_squared: float
    fit_succeeded: bool


@dataclass
class AMROData:
    """Stores an experiment's AMRO data, i.e. resistivity and sample angle."""

    experiment_key: ExperimentKey


@dataclass
class FourierResult:
    """Stores the results of a Fourier Transform."""

    experiment_key: ExperimentKey


@dataclass
class Experiment:
    """Stores the data and results of an AMRO experiment."""

    experiment_key: ExperimentKey
    fit_result: FitResult
    foutier_result: FourierResult
    amro_data: AMROData
    act_val: str
    t_val: float | int
    h_val: float | int
