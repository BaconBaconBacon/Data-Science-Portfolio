"""AMRO Fourier Transform and Fitting Analysis Package"""

__version__ = "0.1.0"

from .data.loader import AMROLoader
from .features.fourier import Fourier
from .models.fitter import AMROFitter
from .data.data_structures import (
    ExperimentKey,
    FitResult,
    ExperimentalData,
    FourierResult,
    Experiment,
)

__all__ = [
    "AMROLoader",
    "Fourier",
    "AMROFitter",
    "ExperimentKey",
    "FitResult",
    "ExperimentalData",
    "FourierResult",
    "Experiment",
]
