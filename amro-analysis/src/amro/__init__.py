"""AMRO Fourier Transform and Fitting Analysis Package"""

__version__ = "0.1.0"


from .data.loader import AMROLoader
from .data.cleaner import AMROCleaner
from .features.fourier import Fourier
from .models.fitter import AMROFitter
from .data.data_structures import (
    OscillationKey,
    FitResult,
    ExperimentalData,
    FourierResult,
    Experiment,
)

__all__ = [
    "AMROCleaner",
    "AMROLoader",
    "Fourier",
    "AMROFitter",
    "OscillationKey",
    "FitResult",
    "ExperimentalData",
    "FourierResult",
    "Experiment",
]
