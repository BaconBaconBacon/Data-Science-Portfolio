"""Shared pytest fixtures for AMRO test suite."""

import pytest
import numpy as np
import pandas as pd
import lmfit as lm
from pathlib import Path

from amro.config import (
    HEADER_EXPERIMENT_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_FREQ_PREFIX,
    HEADER_EXP_LABEL,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_GEO,
    HEADER_ANGLE_DEG,
    HEADER_RES_OHM,
)
from amro.data import (
    OscillationKey,
    ExperimentalData,
    FourierResult,
    AMROscillation,
    Experiment,
    ProjectData,
)


# =============================================================================
# Test Isolation Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def isolate_test_paths(tmp_path, monkeypatch):
    """Redirect all data paths to temp directory to prevent test pollution.

    This fixture runs automatically for all tests, ensuring that:
    - Tests don't write to real data directories
    - Tests don't interfere with each other
    - Test artifacts are automatically cleaned up
    """
    # Patch the paths in all modules that import them
    monkeypatch.setattr("amro.data.data_structures.FINAL_DATA_PATH", tmp_path)
    monkeypatch.setattr("amro.data.loader.RAW_DATA_PATH", tmp_path)
    monkeypatch.setattr("amro.data.loader.PROCESSED_DATA_PATH", tmp_path)
    monkeypatch.setattr("amro.data.cleaner.RAW_DATA_PATH", tmp_path)
    monkeypatch.setattr("amro.data.cleaner.PROCESSED_DATA_PATH", tmp_path)
    monkeypatch.setattr("amro.features.fourier.FINAL_DATA_PATH", tmp_path)
    monkeypatch.setattr("amro.plotting.fitter.PROCESSED_FIGURES_PATH", tmp_path)

    return tmp_path


# =============================================================================
# Basic Data Fixtures
# =============================================================================


@pytest.fixture
def sample_oscillation_key():
    """Standard OscillationKey for testing."""
    return OscillationKey(
        experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
        temperature=2.0,
        magnetic_field=3.0,
    )


@pytest.fixture
def sample_angles():
    """Returns angles from 0 to 360 degrees."""
    return np.linspace(0, 360, 361)


@pytest.fixture
def sample_resistivities():
    """Returns synthetic AMRO-like resistivity data."""
    angles_rad = np.linspace(0, 2 * np.pi, 361)
    mean_res = 1e-5
    # Simulate 4-fold and 2-fold symmetry oscillations
    oscillation = 0.1 * np.sin(4 * angles_rad) + 0.05 * np.sin(2 * angles_rad)
    return mean_res * (1 + oscillation)


@pytest.fixture
def sample_experimental_data(
    sample_oscillation_key, sample_angles, sample_resistivities
):
    """ExperimentalData instance with synthetic AMRO data."""
    return ExperimentalData(
        experiment_key=sample_oscillation_key,
        angles_degs=sample_angles,
        res_ohms=sample_resistivities,
    )


@pytest.fixture
def sample_amro_oscillation(sample_oscillation_key, sample_experimental_data):
    """AMROscillation instance without fourier or fit results."""
    return AMROscillation(
        key=sample_oscillation_key,
        osc_data=sample_experimental_data,
    )


# =============================================================================
# Fourier Fixtures
# =============================================================================


@pytest.fixture
def sample_fourier_xf():
    """Fourier transform frequencies (symmetries)."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture
def sample_fourier_yf():
    """Fourier transform complex amplitudes."""
    magnitudes = np.array([0.01, 0.05, 0.02, 0.10, 0.01, 0.01, 0.01, 0.01])
    phases = np.array([0.5, 1.0, 0.3, 0.8, 0.2, 0.1, 0.4, 0.6])
    return magnitudes * np.exp(1j * phases)


@pytest.fixture
def sample_fourier_result(sample_oscillation_key, sample_fourier_xf, sample_fourier_yf):
    """FourierResult instance."""
    return FourierResult(
        key=sample_oscillation_key,
        xf=sample_fourier_xf,
        yf=sample_fourier_yf,
    )


# =============================================================================
# Fitter Fixtures
# =============================================================================


@pytest.fixture
def sample_lmfit_params():
    """Creates lmfit Parameters for testing."""
    params = lm.Parameters()
    params.add(HEADER_PARAM_MEAN_PREFIX, value=1e-5, min=0)
    params.add(HEADER_PARAM_FREQ_PREFIX + "4", value=4, vary=False)
    params.add(HEADER_PARAM_AMP_PREFIX + "4", value=0.1, min=0)
    params.add(
        HEADER_PARAM_PHASE_PREFIX + "4", value=0.8, min=-2 * np.pi, max=2 * np.pi
    )
    return params


@pytest.fixture
def sample_lmfit_result(sample_lmfit_params):
    """Creates a minimal lmfit MinimizerResult for testing."""
    params = sample_lmfit_params

    x = np.linspace(0, 2 * np.pi, 100)
    y = 1e-5 * (1 + 0.1 * np.sin(4 * x + 0.8))

    def residual_func(p, x, y):
        model = p[HEADER_PARAM_MEAN_PREFIX].value * (
            1
            + p[HEADER_PARAM_AMP_PREFIX + "4"].value
            * np.sin(
                p[HEADER_PARAM_FREQ_PREFIX + "4"].value * x
                + p[HEADER_PARAM_PHASE_PREFIX + "4"].value
            )
        )
        return model - y

    minimizer = lm.Minimizer(residual_func, params, fcn_args=(x, y))
    result = minimizer.minimize()
    return result


# =============================================================================
# Experiment/Project Fixtures
# =============================================================================


@pytest.fixture
def sample_experiment():
    """Experiment instance without oscillations."""
    return Experiment(
        experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
        geometry="perp",
        wire_sep=1.0,
        cross_section=0.05,
    )


@pytest.fixture
def sample_project_data():
    """ProjectData instance without experiments."""
    return ProjectData(project_name="test_project")


@pytest.fixture
def populated_experiment():
    """Experiment with multiple oscillations added."""
    exp = Experiment(
        experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
        geometry="perp",
        wire_sep=1.0,
        cross_section=0.05,
    )
    for t in [2.0, 5.0, 10.0]:
        for h in [3.0, 7.0]:
            key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, h)
            angles = np.linspace(0, 360, 361)
            res = np.full(361, 1e-5)
            data = ExperimentalData(key, angles, res)
            osc = AMROscillation(key, data)
            exp.add_oscillation(osc)
    return exp


@pytest.fixture
def populated_project_data(populated_experiment):
    """ProjectData with experiments and oscillations."""
    project = ProjectData(project_name="test_project")
    project.add_experiment(populated_experiment)
    return project


# =============================================================================
# DataFrame Fixtures (for loader tests)
# =============================================================================


@pytest.fixture
def sample_amro_dataframe():
    """Creates a sample DataFrame mimicking loaded AMRO data."""
    n_points = 361
    angles = np.linspace(0, 360, n_points)
    res = np.full(n_points, 1e-5)

    return pd.DataFrame(
        {
            HEADER_EXP_LABEL: [HEADER_EXPERIMENT_PREFIX + "11"] * n_points,
            HEADER_TEMP: [2.0] * n_points,
            HEADER_MAGNET: [3.0] * n_points,
            HEADER_GEO: ["perp"] * n_points,
            HEADER_ANGLE_DEG: angles,
            HEADER_RES_OHM: res,
        }
    )
