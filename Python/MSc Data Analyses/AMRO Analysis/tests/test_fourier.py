"""Tests for amro.features.fourier module."""

import pytest
import numpy as np

from amro.config import HEADER_EXPERIMENT_PREFIX
from amro.data import (
    OscillationKey,
    ExperimentalData,
    AMROscillation,
    Experiment,
    ProjectData,
    FourierResult,
)
from amro.features import Fourier


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_project_data_with_oscillations():
    """Create ProjectData with oscillations for Fourier testing."""
    project = ProjectData(project_name="test_fourier")
    exp = Experiment(
        experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
        geometry="perp",
        wire_sep=1.0,
        cross_section=0.5,
    )

    # Create oscillation with known frequency content
    for t in [2.0, 5.0]:
        key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, 3.0)
        angles = np.linspace(0, 360, 361)
        angles_rad = np.deg2rad(angles)
        # Create signal with known 4-fold and 2-fold components
        mean_res = 1e-5
        res = mean_res * (
            1 + 0.1 * np.sin(4 * angles_rad) + 0.05 * np.sin(2 * angles_rad)
        )
        data = ExperimentalData(key, angles, res)
        osc = AMROscillation(key, data)
        exp.add_oscillation(osc)

    project.add_experiment(exp)
    return project


@pytest.fixture
def fourier_instance(sample_project_data_with_oscillations):
    """Create a Fourier instance."""
    return Fourier(
        amro_data=sample_project_data_with_oscillations,
        save_name="test_fourier",
        verbose=False,
    )


@pytest.fixture
def sample_experimental_data():
    """Create ExperimentalData with known frequency content."""
    key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
    angles = np.linspace(0, 360, 361)
    angles_rad = np.deg2rad(angles)
    mean_res = 1e-5
    # Known signal: 4-fold dominant, 2-fold secondary
    res = mean_res * (1 + 0.1 * np.sin(4 * angles_rad) + 0.05 * np.sin(2 * angles_rad))
    return ExperimentalData(key, angles, res)


# =============================================================================
# Fourier Initialization Tests
# =============================================================================


class TestFourierInit:
    def test_initialization(self, fourier_instance):
        assert fourier_instance is not None

    def test_stores_project_data(
        self, fourier_instance, sample_project_data_with_oscillations
    ):
        assert fourier_instance.project_data is sample_project_data_with_oscillations

    def test_stores_save_name(self, fourier_instance):
        assert fourier_instance.save_name == "test_fourier"

    def test_verbose_mode(self, sample_project_data_with_oscillations):
        fourier = Fourier(
            amro_data=sample_project_data_with_oscillations,
            save_name="test",
            verbose=True,
        )
        assert fourier.verbose is True


# =============================================================================
# Fourier Transform Tests
# =============================================================================


class TestPerformFourierTransform:
    def test_returns_xf_and_yf(self, fourier_instance, sample_experimental_data):
        xf, yf = fourier_instance._perform_fourier_transform(sample_experimental_data)
        assert xf is not None
        assert yf is not None

    def test_xf_contains_frequencies(self, fourier_instance, sample_experimental_data):
        xf, yf = fourier_instance._perform_fourier_transform(sample_experimental_data)
        # Should contain integer frequencies
        assert len(xf) > 0
        # Frequencies should be positive (DC component removed)
        assert all(xf > 0)

    def test_yf_is_complex(self, fourier_instance, sample_experimental_data):
        xf, yf = fourier_instance._perform_fourier_transform(sample_experimental_data)
        assert np.iscomplexobj(yf)

    def test_removes_dc_component(self, fourier_instance, sample_experimental_data):
        xf, yf = fourier_instance._perform_fourier_transform(sample_experimental_data)
        # xf should not contain 0 (DC component)
        assert 0 not in xf

    def test_detects_dominant_frequency(
        self, fourier_instance, sample_experimental_data
    ):
        """Test that 4-fold symmetry is detected as strongest."""
        xf, yf = fourier_instance._perform_fourier_transform(sample_experimental_data)

        # Find the strongest component
        amplitudes = np.abs(yf)
        max_idx = np.argmax(amplitudes)
        dominant_freq = xf[max_idx]

        # Should be 4 (the strongest component in our test signal)
        assert dominant_freq == 4


# =============================================================================
# Fourier Transform Experiments Tests
# =============================================================================


class TestFourierTransformExperiments:
    def test_adds_fourier_results_to_oscillations(
        self, fourier_instance, sample_project_data_with_oscillations
    ):
        fourier_instance.fourier_transform_experiments()

        # Check that all oscillations now have fourier results
        for exp in sample_project_data_with_oscillations.experiments_dict.values():
            for osc in exp.oscillations_dict.values():
                assert hasattr(osc, "fourier_result")
                assert isinstance(osc.fourier_result, FourierResult)

    def test_summary_statistics_updated(
        self, fourier_instance, sample_project_data_with_oscillations
    ):
        # Before transform
        stats_before = sample_project_data_with_oscillations.get_summary_statistics()
        assert stats_before["n_fourier_completed"] == 0

        # Perform transform
        fourier_instance.fourier_transform_experiments()

        # After transform
        stats_after = sample_project_data_with_oscillations.get_summary_statistics()
        assert stats_after["n_fourier_completed"] == stats_after["n_oscillations"]


# =============================================================================
# Get N Strongest Results Tests
# =============================================================================


class TestGetNStrongestResults:
    def test_returns_list(self, fourier_instance):
        fourier_instance.fourier_transform_experiments()
        results = fourier_instance.get_n_strongest_results(n=2)
        assert isinstance(results, list)

    def test_filters_by_temperature(self, fourier_instance):
        fourier_instance.fourier_transform_experiments()
        results = fourier_instance.get_n_strongest_results(n=2, t=2.0)
        # Should only have results from t=2.0
        for key, _ in results:
            assert key.temperature == 2.0

    def test_filters_by_field(self, fourier_instance):
        fourier_instance.fourier_transform_experiments()
        results = fourier_instance.get_n_strongest_results(n=2, h=3.0)
        # Should only have results from h=3.0
        for key, _ in results:
            assert key.magnetic_field == 3.0

    def test_filters_by_experiment(self, fourier_instance):
        fourier_instance.fourier_transform_experiments()
        results = fourier_instance.get_n_strongest_results(
            n=2, act=HEADER_EXPERIMENT_PREFIX + "11"
        )
        for key, _ in results:
            assert key.experiment_label == HEADER_EXPERIMENT_PREFIX + "11"


# =============================================================================
# Known Signal Tests
# =============================================================================


class TestKnownSignalRecovery:
    """Test that Fourier transform correctly recovers known frequency components."""

    def test_single_frequency_detection(self):
        """Test detection of a single frequency component."""
        project = ProjectData(project_name="test_single")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=1.0,
            cross_section=0.5,
        )

        key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        angles = np.linspace(0, 360, 361)
        angles_rad = np.deg2rad(angles)
        # Pure 4-fold signal
        mean_res = 1e-5
        res = mean_res * (1 + 0.1 * np.sin(4 * angles_rad))
        data = ExperimentalData(key, angles, res)
        osc = AMROscillation(key, data)
        exp.add_oscillation(osc)
        project.add_experiment(exp)

        fourier = Fourier(amro_data=project, save_name="test", verbose=False)
        fourier.fourier_transform_experiments()

        # Check that 4-fold is dominant
        fr = osc.fourier_result
        max_idx = np.argmax(fr.amplitudes)
        assert fr.xf[max_idx] == 4

    def test_multiple_frequency_detection(self):
        """Test detection of multiple frequency components."""
        project = ProjectData(project_name="test_multi")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=1.0,
            cross_section=0.5,
        )

        key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        angles = np.linspace(0, 360, 361)
        angles_rad = np.deg2rad(angles)
        # Signal with 4-fold, 2-fold, and 6-fold components
        mean_res = 1e-5
        res = mean_res * (
            1
            + 0.1 * np.sin(4 * angles_rad)
            + 0.05 * np.sin(2 * angles_rad)
            + 0.02 * np.sin(6 * angles_rad)
        )
        data = ExperimentalData(key, angles, res)
        osc = AMROscillation(key, data)
        exp.add_oscillation(osc)
        project.add_experiment(exp)

        fourier = Fourier(amro_data=project, save_name="test", verbose=False)
        fourier.fourier_transform_experiments()

        # Get n strongest should include 4 and 2
        fr = osc.fourier_result
        strongest = list(fr.get_n_strongest_components(n=3))
        freqs = [item[0] for item in strongest]

        assert 4 in freqs
        assert 2 in freqs
