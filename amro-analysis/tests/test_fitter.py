"""Tests for amro.models.fitter module."""

import pytest
import numpy as np
import lmfit as lm

from amro.config import (
    HEADER_EXPERIMENT_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_FREQ_PREFIX,
)
from amro.data import (
    OscillationKey,
    ExperimentalData,
    FourierResult,
    FitResult,
    AMROscillation,
    Experiment,
    ProjectData,
)
from amro.features.fourier import Fourier
from amro.models.fitter import AMROFitter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_project_data_with_fourier():
    """Create ProjectData with oscillations that have Fourier results."""
    project = ProjectData(project_name="test_fitter")
    exp = Experiment(
        experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
        geometry="perp",
        wire_sep=1.0,
        cross_section=0.5,
    )

    for t in [2.0, 5.0]:
        key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, 3.0)
        angles = np.linspace(0, 360, 361)
        angles_rad = np.deg2rad(angles)
        mean_res = 1e-5
        res = mean_res * (
            1 + 0.1 * np.sin(4 * angles_rad) + 0.05 * np.sin(2 * angles_rad)
        )
        data = ExperimentalData(key, angles, res)
        osc = AMROscillation(key, data)
        exp.add_oscillation(osc)

    project.add_experiment(exp)

    # Run Fourier transform on all oscillations
    fourier = Fourier(amro_data=project, verbose=False)
    fourier.fourier_transform_experiments()

    return project


@pytest.fixture
def fitter_instance(sample_project_data_with_fourier):
    """Create an AMROFitter instance."""
    return AMROFitter(
        amro_data=sample_project_data_with_fourier,
        min_amp_ratio=0.01,  # Low threshold to include most frequencies
        max_freq=10,
        verbose=False,
    )


@pytest.fixture
def sample_fourier_result():
    """Create a FourierResult for testing parameter initialization."""
    key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
    # Frequencies 2 and 4 with different amplitudes
    xf = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    magnitudes = np.array([0.01, 0.05, 0.02, 0.10, 0.01, 0.01, 0.01, 0.01])
    phases = np.array([0.5, 1.0, 0.3, 0.8, 0.2, 0.1, 0.4, 0.6])
    yf = magnitudes * np.exp(1j * phases)
    return FourierResult(key=key, xf=xf, yf=yf)


# =============================================================================
# AMROFitter Initialization Tests
# =============================================================================


class TestAMROFitterInit:
    def test_initialization(self, fitter_instance):
        assert fitter_instance is not None

    def test_stores_min_amp_ratio(self, fitter_instance):
        assert fitter_instance.min_amp_ratio == 0.01

    def test_stores_max_freq(self, fitter_instance):
        assert fitter_instance.max_freq == 10

    def test_stores_verbose(self, fitter_instance):
        assert fitter_instance.verbose is False

    def test_failed_fits_initially_empty(self, fitter_instance):
        assert len(fitter_instance.failed_fits) == 0

    def test_filter_string_created(self, fitter_instance):
        assert "0.01" in fitter_instance.filter_str
        assert "10" in fitter_instance.filter_str


# =============================================================================
# Parameter Initialization Tests
# =============================================================================


class TestInitializeParametersFromFourier:
    def test_returns_params_and_freq_list(self, fitter_instance, sample_fourier_result):
        params, f_list = fitter_instance._initialize_parameters_from_fourier(
            sample_fourier_result, mean_res=1e-5
        )
        assert isinstance(params, lm.Parameters)
        assert isinstance(f_list, list)

    def test_includes_mean_parameter(self, fitter_instance, sample_fourier_result):
        params, _ = fitter_instance._initialize_parameters_from_fourier(
            sample_fourier_result, mean_res=1e-5
        )
        assert HEADER_PARAM_MEAN_PREFIX in params

    def test_mean_value_set(self, fitter_instance, sample_fourier_result):
        params, _ = fitter_instance._initialize_parameters_from_fourier(
            sample_fourier_result, mean_res=1e-5
        )
        assert params[HEADER_PARAM_MEAN_PREFIX].value == pytest.approx(1e-5)

    def test_applies_max_freq_filter(self, sample_project_data_with_fourier):
        """Test that frequencies above max_freq are excluded."""
        fitter = AMROFitter(
            amro_data=sample_project_data_with_fourier,
            min_amp_ratio=0.01,
            max_freq=3,  # Should exclude freq 4
            verbose=False,
        )
        exp = sample_project_data_with_fourier.get_experiment(
            HEADER_EXPERIMENT_PREFIX + "11"
        )
        osc = list(exp.oscillations_dict.values())[0]

        params, f_list = fitter._initialize_parameters_from_fourier(
            osc.fourier_result, osc.osc_data.mean_res_ohms
        )

        # Freq 4 should be excluded
        assert 4 not in f_list

    def test_applies_min_amp_ratio_filter(self, sample_project_data_with_fourier):
        """Test that weak components are excluded by min_amp_ratio."""
        fitter = AMROFitter(
            amro_data=sample_project_data_with_fourier,
            min_amp_ratio=0.5,  # High threshold - should exclude weak components
            max_freq=10,
            verbose=False,
        )
        exp = sample_project_data_with_fourier.get_experiment(
            HEADER_EXPERIMENT_PREFIX + "11"
        )
        osc = list(exp.oscillations_dict.values())[0]

        params, f_list = fitter._initialize_parameters_from_fourier(
            osc.fourier_result, osc.osc_data.mean_res_ohms
        )

        # Only strongest component(s) should remain
        assert len(f_list) <= 2


# =============================================================================
# Normalize/Denormalize Tests
# =============================================================================


class TestNormalizeDenormalize:
    def test_normalize_data(self, fitter_instance):
        y = np.array([1e-5, 2e-5, 3e-5])
        y_norm, scale = fitter_instance._normalize_data(y)

        # Normalized max should be 1
        assert np.abs(y_norm).max() == pytest.approx(1.0)
        assert scale == pytest.approx(3e-5)

    def test_normalize_handles_small_values(self, fitter_instance):
        y = np.array([1e-15, 2e-15, 3e-15])
        y_norm, scale = fitter_instance._normalize_data(y)
        # Should not fail with very small values
        assert not np.any(np.isnan(y_norm))

    def test_denormalize_restores_mean(self, fitter_instance):
        params = lm.Parameters()
        params.add(HEADER_PARAM_MEAN_PREFIX, value=0.5)

        scale = 2e-5
        result = fitter_instance._denormalize_parameters(params, scale)

        assert result[HEADER_PARAM_MEAN_PREFIX].value == pytest.approx(1e-5)


# =============================================================================
# Objective Function Tests
# =============================================================================


class TestObjectiveFunction:
    def test_obj_func_returns_residuals(self, fitter_instance):
        params = lm.Parameters()
        params.add(HEADER_PARAM_MEAN_PREFIX, value=1.0)
        params.add(HEADER_PARAM_FREQ_PREFIX + "4", value=4, vary=False)
        params.add(HEADER_PARAM_AMP_PREFIX + "4", value=0.1)
        params.add(HEADER_PARAM_PHASE_PREFIX + "4", value=0.0)

        fitter_instance.current_f_list = [4]

        x = np.linspace(0, 2 * np.pi, 100)
        y = 1.0 * (1 + 0.1 * np.sin(4 * x))

        residuals = fitter_instance._obj_func(params, x, y)

        # Residuals should be near zero for perfect fit
        assert np.allclose(residuals, 0, atol=1e-10)

    def test_obj_func_returns_correct_shape(self, fitter_instance):
        params = lm.Parameters()
        params.add(HEADER_PARAM_MEAN_PREFIX, value=1.0)
        params.add(HEADER_PARAM_FREQ_PREFIX + "4", value=4, vary=False)
        params.add(HEADER_PARAM_AMP_PREFIX + "4", value=0.1)
        params.add(HEADER_PARAM_PHASE_PREFIX + "4", value=0.0)

        fitter_instance.current_f_list = [4]

        x = np.linspace(0, 2 * np.pi, 100)
        y = np.ones(100)

        residuals = fitter_instance._obj_func(params, x, y)

        assert residuals.shape == x.shape


# =============================================================================
# Fit Oscillation Tests
# =============================================================================


class TestFitOscillation:
    def test_fit_oscillation_returns_result_and_refit_flag(
        self, fitter_instance, sample_project_data_with_fourier
    ):
        exp = sample_project_data_with_fourier.get_experiment(
            HEADER_EXPERIMENT_PREFIX + "11"
        )
        osc = list(exp.oscillations_dict.values())[0]

        result, refit_flag = fitter_instance._fit_oscillation(osc)

        assert isinstance(result, lm.minimizer.MinimizerResult)
        assert isinstance(refit_flag, bool)

    def test_fit_oscillation_result_has_params(
        self, fitter_instance, sample_project_data_with_fourier
    ):
        exp = sample_project_data_with_fourier.get_experiment(
            HEADER_EXPERIMENT_PREFIX + "11"
        )
        osc = list(exp.oscillations_dict.values())[0]

        result, _ = fitter_instance._fit_oscillation(osc)

        assert hasattr(result, "params")
        assert HEADER_PARAM_MEAN_PREFIX in result.params


# =============================================================================
# Fit Experiment Tests
# =============================================================================


class TestFitActExperiment:
    def test_fit_adds_fit_results(
        self, fitter_instance, sample_project_data_with_fourier
    ):
        fitter_instance.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "11")

        exp = sample_project_data_with_fourier.get_experiment(
            HEADER_EXPERIMENT_PREFIX + "11"
        )
        for osc in exp.oscillations_dict.values():
            assert osc.fit_result is not None
            assert isinstance(osc.fit_result, FitResult)

    def test_fit_updates_summary_statistics(
        self, fitter_instance, sample_project_data_with_fourier
    ):
        stats_before = sample_project_data_with_fourier.get_summary_statistics()
        assert stats_before["n_fits_completed"] == 0

        fitter_instance.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "11")

        stats_after = sample_project_data_with_fourier.get_summary_statistics()
        assert stats_after["n_fits_completed"] == stats_after["n_oscillations"]

    def test_skips_already_fitted(
        self, fitter_instance, sample_project_data_with_fourier, capsys
    ):
        # Fit once
        fitter_instance.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "11")

        # Fit again - should skip
        fitter_instance.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "11")

        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    def test_skips_no_fourier(self, sample_project_data_with_fourier, capsys):
        """Test that oscillations without Fourier results are skipped."""
        # Create new experiment without Fourier results
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "12",
            geometry="para",
            wire_sep=1.0,
            cross_section=0.5,
        )
        key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "12", 2.0, 3.0)
        angles = np.linspace(0, 360, 361)
        res = np.full(361, 1e-5)
        data = ExperimentalData(key, angles, res)
        osc = AMROscillation(key, data)
        exp.add_oscillation(osc)
        sample_project_data_with_fourier.add_experiment(exp)

        fitter = AMROFitter(
            amro_data=sample_project_data_with_fourier,
            #
            min_amp_ratio=0.01,
            max_freq=10,
            verbose=False,
        )
        fitter.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "12")

        captured = capsys.readouterr()
        assert "No Fourier" in captured.out


# =============================================================================
# Failed Fits Tracking Tests
# =============================================================================


class TestFailedFitsTracking:
    def test_failed_fits_initially_empty(self, fitter_instance):
        assert len(fitter_instance.failed_fits) == 0

    def test_failed_fits_tracked(
        self, fitter_instance, sample_project_data_with_fourier
    ):
        # After fitting, check failed_fits
        fitter_instance.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "11")
        # With good data, should have few or no failed fits
        # This is more of a smoke test
        assert isinstance(fitter_instance.failed_fits, list)


# =============================================================================
# Refit Tests
# =============================================================================


class TestRefit:
    def test_refit_modifies_phase_bounds(self, fitter_instance):
        params = lm.Parameters()
        params.add(HEADER_PARAM_MEAN_PREFIX, value=1.0)
        params.add(HEADER_PARAM_FREQ_PREFIX + "4", value=4, vary=False)
        params.add(HEADER_PARAM_AMP_PREFIX + "4", value=0.1)
        params.add(
            HEADER_PARAM_PHASE_PREFIX + "4",
            value=0.0,
            min=-2 * np.pi,
            max=2 * np.pi,
        )

        fitter_instance.current_f_list = [4]

        x = np.linspace(0, 2 * np.pi, 100)
        y = np.ones(100)

        result = fitter_instance._refit(params, x, y)

        # Phase bounds should now be infinite
        phase_param = result.params[HEADER_PARAM_PHASE_PREFIX + "4"]
        assert phase_param.min == -np.inf
        assert phase_param.max == np.inf


# =============================================================================
# Integration Tests
# =============================================================================


class TestFitterIntegration:
    def test_full_pipeline(self):
        """Test the full fitting pipeline from data to results."""
        # Create project with known signal
        project = ProjectData(project_name="integration_test")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=1.0,
            cross_section=0.5,
        )

        key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        angles = np.linspace(0, 360, 361)
        angles_rad = np.deg2rad(angles)
        mean_res = 1e-5
        # Known signal
        res = mean_res * (1 + 0.1 * np.sin(4 * angles_rad + 0.5))
        data = ExperimentalData(key, angles, res)
        osc = AMROscillation(key, data)
        exp.add_oscillation(osc)
        project.add_experiment(exp)

        # Run Fourier
        fourier = Fourier(amro_data=project, verbose=False)
        fourier.fourier_transform_experiments()

        # Run Fitter
        fitter = AMROFitter(
            amro_data=project,
            min_amp_ratio=0.05,
            max_freq=10,
            verbose=False,
        )
        fitter.fit_act_experiment(HEADER_EXPERIMENT_PREFIX + "11")

        # Check results
        fit_result = osc.fit_result
        assert fit_result is not None
        assert fit_result.fit_succeeded

        # Mean should be close to original
        assert fit_result.mean == pytest.approx(mean_res, rel=0.1)
