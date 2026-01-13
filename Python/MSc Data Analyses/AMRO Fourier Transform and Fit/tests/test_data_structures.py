"""Tests for amro.data.data_structures module."""

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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_oscillation_key():
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
    return ExperimentalData(
        experiment_key=sample_oscillation_key,
        angles_degs=sample_angles,
        res_ohms=sample_resistivities,
    )


@pytest.fixture
def sample_amro_oscillation(sample_oscillation_key, sample_experimental_data):
    return AMROscillation(
        key=sample_oscillation_key,
        osc_data=sample_experimental_data,
    )


@pytest.fixture
def sample_fourier_xf():
    """Fourier transform frequencies (symmetries)."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture
def sample_fourier_yf():
    """Fourier transform complex amplitudes."""
    # Create complex values with known magnitudes and phases
    magnitudes = np.array([0.01, 0.05, 0.02, 0.10, 0.01, 0.01, 0.01, 0.01])
    phases = np.array([0.5, 1.0, 0.3, 0.8, 0.2, 0.1, 0.4, 0.6])
    return magnitudes * np.exp(1j * phases)


@pytest.fixture
def sample_fourier_result(sample_oscillation_key, sample_fourier_xf, sample_fourier_yf):
    return FourierResult(
        key=sample_oscillation_key,
        xf=sample_fourier_xf,
        yf=sample_fourier_yf,
    )


@pytest.fixture
def sample_lmfit_result():
    """Creates a minimal lmfit MinimizerResult for testing."""
    params = lm.Parameters()
    params.add(HEADER_PARAM_MEAN_PREFIX, value=1e-5, min=0)
    params.add(HEADER_PARAM_FREQ_PREFIX + "4", value=4, vary=False)
    params.add(HEADER_PARAM_AMP_PREFIX + "4", value=0.1, min=0)
    params.add(
        HEADER_PARAM_PHASE_PREFIX + "4", value=0.8, min=-2 * np.pi, max=2 * np.pi
    )

    # Create a mock minimizer result
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


@pytest.fixture
def sample_experiment():
    return Experiment(
        experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
        geometry="perp",
        wire_sep=1.0,
        cross_section=0.5,
    )


@pytest.fixture
def sample_project_data():
    return ProjectData(project_name="test_project")


# =============================================================================
# OscillationKey Tests
# =============================================================================


class TestOscillationKey:
    def test_creation(self, sample_oscillation_key):
        assert (
            sample_oscillation_key.experiment_label == HEADER_EXPERIMENT_PREFIX + "11"
        )
        assert sample_oscillation_key.temperature == 2.0
        assert sample_oscillation_key.magnetic_field == 3.0

    def test_str_format(self, sample_oscillation_key):
        result = str(sample_oscillation_key)
        assert HEADER_EXPERIMENT_PREFIX + "11" in result
        assert "2.0" in result or "2" in result
        assert "3.0" in result or "3" in result

    def test_repr_equals_str(self, sample_oscillation_key):
        assert repr(sample_oscillation_key) == str(sample_oscillation_key)

    def test_frozen_cannot_modify(self, sample_oscillation_key):
        with pytest.raises(AttributeError):
            sample_oscillation_key.temperature = 5.0

    def test_compare_act_true(self, sample_oscillation_key):
        assert (
            sample_oscillation_key.compare_exp_label(HEADER_EXPERIMENT_PREFIX + "11")
            is True
        )

    def test_compare_act_false(self, sample_oscillation_key):
        assert (
            sample_oscillation_key.compare_exp_label(HEADER_EXPERIMENT_PREFIX + "12")
            is False
        )

    def test_compare_temperature_true(self, sample_oscillation_key):
        assert sample_oscillation_key.compare_temperature(2.0) is True

    def test_compare_temperature_false(self, sample_oscillation_key):
        assert sample_oscillation_key.compare_temperature(5.0) is False

    def test_compare_magnetic_field_true(self, sample_oscillation_key):
        assert sample_oscillation_key.compare_magnetic_field(3.0) is True

    def test_compare_magnetic_field_false(self, sample_oscillation_key):
        assert sample_oscillation_key.compare_magnetic_field(7.0) is False

    def test_get_experiment_label(self, sample_oscillation_key):
        assert (
            sample_oscillation_key.get_experiment_label()
            == HEADER_EXPERIMENT_PREFIX + "11"
        )

    def test_get_temperature(self, sample_oscillation_key):
        assert sample_oscillation_key.get_temperature() == 2.0

    def test_get_magnetic_field(self, sample_oscillation_key):
        assert sample_oscillation_key.get_magnetic_field() == 3.0

    def test_hashable_for_dict_key(self, sample_oscillation_key):
        """OscillationKey should be usable as a dictionary key."""
        test_dict = {sample_oscillation_key: "test_value"}
        assert test_dict[sample_oscillation_key] == "test_value"

    def test_equality(self):
        key1 = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        key2 = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        assert key1 == key2

    def test_inequality_different_act(self):
        key1 = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        key2 = OscillationKey(HEADER_EXPERIMENT_PREFIX + "12", 2.0, 3.0)
        assert key1 != key2


# =============================================================================
# ExperimentalData Tests
# =============================================================================


class TestExperimentalData:
    def test_creation(self, sample_experimental_data):
        assert sample_experimental_data is not None

    def test_stores_key(self, sample_experimental_data, sample_oscillation_key):
        assert sample_experimental_data.experiment_key == sample_oscillation_key

    def test_converts_angles_to_array(self, sample_experimental_data):
        assert isinstance(sample_experimental_data.angles_degs, np.ndarray)

    def test_converts_resistivities_to_array(self, sample_experimental_data):
        assert isinstance(sample_experimental_data.res_ohms, np.ndarray)

    def test_calculates_mean_resistivity(
        self, sample_experimental_data, sample_resistivities
    ):
        expected_mean = np.mean(sample_resistivities)
        assert sample_experimental_data.mean_res_ohms == pytest.approx(expected_mean)

    def test_converts_to_radians(self, sample_experimental_data, sample_angles):
        expected_rads = sample_angles * (np.pi / 180)
        np.testing.assert_array_almost_equal(
            sample_experimental_data.angles_rads, expected_rads
        )

    def test_calculates_delta_res_mean(self, sample_experimental_data):
        expected = (
            sample_experimental_data.res_ohms - sample_experimental_data.mean_res_ohms
        )
        np.testing.assert_array_almost_equal(
            sample_experimental_data.delta_res_mean_ohms, expected
        )

    def test_calculates_delta_res_0deg(self, sample_experimental_data):
        expected = (
            sample_experimental_data.res_ohms - sample_experimental_data.deg0_res_ohms
        )
        np.testing.assert_array_almost_equal(
            sample_experimental_data.delta_res_0deg_ohms, expected
        )

    def test_calculates_uohm_values(self, sample_experimental_data):
        expected = sample_experimental_data.mean_res_ohms * 1e6
        assert sample_experimental_data.mean_res_uohms == pytest.approx(expected)

    def test_calculates_normalized_delta_res(self, sample_experimental_data):
        expected = (
            sample_experimental_data.delta_res_mean_ohms
            / sample_experimental_data.mean_res_ohms
        )
        np.testing.assert_array_almost_equal(
            sample_experimental_data.delta_res_mean_norm, expected
        )

    def test_validates_negative_angles_raises(
        self, sample_oscillation_key, sample_resistivities
    ):
        negative_angles = np.linspace(-10, 350, 361)
        with pytest.raises(ValueError, match="non-negative"):
            ExperimentalData(
                experiment_key=sample_oscillation_key,
                angles_degs=negative_angles,
                res_ohms=sample_resistivities,
            )

    def test_validates_negative_resistance_raises(
        self, sample_oscillation_key, sample_angles
    ):
        negative_res = np.full(361, -1e-5)
        with pytest.raises(ValueError, match="non-negative"):
            ExperimentalData(
                experiment_key=sample_oscillation_key,
                angles_degs=sample_angles,
                res_ohms=negative_res,
            )

    def test_get_experiment_label(self, sample_experimental_data):
        assert (
            sample_experimental_data.get_experiment_label()
            == HEADER_EXPERIMENT_PREFIX + "11"
        )

    def test_get_temperature(self, sample_experimental_data):
        assert sample_experimental_data.get_temperature() == 2.0

    def test_get_magnetic_field(self, sample_experimental_data):
        assert sample_experimental_data.get_magnetic_field() == 3.0


# =============================================================================
# FourierResult Tests
# =============================================================================


class TestFourierResult:
    def test_creation(self, sample_fourier_result):
        assert sample_fourier_result is not None

    def test_converts_xf_to_int_array(self, sample_fourier_result):
        assert sample_fourier_result.xf.dtype == int

    def test_calculates_phases(self, sample_fourier_result, sample_fourier_yf):
        expected_phases = np.angle(sample_fourier_yf)
        np.testing.assert_array_almost_equal(
            sample_fourier_result.phases, expected_phases
        )

    def test_calculates_amplitudes(self, sample_fourier_result, sample_fourier_yf):
        expected_amps = np.abs(sample_fourier_yf)
        np.testing.assert_array_almost_equal(
            sample_fourier_result.amplitudes, expected_amps
        )

    def test_calculates_amplitude_ratio(self, sample_fourier_result):
        max_amp = sample_fourier_result.amplitudes.max()
        expected_ratio = sample_fourier_result.amplitudes / max_amp
        np.testing.assert_array_almost_equal(
            sample_fourier_result.amplitudes_ratio, expected_ratio
        )

    def test_amplitude_ratio_max_is_one(self, sample_fourier_result):
        assert sample_fourier_result.amplitudes_ratio.max() == pytest.approx(1.0)

    def test_phases_pos_are_positive(self, sample_fourier_result):
        assert np.all(sample_fourier_result.phases_pos >= 0)

    def test_fourier_results_dict_populated(
        self, sample_fourier_result, sample_fourier_xf
    ):
        for freq in sample_fourier_xf:
            assert freq in sample_fourier_result.fourier_results_dict

    def test_get_fit_guess_returns_ratio_and_phase(self, sample_fourier_result):
        ratio, phase = sample_fourier_result.get_fit_guess(4)
        assert 0 <= ratio <= 1
        assert isinstance(phase, (int, float))

    def test_get_n_strongest_components(self, sample_fourier_result):
        # Frequency 4 has the strongest amplitude in our fixture
        strongest = list(sample_fourier_result.get_n_strongest_components(n=2))
        assert len(strongest) == 2
        # Check that freq 4 is in the top 2
        freqs = [item[0] for item in strongest]
        assert 4 in freqs

    def test_str_format(self, sample_fourier_result):
        result = str(sample_fourier_result)
        assert "Fourier_Result_Object" in result

    def test_compare_methods(self, sample_fourier_result):
        assert (
            sample_fourier_result.compare_act(HEADER_EXPERIMENT_PREFIX + "11") is True
        )
        assert sample_fourier_result.compare_temperature(2.0) is True
        assert sample_fourier_result.compare_magnetic_field(3.0) is True

    def test_invalid_zero_frequency_raises(self, sample_oscillation_key):
        xf_with_zero = np.array([0, 1, 2, 3])
        yf = np.array([0.1, 0.2, 0.3, 0.4]) * np.exp(1j * np.array([0, 0.5, 1.0, 1.5]))
        with pytest.raises(ValueError, match="Invalid symmetry"):
            FourierResult(key=sample_oscillation_key, xf=xf_with_zero, yf=yf)


# =============================================================================
# AMROscillation Tests
# =============================================================================


class TestAMROscillation:
    def test_creation(self, sample_amro_oscillation):
        assert sample_amro_oscillation is not None

    def test_stores_key(self, sample_amro_oscillation, sample_oscillation_key):
        assert sample_amro_oscillation.key == sample_oscillation_key

    def test_stores_osc_data(self, sample_amro_oscillation, sample_experimental_data):
        assert sample_amro_oscillation.osc_data == sample_experimental_data

    def test_hasattr_fit_result_false_before_adding(self, sample_amro_oscillation):
        """Critical test: fit_result should not exist until explicitly added."""
        assert (sample_amro_oscillation.fit_result is not None) is False

    def test_hasattr_fourier_result_false_before_adding(self, sample_amro_oscillation):
        """Critical test: fourier_result should not exist until explicitly added."""
        assert (sample_amro_oscillation.fourier_result is not None) is False

    def test_add_fourier_result(
        self, sample_amro_oscillation, sample_fourier_xf, sample_fourier_yf
    ):
        sample_amro_oscillation.add_fourier_result(sample_fourier_xf, sample_fourier_yf)
        assert hasattr(sample_amro_oscillation, "fourier_result") is True
        assert isinstance(sample_amro_oscillation.fourier_result, FourierResult)

    def test_add_fit_result(
        self,
        sample_amro_oscillation,
        sample_lmfit_result,
        sample_fourier_xf,
        sample_fourier_yf,
    ):
        # Need to add fourier result first for the model calculation
        sample_amro_oscillation.add_fourier_result(sample_fourier_xf, sample_fourier_yf)
        sample_amro_oscillation.add_fit_result(
            lmfit_result=sample_lmfit_result,
            refitted=False,
        )
        assert hasattr(sample_amro_oscillation, "fit_result") is True
        assert isinstance(sample_amro_oscillation.fit_result, FitResult)

    def test_str_format(self, sample_amro_oscillation):
        result = str(sample_amro_oscillation)
        assert "Experiment_Object" in result

    def test_compare_methods(self, sample_amro_oscillation):
        assert (
            sample_amro_oscillation.compare_act(HEADER_EXPERIMENT_PREFIX + "11") is True
        )
        assert sample_amro_oscillation.compare_temperature(2.0) is True
        assert sample_amro_oscillation.compare_magnetic_field(3.0) is True

    def test_get_experiment_label(self, sample_amro_oscillation):
        assert (
            sample_amro_oscillation.get_experiment_label()
            == HEADER_EXPERIMENT_PREFIX + "11"
        )

    def test_get_temperature(self, sample_amro_oscillation):
        assert sample_amro_oscillation.get_temperature() == 2.0

    def test_get_magnetic_field(self, sample_amro_oscillation):
        assert sample_amro_oscillation.get_magnetic_field() == 3.0


# =============================================================================
# Experiment Tests
# =============================================================================


class TestExperiment:
    def test_creation(self, sample_experiment):
        assert sample_experiment.experiment_label == HEADER_EXPERIMENT_PREFIX + "11"
        assert sample_experiment.geometry == "perp"
        assert sample_experiment.oscillations_count == 0

    def test_add_oscillation(self, sample_experiment, sample_amro_oscillation):
        sample_experiment.add_oscillation(sample_amro_oscillation)
        assert sample_experiment.oscillations_count == 1
        assert sample_amro_oscillation.key in sample_experiment.oscillations_dict

    def test_add_duplicate_oscillation_does_not_add(
        self, sample_experiment, sample_amro_oscillation, capsys
    ):
        sample_experiment.add_oscillation(sample_amro_oscillation)
        sample_experiment.add_oscillation(sample_amro_oscillation)
        assert sample_experiment.oscillations_count == 1
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_replace_oscillation(self, sample_experiment, sample_amro_oscillation):
        sample_experiment.add_oscillation(sample_amro_oscillation)
        sample_experiment.replace_oscillation(sample_amro_oscillation)
        assert sample_experiment.oscillations_count == 1

    def test_get_oscillation(self, sample_experiment, sample_amro_oscillation):
        sample_experiment.add_oscillation(sample_amro_oscillation)
        retrieved = sample_experiment.get_oscillation(t=2.0, h=3.0)
        assert retrieved == sample_amro_oscillation

    def test_get_oscillation_from_key(
        self, sample_experiment, sample_amro_oscillation, sample_oscillation_key
    ):
        sample_experiment.add_oscillation(sample_amro_oscillation)
        retrieved = sample_experiment.get_oscillation_from_key(sample_oscillation_key)
        assert retrieved == sample_amro_oscillation

    def test_get_multiple_oscillations_all(self, sample_experiment):
        """Add multiple oscillations and retrieve all."""
        for t in [2.0, 5.0, 10.0]:
            for h in [3.0, 7.0]:
                key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, h)
                angles = np.linspace(0, 360, 361)
                res = np.full(361, 1e-5)
                data = ExperimentalData(key, angles, res)
                osc = AMROscillation(key, data)
                sample_experiment.add_oscillation(osc)

        all_oscs = sample_experiment.get_multiple_oscillations()
        assert len(all_oscs) == 6

    def test_get_multiple_oscillations_by_temperature(self, sample_experiment):
        """Filter oscillations by temperature."""
        for t in [2.0, 5.0]:
            for h in [3.0, 7.0]:
                key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, h)
                angles = np.linspace(0, 360, 361)
                res = np.full(361, 1e-5)
                data = ExperimentalData(key, angles, res)
                osc = AMROscillation(key, data)
                sample_experiment.add_oscillation(osc)

        oscs_at_2K = sample_experiment.get_multiple_oscillations(t=2.0)
        assert len(oscs_at_2K) == 2
        for osc in oscs_at_2K:
            assert osc.get_temperature() == 2.0

    def test_get_multiple_oscillations_by_field(self, sample_experiment):
        """Filter oscillations by magnetic field."""
        for t in [2.0, 5.0]:
            for h in [3.0, 7.0]:
                key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, h)
                angles = np.linspace(0, 360, 361)
                res = np.full(361, 1e-5)
                data = ExperimentalData(key, angles, res)
                osc = AMROscillation(key, data)
                sample_experiment.add_oscillation(osc)

        oscs_at_3T = sample_experiment.get_multiple_oscillations(h=3.0)
        assert len(oscs_at_3T) == 2
        for osc in oscs_at_3T:
            assert osc.get_magnetic_field() == 3.0


# =============================================================================
# ProjectData Tests
# =============================================================================


class TestProjectData:
    def test_creation(self, sample_project_data):
        assert sample_project_data.project_name == "test_project"
        assert sample_project_data.experiments_count == 0

    def test_add_experiment(self, sample_project_data, sample_experiment):
        sample_project_data.add_experiment(sample_experiment)
        assert sample_project_data.experiments_count == 1
        assert HEADER_EXPERIMENT_PREFIX + "11" in sample_project_data.experiments_dict

    def test_get_experiment(self, sample_project_data, sample_experiment):
        sample_project_data.add_experiment(sample_experiment)
        retrieved = sample_project_data.get_experiment(HEADER_EXPERIMENT_PREFIX + "11")
        assert retrieved == sample_experiment

    def test_get_experiment_labels(self, sample_project_data, sample_experiment):
        sample_project_data.add_experiment(sample_experiment)
        labels = list(sample_project_data.get_experiment_labels())
        assert HEADER_EXPERIMENT_PREFIX + "11" in labels

    def test_get_summary_statistics_empty(self, sample_project_data):
        stats = sample_project_data.get_summary_statistics()
        assert stats["n_experiments"] == 0
        assert stats["n_oscillations"] == 0
        assert stats["n_fourier_completed"] == 0
        assert stats["n_fits_completed"] == 0

    def test_get_summary_statistics_with_data(self, sample_project_data):
        """Test that summary statistics correctly count oscillations."""
        exp = Experiment(HEADER_EXPERIMENT_PREFIX + "11", "perp", 1.0, 0.5)

        # Add 4 oscillations
        for t in [2.0, 5.0]:
            for h in [3.0, 7.0]:
                key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, h)
                angles = np.linspace(0, 360, 361)
                res = np.full(361, 1e-5)
                data = ExperimentalData(key, angles, res)
                osc = AMROscillation(key, data)
                exp.add_oscillation(osc)

        sample_project_data.add_experiment(exp)
        stats = sample_project_data.get_summary_statistics()

        assert stats["n_experiments"] == 1
        assert stats["n_oscillations"] == 4
        assert stats["n_fourier_completed"] == 0
        assert stats["n_fits_completed"] == 0

    def test_summary_counts_fourier_correctly(self, sample_project_data):
        """Test that fourier results are counted correctly."""
        exp = Experiment(HEADER_EXPERIMENT_PREFIX + "11", "perp", 1.0, 0.5)

        xf = np.array([1, 2, 3, 4])
        yf = np.array([0.1, 0.2, 0.3, 0.4]) * np.exp(
            1j * np.array([0.1, 0.2, 0.3, 0.4])
        )

        for t in [2.0, 5.0]:
            key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, 3.0)
            angles = np.linspace(0, 360, 361)
            res = np.full(361, 1e-5)
            data = ExperimentalData(key, angles, res)
            osc = AMROscillation(key, data)
            # Add fourier result to first oscillation only
            if t == 2.0:
                osc.add_fourier_result(xf, yf)
            exp.add_oscillation(osc)

        sample_project_data.add_experiment(exp)
        stats = sample_project_data.get_summary_statistics()

        assert stats["n_oscillations"] == 2
        assert stats["n_fourier_completed"] == 1

    def test_filter_oscillations_all(self, sample_project_data):
        """Test filtering returns all oscillations when no filter specified."""
        exp = Experiment(HEADER_EXPERIMENT_PREFIX + "11", "perp", 1.0, 0.5)
        for t in [2.0, 5.0]:
            key = OscillationKey(HEADER_EXPERIMENT_PREFIX + "11", t, 3.0)
            angles = np.linspace(0, 360, 361)
            res = np.full(361, 1e-5)
            data = ExperimentalData(key, angles, res)
            osc = AMROscillation(key, data)
            exp.add_oscillation(osc)

        sample_project_data.add_experiment(exp)
        result = sample_project_data.filter_oscillations()

        assert len(result) == 2

    def test_filter_oscillations_by_experiment(self, sample_project_data):
        """Test filtering by experiment label."""
        exp1 = Experiment(HEADER_EXPERIMENT_PREFIX + "11", "perp", 1.0, 0.5)
        exp2 = Experiment(HEADER_EXPERIMENT_PREFIX + "12", "para", 1.0, 0.5)

        for exp in [exp1, exp2]:
            key = OscillationKey(exp.experiment_label, 2.0, 3.0)
            angles = np.linspace(0, 360, 361)
            res = np.full(361, 1e-5)
            data = ExperimentalData(key, angles, res)
            osc = AMROscillation(key, data)
            exp.add_oscillation(osc)

        sample_project_data.add_experiment(exp1)
        sample_project_data.add_experiment(exp2)

        result = sample_project_data.filter_oscillations(
            experiments=HEADER_EXPERIMENT_PREFIX + "11"
        )
        assert len(result) == 1

    def test_change_project_name(self, sample_project_data):
        sample_project_data.change_project_name("new_name")
        assert sample_project_data.project_name == "new_name"


# =============================================================================
# FitResult Tests
# =============================================================================


class TestFitResult:
    @pytest.fixture
    def sample_fit_result(self, sample_oscillation_key, sample_lmfit_result):
        """Create a FitResult for testing."""
        model_res = np.linspace(0.9e-5, 1.1e-5, 100)
        return FitResult(
            experiment_key=sample_oscillation_key,
            lmfit_result=sample_lmfit_result,
            model_res_ohms=model_res,
            fit_succeeded=True,
            required_refit=False,
        )

    def test_creation(self, sample_fit_result):
        assert sample_fit_result is not None

    def test_stores_chi_squared(self, sample_fit_result):
        assert hasattr(sample_fit_result, "chi_squared")
        assert isinstance(sample_fit_result.chi_squared, float)

    def test_stores_red_chi_squared(self, sample_fit_result):
        assert hasattr(sample_fit_result, "red_chi_squared")
        assert isinstance(sample_fit_result.red_chi_squared, float)

    def test_calculates_model_uohms(self, sample_fit_result):
        expected = sample_fit_result.model_res_ohms * 1e6
        np.testing.assert_array_almost_equal(
            sample_fit_result.model_res_uohms, expected
        )

    def test_parses_symmetries(self, sample_fit_result):
        assert hasattr(sample_fit_result, "symmetries")
        assert isinstance(sample_fit_result.symmetries, np.ndarray)

    def test_parses_amplitudes(self, sample_fit_result):
        assert hasattr(sample_fit_result, "amplitudes")
        assert isinstance(sample_fit_result.amplitudes, np.ndarray)

    def test_parses_phases(self, sample_fit_result):
        assert hasattr(sample_fit_result, "phases")
        assert isinstance(sample_fit_result.phases, np.ndarray)

    def test_parses_mean(self, sample_fit_result):
        assert hasattr(sample_fit_result, "mean")
        assert sample_fit_result.mean == pytest.approx(1e-5, rel=0.1)

    def test_str_format(self, sample_fit_result):
        result = str(sample_fit_result)
        assert "Fit_Result_Object" in result

    def test_compare_methods(self, sample_fit_result):
        assert sample_fit_result.compare_act(HEADER_EXPERIMENT_PREFIX + "11") is True
        assert sample_fit_result.compare_temperature(2.0) is True
        assert sample_fit_result.compare_magnetic_field(3.0) is True

    def test_fitted_params_dict_populated(self, sample_fit_result):
        assert len(sample_fit_result.fitted_params_dict) > 0

    def test_get_experiment_label(self, sample_fit_result):
        assert (
            sample_fit_result.get_experiment_label() == HEADER_EXPERIMENT_PREFIX + "11"
        )

    def test_get_temperature(self, sample_fit_result):
        assert sample_fit_result.get_temperature() == 2.0

    def test_get_magnetic_field(self, sample_fit_result):
        assert sample_fit_result.get_magnetic_field() == 3.0
