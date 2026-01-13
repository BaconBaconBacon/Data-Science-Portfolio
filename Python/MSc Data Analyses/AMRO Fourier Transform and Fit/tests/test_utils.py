"""Tests for amro.utils.utils module."""

import pytest
import numpy as np
import pandas as pd
import lmfit as lm

from amro.config import (
    HEADER_EXPERIMENT_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_FREQ_PREFIX,
    HEADER_EXP_LABEL,
    HEADER_TEMP,
    HEADER_MAGNET,
)
from amro.utils import (
    sine_builder,
    query_dataframe,
    build_query_string,
    convert_params_to_ndarrays,
    calculate_model_resistivities,
    format_oscillation_key,
)


# =============================================================================
# sine_builder Tests
# =============================================================================


class TestSineBuilder:
    def test_single_frequency_zero_phase(self):
        """Test sine wave with single frequency and zero phase."""
        x = np.linspace(0, 2 * np.pi, 100)
        amps = np.array([0.1])
        freqs = np.array([4])
        phases = np.array([0])
        mean = 1.0

        result = sine_builder(x, amps, freqs, phases, mean)

        # At x=0, sin(0)=0, so result should be mean * (0 + 1) = mean
        assert result[0] == pytest.approx(mean)

    def test_single_frequency_with_phase(self):
        """Test sine wave with phase offset."""
        x = np.array([0])
        amps = np.array([0.5])
        freqs = np.array([1])
        phases = np.array([np.pi / 2])  # Phase shift makes sin -> cos-like
        mean = 1.0

        result = sine_builder(x, amps, freqs, phases, mean)

        # sin(0 + pi/2) = 1, so result = mean * (0.5 * 1 + 1) = 1.5
        assert result[0] == pytest.approx(1.5)

    def test_multiple_frequencies(self):
        """Test superposition of multiple frequencies."""
        x = np.array([0])
        amps = np.array([0.1, 0.05])
        freqs = np.array([4, 2])
        phases = np.array([0, 0])
        mean = 1.0

        result = sine_builder(x, amps, freqs, phases, mean)

        # At x=0, both sines are 0, so result = mean * (0 + 1) = mean
        assert result[0] == pytest.approx(mean)

    def test_output_shape(self):
        """Test that output has same shape as input x."""
        x = np.linspace(0, 2 * np.pi, 361)
        amps = np.array([0.1, 0.05])
        freqs = np.array([4, 2])
        phases = np.array([0, 0])
        mean = 1e-5

        result = sine_builder(x, amps, freqs, phases, mean)

        assert result.shape == x.shape

    def test_mean_scaling(self):
        """Test that mean parameter scales the output correctly."""
        x = np.linspace(0, 2 * np.pi, 100)
        amps = np.array([0.1])
        freqs = np.array([4])
        phases = np.array([0])

        result_mean1 = sine_builder(x, amps, freqs, phases, 1.0)
        result_mean2 = sine_builder(x, amps, freqs, phases, 2.0)

        # Doubling mean should double the output
        np.testing.assert_array_almost_equal(result_mean2, 2 * result_mean1)

    def test_periodicity(self):
        """Test that 4-fold symmetry repeats every pi/2."""
        x = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        amps = np.array([0.1])
        freqs = np.array([4])
        phases = np.array([0])
        mean = 1.0

        result = sine_builder(x, amps, freqs, phases, mean)

        # All points should have same value for 4-fold symmetry
        np.testing.assert_array_almost_equal(result, np.full(4, mean))


# =============================================================================
# query_dataframe Tests
# =============================================================================


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for query tests."""
    return pd.DataFrame(
        {
            HEADER_EXP_LABEL: [
                HEADER_EXPERIMENT_PREFIX + "11",
                HEADER_EXPERIMENT_PREFIX + "11",
                HEADER_EXPERIMENT_PREFIX + "12",
                HEADER_EXPERIMENT_PREFIX + "12",
            ],
            HEADER_TEMP: [2.0, 5.0, 2.0, 5.0],
            HEADER_MAGNET: [3.0, 3.0, 7.0, 7.0],
            "value": [1, 2, 3, 4],
        }
    )


class TestQueryDataframe:
    def test_query_by_act(self, sample_df):
        result = query_dataframe(sample_df, act=HEADER_EXPERIMENT_PREFIX + "11")
        assert len(result) == 2
        assert all(result[HEADER_EXP_LABEL] == HEADER_EXPERIMENT_PREFIX + "11")

    def test_query_by_temperature(self, sample_df):
        result = query_dataframe(sample_df, t=2.0)
        assert len(result) == 2
        assert all(result[HEADER_TEMP] == 2.0)

    def test_query_by_magnetic_field(self, sample_df):
        result = query_dataframe(sample_df, h=3.0)
        assert len(result) == 2
        assert all(result[HEADER_MAGNET] == 3.0)

    def test_query_combined(self, sample_df):
        result = query_dataframe(sample_df, act=HEADER_EXPERIMENT_PREFIX + "11", t=2.0)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 1

    def test_query_no_filter(self, sample_df):
        result = query_dataframe(sample_df)
        assert len(result) == len(sample_df)

    def test_query_empty_result(self, sample_df):
        result = query_dataframe(sample_df, t=999.0)
        assert len(result) == 0


class TestBuildQueryString:
    def test_act_only(self):
        query = build_query_string(act=HEADER_EXPERIMENT_PREFIX + "11")
        assert HEADER_EXP_LABEL in query
        assert HEADER_EXPERIMENT_PREFIX + "11" in query

    def test_temperature_only(self):
        query = build_query_string(t=2.0)
        assert HEADER_TEMP in query
        assert "2.0" in query

    def test_magnetic_field_only(self):
        query = build_query_string(h=3.0)
        assert HEADER_MAGNET in query
        assert "3.0" in query

    def test_combined_query(self):
        query = build_query_string(act=HEADER_EXPERIMENT_PREFIX + "11", t=2.0, h=3.0)
        assert "&" in query
        assert HEADER_EXP_LABEL in query
        assert HEADER_TEMP in query
        assert HEADER_MAGNET in query

    def test_empty_query(self):
        query = build_query_string()
        assert query == ""


# =============================================================================
# convert_params_to_ndarrays Tests
# =============================================================================


class TestConvertParamsToNdarrays:
    @pytest.fixture
    def sample_params(self):
        """Create sample lmfit Parameters."""
        params = lm.Parameters()
        params.add(HEADER_PARAM_MEAN_PREFIX, value=1e-5, min=0)
        params.add(HEADER_PARAM_FREQ_PREFIX + "4", value=4, vary=False)
        params.add(HEADER_PARAM_AMP_PREFIX + "4", value=0.1, min=0)
        params.add(HEADER_PARAM_PHASE_PREFIX + "4", value=0.8)
        params.add(HEADER_PARAM_FREQ_PREFIX + "2", value=2, vary=False)
        params.add(HEADER_PARAM_AMP_PREFIX + "2", value=0.05, min=0)
        params.add(HEADER_PARAM_PHASE_PREFIX + "2", value=0.3)
        return params

    def test_returns_tuple_without_errors(self, sample_params):
        result = convert_params_to_ndarrays(sample_params, include_errs=False)
        assert isinstance(result, tuple)
        assert len(result) == 4  # amps, freqs, phases, mean

    def test_returns_tuple_with_errors(self, sample_params):
        result = convert_params_to_ndarrays(sample_params, include_errs=True)
        assert isinstance(result, tuple)
        assert (
            len(result) == 7
        )  # amps, amps_err, freqs, phases, phases_err, mean, mean_err

    def test_extracts_frequencies(self, sample_params):
        amps, freqs, phases, mean = convert_params_to_ndarrays(
            sample_params, include_errs=False
        )
        assert 4 in freqs
        assert 2 in freqs

    def test_extracts_mean(self, sample_params):
        amps, freqs, phases, mean = convert_params_to_ndarrays(
            sample_params, include_errs=False
        )
        assert mean == pytest.approx(1e-5)

    def test_amplitudes_match_frequencies(self, sample_params):
        amps, freqs, phases, mean = convert_params_to_ndarrays(
            sample_params, include_errs=False
        )
        # Find index of freq 4
        idx_4 = np.where(freqs == 4)[0][0]
        assert amps[idx_4] == pytest.approx(0.1)

        # Find index of freq 2
        idx_2 = np.where(freqs == 2)[0][0]
        assert amps[idx_2] == pytest.approx(0.05)


# =============================================================================
# calculate_model_resistivities Tests
# =============================================================================


class TestCalculateModelResistivities:
    def test_returns_array(self):
        x = np.linspace(0, 2 * np.pi, 100)
        params = (
            np.array([0.1]),  # amps
            np.array([4]),  # freqs
            np.array([0]),  # phases
            1e-5,  # mean
        )
        result = calculate_model_resistivities(x, params)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_output_near_mean(self):
        x = np.linspace(0, 2 * np.pi, 100)
        params = (
            np.array([0.1]),
            np.array([4]),
            np.array([0]),
            1e-5,
        )
        result = calculate_model_resistivities(x, params)
        # Mean of sine is 0, so average should be close to mean
        assert np.mean(result) == pytest.approx(1e-5, rel=0.1)


# =============================================================================
# format_oscillation_key Tests
# =============================================================================


class TestFormatOscillationKey:
    def test_basic_format(self):
        result = format_oscillation_key(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        assert HEADER_EXPERIMENT_PREFIX + "11" in result
        assert "2" in result or "2.0" in result
        assert "3" in result or "3.0" in result

    def test_contains_temperature_marker(self):
        result = format_oscillation_key(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        assert "T" in result or "K" in result

    def test_contains_field_marker(self):
        result = format_oscillation_key(HEADER_EXPERIMENT_PREFIX + "11", 2.0, 3.0)
        # Should contain H or T (Tesla) marker for field
        assert "H" in result or "T" in result
