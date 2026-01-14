"""Tests for AMROCleaner class."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

from amro.data.cleaner import AMROCleaner
from amro.data import OscillationKey
from amro.config import (
    HEADER_EXPERIMENT_PREFIX,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_ANGLE_DEG,
    HEADER_RES_OHM,
    HEADER_TEMP_RAW,
    HEADER_MAGNET_RAW_OE,
    HEADER_MAGNET_RAW_OE_ABS,
    CLEANER_HEADER_LENGTH,
    CLEANER_OPTION_LABEL,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cleaner():
    """Basic AMROCleaner instance."""
    return AMROCleaner(datafile_type=".dat", verbose=False)


@pytest.fixture
def verbose_cleaner():
    """Verbose AMROCleaner instance."""
    return AMROCleaner(datafile_type=".dat", verbose=True)


@pytest.fixture
def sample_header():
    """Sample header mimicking QD USA PPMS ACT Option format.

    Uses coordinates from config:
    - CLEANER_OPTION_COORD = (5, 1)
    - CLEANER_LABEL_COORD = (11, 1)
    - CLEANER_GEOM_COORD = (12, 1)
    - CLEANER_WIRE_SEP_COORD = (13, 1)
    - CLEANER_CROSS_SEC_COORD = (14, 1)
    """
    header = [[""] * 5 for _ in range(CLEANER_HEADER_LENGTH)]
    # Row 5, Col 1: ACT Option label
    header[5][1] = CLEANER_OPTION_LABEL
    # Row 11, Col 1: Sample name with experiment label
    header[11][1] = f"YbPdBi {HEADER_EXPERIMENT_PREFIX}11"
    # Row 12, Col 1: Geometry
    header[12][1] = "perpendicular"
    # Row 13, Col 1: Wire separation
    header[13][1] = "0.5"
    # Row 14, Col 1: Cross section
    header[14][1] = "0.025"
    return header


@pytest.fixture
def sample_raw_dataframe():
    """Sample raw DataFrame with oscillation data for anti-symmetrization.

    Creates data with:
    - Angles from 0 to 360 degrees
    - Measurements at +H and -H for each angle
    - Constant T and |H|
    """
    n_angles = 361
    angles = np.linspace(0, 360, n_angles)

    # Create +H measurements
    pos_field_oe = 30000.0  # +3T in Oe
    # Create -H measurements
    neg_field_oe = -30000.0  # -3T in Oe

    temp = 2.0
    mean_res = 1e-5

    # Generate resistivity with small oscillation
    angles_rad = np.deg2rad(angles)
    res_base = mean_res * (1 + 0.1 * np.sin(4 * angles_rad))

    # Small asymmetry between +H and -H (to be averaged out)
    res_pos = res_base * 1.001
    res_neg = res_base * 0.999

    # Combine into DataFrame
    df_pos = pd.DataFrame(
        {
            HEADER_ANGLE_DEG: angles,
            HEADER_TEMP_RAW: temp,
            HEADER_MAGNET_RAW_OE: pos_field_oe,
            HEADER_MAGNET_RAW_OE_ABS: abs(pos_field_oe),
            HEADER_TEMP: temp,
            HEADER_MAGNET: 3.0,  # Tesla
            HEADER_RES_OHM: res_pos,
        }
    )

    df_neg = pd.DataFrame(
        {
            HEADER_ANGLE_DEG: angles,
            HEADER_TEMP_RAW: temp,
            HEADER_MAGNET_RAW_OE: neg_field_oe,
            HEADER_MAGNET_RAW_OE_ABS: abs(neg_field_oe),
            HEADER_TEMP: temp,
            HEADER_MAGNET: 3.0,  # Tesla
            HEADER_RES_OHM: res_neg,
        }
    )

    return pd.concat([df_pos, df_neg], ignore_index=True)


@pytest.fixture
def sample_dataframe_with_outliers(sample_raw_dataframe):
    """Sample DataFrame with some outlier resistivity values."""
    df = sample_raw_dataframe.copy()
    # Add outliers at specific indices
    outlier_indices = [10, 50, 100]
    for idx in outlier_indices:
        df.loc[idx, HEADER_RES_OHM] = df[HEADER_RES_OHM].mean() * 100  # 100x mean
    return df


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAMROCleanerInit:

    def test_default_initialization(self, cleaner):
        """Test default initialization values."""
        assert cleaner.datafile_type == ".dat"
        assert cleaner.verbose is False
        assert cleaner.experiment_labels == []

    def test_csv_datafile_type(self):
        """Test initialization with CSV datafile type."""
        cleaner = AMROCleaner(datafile_type=".csv")
        assert cleaner.datafile_type == ".csv"

    def test_verbose_mode(self, verbose_cleaner):
        """Test verbose mode initialization."""
        assert verbose_cleaner.verbose is True

    def test_get_experiment_labels_empty(self, cleaner):
        """Test get_experiment_labels returns empty list initially."""
        assert cleaner.get_experiment_labels() == []


# =============================================================================
# Header Parsing Tests
# =============================================================================


class TestHeaderParsing:

    def test_get_header_element(self, cleaner, sample_header):
        """Test extracting element from header at specific coordinates."""
        # Get element at row 11, col 1
        element = cleaner._get_header_element(sample_header, (11, 1))
        assert HEADER_EXPERIMENT_PREFIX in element

    def test_get_header_element_geometry(self, cleaner, sample_header):
        """Test extracting geometry from header."""
        element = cleaner._get_header_element(sample_header, (12, 1))
        assert "perp" in element.lower()

    def test_parse_and_verify_header_valid(self, cleaner, sample_header):
        """Test parsing valid header returns expected values."""
        exp_label, geom, wire_sep, cross_section = cleaner._parse_and_verify_header(
            sample_header
        )
        assert exp_label == f"{HEADER_EXPERIMENT_PREFIX}11"
        assert "perp" in geom.lower()
        assert wire_sep == 0.5
        assert cross_section == 0.025

    def test_parse_and_verify_header_invalid_option(self, cleaner):
        """Test parsing header with wrong option label raises error."""
        bad_header = [[""] * 5 for _ in range(CLEANER_HEADER_LENGTH)]
        bad_header[10][1] = "Wrong Option Label"

        with pytest.raises(FileNotFoundError, match="ACT Option"):
            cleaner._parse_and_verify_header(bad_header)

    def test_parse_and_verify_header_warns_default_values(self, cleaner, sample_header):
        """Test warning when wire_sep or cross_section is 1 (default)."""
        sample_header[13][1] = "1"  # Default wire sep

        with pytest.warns(UserWarning, match="Default value"):
            cleaner._parse_and_verify_header(sample_header)

    def test_parse_and_verify_header_warns_unexpected_geometry(
        self, cleaner, sample_header
    ):
        """Test warning for unexpected geometry string."""
        sample_header[12][1] = "unknown_geometry"

        with pytest.warns(UserWarning, match="Unexpected geometry"):
            cleaner._parse_and_verify_header(sample_header)


# =============================================================================
# Experiment Label Tests
# =============================================================================


class TestExperimentLabels:

    def test_get_experiment_label_from_fn_valid(self, cleaner):
        """Test extracting experiment label from valid filename."""
        filename = f"sample_{HEADER_EXPERIMENT_PREFIX}11_data.dat"
        label = cleaner._get_experiment_label_from_fn(filename)
        assert label == f"{HEADER_EXPERIMENT_PREFIX}11"

    def test_get_experiment_label_from_fn_no_prefix(self, cleaner):
        """Test returns None when prefix not in filename."""
        filename = "sample_data.dat"
        label = cleaner._get_experiment_label_from_fn(filename)
        assert label is None

    def test_get_experiment_label_from_fn_multiple_underscores(self, cleaner):
        """Test extracting label from filename with multiple underscores."""
        filename = f"sample_name_{HEADER_EXPERIMENT_PREFIX}12_extra_info.dat"
        label = cleaner._get_experiment_label_from_fn(filename)
        assert label == f"{HEADER_EXPERIMENT_PREFIX}12"

    def test_compare_labels_both_match(self, cleaner):
        """Test compare_labels when both labels match."""
        label = f"{HEADER_EXPERIMENT_PREFIX}11"
        result = cleaner._compare_labels(label, label)
        assert result == label

    def test_compare_labels_fn_only(self, cleaner):
        """Test compare_labels when only filename label exists."""
        label = f"{HEADER_EXPERIMENT_PREFIX}11"
        result = cleaner._compare_labels(label, None)
        assert result == label

    def test_compare_labels_header_only(self, cleaner):
        """Test compare_labels when only header label exists."""
        label = f"{HEADER_EXPERIMENT_PREFIX}11"
        result = cleaner._compare_labels(None, label)
        assert result == label

    def test_compare_labels_both_none_raises(self, cleaner):
        """Test compare_labels raises when both labels are None."""
        with pytest.raises(ValueError, match="Could not extract"):
            cleaner._compare_labels(None, None)

    def test_compare_labels_mismatch_uses_filename(self, cleaner, capsys):
        """Test compare_labels prefers filename when labels don't match."""
        fn_label = f"{HEADER_EXPERIMENT_PREFIX}11"
        head_label = f"{HEADER_EXPERIMENT_PREFIX}12"
        result = cleaner._compare_labels(fn_label, head_label)
        assert result == fn_label
        captured = capsys.readouterr()
        assert "does not match" in captured.out


# =============================================================================
# Oscillation Label Tests
# =============================================================================


class TestOscillationLabels:

    def test_get_oscillation_labels(self, cleaner):
        """Test extracting unique oscillation labels from data."""
        data = pd.DataFrame(
            {
                HEADER_TEMP: [2.0, 2.0, 5.0, 5.0],
                HEADER_MAGNET: [3.0, 3.0, 3.0, 7.0],
            }
        )
        exp_label = f"{HEADER_EXPERIMENT_PREFIX}11"

        labels = cleaner._get_oscillation_labels(data, exp_label)

        assert len(labels) == 3  # (2.0, 3.0), (5.0, 3.0), (5.0, 7.0)
        assert all(isinstance(label, OscillationKey) for label in labels)
        assert all(label.experiment_label == exp_label for label in labels)

    def test_get_oscillation_labels_single(self, cleaner):
        """Test with single unique T/H combination."""
        data = pd.DataFrame(
            {
                HEADER_TEMP: [2.0, 2.0, 2.0],
                HEADER_MAGNET: [3.0, 3.0, 3.0],
            }
        )
        exp_label = f"{HEADER_EXPERIMENT_PREFIX}11"

        labels = cleaner._get_oscillation_labels(data, exp_label)

        assert len(labels) == 1
        assert labels[0].temperature == 2.0
        assert labels[0].magnetic_field == 3.0


# =============================================================================
# Outlier Cleaning Tests
# =============================================================================


class TestOutlierCleaning:

    def test_clean_outliers_removes_extreme_values(
        self, cleaner, sample_dataframe_with_outliers
    ):
        """Test that extreme outliers are removed."""
        original_len = len(sample_dataframe_with_outliers)
        cleaned = cleaner._clean_outliers(sample_dataframe_with_outliers)

        # Should have removed outliers
        assert len(cleaned) < original_len

    def test_clean_outliers_preserves_normal_data(self, cleaner, sample_raw_dataframe):
        """Test that normal data is preserved."""
        original_len = len(sample_raw_dataframe)
        cleaned = cleaner._clean_outliers(sample_raw_dataframe)

        # Should not remove any rows
        assert len(cleaned) == original_len

    def test_clean_outliers_verbose_output(
        self, verbose_cleaner, sample_dataframe_with_outliers, capsys
    ):
        """Test verbose mode prints removal count."""
        verbose_cleaner._clean_outliers(sample_dataframe_with_outliers)
        captured = capsys.readouterr()
        assert "outliers" in captured.out.lower()


# =============================================================================
# Anti-symmetrization Tests
# =============================================================================


class TestAntiSymmetrization:

    def test_anti_symmetrize_oscillation_basic(self, cleaner, sample_raw_dataframe):
        """Test basic anti-symmetrization produces averaged data."""
        result = cleaner._anti_symmetrize_oscillation(sample_raw_dataframe)

        assert result is not None
        # Should have half the rows (averaged +H and -H)
        assert len(result) == len(sample_raw_dataframe) // 2

    def test_anti_symmetrize_oscillation_preserves_angles(
        self, cleaner, sample_raw_dataframe
    ):
        """Test that all unique angles are preserved."""
        result = cleaner._anti_symmetrize_oscillation(sample_raw_dataframe)

        original_angles = sample_raw_dataframe[HEADER_ANGLE_DEG].unique()
        result_angles = result[HEADER_ANGLE_DEG].unique()

        np.testing.assert_array_equal(np.sort(original_angles), np.sort(result_angles))

    def test_anti_symmetrize_oscillation_averages_resistivity(
        self, cleaner, sample_raw_dataframe
    ):
        """Test that resistivity is properly averaged."""
        result = cleaner._anti_symmetrize_oscillation(sample_raw_dataframe)

        # Average of +H and -H should be close to base value
        mean_res = result[HEADER_RES_OHM].mean()
        expected_mean = sample_raw_dataframe[HEADER_RES_OHM].mean()

        np.testing.assert_almost_equal(mean_res, expected_mean, decimal=10)


# =============================================================================
# Missing/Extra Measurement Handling Tests
# =============================================================================


class TestMeasurementHandling:

    def test_handle_missing_measurements(self, cleaner):
        """Test handling of missing measurements removes incomplete angles."""
        # Create data with some angles having only 1 measurement
        df = pd.DataFrame(
            {
                HEADER_TEMP: [2.0] * 5,
                HEADER_MAGNET: [3.0] * 5,
                HEADER_ANGLE_DEG: [0, 0, 1, 2, 2],  # Angle 1 has only 1 measurement
                HEADER_RES_OHM: [1e-5] * 5,
            }
        )

        counted_df = df.groupby(
            [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
        ).count()

        result = cleaner._handle_missing_measurements(df, counted_df)

        # Angle 1 should be removed
        assert 1 not in result[HEADER_ANGLE_DEG].values
        assert 0 in result[HEADER_ANGLE_DEG].values
        assert 2 in result[HEADER_ANGLE_DEG].values

    def test_handle_extra_measurements(self, cleaner):
        """Test handling of extra measurements keeps first two per angle."""
        df = pd.DataFrame(
            {
                HEADER_TEMP: [2.0] * 5,
                HEADER_MAGNET: [3.0] * 5,
                HEADER_MAGNET_RAW_OE: [30000, -30000, 30000, 30000, -30000],
                HEADER_ANGLE_DEG: [0, 0, 0, 1, 1],  # Angle 0 has 3 measurements
                HEADER_RES_OHM: [1e-5, 1.1e-5, 1.2e-5, 1e-5, 1.1e-5],
            }
        )

        counted_df = df.groupby(
            [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
        ).count()

        result = cleaner._handle_extra_measurements(df, counted_df)

        # Should have at most 2 measurements per angle
        counts = result.groupby(HEADER_ANGLE_DEG).size()
        assert all(counts <= 2)


# =============================================================================
# Verification Tests
# =============================================================================


class TestVerification:

    def test_verify_averaged_df_valid(self, cleaner, sample_raw_dataframe):
        """Test verification passes for valid averaged data."""
        grouped = sample_raw_dataframe.groupby(
            [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
        )
        averaged = grouped.mean()

        result = cleaner._verify_averaged_df(averaged, sample_raw_dataframe)

        assert result is not None

    def test_verify_averaged_df_invalid_row_count(self, cleaner, sample_raw_dataframe):
        """Test verification fails when row count doesn't match."""
        grouped = sample_raw_dataframe.groupby(
            [HEADER_TEMP, HEADER_MAGNET, HEADER_ANGLE_DEG], as_index=False
        )
        averaged = grouped.mean()

        # Remove a row to make counts invalid
        averaged_bad = averaged.iloc[:-1]

        result = cleaner._verify_averaged_df(averaged_bad, sample_raw_dataframe)

        assert result is None


# =============================================================================
# Filter Tests
# =============================================================================


class TestFilterForOscillationData:

    def test_filter_removes_constant_angle_sweeps(self, cleaner):
        """Test that constant angle sweeps are filtered out."""
        # Create data with oscillation (changing angles) and sweep (constant angle)
        oscillation_data = pd.DataFrame(
            {
                HEADER_TEMP: [2.0] * 10,
                HEADER_MAGNET: [3.0] * 10,
                HEADER_ANGLE_DEG: np.linspace(0, 90, 10),  # Changing angles
                HEADER_RES_OHM: [1e-5] * 10,
            }
        )

        sweep_data = pd.DataFrame(
            {
                HEADER_TEMP: [2.0] * 10,
                HEADER_MAGNET: [3.0] * 10,
                HEADER_ANGLE_DEG: [45.0] * 10,  # Constant angle
                HEADER_RES_OHM: [1e-5] * 10,
            }
        )

        combined = pd.concat([oscillation_data, sweep_data], ignore_index=True)

        result = cleaner._filter_for_oscillation_data(combined)

        # Should have filtered out most of the constant angle data
        assert len(result) < len(combined)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:

    def test_cleaner_with_isolated_paths(self, cleaner, tmp_path, monkeypatch):
        """Test that cleaner uses isolated paths properly."""
        # Monkeypatch paths
        monkeypatch.setattr("amro.data.cleaner.RAW_DATA_PATH", tmp_path)
        monkeypatch.setattr("amro.data.cleaner.PROCESSED_DATA_PATH", tmp_path)

        # Create a new cleaner after patching
        test_cleaner = AMROCleaner(datafile_type=".dat", verbose=False)

        assert test_cleaner.load_path == tmp_path
        assert test_cleaner.save_path == tmp_path
