"""Tests for amro.data.loader module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from amro.config.settings import (
    HEADER_EXPERIMENT_PREFIX,
    HEADER_ACT,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_GEO,
    HEADER_ANGLE_DEG,
    HEADER_RES_OHM,
    HEADER_LENGTH,
    HEADER_WIDTH,
    HEADER_HEIGHT,
    HEADER_TEMP_RAW,
    HEADER_MAGNET_RAW_OE,
)
from amro.data.loader import AMROLoader
from amro.data.data_structures import ProjectData


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def loader():
    """Create an AMROLoader instance."""
    return AMROLoader(project_name="test_project", verbose=False)


@pytest.fixture
def sample_csv_data():
    """Create sample data that mimics a raw AMRO CSV file."""
    n_points = 361
    angles = np.linspace(0, 360, n_points)
    res = np.full(n_points, 1e-5) * (
        1 + 0.1 * np.sin(4 * np.deg2rad(angles))
    )

    return pd.DataFrame(
        {
            HEADER_ANGLE_DEG: angles,
            HEADER_RES_OHM: res,
            HEADER_TEMP_RAW: [2.0] * n_points,
            HEADER_MAGNET_RAW_OE: [30000] * n_points,  # 3T
            HEADER_LENGTH: [1.0] * n_points,
            HEADER_WIDTH: [0.5] * n_points,
            HEADER_HEIGHT: [0.1] * n_points,
        }
    )


# =============================================================================
# AMROLoader Initialization Tests
# =============================================================================


class TestAMROLoaderInit:
    def test_initialization(self, loader):
        assert loader.project_name == "test_project"
        assert isinstance(loader.project_data, ProjectData)

    def test_verbose_mode(self):
        loader = AMROLoader(project_name="test", verbose=True)
        assert loader.verbose is True

    def test_project_data_created(self, loader):
        assert loader.project_data.project_name == "test_project"


# =============================================================================
# Filename Validation Tests
# =============================================================================


class TestFilenameValidation:
    def test_valid_amro_filename(self, loader):
        valid_path = Path("ACTRot11_2K_3T_AMRO.csv")
        assert loader._is_valid_amro_filename(valid_path) is True

    def test_valid_amro_filename_variant(self, loader):
        valid_path = Path("ACTRot12_Dres2_10K_3.0T_AMRO.csv")
        assert loader._is_valid_amro_filename(valid_path) is True

    def test_invalid_missing_act(self, loader):
        invalid_path = Path("Rotation11_2K_3T_AMRO.csv")
        assert loader._is_valid_amro_filename(invalid_path) is False

    def test_invalid_missing_amro(self, loader):
        invalid_path = Path("ACTRot11_2K_3T.csv")
        assert loader._is_valid_amro_filename(invalid_path) is False

    def test_invalid_both_missing(self, loader):
        invalid_path = Path("random_file.csv")
        assert loader._is_valid_amro_filename(invalid_path) is False


# =============================================================================
# Degree to Radian Conversion Tests
# =============================================================================


class TestDegreeConversion:
    def test_convert_zero(self, loader):
        result = loader._convert_degs_to_rads(np.array([0]))
        assert result[0] == pytest.approx(0)

    def test_convert_90(self, loader):
        result = loader._convert_degs_to_rads(np.array([90]))
        assert result[0] == pytest.approx(np.pi / 2)

    def test_convert_180(self, loader):
        result = loader._convert_degs_to_rads(np.array([180]))
        assert result[0] == pytest.approx(np.pi)

    def test_convert_360(self, loader):
        result = loader._convert_degs_to_rads(np.array([360]))
        assert result[0] == pytest.approx(2 * np.pi)

    def test_convert_array(self, loader):
        degs = np.array([0, 90, 180, 270, 360])
        result = loader._convert_degs_to_rads(degs)
        expected = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Microohm Calculation Tests
# =============================================================================


class TestMicroohmCalculation:
    def test_calculate_uohm_cols(self, loader):
        df = pd.DataFrame(
            {
                "Res (ohm)": [1e-5, 2e-5, 3e-5],
                "other_col": [1, 2, 3],
            }
        )
        result = loader._calculate_uohm_cols(df)
        assert "Res (uohm)" in result.columns
        np.testing.assert_array_almost_equal(
            result["Res (uohm)"].values, [10, 20, 30]
        )

    def test_calculate_uohm_preserves_other_cols(self, loader):
        df = pd.DataFrame(
            {
                "Res (ohm)": [1e-5],
                "other_col": [1],
            }
        )
        result = loader._calculate_uohm_cols(df)
        assert "other_col" in result.columns
        assert result["other_col"].iloc[0] == 1


# =============================================================================
# Project Data Access Tests
# =============================================================================


class TestGetAmroData:
    def test_get_amro_data_returns_project_data(self, loader):
        result = loader.get_amro_data()
        assert result is loader.project_data
        assert isinstance(result, ProjectData)


# =============================================================================
# Integration Tests (with mocking)
# =============================================================================


class TestLoadAmroIntegration:
    @patch.object(Path, "is_file")
    @patch.object(ProjectData, "load_project_from_pickle")
    def test_load_from_pickle_when_exists(
        self, mock_load, mock_is_file, loader
    ):
        """Test that existing pickle file is loaded."""
        mock_is_file.return_value = True

        loader.load_amro()

        mock_load.assert_called_once()

    @patch.object(Path, "is_file")
    @patch.object(AMROLoader, "_run_amro_etl")
    def test_run_etl_when_no_pickle(
        self, mock_etl, mock_is_file, loader
    ):
        """Test that ETL runs when no pickle exists."""
        mock_is_file.return_value = False

        loader.load_amro()

        mock_etl.assert_called_once()


# =============================================================================
# Data Structure Creation Tests
# =============================================================================


class TestDataStructureCreation:
    def test_project_data_has_correct_name(self, loader):
        assert loader.project_data.project_name == "test_project"

    def test_project_data_initially_empty(self, loader):
        stats = loader.project_data.get_summary_statistics()
        assert stats["n_experiments"] == 0
        assert stats["n_oscillations"] == 0
