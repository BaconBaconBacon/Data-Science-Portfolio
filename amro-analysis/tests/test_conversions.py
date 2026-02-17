"""Tests for amro.utils.conversions module."""

import pytest
import numpy as np
import pandas as pd

from amro.utils import (
    convert_degs_to_rads,
    convert_rads_to_degs,
    convert_ohms_to_uohms,
    convert_uohms_to_ohms,
    convert_oe_to_teslas,
    convert_teslas_to_oe,
)


# =============================================================================
# Degree/Radian Conversion Tests
# =============================================================================


class TestDegsToRads:
    def test_zero_degrees(self):
        assert convert_degs_to_rads(0) == 0

    def test_90_degrees(self):
        assert convert_degs_to_rads(90) == pytest.approx(np.pi / 2)

    def test_180_degrees(self):
        assert convert_degs_to_rads(180) == pytest.approx(np.pi)

    def test_360_degrees(self):
        assert convert_degs_to_rads(360) == pytest.approx(2 * np.pi)

    def test_negative_degrees(self):
        assert convert_degs_to_rads(-90) == pytest.approx(-np.pi / 2)

    def test_array_input(self):
        degs = np.array([0, 90, 180, 270, 360])
        expected = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        np.testing.assert_array_almost_equal(convert_degs_to_rads(degs), expected)

    def test_list_input(self):
        degs = [0, 90, 180]
        result = convert_degs_to_rads(degs)
        assert result[1] == pytest.approx(np.pi / 2)

    def test_pandas_series_input(self):
        degs = pd.Series([0, 90, 180])
        result = convert_degs_to_rads(degs)
        assert result.iloc[1] == pytest.approx(np.pi / 2)


class TestRadsToDeg:
    def test_zero_rads(self):
        assert convert_rads_to_degs(0) == 0

    def test_pi_over_2(self):
        assert convert_rads_to_degs(np.pi / 2) == pytest.approx(90)

    def test_pi(self):
        assert convert_rads_to_degs(np.pi) == pytest.approx(180)

    def test_2pi(self):
        assert convert_rads_to_degs(2 * np.pi) == pytest.approx(360)

    def test_negative_rads(self):
        assert convert_rads_to_degs(-np.pi / 2) == pytest.approx(-90)

    def test_array_input(self):
        rads = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        expected = np.array([0, 90, 180, 270, 360])
        np.testing.assert_array_almost_equal(convert_rads_to_degs(rads), expected)


class TestDegsRadsRoundtrip:
    def test_scalar_roundtrip(self):
        original = 45.0
        result = convert_rads_to_degs(convert_degs_to_rads(original))
        assert result == pytest.approx(original)

    def test_array_roundtrip(self):
        original = np.linspace(0, 360, 37)
        result = convert_rads_to_degs(convert_degs_to_rads(original))
        np.testing.assert_array_almost_equal(result, original)

    def test_rads_roundtrip(self):
        original = np.pi / 3
        result = convert_degs_to_rads(convert_rads_to_degs(original))
        assert result == pytest.approx(original)


# =============================================================================
# Ohms/Microohms Conversion Tests
# =============================================================================


class TestOhmsToUohms:
    def test_zero_ohms(self):
        assert convert_ohms_to_uohms(0) == 0

    def test_one_ohm(self):
        assert convert_ohms_to_uohms(1) == 1e6

    def test_typical_amro_value(self):
        # Typical AMRO resistance ~1e-5 ohm-cm
        ohms = 1e-5
        expected = 10  # microohm-cm
        assert convert_ohms_to_uohms(ohms) == pytest.approx(expected)

    def test_array_input(self):
        ohms = np.array([1e-6, 1e-5, 1e-4])
        expected = np.array([1, 10, 100])
        np.testing.assert_array_almost_equal(convert_ohms_to_uohms(ohms), expected)


class TestUohmsToOhms:
    def test_zero_uohms(self):
        assert convert_uohms_to_ohms(0) == 0

    def test_one_million_uohms(self):
        assert convert_uohms_to_ohms(1e6) == pytest.approx(1)

    def test_typical_amro_value(self):
        uohms = 10  # microohm-cm
        expected = 1e-5  # ohm-cm
        assert convert_uohms_to_ohms(uohms) == pytest.approx(expected)

    def test_array_input(self):
        uohms = np.array([1, 10, 100])
        expected = np.array([1e-6, 1e-5, 1e-4])
        np.testing.assert_array_almost_equal(convert_uohms_to_ohms(uohms), expected)


class TestOhmsUohmsRoundtrip:
    def test_scalar_roundtrip(self):
        original = 1e-5
        result = convert_uohms_to_ohms(convert_ohms_to_uohms(original))
        assert result == pytest.approx(original)

    def test_array_roundtrip(self):
        original = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        result = convert_uohms_to_ohms(convert_ohms_to_uohms(original))
        np.testing.assert_array_almost_equal(result, original)


# =============================================================================
# Oersted/Tesla Conversion Tests
# =============================================================================


class TestOeToTeslas:
    def test_zero_oe(self):
        assert convert_oe_to_teslas(0) == 0

    def test_10000_oe_is_1_tesla(self):
        assert convert_oe_to_teslas(10000) == pytest.approx(1.0)

    def test_typical_amro_field(self):
        # 30000 Oe = 3 T
        oe = 30000
        expected = 3.0
        assert convert_oe_to_teslas(oe) == pytest.approx(expected)

    def test_array_input(self):
        oe = np.array([5000, 30000, 70000, 90000])
        expected = np.array([0.5, 3.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(convert_oe_to_teslas(oe), expected)

    def test_pandas_series_input(self):
        oe = pd.Series([10000, 20000, 30000])
        result = convert_oe_to_teslas(oe)
        assert result.iloc[2] == pytest.approx(3.0)


class TestTeslasToOe:
    def test_zero_tesla(self):
        assert convert_teslas_to_oe(0) == 0

    def test_1_tesla_is_10000_oe(self):
        assert convert_teslas_to_oe(1.0) == pytest.approx(10000)

    def test_typical_amro_field(self):
        teslas = 3.0
        expected = 30000
        assert convert_teslas_to_oe(teslas) == pytest.approx(expected)

    def test_array_input(self):
        teslas = np.array([0.5, 3.0, 7.0, 9.0])
        expected = np.array([5000, 30000, 70000, 90000])
        np.testing.assert_array_almost_equal(convert_teslas_to_oe(teslas), expected)


class TestOeTeslaRoundtrip:
    def test_scalar_roundtrip_oe(self):
        original = 50000
        result = convert_teslas_to_oe(convert_oe_to_teslas(original))
        assert result == pytest.approx(original)

    def test_scalar_roundtrip_tesla(self):
        original = 5.0
        result = convert_oe_to_teslas(convert_teslas_to_oe(original))
        assert result == pytest.approx(original)

    def test_array_roundtrip(self):
        original = np.array([5000, 30000, 70000, 90000])
        result = convert_teslas_to_oe(convert_oe_to_teslas(original))
        np.testing.assert_array_almost_equal(result, original)
