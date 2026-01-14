import numpy as np
import pandas as pd

"""Functions to convert between units"""


def convert_degs_to_rads(
    degs: list | np.ndarray | float | pd.Series,
) -> np.ndarray | float:
    """Convert angle values from degrees to radians.

    Args:
        degs: Angle value(s) in degrees.

    Returns:
        Angle value(s) converted to radians.
    """
    if isinstance(degs, list):
        degs = np.asarray(degs)
    return degs * (np.pi / 180)


def convert_rads_to_degs(
    rads: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    """Convert angle values from radians to degrees.

    Args:
        rads: Angle value(s) in radians.

    Returns:
        Angle value(s) converted to degrees.
    """
    return rads * (180 / np.pi)


def convert_ohms_to_uohms(
    ohms: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    """Convert resistivity values from ohm-cm to micro-ohm-cm.

    Args:
        ohms: Resistivity value(s) in ohm-cm.

    Returns:
        Resistivity value(s) converted to micro-ohm-cm.
    """
    return ohms * (10**6)


def convert_uohms_to_ohms(
    uohms: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    """Convert resistivity values from micro-ohm-cm to ohm-cm.

    Args:
        uohms: Resistivity value(s) in micro-ohm-cm.

    Returns:
        Resistivity value(s) converted to ohm-cm.
    """
    return uohms * (10 ** (-6))


def convert_oe_to_teslas(
    oe: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    """Convert magnetic field values from Oersted to Tesla.

    Args:
        oe: Magnetic field value(s) in Oersted.

    Returns:
        Magnetic field value(s) converted to Tesla.
    """
    return oe / (10**4)


def convert_teslas_to_oe(
    teslas: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    """Convert magnetic field values from Tesla to Oersted.

    Args:
        teslas: Magnetic field value(s) in Tesla.

    Returns:
        Magnetic field value(s) converted to Oersted.
    """
    return teslas * (10**4)
