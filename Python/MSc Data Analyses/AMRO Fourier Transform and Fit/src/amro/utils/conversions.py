import numpy as np

"""Functions to convert between units"""


def convert_degs_to_rads(degs: list | np.ndarray | float) -> list | np.ndarray | float:

    return degs * (np.pi / 180)


def convert_rads_to_degs(rads: list | np.ndarray | float) -> list | np.ndarray | float:

    return rads * (180 / np.pi)


def convert_ohms_to_uohms(ohms: list | np.ndarray | float) -> list | np.ndarray | float:

    return ohms * (10**6)


def convert_uohms_to_ohms(
    uohms: list | np.ndarray | float,
) -> list | np.ndarray | float:
    return uohms * (10 ** (-6))
