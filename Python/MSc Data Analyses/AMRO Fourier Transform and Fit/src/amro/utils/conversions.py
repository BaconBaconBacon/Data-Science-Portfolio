import numpy as np
import pandas as pd

"""Functions to convert between units"""


def convert_degs_to_rads(
    degs: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:

    return degs * (np.pi / 180)


def convert_rads_to_degs(
    rads: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:

    return rads * (180 / np.pi)


def convert_ohms_to_uohms(
    ohms: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:

    return ohms * (10**6)


def convert_uohms_to_ohms(
    uohms: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    return uohms * (10 ** (-6))


def convert_oe_to_teslas(
    oe: list | np.ndarray | float | pd.Series,
):
    return oe / (10**4)


def convert_teslas_to_oe(
    teslas: list | np.ndarray | float | pd.Series,
) -> list | np.ndarray | float:
    return teslas * (10**4)
