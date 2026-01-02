import pandas as pd
import numpy as np

from amro.config.settings import HEADER_ACT, HEADER_TEMP, HEADER_MAGNET


def query_dataframe(
    df: pd.DataFrame,
    act: str | list | None = None,
    h: float | int | list | None = None,
    t: float | int | list | None = None,
):
    """
    Utility function to build strings to query Pandas DataFrames.

    Args:
        act: The ACT label(s)
        h: The magnetic field strength(s)
        t: The temperature(s)

    Returns:
        q: A string used to query a Pandas DataFrame.

    """
    q = []
    if type(act) is str:
        q.append(HEADER_ACT + f' == "{act}"')
    elif type(act) is list:
        q.append(HEADER_ACT + f"== {act}")
    if h is not None:
        q.append(HEADER_MAGNET + f"== {h}")
    if t is not None:
        q.append(HEADER_TEMP + f"== {t}")
    if len(q) > 0:
        q = " & ".join(q)
        return df.query(q)
    else:
        return df


def sine_builder(
    rads, amps: np.ndarray, freqs: np.ndarray, phases: np.ndarray, mean: float | int
):
    """Returns a Fourier series consisting of sine terms and an offset."""
    summation = np.sum(
        amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
    )

    return mean * (summation + 1)
