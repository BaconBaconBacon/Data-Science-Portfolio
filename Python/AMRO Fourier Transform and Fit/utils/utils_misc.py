import pandas as pd
import numpy as np


def QueryDataFrame(
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
        q.append('ACT_str == "{}"'.format(act))
    elif type(act) is list:
        q.append("ACT_str == {}".format(act))
    if h is not None:
        q.append("H == {}".format(h))
    if t is not None:
        q.append("T == {}".format(t))
    if len(q) > 0:
        q = " & ".join(q)
        return df.query(q)
    else:
        return df


def DictBuilder():
    """
        Utility function to pre-build dictionary for storing meta-data regarding what ACT labels, H-values, and T-values
        are present in the data. Minimizes reliance on nested dictionaries.

        using defaultdict?
    Returns:

    """
    return


def SineBuilder(
    rads, amps: np.ndarray, freqs: np.ndarray, phases: np.ndarray, mean: float | int
):
    """Returns a Fourier series consisting of sine terms and an offset."""
    summation = np.sum(
        amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
    )

    return mean * (summation + 1)
