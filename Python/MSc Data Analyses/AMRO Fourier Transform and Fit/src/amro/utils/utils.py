import pandas as pd
import numpy as np
import lmfit as lm

from amro.config.settings import (
    HEADER_ACT,
    HEADER_TEMP,
    HEADER_MAGNET,
    HEADER_PARAM_FREQ_PREFIX,
    HEADER_PARAM_AMP_PREFIX,
    HEADER_PARAM_PHASE_PREFIX,
    HEADER_PARAM_MEAN_PREFIX,
)


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


def convert_params_to_ndarrays(params: lm.parameter.Parameters):
    """
    Ensures the parameters are correctly ordered for sine_builder. Aside from the
    'mean' parameter, each 'phase' and 'freq' are paired based on the 'freq' value.
    This function ensures the amps and phases are in the correct order relative to
    the frequencies.

    """
    params_dict = params.valuesdict()

    freqs_list = []
    for key in params_dict.keys():
        if HEADER_PARAM_FREQ_PREFIX in key:
            freqs_list.append(int(params_dict[key]))
    amps_list = []
    phases_list = []
    for freq in freqs_list:
        amps_list.append(params_dict[HEADER_PARAM_AMP_PREFIX + f"{freq}"])
        phases_list.append(params_dict[HEADER_PARAM_PHASE_PREFIX + f"{freq}"])

    return (
        np.asarray(amps_list),
        np.asarray(freqs_list),
        np.asarray(phases_list),
        params_dict[HEADER_PARAM_MEAN_PREFIX],
    )
