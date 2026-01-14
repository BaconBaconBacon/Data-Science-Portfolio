import pandas as pd
import numpy as np
import lmfit as lm

from ..config import (
    HEADER_EXP_LABEL,
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
) -> pd.DataFrame:
    """
    Utility function to build strings to query Pandas DataFrames.

    Args:
        df: Pandas DataFrame to be queried.
        act: The ACT label(s)
        h: The magnetic field strength(s)
        t: The temperature(s)

    Returns:
        q: A string used to query a Pandas DataFrame.

    """
    q = build_query_string(act, h, t)
    if len(q) > 0:
        return df.query(q)
    else:
        return df


def build_query_string(
    act: str | list | None = None,
    h: float | int | list | None = None,
    t: float | int | list | None = None,
) -> str:
    """To query DataFrame representations of the AMRO data."""
    query = []
    if isinstance(act, str):
        query.append(HEADER_EXP_LABEL + f' == "{act}"')
    elif isinstance(act, list):
        query.append(HEADER_EXP_LABEL + f"== {act}")
    if h is not None:
        query.append(HEADER_MAGNET + f"== {h}")
    if t is not None:
        query.append(HEADER_TEMP + f"== {t}")
    return " & ".join(query)


def sine_builder(
    rads, amps: np.ndarray, freqs: np.ndarray, phases: np.ndarray, mean: float | int
) -> np.ndarray:
    """Returns a Fourier series consisting of sine terms and an offset."""
    summation = np.sum(
        amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
    )

    return mean * (summation + 1)


def flatten_list(lst: list) -> list:
    """Flatten a nested list."""
    return [item for sublist in lst for item in sublist]


def calculate_model_resistivities(x, params: tuple) -> np.ndarray:
    """To be used with the output of convert_params_to_ndarrays(). Assumes x the x variable is in units of rads.
    The units of y_fit will depend on those of the 'mean' parameter."""
    (
        amps_list,
        freqs_list,
        phase_list,
        mean,
    ) = params

    # Calculate model's values
    y_fit = sine_builder(
        x,
        amps_list,
        freqs_list,
        phase_list,
        mean,
    )
    return y_fit


def convert_params_to_ndarrays(
    params: lm.parameter.Parameters, include_errs: bool = False
) -> tuple:
    """
    Ensures the parameters are correctly ordered for sine_builder. Aside from the
    'mean' parameter, each 'phase' and 'freq' are paired based on the 'freq' value.
    This function ensures the amps and phases are in the correct order relative to
    the frequencies.

    """
    params_dict = params.create_uvars()

    freqs_list = []
    for key in params_dict.keys():
        if HEADER_PARAM_FREQ_PREFIX in key:
            f = params_dict[key].nominal_value
            freqs_list.append(int(f))

    amps_list = []
    phases_list = []
    amps_errs_list = []
    phases_errs_list = []

    mean = params_dict[HEADER_PARAM_MEAN_PREFIX].nominal_value
    mean_err = params_dict[HEADER_PARAM_MEAN_PREFIX].std_dev

    for freq in freqs_list:
        amps_list.append(params_dict[HEADER_PARAM_AMP_PREFIX + f"{freq}"].nominal_value)
        phases_list.append(
            params_dict[HEADER_PARAM_PHASE_PREFIX + f"{freq}"].nominal_value
        )

        amps_errs_list.append(params_dict[HEADER_PARAM_AMP_PREFIX + f"{freq}"].std_dev)
        phases_errs_list.append(
            params_dict[HEADER_PARAM_PHASE_PREFIX + f"{freq}"].std_dev
        )

    if include_errs:
        return (
            np.asarray(amps_list),
            np.asarray(amps_errs_list),
            np.asarray(freqs_list),
            np.asarray(phases_list),
            np.asarray(phases_errs_list),
            mean,
            mean_err,
        )
    else:
        return (
            np.asarray(amps_list),
            np.asarray(freqs_list),
            np.asarray(phases_list),
            mean,
        )


def format_oscillation_key(act: str, t: float, h: float) -> str:
    return f"{act}_T{t}K_H{h}T"
