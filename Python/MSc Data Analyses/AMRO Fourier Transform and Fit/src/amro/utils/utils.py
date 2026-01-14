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
    """Filter a DataFrame by experiment label, magnetic field, and/or temperature.

    Args:
        df: Pandas DataFrame to be queried.
        act: Experiment label(s) to filter by.
        h: Magnetic field strength(s) to filter by.
        t: Temperature(s) to filter by.

    Returns:
        Filtered DataFrame matching the specified criteria.
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
    """Build a query string for filtering AMRO DataFrames.

    Args:
        act: Experiment label(s) to include in query.
        h: Magnetic field value(s) to include in query.
        t: Temperature value(s) to include in query.

    Returns:
        Query string formatted for pandas DataFrame.query().
    """
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
    """Construct a Fourier series model from sine components.

    Computes: mean * (1 + sum(amp_i * sin(freq_i * rads + phase_i)))

    Args:
        rads: Array of angle values in radians.
        amps: Array of amplitude ratios for each frequency component.
        freqs: Array of frequencies (cycles per rotation).
        phases: Array of phase offsets in radians.
        mean: Mean resistivity value (offset).

    Returns:
        Array of model resistivity values.
    """
    summation = np.sum(
        amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
    )

    return mean * (summation + 1)


def flatten_list(lst: list) -> list:
    """Flatten a nested list into a single-level list.

    Args:
        lst: Nested list to flatten.

    Returns:
        Flattened list containing all elements.
    """
    return [item for sublist in lst for item in sublist]


def calculate_model_resistivities(x, params: tuple) -> np.ndarray:
    """Calculate model resistivity values using fitted parameters.

    Args:
        x: Array of angle values in radians.
        params: Tuple of (amplitudes, frequencies, phases, mean) from convert_params_to_ndarrays().

    Returns:
        Array of model resistivity values in the same units as the mean parameter.
    """
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
    """Convert lmfit Parameters object to numpy arrays for model calculation.

    Extracts amplitude, frequency, and phase values, ensuring correct ordering
    for use with sine_builder().

    Args:
        params: lmfit Parameters object containing fitted values.
        include_errs: If True, also return error estimates for each parameter.

    Returns:
        If include_errs is False: (amplitudes, frequencies, phases, mean).
        If include_errs is True: (amplitudes, amp_errors, frequencies, phases,
            phase_errors, mean, mean_error).
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
    """Format oscillation identifiers into a standardized string key.

    Args:
        act: Experiment label.
        t: Temperature in Kelvin.
        h: Magnetic field in Tesla.

    Returns:
        Formatted string in the form '{act}_T{t}K_H{h}T'.
    """
    return f"{act}_T{t}K_H{h}T"
