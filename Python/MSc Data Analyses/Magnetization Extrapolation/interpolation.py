from numpy.polynomial.chebyshev import chebfit, chebval
import pandas as pd
import numpy as np


def cheb_interpolation(
    df: pd.DataFrame,
    t_vals: list,
    degrees: float = 15,
    x_label="Magnetic Flux Density (G)",
    y_label="Sym Res. (ohm-cm)",
) -> tuple[dict, dict]:

    RvB_cheb_coeffs = {}
    RvB_cheb_residuals = {}

    for T in t_vals:
        try:
            this_df = df.query(f"T == {T}")
            x_data = this_df[x_label].values
            y_data = this_df[y_label].values

            coeffs = chebfit(x_data, y_data, deg=degrees)
            RvB_cheb_coeffs[T] = coeffs

            residuals = 100.0 * (chebval(x_data, coeffs) - y_data) / y_data
            RvB_cheb_residuals[T] = residuals
        except Exception as e:
            print(T)
            print(x_data)
            print(y_data)
            print(e)
            break
    return RvB_cheb_coeffs, RvB_cheb_residuals


def evaluate_cheb(x: float | list | pd.Series | int | np.ndarray, coeffs):

    return chebval(x, coeffs)
