import pandas as pd
import numpy as np
import formulas as fo
from scipy.optimize import fsolve


def solve_for_magnetization(H, D, T, J, C, n):
    """
    Based on inputted values, we numerically solve this to extract a magnetization.

    # TODO: Why is this necessary?
    """
    if isinstance(H, float) or isinstance(H, int):

        def func(M):
            return fo.magnetization(H, M[0], D, T, J, C, n) - M[0]

        return fsolve(func, [20])[0]
    else:
        M_list = []
        for H in H:

            def func(M):
                return fo.magnetization(H, M[0], D, T, J, C, n) - M[0]

            M_list.append(fsolve(func, [20])[0])
        return M_list
