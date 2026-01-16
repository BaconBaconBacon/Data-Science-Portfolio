import numpy as np
import pandas as pd
import constants as co


def DemagFactor(a, b, c):
    """
    c must be parallel to the applied magnetic field.

    Finds the demagnetizing factor for a rectangular prism onto which a
    magnetic field is applied. The field is assumed to be applied parallel
    to the z-axis, and therefore perpendicularly to the x- and y-axes.

    The values of the input variables are taken to be half the length
    of the rectangular prism, with one length for each spatial dimension.
    -a < x < a
    -b < y < b
    -c < z < c

    Thus, each a, b, and c are half the sample length in each direction.

    TODO: Make more readable.
    """
    return (1 / np.pi) * (
        (
            (b**2 - c**2)
            * np.log(
                (np.sqrt(a**2 + b**2 + c**2) - a) / (np.sqrt(a**2 + b**2 + c**2) + a)
            )
        )
        / (2 * b * c)
        + (
            (a**2 - c**2)
            * np.log(
                (np.sqrt(a**2 + b**2 + c**2) - b) / (np.sqrt(a**2 + b**2 + c**2) + b)
            )
        )
        / (2 * a * c)
        + (b * np.log((np.sqrt(a**2 + b**2) + a) / (np.sqrt(a**2 + b**2) - a)))
        / (2 * c)
        + (a * np.log((np.sqrt(a**2 + b**2) + b) / (np.sqrt(a**2 + b**2) - b)))
        / (2 * c)
        + (c * np.log((np.sqrt(b**2 + c**2) - b) / (np.sqrt(b**2 + c**2) + b)))
        / (2 * a)
        + (c * np.log((np.sqrt(a**2 + c**2) - a) / (np.sqrt(a**2 + c**2) + a)))
        / (2 * b)
        + 2 * np.arctan((a * b) / (c * np.sqrt(a**2 + b**2 + c**2)))
        + (a**3 + b**3 - 2 * c**3) / (3 * a * b * c)
        + ((a**2 + b**2 - 2 * c**2) * np.sqrt(a**2 + b**2 + c**2)) / (3 * a * b * c)
        + (c * (np.sqrt(a**2 + c**2) + np.sqrt(b**2 + c**2))) / (a * b)
        - (
            (a**2 + b**2) ** (3 / 2)
            + (b**2 + c**2) ** (3 / 2)
            + (c**2 + a**2) ** (3 / 2)
        )
        / (3 * a * b * c)
    )


def lange_g_factor(J, S=0.5):
    """
    https://en.wikipedia.org/wiki/Land%C3%A9_g-factor

    L, orbital angular momentum of the magnetic ion
    S = 1/2, spin of an electron
    J = L+S, total quantum angular momentum
    """
    L = J - S
    top = S * (S + 1) - L * (L + 1)
    bottom = 2 * J * (J + 1)

    # The 1 on wiki comes from Kittel, Blundell has 3/2 but that is equivalent based on his second term.
    gj = 3 / 2 + top / bottom
    return gj


def majumdar_func(M_of_H, rho_0, M_0, A, exp):
    """TODO: Add the citation to majumdar et al."""
    # The actual paper seems to imply that there is no -1 term
    # but! \Delta \rho / \rho_mean = (rho_H -\rho_mean)/\rho_mean , so it's ok
    rho_of_H = (1 - A * (M_of_H / M_0) ** 2) * rho_0
    return rho_of_H


def brillouin_function(x, J):
    """
    TODO: Check this against Blundell.
    """
    # J = L + S
    # L = J - S

    J1 = (2 * J + 1) / (2 * J)
    J2 = 1 / (2 * J)

    coth1 = 1 / np.tanh(J1 * x)
    coth2 = 1 / np.tanh(J2 * x)

    Bj = J1 * coth1 - J2 * coth2

    return Bj


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


def magnetization(H, M, D, T, J, C, n):
    """

    M0 : Saturization magnetization
    H  : Applied Magnetic Field, but perhaps this should be magnetic flux density, B
    T  : Temperature
    A : Arbitrary scaling factor to absorb the details (since we just want a close fit)

    NOTE: May need to add a demagnetization factor to this,
             B = H + 4\pi (1-D)M
           and solve it self-consistently, which would allow the inclusion of both B and
           demagnetization effects...

    Both M0 and A are temperature independent

    Also check out eqn 5.6 and eqn 5.7, which gives the brillouin function for Weiss
    ferromagnetism.
    """

    # J = L+S

    B = H + 4 * np.pi * (1 - D) * M

    x = (
        lange_g_factor(J, S=1 / 2) * co.mub_CGS * J * B / (co.kb_CGS * T)
    )  # dimensionless

    m_0 = n * lange_g_factor(J, S=1 / 2) * co.mub_CGS * J  # emu/cm^3

    pauli_mag = C * H

    return m_0 * brillouin_function(x, J) + pauli_mag
