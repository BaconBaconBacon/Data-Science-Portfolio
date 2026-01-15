""" Code to validate any loaded data."""

# TODO:
# Is is worth implementing this kind of code?
# Claude suggestions:
#
# import numpy as np
#
#
# def validate_angles(angles: np.ndarray) -> bool:
#     """Check angles are within valid range [0, 360]."""
#     if np.any(angles < 0) or np.any(angles > 360):
#         raise ValueError("Angles must be between 0 and 360 degrees")
#     return True
#
#
# def validate_resistivity(res: np.ndarray) -> bool:
#     """Check resistivity values are positive."""
#     if np.any(res < 0):
#         raise ValueError("Resistivity must be non-negative")
#     return True
#
#
# def validate_oscillation_data(angles: np.ndarray, res: np.ndarray) -> bool:
#     """Validate paired angle/resistivity data."""
#     if len(angles) != len(res):
#         raise ValueError("Angles and resistivity arrays must have same length")
#     validate_angles(angles)
#     validate_resistivity(res)
#     return True
