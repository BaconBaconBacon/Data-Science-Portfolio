H_PALETTE = {0.5: "tab:red", 3: "tab:green", 7: "tab:orange", 9: "tab:blue"}

HEADER_EXPERIMENT_PREFIX = "ACTRot"

CLEANER_ANG_CHANGE_THRESH = 0.001  # deg
CLEANER_TEMP_STABLE_THRESH = 0.05  # K
CLEANER_MAG_FIELD_STABLE_THRESH = 0.01  # T
CLEANER_OUTLIER_RES_STD = 5  # many standard deviations
CLEANER_SAVE_FN_SUFFIX = "_antisymmetrized.csv"


# The loader functionality reads only these from the cleaned AMRO data
# TODO: Replace this in the code with the respective HEADER_X_ stuff
LOADER_DESIRED_COLS = [
    "Temperature (K)",
    "Sample Position (deg)",
    "Res. (ohm-cm)",
    "ACT_str",
    "T",
    "H",
    "geo",
]
