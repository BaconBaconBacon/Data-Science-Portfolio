from pathlib import Path


BASE_PATH = Path(__file__).parent.parent.parent.parent

CONFIG_PATH = BASE_PATH / "config"
TESTS_PATH = BASE_PATH / "tests"
UTILS_PATH = BASE_PATH / "utils"
DATA_PATH = BASE_PATH / "data"
FIGURES_PATH = BASE_PATH / "figures"
NOTEBOOKS_PATH = BASE_PATH / "notebooks"

RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
FINAL_DATA_PATH = DATA_PATH / "final"

RAW_FIGURES_PATH = FIGURES_PATH / "raw"
PROCESSED_FIGURES_PATH = FIGURES_PATH / "processed"
FINAL_FIGURES_PATH = FIGURES_PATH / "final"

H_PALETTE = {0.5: "tab:red", 3: "tab:green", 7: "tab:orange", 9: "tab:blue"}


# The loader functionality reads only these from the cleaned AMRO data
LOADER_DESIRED_COLS = [
    "Temperature (K)",
    "Sample Position (deg)",
    "Res. (ohm-cm)",
    "ACT_str",
    "T",
    "H",
    "geo",
]


HEADER_ANGLE_DEG = "Sample Position (deg)"
HEADER_ANGLE_RAD = "Sample Position (rads)"
HEADER_RES_OHM = "Res. (ohm-cm)"
HEADER_RES_UOHM = "Res. (uohm-cm)"

HEADER_EXPERIMENT_PREFIX = "ACTRot"


# AMRO DataFrame header labels
HEADER_TEMP = "T"
HEADER_MAGNET = "H"
HEADER_EXP_LABEL = "ACT_str"
HEADER_GEO = "geo"
HEADER_WIRE_SEP = "L (cm)"

# TODO: Sort out the cross section vs width/height stuff.
HEADER_WIDTH = "W (cm)"
HEADER_HEIGHT = "H (cm)"
HEADER_CROSS_SECTION = "cross (cm^2)"

HEADER_TEMP_RAW = "Temperature (K)"
HEADER_MAGNET_RAW_OE = "Magnetic Field (Oe)"
HEADER_MAGNET_RAW_OE_ABS = "Abs. Magnetic Field (Oe)"

# Fourier DataFrame header labels
HEADER_MAG = "mag (ohm-cm)"
HEADER_MAG_RATIO = "amp_ratio"
HEADER_FREQ = "freqs (cycles/rot)"
HEADER_FREQ_LIST = "f_list"
HEADER_PHASE = "phase"
HEADER_PHASE_RAW = "phase_raw"

#  Fitter DF header Labels
HEADER_FIT_CHISQ = "chi_squared"
HEADER_FIT_RED_CHISQ = "red_chi_squared"
HEADER_PARAM_AMP_PREFIX = "amp"
HEADER_PARAM_FREQ_PREFIX = "freq"
HEADER_PARAM_PHASE_PREFIX = "phase"
HEADER_PARAM_MEAN_PREFIX = "mean"


# Loader DF Header Labels
HEADER_MEAN = "Mean (ohm-cm)"
HEADER_0DEG = "0deg (ohm-cm)"

# TODO: Once the data classes replace the META_DATA dict, these can be removed
KEY_RES_CONSTANTS = "res_constants"
KEY_TEMP_LABELS = "T_labels"
KEY_MAGNET_LABELS = "H_labels"

### Alternative resistivity units (used mostly in loader and plotting functions)
# value = (res-res_{constant})
HEADER_RES_DEL_MEAN_OHM = f"Delta Res. {HEADER_MEAN}"
HEADER_RES_DEL_MEAN_UOHM = HEADER_RES_DEL_MEAN_OHM.replace("ohm", "uohm")

HEADER_RES_DEL_0DEG_OHM = HEADER_RES_DEL_MEAN_OHM.replace(HEADER_MEAN, HEADER_0DEG)
HEADER_RES_DEL_0DEG_UOHM = HEADER_RES_DEL_0DEG_OHM.replace("ohm", "uohm")

# value = (res-res_{constant})/res_{constant} (unitless)
HEADER_RES_DEF_MEAN_NORM = f"Delta Res./R0 {HEADER_MEAN}".replace("(ohm-cm)", "")
HEADER_RES_DEL_0DEG_NORM = f"Delta Res./R0 {HEADER_0DEG}".replace("(ohm-cm)", "")

# value = (res-res_{constant})/res_{constant}*100
HEADER_RES_DEL_MEAN_NORM_PCT = HEADER_RES_DEF_MEAN_NORM + " (%)"
HEADER_RES_DEL_0DEG_NORM_PCT = HEADER_RES_DEL_0DEG_NORM + " (%)"


# For use in cleaner.py
CLEANER_HEADER_LENGTH = 25
# (row, col), zero indexed
CLEANER_WIRE_SEP_COORD = (13, 1)
CLEANER_CROSS_SEC_COORD = (14, 1)
CLEANER_LABEL_COORD = (11, 1)
CLEANER_GEOM_COORD = (12, 1)

CLEANER_OPTION_COORD = (5, 1)
CLEANER_OPTION_LABEL = "ACTRANSPORT"

CLEANER_T_MIN_RESOLUTION = 1  # round digit places

CLEANER_COL_RENAME_DICT = {"Res. ch2 (ohm-cm)": "Res. (ohm-cm)"}
