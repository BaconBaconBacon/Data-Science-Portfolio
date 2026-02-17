HEADER_ANGLE_DEG = "Sample Position (deg)"
HEADER_ANGLE_RAD = "Sample Position (rads)"
HEADER_RES_OHM = "Res. (ohm-cm)"
HEADER_RES_UOHM = "Res. (uohm-cm)"


# AMRO DataFrame header labels
HEADER_TEMP = "T"
HEADER_MAGNET = "H"
HEADER_EXP_LABEL = "ACT_str"
HEADER_GEO = "geo"
HEADER_WIRE_SEP = "L (cm)"
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

# Amplitude ratio column naming
HEADER_AMP_RATIO_PREFIX = "ratio_"
HEADER_AMP_RATIO_ERR_SUFFIX = "_err"


# Loader DF Header Labels
HEADER_MEAN = "Mean (ohm-cm)"
HEADER_0DEG = "0deg (ohm-cm)"

# Alternative resistivity units (used mostly in loader and plotting functions)
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
