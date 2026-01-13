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
CLEANER_DROP_COLS = [
    "Comment",
    "Time Stamp (sec)",
    "Status (code)",
    "Volts ch1",
    "V Std.Dev. ch1",
    "Res. ch1 (ohm-cm)",
    "Res. Std.Dev. ch1",
    "Hall ch1 (cm^3/coul)",
    "Hall ch2 (cm^3/coul)",
    "Hall Std.Dev. ch1",
    "Hall Std.Dev. ch2",
    "Crit.Cur. ch1 (mA)",
    "Crit.Cur. ch2 (mA)",
    "C.Cur. Std.Dev. ch1",
    "C.Cur. Std.Dev. ch2",
    "2nd Harm. ch1 (dB)",
    "3rd Harm. ch1 (dB)",
    "Quad.Error ch1 (ohm-cm-rad)",
    "Quad.Error ch2 (ohm-cm-rad)",
    "Drive Signal ch1 (V)",
    "Bridge 1 Resistance (ohms)",
    "Bridge 1 Excitation (uA)",
    "Bridge 2 Resistance (ohms)",
    "Bridge 2 Excitation (uA)",
    "Bridge 3 Resistance (ohms)",
    "Bridge 3 Excitation (uA)",
    "Bridge 4 Resistance (ohms)",
    "Bridge 4 Excitation (uA)",
    "Signal 1 Vin (V)",
    "Signal 2 Vin (V)",
    "Digital Inputs (code)",
    "Drive 1 Iout (mA)",
    "Drive 1 Ipower (watts)",
    "Drive 2 Iout (mA)",
    "Drive 2 Ipower (watts)",
    "Pressure ()",
    "Map 20 ()",
    "Map 21 ()",
    "Map 22 ()",
    "Map 23 ()",
    "Map 24 ()",
    "Map 25 ()",
    "Map 26 ()",
    "Map 27 ()",
    "Map 28 ()",
    "Map 29 ()",
    "Excitation (mA)",
    "Frequency (Hz)",
    "Volts ch2",
    "V Std.Dev. ch2",
    "Res. Std.Dev. ch2",
    "ACT Status (code)",
    "ACT Gain",
    "2nd Harm. ch2 (dB)",
    "3rd Harm. ch2 (dB)",
    "Drive Signal ch2 (V)",
]
