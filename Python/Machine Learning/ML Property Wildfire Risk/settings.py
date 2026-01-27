from pathlib import Path

GIS_DESIRED_COLS = [
    "LATITUDE",  # Center of nominal 375 m fire pixel
    "LONGITUDE",  # Center of nominal 375 m fire pixel
    # 'BRIGHTNESS',
    # 'SCAN',
    # 'TRACK',
    "ACQ_DATE",
    # 'ACQ_TIME',
    "SAT_ID",  # N21 = NOAA-21, N=SNPP
    # 'INSTRUMENT',
    "CONFIDENCE",  # It is intended to help users gauge the quality of individual hotspot/fire pixels.
    # 'VERSION',
    # "BRIGHT_T31",  # T31 Channel brightness temperature of the fire pixel measured in Kelvin
    "FRP",  # FRP depicts the pixel-integrated fire radiative power in MW (megawatts)
    # 'DAYNIGHT',
    "TYPE",  # Inferred hot spot type: 0 is presumed vegetation fire
    "geometry",
]

# Set the coordinate system
GIS_DEFAULT_CRS = 5070

SQL_ENGINE_STR = (
    "postgresql+psycopg2://postgres:postgres@localhost:5432/wildfire_risk_project"
)
TEST_SQL_ENGINE_STR = (
    "postgresql+psycopg2://postgres:postgres@localhost:5432/wildfire_risk_project_TEST"
)

CENSUS_FEATURES = [
    "B25014",
    "B25015",
    "B25016",
    "B25017",
    "B25024",
    "B25032",
    "B25033",
    "B25034",
    "B25035",
    "B25040",
    "B25041",
    "B25047",
    "B25063",
    "B25070",
    "B25091",
    "B01001",
    "B01002",
    "B01003",
    "B01005",
    "B01009",
    "B02001",
    "B03001",
    "B05001",
    "B05002",
    "B06001",
    "B19001",
    "B19013",
    "B19019",
    "B19020",
    "B19083",
    "B19301",
    "B19326",
    "B23025",
    "B23032",
    "B24010",
    "B24050",
    "B08201",
    "B08202",
    "B08203",
    "B08204",
    "B25031",
]


PATH_DATA = Path("data")
PATH_DATA_WILDFIRES = PATH_DATA / "wildfires"
PATH_DATA_PROPERTIES = PATH_DATA / "properties"
PATH_DATA_CENSUS = PATH_DATA / "census"
PATH_DATA_MODELS = PATH_DATA / "models"


PROP_TABLE_NAME = "properties"
PROP_LABELS_KEYS_MAP = {
    "geoid": "GEOID",
    "block_id": "BLOCK",
    "block_grp": "BLKGRP",
    "tract_id": "TRACT",
    "county_id": "COUNTY",
    "state_id": "STATE",
}

# Scale time so 3 days maps to ~750m (same as spatial eps)
# This means fires within 750m AND within 3 days cluster together
WILDFIRES_CLUSTER_TIME_SCALE = 250  # meters per day (750m / 3 days)
WILDFIRES_CLUSTER_SPATIAL_EPS = 750  # meters (2x pixel size)


USA_MAX_LAT = 49  # °23′04.08″N 95°9′12.16″W, Northwest Angle, Minnesota
USA_MIN_LAT = 24  # °31′15″N 81°57′49″W, Ballast Key, Florida
USA_MAX_LON = -66  # °57′02″W 44°48′54″N # West Quoddy Head Light, Maine
USA_MIN_LON = -124  # °43′37″W 48°23′09″N, Cape Flattery, Washington


SAVENAME_WILDFIRES = "combined_wildfire_data.parquet"
SAVENAME_CENSUS = ""
SAVENAME_PROPERTIES = ""
