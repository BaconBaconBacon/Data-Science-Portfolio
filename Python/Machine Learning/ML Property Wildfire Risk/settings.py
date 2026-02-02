from pathlib import Path

HEADER_GEOM = "geometry"
HEADER_SAT_ID = "SAT_ID"
GIS_DESIRED_COLS = [
    "LATITUDE",  # Center of nominal 375 m fire pixel
    "LONGITUDE",  # Center of nominal 375 m fire pixel
    # 'BRIGHTNESS',
    # 'SCAN',
    # 'TRACK',
    "ACQ_DATE",
    # 'ACQ_TIME',
    HEADER_SAT_ID,  # N21 = NOAA-21, N=SNPP
    # 'INSTRUMENT',
    "CONFIDENCE",  # It is intended to help users gauge the quality of individual hotspot/fire pixels.
    # 'VERSION',
    # "BRIGHT_T31",  # T31 Channel brightness temperature of the fire pixel measured in Kelvin
    "FRP",  # FRP depicts the pixel-integrated fire radiative power in MW (megawatts)
    # 'DAYNIGHT',
    "TYPE",  # Inferred hot spot type: 0 is presumed vegetation fire
    HEADER_GEOM,
]

# Set the coordinate system
GIS_DEFAULT_CRS = 5070

SQL_ENGINE_STR = (
    "postgresql+psycopg2://postgres:postgres@localhost:5432/wildfire_risk_project"
)
TEST_SQL_ENGINE_STR = SQL_ENGINE_STR
#     (
#     "postgresql+psycopg2://postgres:postgres@localhost:5432/wildfire_risk_project_TEST"
# ))

TABLE_NAME_CACHE = "census_cache"
TABLE_NAME_PROPERTIES = "properties"
TABLE_NAME_PROPERTIES_TEST = "properties_test"
TABLE_NAME_CENSUS = "census"
TABLE_NAME_CENSUS_TEST = "census_test"

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
    # "B01003",  # total pop, redundant with b01001_001E
    # "B01005",  # Ancestry pop, redundant pop total
    # "B01009",  # pop total, redundant pop total
    "B02001",
    "B03001",
    "B05001",
    "B05002",
    # "B06001",  # place of birth by age, low wildfire relevance
    "B19001",
    "B19013",
    "B19019",
    "B19020",
    "B19083",
    "B19301",
    "B19326",
    "B23025",
    # "B23032",  #Weeks worked by sex, low wf relevance
    # "B24010",# Sex by occupation, large, low wf relevance
    # "B24050",#Industry by occupations, low wf relevance
    # "B08201", transportation, marginal relevance
    # "B08202", transportation, marginal relevance
    # "B08203", transportation, marginal relevance
    # "B08204", transportation, marginal relevance
    "B25031",
]

# not to be converted into percentages!
CENSUS_SUMMARY_TABLES = {
    "B01002",  # Median Age by Sex
    "B19013",  # Median Household Income
    "B19019",  # Median Household Income by Presence of Children
    "B19020",  # Aggregate Household Income
    "B19083",  # Gini Index
    "B19301",  # Per Capita Income
    "B19326",  # Median Income by Sex
    "B25035",  # Median Year Structure Built
}

CENSUS_VALID_GRANULARITY_LEVELS = {"block_group", "tract", "county"}


PATH_DATA = Path("data")
PATH_DATA_WILDFIRES = PATH_DATA / "wildfires"
PATH_DATA_PROPERTIES = PATH_DATA / "properties"
PATH_DATA_CENSUS = PATH_DATA / "census"
PATH_DATA_MODELS = PATH_DATA / "models"


PROP_TABLE_NAME = "properties"
PROP_TABLE_NAME_TEST = PROP_TABLE_NAME + "_test"
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


# SAVENAME_WILDFIRES = "combined_wildfire_data.parquet"
WILDFIRES_TABLE_NAME = "wildfires"

PROPERTIES_INIT_COUNT = 10

# GIS proximity scoring defaults
GIS_SCORING_DEFAULT_RADIUS_M = 80467  # 50 miles
GIS_SCORING_DEFAULT_POWER = 2
GIS_SCORING_DEFAULT_BANDWIDTH_M = 25000  # 25km
GIS_SCORING_DEFAULT_RINGS_M = [10_000, 25_000, 50_000, 100_000]
