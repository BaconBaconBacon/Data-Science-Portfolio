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
