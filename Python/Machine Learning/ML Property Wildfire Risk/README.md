# Wildfire Risk Prediction

Predicts wildfire risk for US properties using satellite fire detections and census data. The model takes a property location, calculates its proximity to historical wildfires, enriches it with neighborhood demographics from the Census ACS5, and outputs a risk score.

Built as a portfolio project to demonstrate GIS data pipelines and ML workflows.

## How it works

The goal is to predict a property's wildfire risk from its location alone. Given a set of GPS coordinates (e.g., a real estate portfolio), the trained model estimates each property's proximity to wildfire activity using neighborhood characteristics as features.

**Training data generation** uses randomly sampled US property locations to build a large, geographically diverse dataset. **Inference** takes real GPS coordinates and runs them through the same pipeline.

### Pipeline stages

```
Properties ──> Census Merge ──> Wildfires ──> Proximity Features ──> Model Training
  (GPS)         (ACS5 API)     (NASA FIRMS)    (distance scores)    (XGBoost / RF)
```

1. **Properties** -- Generates random US locations for training data, or accepts real GPS coordinates via CSV for inference. Locations are geocoded to Census geographies (county/tract/block group) using TIGER shapefiles.

2. **Wildfires** -- Loads NASA FIRMS VIIRS satellite fire detections. Raw detections are clustered spatiotemporally using DBSCAN (750m spatial, 3-day temporal window) to deduplicate overlapping satellite passes into discrete fire events.

3. **Census** -- Pulls ACS5 socioeconomic data for each property's neighborhood: household income, housing age, heating fuel type, tenure, employment, demographics, and more (~850 features after expansion). Data is cached in PostGIS to avoid redundant API calls.

4. **Proximity** -- Calculates 12 distance-based features for each property:
   - `nearest_fire_km` -- distance to closest fire event (the prediction target)
   - `kde_density` -- kernel density estimate of fire activity
   - `idw_score` -- inverse-distance-weighted fire intensity
   - `exp_decay_score` -- exponential decay fire score
   - `fire_count_*` -- fire counts in 10km, 25km, 50km, 100km rings
   - `fire_FRP_*` -- cumulative fire radiative power in those same rings

5. **Model** -- Trains XGBoost and RandomForest regressors to predict `nearest_fire_km` from census features. Uses RandomizedSearchCV for hyperparameter tuning with cross-validation.

### Adaptive imputation

Missing census data is handled through missingness mechanism detection. Each feature's missing values are correlated with all other features:
- **MCAR** (Missing Completely At Random): low correlation -> median imputation
- **MAR** (Missing At Random): high correlation -> iterative imputation that preserves feature relationships

This avoids the common mistake of applying a single imputation strategy to all features.

## Setup

### Requirements

- Python 3.11+
- PostgreSQL with PostGIS extension
- Census API key (free from [census.gov](https://api.census.gov/data/key_signup.html))
- CUDA-capable GPU (optional, for XGBoost GPU acceleration)

### Installation

```bash
pip install numpy pandas geopandas scikit-learn xgboost sqlalchemy psycopg2-binary census censusgeocode folium shapely scipy requests joblib
```

### Environment

```bash
# Set your Census API key
export US_CENSUS_API_KEY="your_key_here"
```

### Database

Create a PostgreSQL database called `wildfire_risk_project`. The connection string is in `settings.py`:

```
postgresql+psycopg2://postgres:postgres@localhost:5432/wildfire_risk_project
```

PostGIS extension is installed automatically on first connection.

### Wildfire data

Download VIIRS fire data from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) and extract shapefiles to `data/wildfires/`. The pipeline expects VIIRS detections from NOAA-21 (J2) and Suomi NPP (J1).

## Running the pipeline

### ETL pipeline (CLI)

The ETL pipeline generates training data and saves it as parquet files for the notebook.

```bash
# Build training data: generate random properties + census + wildfires + proximity
python run_etl_pipeline.py --num-properties 10000 --granularity county

# Score real properties: load GPS coordinates and run them through the pipeline
python run_etl_pipeline.py --coords-file my_properties.csv --granularity county

# Census merge only (use existing properties in database)
python run_etl_pipeline.py --granularity county

# Properties only (skip census merge)
python run_etl_pipeline.py --num-properties 5000 --skip-census
```

The pipeline runs property generation and census fetching in parallel (producer/consumer threads) for faster execution on full runs.

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-properties N` | 0 | Number of random properties to generate |
| `--coords-file PATH` | None | CSV with `latitude,longitude` columns |
| `--granularity` | county | Census level: `county`, `tract`, or `block_group` |
| `--year` | 2023 | ACS5 survey year |
| `--skip-census` | false | Skip census enrichment |
| `--skip-wildfires` | false | Skip wildfire + proximity computation |
| `--n-jobs` | 5 | Parallel jobs for proximity computation |
| `--test` | false | Use test database tables |

**CSV format for `--coords-file`:**
```csv
latitude,longitude
34.0522,-118.2437
40.7128,-74.0060
```

**Output files:**
- `data/merged_properties_census.parquet` -- properties with census features
- `data/targets_features.parquet` -- above plus proximity features (ready for training)

### Jupyter notebook

`fire_risk_ML.ipynb` walks through the full pipeline interactively. It loads pre-computed parquet files from the ETL pipeline when available, or falls back to computing each step inline.

Notebook sections:
1. Setup and database connection
2. Property loading/generation
3. Census data merge
4. Wildfire data and proximity features
5. Preprocessing (adaptive imputation, feature selection)
6. Model training (RandomForest, XGBoost)
7. Feature importance and evaluation
8. Visualization (maps, charts)

### Model training (standalone)

```bash
python train.py \
  --input data/targets_features.parquet \
  --output data/models/best_model.pkl \
  --n-iter 50 \
  --model both
```

## Testing

Tests use pytest with markers for selective execution. Shared fixtures are in `conftest.py`.

```bash
# Fast unit tests only — no DB needed (~10s)
pytest test_etl_pipeline.py -m "not db and not e2e" -v

# DB integration tests (~30s)
pytest test_etl_pipeline.py -m "db and not e2e" -v

# Full ETL test suite (~2 min)
pytest test_etl_pipeline.py -v

# Training smoke tests — XGBoost + RandomForest on reduced features
pytest test_train.py -v

# Everything
pytest test_etl_pipeline.py test_train.py -v
```

### Test structure

**`test_etl_pipeline.py`** -- ETL and data pipeline tests:
- Settings validation (census features, table names, CRS)
- Missingness analysis (MCAR/MAR detection, edge cases)
- Preprocessing helpers (NaN column dropping, correlation filtering)
- GIS validation (CRS checks, proximity feature shapes/values)
- SQL round-trips (DataFrame, GeoDataFrame, column case preservation)
- Table auto-creation and duplicate removal
- Mini end-to-end pipeline (20 properties through all stages)

**`test_train.py`** -- Model training tests:
- Preprocessing output validation (shapes, feature names)
- XGBoost smoke test (n_iter=2, cv=2)
- RandomForest smoke test (n_iter=2, cv=2)
- Evaluation metrics and feature importance extraction

## Project structure

```
├── fire_risk_ML.ipynb      # Main notebook — interactive walkthrough
├── run_etl_pipeline.py     # CLI pipeline: properties -> census -> wildfires -> proximity
├── train.py                # Preprocessing + model training + evaluation
├── load_properties.py      # Property generation, TIGER shapefiles, geocoding
├── load_census.py          # Census ACS5 API, feature normalization, label translation
├── load_wildfires.py       # NASA FIRMS ETL, DBSCAN spatiotemporal clustering
├── gis.py                  # Proximity scoring: IDW, KDE, decay, distance rings
├── sql_funcs.py            # PostgreSQL/PostGIS connection and query interface
├── missing_analysis.py     # MCAR/MAR missingness detection for imputation strategy
├── visualize.py            # Folium interactive maps + matplotlib charts
├── settings.py             # Configuration: CRS, table names, feature codes, paths
├── conftest.py             # Shared pytest fixtures
├── test_etl_pipeline.py    # ETL + data pipeline tests
├── test_train.py           # Model training smoke tests
├── pytest.ini              # Test marker registration
└── data/
    ├── wildfires/           # NASA FIRMS shapefiles (not tracked)
    ├── census/              # Census API metadata cache
    ├── cache/               # Preprocessing cache (MD5-keyed)
    └── *.parquet            # Generated datasets (not tracked)
```

## Key design decisions

**Projected CRS (EPSG:5070)** -- All spatial operations use CONUS Albers Equal Area projection. Distances are in meters, which avoids the distortion of lat/lon-based distance calculations.

**PostGIS for persistence** -- Property locations, census data, and wildfire detections are stored in PostgreSQL with PostGIS. This enables spatial indexing and avoids re-downloading data on subsequent runs.

**Census data in long format** -- Census features are stored in narrow format (geoid, variable_code, estimate, percent) for efficient storage. They're pivoted to wide format only when merged onto properties for training.

**Preprocessing caching** -- The full preprocessing pipeline (imputation, scaling, feature selection) is cached to disk using MD5 hashes of the input data shape and parameters. Re-runs with the same data load in under a second.

**Parallel ETL** -- For full pipeline runs, property generation and census fetching run in parallel threads. The census consumer polls for new geographies while the property producer is still generating locations.

## Future work

- PyTorch tabular MLP as a third model for comparison
- LANDFIRE vegetation/fuels layers as additional features
- GRIDMET weather data (temperature, precipitation, wind)
- DEM-derived topography (slope, aspect, elevation)
- WUI (Wildland-Urban Interface) boundary classification
- Airflow DAG for pipeline orchestration
