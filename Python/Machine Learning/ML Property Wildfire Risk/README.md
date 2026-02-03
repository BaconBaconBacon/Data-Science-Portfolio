# Wildfire Risk Prediction

Predicts wildfire risk for US properties using satellite fire detections and census data. The model takes a property location, calculates its proximity to historical wildfires, enriches it with neighborhood demographics from the Census ACS5, and outputs a risk score.

Built as a portfolio project to demonstrate GIS data pipelines, ML workflows, and AWS deployment.

## What it does

1. **Properties** — Generates random US addresses, reverse-geocodes them to get Census block/tract IDs
2. **Wildfires** — Loads NASA FIRMS satellite fire detections, clusters nearby points into discrete fire events
3. **Census** — Pulls ACS5 socioeconomic data (income, housing age, heating fuel, etc.) for each property's neighborhood
4. **Proximity** — Calculates distance-based features: nearest fire, fire counts in distance rings, inverse-distance-weighted scores
5. **Model** — Trains XGBoost/RandomForest to predict `nearest_fire_km` from census + proximity features

## Setup

**Requirements:**
- Python 3.11+
- PostgreSQL with PostGIS extension
- Census API key (free from [census.gov](https://api.census.gov/data/key_signup.html))

```bash
# Install dependencies
pip install numpy pandas geopandas scikit-learn xgboost sqlalchemy psycopg2-binary census censusgeocode folium boto3

# Set your Census API key
export US_CENSUS_API_KEY="your_key_here"
```

**Database:**
Create a PostgreSQL database called `wildfire_risk_project`. The connection string is in `settings.py`.

**Data:**
Download VIIRS fire data from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) and extract to `data/wildfires/`.

## Running the pipeline

The Jupyter notebook `fire_risk_ML.ipynb` walks through the full pipeline. Or run the modules directly:

```bash
# Generate properties (this takes a while — uses Census geocoder API)
python load_properties.py 1000

# The notebook handles the rest: loading wildfires, fetching census data,
# computing proximity features, joining everything, and training
```

## Training on AWS

For larger runs, train on EC2 instead of locally:

```powershell
# Check your AWS setup
.\aws\check_aws.ps1

# Upload code and launch a spot instance
.\aws\deploy.ps1 -Code -Launch
```

The EC2 instance runs `train.py`, saves the model to S3, then auto-terminates. Costs ~$0.05-0.10/hr for a c6i.4xlarge spot instance.

```bash
# train.py usage
python train.py \
  --input s3://wildfire-risk-ml/data/model_joined.parquet \
  --output s3://wildfire-risk-ml/models/best_model.pkl \
  --n-iter 50 \
  --model both
```

## Project structure

```
├── fire_risk_ML.ipynb   # Main notebook — run this
├── train.py             # Standalone training script (local or AWS)
├── load_properties.py   # Property generation + geocoding
├── load_wildfires.py    # NASA FIRMS data ETL
├── load_census.py       # Census ACS5 API wrapper
├── gis.py               # Proximity scoring functions
├── visualize.py         # Folium maps + matplotlib charts
├── sql_funcs.py         # PostgreSQL/PostGIS interface
├── settings.py          # Config: paths, table names, parameters
├── aws/
│   ├── deploy.ps1       # Upload to S3 + launch EC2
│   ├── check_aws.ps1    # Validate AWS setup
│   ├── setup_ec2.sh     # EC2 bootstrap script
│   └── spot-options.json
└── data/
    ├── wildfires/       # NASA FIRMS shapefiles (not tracked)
    └── *.parquet        # Generated datasets (not tracked)
```

## Future work

- LANDFIRE vegetation/fuels layers
- GRIDMET weather data
- DEM-derived topography (slope, aspect, elevation)
- WUI boundary classification
