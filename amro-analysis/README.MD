# AMRO Analysis

**Fermi Surface Symmetry Analysis via Angle-Resolved Magnetoresistance Oscillations**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Introduction

The purpose of this project is to provide a package to quickly analyze angle-resolved magnetoresistance oscillation (AMRO) data, wherein we measure the longitudinal resistivity of a rectangular prism sample as we rotate it within a strong applied magnetic field. Resistivity is an intrinsic analogue to resistance. The resistivity is measured at both positive and negative magnetic field values, and then anti-symmetrized to minimize the influence of sample misalignment and any transverse resistivity that may be present. From this, we can gain insight on the [Fermi surface](https://en.wikipedia.org/wiki/Fermi_surface), a quantum mechanical structure that determines many of the electronic transport properties of a material (that is to say, the way electricity runs through it). For example, this surface will be different for metals, semi-conductors, and insulators.

The symmetry of the AMRO (how many times the resistivity repeats per sample rotation) is determined by the Fermi surface. A significant change in the AMRO symmetry indicates a significant change in the Fermi surface structure—an unusual occurrence indicative of interesting quantum physics. Investigations of such behaviour can help discover new materials for use in future technologies.

## Features

- **Data Cleaning**: Automated preprocessing of raw PPMS ACT Option data files with anti-symmetrization
- **Fourier Transform Analysis**: Extract rotational symmetry components from AMRO oscillations
- **Multi-Frequency Fitting**: Fit sinusoidal models with multiple symmetry terms using least-squares optimization
- **Publication-Ready Plotting**: Generate faceted plots of fits with residuals
- **Data Persistence**: Save/load project data via pickle and CSV formats
- **CLI Tools**: Command-line scripts for batch processing

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "AMRO Analysis"

# Install the package in development mode
pip install -e .

# Install development dependencies (for testing, notebooks)
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- numpy, pandas, scipy
- matplotlib, seaborn
- lmfit

## Quick Start

```python
from amro import AMROLoader, Fourier, AMROFitter

project_name = 'YbPdBi_AMRO' 

# Load cleaned AMRO data
loader = AMROLoader(project_name, verbose=True)
project_data = loader.load_amro()

# Perform Fourier analysis
fourier = Fourier(project_data, project_name)
fourier.fourier_transform_experiments()

# Fit oscillations with sinusoidal model
fitter = AMROFitter(project_data, save_name=project_name, min_amp_ratio=0.075, max_freq=8)
for exp_label in project_data.get_experiment_labels():
    fitter.fit_act_experiment(exp_label)

# View results
print(project_data.get_summary_statistics())
```

## Expected Data Format

### Input Files

This package expects data files from a **QD USA PPMS ACT (AC Transport) Option**. Files should be `.dat` or `.csv` format with:

- Header containing experiment metadata (experiment label, geometry, wire separation)
  - The 'Material' field should also include the experiment label, with a prefix that matches HEADER_EXPERIMENT_PREFIX in config/settings.py
  - The 'Comment' field should have the experiment's geometry, either parallel or perpendicular. Functionality for alternatives can be added to the repo later.
- Columns for temperature, magnetic field, sample angle, and resistance

For an example of the expected input data, see the `.dat` file in /tests/figures.

Other data files from other measurement devices would need to have their own cleaner code written to massage the data into the format that the loader code requires as input.

### File Naming

Files should include an experiment label with the prefix given by the variable HEADER_EXPERIMENT_PREFIX in config/settings.py.

The examples in this repo use `ACTRot` (e.g., `ACTRot11_sample_data.dat`). This prefix can be changed to match your project's naming conventions.

This experiment label should match the one in the 'Material' field of the header of the ACT option.

### Automatic Oscillation Detection
The cleaner code has functionality to automatically detect oscillations in a data file with mixed measurements (i.e., a sweep of H at fixed angle), but be sure to review its output to be confident in its results. I can't exactly claim there was thorough testing on a plethora of input file types.

### Geometry Correction

If measurements were taken with default geometry values (wire_sep=1, cross_section=1), use:

```python
project_data.correct_geometry_scaling(
    experiment_label="ACTRot11",
    wire_sep=0.15,      # cm
    cross_section=0.02  # cm^2
)
```
## Naming Convention

Each set of resistivity measurements as the sample rotation angle is changed, for a given |H| and T, is considered one AMROscillation. Each AMROscillation has an ExperimentalData, FourierResult and FitResult associated with it.

Multiple AMROscillation's constitute one Experiment, which has an associated experiment label, geometry, wire separation, and cross-section. Multiple Experiments constitute one ProjectData. 

The data is stored in custom data classes that reflect this hierarchy.

## Usage

### Step 1: Data Cleaning

Place raw `.dat` files from the QD USA PPMS ACT Option in `data/raw/`, then run:

```bash
python scripts/run_cleaner.py --datafile-type .dat --verbose
```

This will:
- Extract metadata from file headers
- Filter for oscillation data
- Remove outliers
- Anti-symmetrize measurements (average resistivity measurements taken at +H and -H)
- Save cleaned data to `data/processed/`

**Options:**
| Flag | Description |
|------|-------------|
| `--datafile-type` | File extension: `.dat` (default) or `.csv` |
| `--verbose` | Print detailed processing information |

### Step 2: Analysis Pipeline

Run the full Fourier transform and fitting pipeline:

```bash
python scripts/run_pipeline.py --project-name YbPdBi_AMRO --verbose
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--project-name` | (required) | Project file identifier |
| `--fourier-only` | False | Only run Fourier transform, skip fitting |
| `--fit-only` | False | Only run fitting (requires prior Fourier results) |
| `--min-amp-ratio` | 0.075 | Minimum amplitude ratio threshold for fitting |
| `--max-freq` | 8 | Maximum frequency to include in fit |
| `--force-symmetry` | True | Always include 2-fold and 4-fold symmetry terms |
| `--verbose` | False | Print detailed output |
| `--plot` | False | Generate plots |

### Interactive Analysis

For interactive exploration, use the Jupyter notebook:

```bash
jupyter notebook notebooks/01-example-analysis.ipynb
```

## Project Structure

```
AMRO Analysis/
├── src/amro/               # Source code package
│   ├── config/             # Configuration settings (paths, headers, params)
│   ├── data/               # Data loading, cleaning, and data structures
│   ├── features/           # Fourier transform analysis
│   ├── models/             # Fitting algorithms (AMROFitter)
│   ├── plotting/           # Visualization functions
│   └── utils/              # Helper functions and unit conversions
├── scripts/                # CLI entry points
│   ├── run_cleaner.py      # Data preprocessing script
│   └── run_pipeline.py     # Analysis pipeline script
├── notebooks/              # Jupyter notebooks for interactive analysis
├── tests/                  # Unit tests (pytest)
├── data/
│   ├── raw/                # Input: raw .dat files from PPMS
│   ├── processed/          # Cleaned and anti-symmetrized data
│   └── final/              # Analysis outputs (pickle, CSV)
└── figures/                # Generated plots
    ├── processed/          # Analysis figures
    └── raw/                # Raw data plots
```



## API Overview

### Main Classes

| Class | Module | Description |
|-------|--------|-------------|
| `AMROCleaner` | `amro.data` | Preprocess raw PPMS data files |
| `AMROLoader` | `amro.data` | Load cleaned data and run ETL pipeline |
| `ProjectData` | `amro.data` | Container for experiments and oscillations |
| `Fourier` | `amro.features` | Perform FFT analysis on oscillations |
| `AMROFitter` | `amro.models` | Fit oscillations with sinusoidal models |

### Data Structures

| Class | Description |
|-------|-------------|
| `Experiment` | Collection of oscillations at different T and H |
| `AMROscillation` | Single oscillation with data, Fourier, and fit results |
| `OscillationKey` | Identifier tuple (experiment_label, temperature, magnetic_field) |
| `FourierResult` | Fourier transform output (frequencies, amplitudes, phases) |
| `FitResult` | Fitting output (parameters, residuals, statistics) |

## Configuration

Configuration files are located in `src/amro/config/`:

| File | Purpose |
|------|---------|
| `paths.py` | Data directory paths |
| `headers.py` | Column header names and metadata keys |
| `cleaner.py` | Data cleaning parameters |
| `settings.py` | General settings |

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_cleaner.py

# Run with coverage (if installed)
pytest --cov=amro
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit with descriptive message
6. Push to your fork
7. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```
James Fraser
Email: james.fraser@umontreal.ca
ORCID: 0000-0003-2611-9686
```

## Acknowledgments

This project was developed as part of graduate research investigating the quantum electronic properties of materials at l'Université de Montréal under the supervision of Prof. Andrea Bianchi.
