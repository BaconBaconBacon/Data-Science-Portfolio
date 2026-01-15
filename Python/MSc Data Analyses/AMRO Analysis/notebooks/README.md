# Notebooks

Jupyter notebooks for interactive AMRO data analysis and exploration.

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| `01-load-fourier-and-fit.ipynb` | Complete analysis workflow: load cleaned data, perform Fourier transforms, fit oscillations, and visualize results |

## Usage

```bash
# Start Jupyter from the project root
jupyter notebook notebooks/

# Or open a specific notebook
jupyter notebook notebooks/01-load-fourier-and-fit.ipynb
```

## Prerequisites

Before running notebooks:

1. Install the package: `pip install -e .`
2. Ensure cleaned data exists in `data/processed/`
   - Run `python scripts/run_cleaner.py` if needed

## Notes

- Notebooks use the `amro` package for all analysis functions
- Results are saved to `data/final/` and `figures/processed/`
- Cell outputs are preserved for reference; re-run cells to update with your data
