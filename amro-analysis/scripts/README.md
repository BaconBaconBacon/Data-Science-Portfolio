# Scripts

Command-line scripts for batch processing AMRO data.

## Available Scripts

### `run_cleaner.py`

Preprocess raw PPMS data files: extract metadata, remove outliers, and anti-symmetrize measurements.

```bash
python scripts/run_cleaner.py --datafile-type .dat --verbose
```

**Options:**
- `--datafile-type`: Input file extension (`.dat` or `.csv`, default: `.dat`)
- `--verbose`: Print detailed processing information

### `run_pipeline.py`

Run the complete Fourier transform and fitting analysis pipeline.

```bash
python scripts/run_pipeline.py --project-name YbPdBi_amro --verbose
```

**Options:**
- `--project-name`: Project/data identifier (required)
- `--fourier-only`: Only run Fourier analysis
- `--fit-only`: Only run fitting (requires prior Fourier results)
- `--min-amp-ratio`: Amplitude threshold for fitting (default: 0.075)
- `--max-freq`: Maximum frequency to fit (default: 8)
- `--force-symmetry`: Include 2-fold and 4-fold terms (default: True)
- `--verbose`: Print detailed output
- `--plot`: Generate plots

## Workflow

1. Place raw `.dat` files in `data/raw/`
2. Run `run_cleaner.py` to preprocess data
3. Run `run_pipeline.py` to analyze cleaned data

See the main [README](../README.MD) for complete documentation.
