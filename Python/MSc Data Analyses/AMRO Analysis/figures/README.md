# Figures

Generated plots and visualizations from AMRO analysis.

NOTE: The plotting functionality is limited, and can be expanded in the future should there be interest. 

## Directory Structure

### `raw/`

Plots of raw or minimally processed data for quality checking.

- Quick visualization of loaded AMRO data
- Useful for identifying data issues before analysis

### `processed/`

Publication-quality figures from analysis results.

- Fit plots with residuals
- Fourier component bar charts
- Multi-panel figures organized by temperature and magnetic field

## Generated Figures

Figures are created by:
- `AMROFitter.plot_fits_with_residuals()` - Fit quality visualization
- `AMROFitter.plot_fits_with_residuals_uohm()` - Same, with micro-ohm units
- `Fourier.plot_n_strongest()` - Strongest symmetry components
- `AMROLoader.quick_plot_amro()` - Raw data overview

## Naming Convention

```
{experiment_label}_figure_{type}_ratio_{min_amp_ratio}_maxf_{max_freq}.pdf
```

Example: `ACTRot11_figure_amro_fits_ratio_0.075_maxf_8.pdf`

## Notes

- Figures are excluded from version control via `.gitignore`
- Regenerate by running analysis scripts with `--plot` flag or notebook cells
