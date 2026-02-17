# Tests

Unit tests for the AMRO analysis package using pytest.

## Running Tests

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_cleaner.py

# Run tests matching a pattern
pytest -k "test_fourier"

# Run with coverage report
pytest --cov=amro
```

## Test Files

| File | Module Tested | Description |
|------|---------------|-------------|
| `test_cleaner.py` | `amro.data.cleaner` | Data cleaning and anti-symmetrization |
| `test_loader.py` | `amro.data.loader` | Data loading and ETL pipeline |
| `test_data_structures.py` | `amro.data.data_structures` | Dataclass functionality |
| `test_fourier.py` | `amro.features.fourier` | Fourier transform analysis |
| `test_fitter.py` | `amro.models.fitter` | Sinusoidal curve fitting |
| `test_conversions.py` | `amro.utils.conversions` | Unit conversion functions |
| `test_utils.py` | `amro.utils.utils` | Utility functions |

## Configuration

- `conftest.py`: Shared pytest fixtures and test path isolation
- Tests use monkeypatched paths to avoid modifying real data directories

## Writing Tests

Tests follow these conventions:
- One test file per source module
- Fixtures defined in `conftest.py` for reusability
- Test isolation: no side effects on real data folders
