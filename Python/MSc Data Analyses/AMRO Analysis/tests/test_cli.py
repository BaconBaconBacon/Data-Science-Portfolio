"""Tests for CLI scripts (run_pipeline.py and run_cleaner.py)."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from amro.config import HEADER_EXPERIMENT_PREFIX
from amro.data import Experiment, ProjectData


# =============================================================================
# Import helpers — scripts live outside the package, so import their functions
# =============================================================================

sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
from run_pipeline import parse_args as pipeline_parse_args, check_geometry_defaults
from run_cleaner import parse_args as cleaner_parse_args


# =============================================================================
# run_cleaner: Argument Parsing Tests
# =============================================================================


class TestCleanerParseArgs:
    """Tests for run_cleaner.py argument parsing."""

    def test_defaults(self):
        with patch("sys.argv", ["run_cleaner.py"]):
            args = cleaner_parse_args()
        assert args.datafile_type == ".dat"
        assert args.verbose is False

    def test_datafile_type_csv(self):
        with patch("sys.argv", ["run_cleaner.py", "--datafile-type", ".csv"]):
            args = cleaner_parse_args()
        assert args.datafile_type == ".csv"

    def test_datafile_type_invalid(self):
        with patch("sys.argv", ["run_cleaner.py", "--datafile-type", ".xlsx"]):
            with pytest.raises(SystemExit):
                cleaner_parse_args()

    def test_verbose_flag(self):
        with patch("sys.argv", ["run_cleaner.py", "--verbose"]):
            args = cleaner_parse_args()
        assert args.verbose is True


# =============================================================================
# run_pipeline: Argument Parsing Tests
# =============================================================================


class TestPipelineParseArgs:
    """Tests for run_pipeline.py argument parsing."""

    def test_project_name_required(self):
        with patch("sys.argv", ["run_pipeline.py"]):
            with pytest.raises(SystemExit):
                pipeline_parse_args()

    def test_minimal_args(self):
        with patch("sys.argv", ["run_pipeline.py", "--project-name", "my_project"]):
            args = pipeline_parse_args()
        assert args.project_name == "my_project"

    def test_defaults(self):
        with patch("sys.argv", ["run_pipeline.py", "--project-name", "test"]):
            args = pipeline_parse_args()
        assert args.fourier_only is False
        assert args.fit_only is False
        assert args.min_amp_ratio == 0.075
        assert args.max_freq == 8
        assert args.force_symmetry is True
        assert args.save_name is None
        assert args.verbose is False
        assert args.plot is False

    def test_fourier_only(self):
        with patch("sys.argv", ["run_pipeline.py", "--project-name", "test", "--fourier-only"]):
            args = pipeline_parse_args()
        assert args.fourier_only is True

    def test_fit_only(self):
        with patch("sys.argv", ["run_pipeline.py", "--project-name", "test", "--fit-only"]):
            args = pipeline_parse_args()
        assert args.fit_only is True

    def test_no_force_symmetry(self):
        with patch("sys.argv", ["run_pipeline.py", "--project-name", "test", "--no-force-symmetry"]):
            args = pipeline_parse_args()
        assert args.force_symmetry is False

    def test_force_symmetry_default_true(self):
        """Ensure symmetry is forced by default (user must opt out)."""
        with patch("sys.argv", ["run_pipeline.py", "--project-name", "test"]):
            args = pipeline_parse_args()
        assert args.force_symmetry is True

    def test_custom_amp_ratio_and_freq(self):
        with patch("sys.argv", [
            "run_pipeline.py", "--project-name", "test",
            "--min-amp-ratio", "0.1", "--max-freq", "12",
        ]):
            args = pipeline_parse_args()
        assert args.min_amp_ratio == 0.1
        assert args.max_freq == 12

    def test_all_flags(self):
        with patch("sys.argv", [
            "run_pipeline.py", "--project-name", "test",
            "--fourier-only", "--verbose", "--plot",
            "--save-name", "output",
        ]):
            args = pipeline_parse_args()
        assert args.fourier_only is True
        assert args.verbose is True
        assert args.plot is True
        assert args.save_name == "output"


# =============================================================================
# run_pipeline: check_geometry_defaults Tests
# =============================================================================


class TestCheckGeometryDefaults:
    """Tests for the geometry default-value warning in run_pipeline.py."""

    def test_no_warning_with_non_default_values(self):
        project = ProjectData(project_name="test")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=0.8,
            cross_section=0.05,
        )
        project.add_experiment(exp)

        warnings = check_geometry_defaults(project, verbose=False)
        assert warnings == []

    def test_warning_on_default_wire_sep(self):
        project = ProjectData(project_name="test")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=1,
            cross_section=0.05,
        )
        project.add_experiment(exp)

        warnings = check_geometry_defaults(project, verbose=False)
        assert len(warnings) == 1
        assert "default geometry" in warnings[0]

    def test_warning_on_default_cross_section(self):
        project = ProjectData(project_name="test")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=0.8,
            cross_section=1,
        )
        project.add_experiment(exp)

        warnings = check_geometry_defaults(project, verbose=False)
        assert len(warnings) == 1

    def test_warning_prints_when_verbose(self, capsys):
        project = ProjectData(project_name="test")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=1,
            cross_section=1,
        )
        project.add_experiment(exp)

        check_geometry_defaults(project, verbose=True)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_no_warning_prints_when_not_verbose(self, capsys):
        project = ProjectData(project_name="test")
        exp = Experiment(
            experiment_label=HEADER_EXPERIMENT_PREFIX + "11",
            geometry="perp",
            wire_sep=1,
            cross_section=1,
        )
        project.add_experiment(exp)

        check_geometry_defaults(project, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_multiple_experiments_multiple_warnings(self):
        project = ProjectData(project_name="test")
        for i in range(3):
            exp = Experiment(
                experiment_label=HEADER_EXPERIMENT_PREFIX + str(i),
                geometry="perp",
                wire_sep=1,
                cross_section=1,
            )
            project.add_experiment(exp)

        warnings = check_geometry_defaults(project, verbose=False)
        assert len(warnings) == 3

    def test_empty_project_no_warnings(self):
        project = ProjectData(project_name="test")
        warnings = check_geometry_defaults(project, verbose=False)
        assert warnings == []