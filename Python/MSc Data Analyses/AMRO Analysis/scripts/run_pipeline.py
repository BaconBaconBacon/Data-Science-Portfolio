import argparse
from amro import AMROFitter, AMROLoader, Fourier


def parse_args():
    """Parse command line arguments for the analysis pipeline.

    Returns:
        Namespace object containing all pipeline configuration arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", required=True, help="Project file name")
    # parser.add_argument("--experiments", nargs="+", help="Experiment labels")
    parser.add_argument("--fourier-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--min-amp-ratio", type=float, default=0.075)
    parser.add_argument("--max-freq", type=int, default=8)
    parser.add_argument("--force-symmetry", action="store_true", default=True)
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def check_geometry_defaults(project_data, verbose=True):
    """Warn if any experiment has default geometry values."""
    warnings_issued = []
    for exp_label in project_data.get_experiment_labels():
        exp = project_data.experiments_dict[exp_label]
        if exp.wire_sep == 1 or exp.cross_section == 1:
            msg = (
                f"WARNING: Experiment '{exp_label}' has default geometry values "
                f"(wire_sep={exp.wire_sep}, cross_section={exp.cross_section}). "
                f"Consider using project_data.correct_geometry_scaling() to rescale"
                f"the AMRO data, then re-run this script."
            )
            warnings_issued.append(msg)
            if verbose:
                print(msg)
    return warnings_issued


def main():
    """Run the AMRO Fourier transform and fitting pipeline.

    Loads cleaned AMRO data, performs Fourier transforms on oscillations,
    fits the data using sine series, and optionally generates plots.
    """
    args = parse_args()

    loader = AMROLoader(args.project_name, verbose=args.verbose)
    project_data = loader.load_amro()

    check_geometry_defaults(project_data, verbose=True)

    if args.verbose:
        print(project_data.get_summary_statistics())
    if not args.fit_only:
        fourier = Fourier(project_data, args.project_name, verbose=args.verbose)
        fourier.fourier_transform_experiments()
        project_data.save_fourier_results_to_csv()
        if args.verbose:
            print(project_data.get_summary_statistics())
    if not args.fourier_only:
        fitter = AMROFitter(
            project_data,
            save_name=args.project_name,
            min_amp_ratio=args.min_amp_ratio,
            max_freq=args.max_freq,
            force_four_and_two_sym=args.force_symmetry,
            verbose=args.verbose,
        )
        experiments = list(project_data.get_experiment_labels())
        for exp_label in experiments:
            fitter.fit_act_experiment(exp_label)
    print(project_data.get_summary_statistics())


if __name__ == "__main__":
    main()
