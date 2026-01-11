import argparse
from amro import AMROFitter, AMROLoader, Fourier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", required=True, help="Data file name")
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


def main():
    args = parse_args()

    loader = AMROLoader(args.data_name, verbose=args.verbose)
    project_data = loader.load_amro()
    if args.verbose:
        print(project_data.get_summary_statistics())
    if not args.fit_only:
        fourier = Fourier(project_data, args.data_name, verbose=args.verbose)
        fourier.fourier_transform_experiments()
        project_data.save_fourier_results_to_csv()
        if args.verbose:
            print(project_data.get_summary_statistics())
    if not args.fourier_only:
        fitter = AMROFitter(
            project_data,
            save_name=args.data_name,
            min_amp_ratio=args.min_amp_ratio,
            max_freq=args.max_freq,
            force_four_and_two_sym=args.force_symmetry,
            verbose=args.verbose,
        )
        experiments = list(project_data.get_experiment_labels())
        for exp_label in experiments:
            fitter.fit_act_experiment(exp_label)
    print(len(fitter.failed_fits))
    print(project_data.get_summary_statistics())


if __name__ == "__main__":
    main()
