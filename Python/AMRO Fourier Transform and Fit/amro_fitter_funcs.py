import lmfit as lm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


class FitAMRO:
    # TODO: Consider removing this functionality.
    FIT_SYMMETRIES = [2, 4, 6, 8]  # [2,4, 8]  # [2, 4, 6]  #

    def __init__(self, amro, fourier_results: pd.DataFrame, save_name: str) -> None:
        self.ft_results_df = fourier_results
        self.amro_df = amro.AMRO
        self.save_name = save_name
        self.save_path = os.path.join("Data", save_name)
        self.act_choices = self._get_H_T_data()
        return

    def SineBuilder(
        self, rads, amps: np.array, freqs: np.array, phases: np.array, mean: float | int
    ):
        summation = np.sum(
            amps[:, None] * np.sin(freqs[:, None] * rads + phases[:, None]), axis=0
        )
        result = mean * summation + mean

        return result

    def ObjFcn(self, params, deg, res_data):
        """
        The sinebuilder is fitted by minimizing this objective function.
        """
        amps_list = []
        freqs_list = []
        phase_list = []

        for key in params.keys():
            if "amp" in key:
                amps_list.append(params[key].value)
            elif "freq" in key:
                freqs_list.append(params[key].value)
            elif "phase" in key:
                phase_list.append(params[key].value)

        offset = params["mean"].value
        amps_list = np.asarray(amps_list)
        freqs_list = np.asarray(freqs_list)
        phase_list = np.asarray(phase_list)

        res_model = self.SineBuilder(deg, amps_list, freqs_list, phase_list, offset)

        # Want to minimize least squares
        return (res_model - res_data) ** 2

    def FitAMROData(
        self, f_list: list, ACT: str, H: int | float, T: int | float
    ) -> lm.minimizer.MinimizerResult:
        """
        TODO: Refactor and encapsulate as much as possible.
        """

        # Select the experimental data to be fitted
        q = 'ACT_str == "{}" & H == {} & T== {}'.format(ACT, H, T)
        fit_df = self.amro_df.query(q)
        q = 'ACT_str == "{}" & H == {} & T== {} & `freqs (cycles/rot)` in @f_list'.format(
            ACT, H, T
        )
        guess_df = self.ft_results_df.query(q)

        # # Query initial values from FT_guesses using frequencies list
        # guess_df = guess_df.query('')

        # Extract data we are going to fit
        x = fit_df["Sample Position (rads)"].values
        y = fit_df["Res. (ohm-cm)"].values
        y_mean = y.mean()

        initial_params = self._initialize_parameters(f_list, guess_df, y_mean)

        # Perform the minimization
        minner = lm.Minimizer(self.ObjFcn, initial_params, fcn_args=(x, y))
        # kws = {'options': {'maxiter':5000}}

        results = minner.minimize()
        print(lm.fit_report(results))

        return results

    def _initialize_parameters(
        self, freqs_list: list, guesses: pd.DataFrame, mean: float
    ) -> lm.Parameters:
        # Generate a Parameters ordered dictionary, to which we add Parameter objects
        initial_p_guesses = lm.Parameters()
        initial_p_guesses.add("mean", value=mean)

        # Append all Parameter objects, except for the last one (must deal with appended 2)
        i = 0
        while i < (len(freqs_list) - 1):  # Extra 2 will always be at the end of f_list
            freq = int(freqs_list[i])
            self._add_freq_guess(freq, initial_p_guesses, mean, guesses)
            i += 1

        # Deal with the final element
        # TODO: Surely this is unnecessary?
        if len(freqs_list) == len(guesses):
            print("EQUAL")

            # If nothing has been appended, just add the information for the final FT guess
            freq = int(freqs_list[i])
            self._add_freq_guess(freq, initial_p_guesses, mean, guesses)

        return initial_p_guesses

    def _add_freq_guess(
        self,
        frequency: int,
        params: lm.Parameters,
        mean_val: float,
        guesses_df: pd.DataFrame,
    ) -> None:
        """
        Forcing all amplitudes to be positive, negative values show up as
        a pi-sized phase offset.
        """
        temp_df = guesses_df.query("`freqs (cycles/rot)` == {}".format(frequency))
        params.add(
            "amp" + str(frequency),
            value=temp_df["mag (ohm-cm)"].values[0] / mean_val,
            min=0,
        )

        params.add(
            "freq" + str(frequency),
            value=temp_df["freqs (cycles/rot)"].values[0],
            vary=False,
        )

        params.add(
            "phase" + str(frequency),
            value=temp_df["phase"].values[0],
            min=-2 * np.pi,
            max=2 * np.pi,
        )
        return

    def FitACTExperiment(
        self, label, f_rank_min, f_ratio_min_ratio, f_max, show_fig=False
    ):
        all_fits_df = pd.DataFrame()
        all_results_dict = {}
        # fitted_amps = pd.DataFrame()
        i = 0
        for T_label, H_label in self.act_choices[label]:
            print("Fitting {}, {}K, {}T.".format(label, T_label, H_label))
            # Maybe select the frequencies based on their rank and ratio wrt strongest frequency
            q = 'ACT_str == "{}" & H == {} & T == {} & `freqs (cycles/rot)`in @self.FIT_SYMMETRIES'.format(
                label, H_label, T_label
            )

            f_info = self.ft_results_df.query(q)[
                ["freqs (cycles/rot)", "amp_ratio"]
            ]  # 'rank', 'amp_ratio']]
            f = f_info["freqs (cycles/rot)"].values
            print("F LIST:", f)
            if 2 not in f:
                print("{}, T = {}, H = {}".format(label, T_label, H_label))
                print("2 not in f.")

            if 4 not in f:
                print("{}, T = {}, H = {}".format(label, T_label, H_label))
                print("4 not in f.")

            results_obj = self.FitAMROData(f, label, H_label, T_label)

            # Pack results to add to a larger dataframe
            var_names = results_obj.var_names
            param_results = results_obj.params

            # Store fitted values in a dictionary, which will be turned into a dataframe and concatenated to ACT's data
            results_dict = {}

            f_info["act"] = label
            f_info["T (K)"] = T_label
            f_info["H (T)"] = H_label
            f_info["f_list"] = f

            if i == 0:
                fitted_amps = f_info
                i += 1
            else:
                fitted_amps = pd.concat([fitted_amps, f_info])
            # Add variables to
            for var in var_names:
                results_dict[var] = param_results[var].value
                results_dict[var + " err"] = param_results[var].stderr
                # i+=1

            results_dict["ACT"] = label
            results_dict["H"] = H_label
            results_dict["T"] = T_label
            results_dict["chi squared"] = results_obj.redchi

            # Add to the larger dataframe
            results_df = pd.DataFrame(results_dict, index=[0])
            all_fits_df = pd.concat([all_fits_df, results_df], ignore_index=True)
            all_results_dict[
                label + "T" + str(T_label) + "H" + str(H_label)
            ] = results_obj

        # replace all NaNs as zeros, assuming the problem was a mismatch between requested frequencies and FT guesses for the given experiment
        all_fits_df = all_fits_df.fillna(0)

        return all_fits_df, all_results_dict, fitted_amps

    def _get_H_T_data(self):
        H_T_dict = {}
        grouped = self.amro_df[
            ["ACT_str", "H", "T"]
        ].drop_duplicates()  # .groupby(, as_index=False)

        for act in grouped["ACT_str"].unique():
            tmp_list = []
            for _, row in grouped.query('ACT_str=="{}"'.format(act)).iterrows():
                tmp_list.append((row["T"], row["H"]))
            H_T_dict[act] = tmp_list
        return H_T_dict

    def _test_plot_sinebuilder(self):
        # Test function
        f = [4, 2]
        amp = [2, 1]
        phase = [0, 0]
        offset = 1

        x = np.linspace(0, 2 * np.pi, 1000)
        y = self.SineBuilder(x, amp, f, phase, offset)

        plt.scatter(x, y)
        return
