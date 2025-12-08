import numpy as np
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.fft import fft, fftfreq, rfft, rfftfreq

H_PALETTE = {
	0.5: 'tab:red',
	3: 'tab:green',
	7: 'tab:orange',
	9: 'tab:blue'
}

class LoadAMRO:
	'''
		Here we load the pre-cleaned and symmetrized data into a single DataFrame. 
		We have already checked the data for NaNs, and handled them when they appeared. 

		As this is only a demonstration, we are only using a subset of the total 
		available data in order to simplify the output. 

		Nonetheless, the code is capable of handling all of the experimental data.

		We extract experimental information about temperature ($T$) and magnetic field
		strength ($H$) from the filenames, but we must account for an inconsistent 
		naming scheme.

		The 'geo' label indicates the experimental geometry that was used. In 'para', 
		the rotation of the sample brings the electrical current vector parallel with the 
		magnetic field at 90deg. For the 'perp' geometry, the current vector is held 
		orthogonal to the magnetic field for the entire rotation of the sample.

		TODO: Add the cleaning and symmetrization functionality into the ETL pipeline.
	'''
	
	META_DATA ={
		'ACTRot11':{'geo':'perp', 'ACT':11},
		'ACTRot12':{'geo':'para', 'ACT':12},
		'ACTRot13':{'geo':'para', 'ACT':13},
		'ACTRot14':{'geo':'perp', 'ACT':14}
	}
	DESIRED_COLS = [
		'Temperature (K)', 'Sample Position (deg)', #'Magnetic Field (Oe)',
		'Res. (ohm-cm)', 'ACT', 'ACT_str', 'T', 'H','geo' #, 'L (cm)', 'W (cm)', 'H (cm)'
	]
	COL_RENAMES = {
		'Res. ch2 (ohm-cm)':'Res. (ohm-cm)'
	}

	def __init__(self, file_name:str, save_folder:str):

		self.file_name = file_name
		self.save_folder = save_folder
		self.file_path = os.path.join(save_folder, file_name)
		self.AMRO = pd.DataFrame()


	def getAMRO(self)->pd.DataFrame:
		'''
		'''
		if self.file_name.endswith('.csv'):
			if os.path.exists(self.file_path):
				print('Loading : {}'.format(self.file_name))
				self.AMRO = pd.read_csv(self.file_path)
				# GEt META DATA
				
				return self.AMRO
				
			elif self.file_name.endswith('.csv') and not os.path.exists(self.file_path):
				print('Combining AMRO files into: {}'.format(self.file_name))
				self.AMRO = self.combineAMRO(self.save_folder)
				# self.AMRO.meta_data = self.META_DATA
				return self.AMRO
		else:
			raise TypeError("Wrong file type: {}".format(self.file_name))



	def combineAMRO(self, data_dir:str)->pd.DataFrame:
		'''
		'''

		# Get names of files in the folder
		filenames = os.listdir(data_dir)
		amro_df = pd.DataFrame()

		for filename in filenames:
			# Ensure we are selecting only AMRO data
			if any(key in filename for key in self.META_DATA.keys()) and 'AMRO' in filename:
				### EXTRACT
				fp = os.path.join(data_dir, filename)
				temp_df = self._extract(fp)
				
				### TRANSFORM
				temp_df = self._transform(temp_df)

				# LOAD
				amro_df = pd.concat([amro_df, temp_df], ignore_index=True)
				amro_df.to_csv(self.file_path, sep=',')
		print('Done combining AMRO files.')
		return amro_df

	def _extract(self,file_path:str)->pd.DataFrame:
		'''
		'''
		temp_df = pd.read_csv(file_path, sep=',')
				
		# Extract experimental info from file_path
		fn = file_path.split(os.path.sep)[-1]
		temp_name = fn.replace('.csv','').replace('0_5','0.5').replace('1p9','1.9')
		conds = lambda x: ('ACT' in x or x.endswith('T') or x.endswith('K'))
		for label in [a for a in temp_name.split('_') if conds(a)]:
			if label.endswith('K'):
				T_label = float(label.replace('K',''))
				temp_df['T'] = T_label
			elif label.endswith('T'):
				H_label = float(label.replace('T',''))
				temp_df['H'] = H_label
			elif 'ACT' in label:
				act_label = label
				# print(act_label)
				# raise
				temp_df['ACT_str']= act_label
				temp_df['geo'] = self.META_DATA[label]['geo']
				temp_df['ACT'] = self.META_DATA[label]['ACT']
				
				# Update meta data 
				if ('L (cm)' not in self.META_DATA[label].keys()):
					self.META_DATA[label]['L (cm)'] = temp_df['L (cm)'].values[0]
					self.META_DATA[label]['W (cm)'] = temp_df['W (cm)'].values[0]
					self.META_DATA[label]['H (cm)'] = temp_df['H (cm)'].values[0]
		  
			else:
				print("Filename parsing error, fix filename for:\t"+filename)
				raise ValueError

		# Create additional meta data dictionaries
		self.META_DATA[act_label]['T_vals'] = []
		self.META_DATA[act_label]['H_vals'] = []
		self.META_DATA[act_label][H_label] = {}
		self.META_DATA[act_label][H_label][T_label] = {'res. units':{}}
		# print('bob', self.META_DATA)
		
		# Select desired columns, rename as needed
		temp_df = temp_df.rename(columns=self.COL_RENAMES)[self.DESIRED_COLS]
		return temp_df

	def _transform(self, temp_df:pd.DataFrame)->pd.DataFrame:
		'''
		'''

		self._genMetaData(temp_df)
		temp_df = self._createAltResistanceUnits(temp_df)
		temp_df['Sample Position (rads)'] = temp_df['Sample Position (deg)']*2*np.pi/360
		return temp_df

	def _genMetaData(self, df:pd.DataFrame)->None:

		# print(self.META_DATA)
		# raise
		act_label='ACTRot' + str(df['ACT'].values[0])
		H_label=df['H'].values[0]
		T_label=df['T'].values[0]
		# print(act_label, H_label, T_label)

		# Calc for additional columns as needed
		mean_res = df['Res. (ohm-cm)'].mean()
		zero_deg_res= df.loc[df['Sample Position (deg)'].idxmin(), 'Res. (ohm-cm)']
		
		# Store additional meta data
		# print(self.META_DATA[act_label])
		# print(self.META_DATA[act_label][H_label])
		# print(self.META_DATA[act_label][H_label][T_label])
		this_meta_data = self.META_DATA[act_label][H_label][T_label]
		# print(this_meta_data)
		this_meta_data['res. units']['Mean res (ohm-cm)'] = mean_res
		this_meta_data['res. units']['0deg res (ohm-cm)'] = zero_deg_res
		# print(self.META_DATA)
		return


	def _createAltResistanceUnits(self, df:pd.DataFrame)->pd.DataFrame:
		'''
			Calculates alternative resistivity units based on the new meta data
		'''
		act_label='ACTRot' + str(df['ACT'].values[0])
		H_label=df['H'].values[0]
		T_label=df['T'].values[0]

		res_meta_data = self.META_DATA[act_label][H_label][T_label]['res. units']

		for key in res_meta_data:
			label = key.split(' ')[0]
			new_labels = [
				'Delta Res. {} (ohm-cm)'.format(label),
				'Delta Res./R0 {} (ohm-cm)'.format(label),
				'Delta Res./R0 {} (%)'.format(label)
			]
			df[new_labels[0]] =  df['Res. (ohm-cm)'] - res_meta_data[key]
			df[new_labels[1]] =  df[new_labels[0]] / res_meta_data[key]
			df[new_labels[2]] =  df[new_labels[1]] * 100

		# uohms
		for col in df.columns:
			if 'Res' in col:
				new_col = col.replace('ohm','uohm')
				df[new_col] = df[col]*10**6

		return df


	def QuickPlotAMRO(self)->None:
		_ = sns.relplot(x='Sample Position (rads)', 
						y='Delta Res./R0 Mean (ohm-cm)', 
						hue='H',
						col='T',
						# hue='ACT',
						row='ACT',
						palette = H_PALETTE,
						facet_kws={'sharey':False},
						data = self.AMRO)
		return


class Fourier:

	def __init__(self, amro:LoadAMRO, save_name :str, save_dir:str):

		# Get the AMRO
		self.amro_data = amro.AMRO
		# self.meta_data = amro.META_DATA
		self.labels = self.amro_data[['ACT', 'T', 'H']].drop_duplicates()

		# Iterate through DataFrame entries, appending results to a new DataFrame
		self.FT_results_df = pd.DataFrame()
		self.save_name = save_name
		self.save_dir = save_dir
		self.save_fp = os.path.join(save_dir, save_name)

		if os.path.exists(self.save_fp):
			# TODO: Need to check and make sure it's loading the same data as the AMRO
			print("loading {}".format(save_name))
			self.FT_results_df = pd.read_csv(self.save_fp)

			return
		else:
			for act_label in self.amro_data['ACT_str'].unique():
				print("FT'ing: " + act_label)
				
				act_df = self.amro_data.query('ACT_str=="{}"'.format(act_label))
				t_vals = act_df['T'].unique()
				h_vals = act_df['H'].unique()
				geo_label = act_df['geo'].unique()[0]
				
				for t, h in itertools.product(t_vals, h_vals):
				
					# Query the correct dataframe using the experiment labels
					ft_df = self.amro_data.query('ACT_str=="{}" & T =={} & H == {}'.format(act_label, t, h))  # 'ACT_str=="{}"'.format(act_label))  # 
					
					freq_df = self._fourier_transform(ft_df)

					freq_df['ACT_str'] = act_label 
					freq_df['ACT'] = float(act_label.replace("ACTRot",""))
					freq_df['T'] = t 
					freq_df['H'] = h 
					freq_df['geo'] = geo_label#[0]

					self.FT_results_df = pd.concat([self.FT_results_df, freq_df], ignore_index=True)#.reset_index(drop=True)
			# except KeyError as e:
			# 	print(e)
			# 	print('No data for: '+act_label+'. Skipping...')
				
			# Save the results of the FT
			self.FT_results_df.to_csv(self.save_fp, sep=',',index=False)
			print('Results saved to: {}'.format(self.save_name))
		return


	def GetNStrongest(self, n:int):
		'''
			Queries the n strongest contributions for each experiment in the data set.
		''' 
		# self.get_n_strongest()
		# strongest_df = self.FT_results_df.query("`freqs (cycles/rot)`<{}".format(max_sym))
		# # freq_df = freq_df.reset_index(drop=True)
		# print(self.FT_results_df.head())#.columns)
		strongest_df = self.FT_results_df.sort_values(by=['ACT','H','T', 'mag (ohm-cm)'], ascending=False)
		strongest_freqs = strongest_df.groupby(['ACT','H','T']).head(n)
		# .sort_values(by='mag (ohm-cm)', ascending=False)
		# strongest_freqs =  strongest_df.head(n)

		return strongest_freqs.reset_index(drop=True)


	def PlotNStrongest(self, n:int, T:list|float, H:list|float)->None:
		'''
		Plots the n-strongest. 

		If n=0, then plots all available contributions.
		'''
		if isinstance(T, list):
			q = 'T in {}'.format(T)  
		else: 
			q = 'T == {}'.format(T)  

		if isinstance(H, list):
			q += ' & H in {}'.format(H)  
		else: 
			q += ' & H == {}'.format(H)  


		plot_df = self.GetNStrongest(n).query(q)
		# Bypass a formatting bug in catplot
		hue_choice = 'H'
		plot_df = plot_df.sort_values(hue_choice)
		plot_df[hue_choice] = plot_df[hue_choice].astype(str)
		sns.set_context('poster')
		g = sns.catplot(
		    x='freqs (cycles/rot)',
		    y='amp_ratio',
		    data=plot_df,
		    col='T',
		    row='ACT',    
		    kind='bar',
		    hue=hue_choice,

		)
		g.set(xlim=(0.1,None))
		return


	def _fourier_transform(self, df:pd.DataFrame)->pd.DataFrame:
		fftdata = df['Delta Res. Mean (ohm-cm)'].values

		# Perform the FFT, where yf is the amplitudes and xf are the frequencies
		yf = rfft(fftdata, n= len(fftdata), norm='ortho')
		xf = rfftfreq(len(fftdata), 1/len(fftdata))

		# Package the results
		freq_df = pd.DataFrame({'freqs (cycles/rot)':xf,
								'amps':yf,
								'mag (ohm-cm)':np.abs(yf),
								'phase':np.angle(yf)
							   })

		# Amplitudes relative to the strongest
		freq_df['amp_ratio'] = freq_df['mag (ohm-cm)']/freq_df['mag (ohm-cm)'].max()
		freq_df['freqs (cycles/rot)'] = freq_df['freqs (cycles/rot)'].astype(int)

		# Force positive phase values
		freq_df['phase_raw'] = freq_df['phase'].copy()
		freq_df['phase'] = np.select(
			freq_df['phase_raw']<0,
			freq_df['phase_raw']+2*np.pi,
			freq_df['phase_raw']
		)
		return freq_df

class FitAMRO:

	def __init__(self):


		return

	def SineBuilder(rads, amps_list : list, freq_list : list, phase: list, mean):
	    # Want this to be as fast as possible,
	    # because it will be called a lot during the curve_fit regression
	    # Make these amplitudes be as a fraction of the mean
	    summation = 0
	    for amp, f, p in zip(amps_list, freq_list, phase):
	        # res += amp*np.cos(f*rads+ p)# cos(rads, amp, f, phase)
	        summation += amp*np.sin(f*rads+ p)# cos(rads, amp, f, phase)
	        # res += amp*np.sin(rads+ p)**f# cos(rads, amp, f, phase)
	    return mean * summation + mean


    def _test_plot_sinebuilder(self):
		# Test function
		f = [4, 2]
		amp = [2, 1]
		phase = [0,0]
		offset = 1 

		x = np.linspace(0,2*np.pi, 1000)
		y = SineBuilder(x, amp, f, phase, offset)

		print(max(y))
		plt.scatter(x, y)
    	return

	def ObjFcn(params, deg, res_data):
	    
	    amps_list = []
	    freqs_list = []
	    phase_list = []
	    
	    for key in params.keys():
	        if 'amp' in key:
	            amps_list.append(params[key].value)
	        elif 'freq' in key:
	            freqs_list.append(params[key].value)
	        elif 'phase' in key:
	            phase_list.append(params[key].value)

	    offset = params['mean']
	    
	    res_model = CosineBuilder(deg, amps_list, freqs_list, phase_list, offset)

	    # Want to minimize least squares
	    return (res_model-res_data)**2



	def FitAMROData(data_df, guesses_df, f_list, ACT, H, T,
	                print_results = True, plot_results = True, savefig=False, plot_residuals = False):
	    # raise
	    sns.set_context("paper")# rc={"axes.labelsize":20}
	    # Select the experimental data to be fitted
	    fit_df = data_df.query('ACT == "{}" & H == {} & T== {}'.format(ACT, H, T))
	    guess_df = guesses_df.query('ACT == "{}" & H == {} & T== {}'.format(ACT, H, T))
	    
	    
	    # Query initial values from FT_guesses using frequencies list
	    freq_query = ""
	    for f in f_list:
	        freq_query += '`freqs (cycles/rot)` == {} '.format(f)
	    freq_query = freq_query.replace(' `', '|`')  # Add OR operators between query terms
	    guess_df = guess_df.query(freq_query)
	    
	    # raise
	    # Extract data we are going to fit
	    x = fit_df['Sample Position (rads)']
	    y = fit_df['Res. ch2 (ohm-cm)']

	    # Generate a Parameters ordered dictionary, to which we add Parameter objects
	    initial_p_guesses = lm.Parameters()  

	    # Calculate mean with which we can prepare the amplitudes for the cosine builder function
	    y_mean = y.mean()
	    
	    # Append all Parameter objects, except for the last one (must deal with appended 2)
	    i=0
	    while i < (len(f_list)-1):  # Extra 2 will always be at the end of f_list
	        freq = int(f_list[i])
	        temp_df = guess_df.query('`freqs (cycles/rot)` == {}'.format(freq))
	        initial_p_guesses.add('amp'+str(freq),
	                             value = temp_df['mag (ohm-cm)'].values[0]/y_mean,
	                            min=0)  # Forcing all amplitudes to be positive, negative values show up as a pi-large phase offset. absolute value

	        initial_p_guesses.add('freq'+str(freq),
	                            value = temp_df['freqs (cycles/rot)'].values[0],
	                            vary=False)

	        initial_p_guesses.add('phase'+str(freq),
	                              value = temp_df['phase'].values[0],
	                              min = -2*np.pi,
	                              max=2*np.pi)
	        i+=1
	    
	    # Deal with the final element
	    if len(f_list)==len(guess_df): 
	        # print('EQUAL')
	        
	        # If nothing has been appended, just add the information for the final FT guess
	        freq = int(f_list[i])
	        temp_df = guess_df.query('`freqs (cycles/rot)` == {}'.format(freq))
	        # print(guess_df)
	        #amp_frac = 0.1  # Fraction of strongest FT guess amplitude
	        
	        # Create last Parameter object
	        initial_p_guesses.add('amp'+str(freq),
	                            value = temp_df['mag (ohm-cm)'].values[0]/y_mean,
	                            min=0)  # Forcing all amplitudes to be positive, negative values show up as a pi-large phase offset. absolute value

	        initial_p_guesses.add('freq'+str(freq),
	                            value = temp_df['freqs (cycles/rot)'].values[0],
	                            vary=False)

	        initial_p_guesses.add('phase'+str(freq),
	                            value = temp_df['phase'].values[0],
	                            min =  -2*np.pi, 
	                              max=2*np.pi) 
	    else:
	        raise print('UNEQUAL')
	#         print('UNEQUAL')
	        # print(f_list)

	        # Deal with the appended 2 and/or 4
	        num_appended_f = len(f_list) - len(guess_df)

	        appended_freqs = f_list[-num_appended_f:]     
	        print('appended freqs:', appended_freqs)
	        
	        # We'll assume that if 2 wasn't detected by the FT transform,then we can
	        # definitely/maybe/hopefully use the initial guesses for the strongest frequencies
	        freq = 2
	        temp_df = guess_df.query('amp_ratio == 1')#.format())
	        amp_frac = 0.3  # Fraction of strongest FT guess amplitude
	        
	        for f in appended_freqs:
	            # Create last Parameter object
	            initial_p_guesses.add('amp'+str(int(f)),
	                                value=temp_df['mag (ohm-cm)'].values[0]/y_mean*amp_frac,
	                                min=0)  # Forcing all amplitudes to be positive, negative values show up as a pi-large phase offset. absolute value

	            initial_p_guesses.add('freq'+str(int(f)),
	                                value=temp_df['freqs (cycles/rot)'].values[0],
	                                vary=False)

	            initial_p_guesses.add('phase'+str(int(f)),
	                                value=temp_df['phase'].values[0],
	                                min=-2*np.pi, 
	                                  max=2*np.pi) 


	    #print(initial_p_guesses.pretty_print())
	    initial_p_guesses.add('mean', value = y_mean)
	    
	#     x = x[~x.isna()]
	#     y = y[~y.isna()]
	    # Perform the minimization
	    minner = lm.Minimizer(ObjFcn, initial_p_guesses, fcn_args=(x, y))
	    kws  = {'options': {'maxiter':5000}}
	    
	    #try:
	    results = minner.minimize()
	    # except ValueError as e:
	    #     print('Value Error')
	    #     print('x', x[x.isna()])
	    #     print('y', y[x.isna()])
	    if print_results : print(lm.fit_report(results))
	    
	#     if plot_residuals:
	        
	#         fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
	#         PlotFitOverData(fit_df, results.params, title='Best Fit Results', ax=ax1)
	#         ax2.scatter(x, results.residual)
	#         #ax2.set_yscale('log')
	#         ax2.axhline(0)
	        
	#         # Formatting
	#         fig.set_size_inches(10,10)
	#         title = '{} Best Fit, H={}T, T={}K, f\'s='.format(ACT, H, T)#
	#         for f in f_list:
	#             title +=  str(int(f)) + ', '
	#         title = title[:-2]

	#         fig.suptitle(title, fontsize=30)  
	#         plt.tight_layout()
	        
	#     elif not plot_residuals:
	#         # Plot fitted parameters next to guessed parameters
	#         fig, (ax1, ax2) = plt.subplots(1,2)
	#         PlotFitOverData(fit_df, initial_p_guesses, title='Initial Guesses', ax=ax1)
	#         PlotFitOverData(fit_df, results.params, title='Best Fit Results', ax=ax2)

	#         # Formatting
	#         fig.set_size_inches(15,5)
	#         title = '{} Best Fit, H={}T, T={}K, f\'s='.format(ACT, H, T)#
	#         for f in f_list:
	#             title +=  str(int(f)) + ', '
	#         title = title[:-2]

	#         fig.suptitle(title, fontsize=30)  
	#         plt.tight_layout()
	    

	        # Deal with figure based on input parameters
	    # if savefig : 
	    #     outdir='./AMRO Best Fit Plots/{} Best Fits/'.format(ACT_label)
	    #     if not os.path.exists(outdir):
	    #         os.mkdir(outdir)
	    #     fig_file_name = outdir+title.replace('=',' ').replace(".","_").replace("\'s","").replace(',','').replace(" ","_")+".pdf"
	    #     fig.savefig(fig_file_name, dpi=300, transparent=False, bbox_inches='tight')
	    #     #print(ACT,H,T,'Best Fit Fig Saved')
	    # if plot_results: 
	    #     plt.show()
	    # else:
	    #     plt.close(fig)
	    # print(results.params)
	    # raise

	    return results

	    
	def FitACTExperiments(label, f_rank_min, f_ratio_min_ratio, f_max, show_fig = False): #(label, all_fits_df, f, show_fig=False):
	    all_fits_df = pd.DataFrame()
	    all_results_dict = {}
	    # fitted_amps = pd.DataFrame()
	    i=0
	    for T_label, H_label in ACT_choices[label]: 
	        
	        # Maybe select the frequencies based on their rank and ratio wrt strongest frequency
	        #q ='ACT == "{}" & H == {} & T == {} & amp_ratio >= {} & rank <= {} & `freqs (cycles/rot)`<={}'.format(label, H_label, T_label, f_ratio_min_ratio, f_rank_min, f_max)
	        # q ='ACT == "{}" & H == {} & T == {} & `freqs (cycles/rot)`in (2,4,6,8)'.format(label, H_label, T_label)
	        q ='ACT == "{}" & H == {} & T == {} & `freqs (cycles/rot)`in @FIT_SYMMETRIES'.format(label, H_label, T_label)

	        f_info = FT_results.query(q)[['freqs (cycles/rot)',  'amp_ratio']]  # 'rank', 'amp_ratio']]
	        f = f_info['freqs (cycles/rot)'].values
	        print("F LIST:", f)
	        # print(f)
	        # raise
	#         # We always want to fit f=2 and f=4
	        if 2 not in f: 
	            print('{}, T = {}, H = {}'.format(label, T_label, H_label))
	            print("2 not in f.")
	            # f =  np.append(f, 2)
	            
	#             print('\n**** 2-fold FT guess not present!!! ****')
	#             print('{}, T = {}, H = {}'.format(label, T_label, H_label))
	#             print("Appending 2... ")
	            
	        if 4 not in f:
	            print('{}, T = {}, H = {}'.format(label, T_label, H_label))
	            print("4 not in f.")
	#             f = np.append(f, 4)
	        
	#             print('\n**** 4-fold FT guess not present!!! ****')
	#             print('{}, T = {}, H = {}'.format(label, T_label, H_label))
	#             print("Appending 4... ")

	        results_obj = FitAMROData(amro_df, FT_results,
	                                  f, label, H_label, T_label,
	                                  print_results = False,
	                                  plot_results = show_fig,
	                                  savefig=True)

	        # Pack results to add to a larger dataframe
	        var_names = results_obj.var_names
	        param_results = results_obj.params
	        
	        # Store fitted values in a dictionary, which will be turned into a dataframe and concatenated to ACT's data
	        results_dict = {}
	        
	        f_info['act'] =  label
	        f_info['T (K)'] = T_label
	        f_info['H (T)'] =  H_label
	        f_info['f_list'] = f

	        if i == 0: 
	            fitted_amps = f_info
	            i+=1
	        else:
	            fitted_amps = pd.concat([fitted_amps, f_info])
	        # Add variables to
	        # i = 0
	        # while i < len(var_names):
	        # print(var_names)
	        for var in var_names:
	            results_dict[var] = param_results[var].value
	            results_dict[var + " err"] = param_results[var].stderr
	            # i+=1

	        results_dict['ACT'] = label
	        results_dict['H'] = H_label
	        results_dict['T'] = T_label
	        results_dict['chi squared'] = results_obj.redchi

	        # Add to the larger dataframe
	        results_df = pd.DataFrame(results_dict, index=[0])
	        all_fits_df = pd.concat([all_fits_df, results_df], ignore_index=True)
	        all_results_dict[label+"T"+str(T_label)+"H"+str(H_label)] = results_obj
	        
	    # replace all NaNs as zeros, assuming the problem was a mismatch between requested frequencies and FT guesses for the given experiment
	    all_fits_df = all_fits_df.fillna(0)
	    


	    return all_fits_df, all_results_dict, fitted_amps
if __name__ == "__main__":
	import sys
	load = LoadAMRO(sys.argv[1],sys.argv[2])
	_ = load.combineAMRO()
	