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
		this_meta_data['res. units']['mean res (ohm-cm)'] = mean_res
		this_meta_data['res. units']['0deg res (ohm-cm)'] = zero_deg_res
		
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
						y='Delta Res./R0 mean (ohm-cm)', 
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
		self.meta_data = amro.META_DATA
		self.labels = self.amro_data[['ACT', 'T', 'H']].drop_duplicates()

		# Iterate through DataFrame entries, appending results to a new DataFrame
		self.FT_results_df = pd.DataFrame()
		self.save_name = save_name
		self.save_dir = save_dir
		self.save_fp = os.path.join(save_dir, save_name)

		if os.path.exists(self.save_fp):
			print("loading {}".format(save_name))
			self.FT_results_df = pd.read_csv(self.save_fp)
			print(self.FT_results_df.columns)
			return
		else:
			# for i in range(len(experiment_labels)):
			for act_label in self.meta_data.keys():
				print("FT'ing: " + act_label)
				print(self.meta_data)
				act_meta_data = self.meta_data[act_label]
				print(act_meta_data)
				# try:
				t_vals = act_meta_data['T_vals']
				h_vals = act_meta_data['H_vals']
				geo_label = act_meta_data['geo']
				print(t_vals, h_vals, geo_label)
				for t, h in itertools.product(t_vals, h_vals):
			
					ft_df = self._fourier_transform(act_label, t, h)


					# Query the correct dataframe using the experiment labels
					ft_df=self.amro_df.query('ACT_str=="{}" & T =={} & H == {}'.format(act_label, t, h))  # 'ACT_str=="{}"'.format(act_label))  # 
			
					# label = act_label+", "+str(t)+"K, "+str(h)+"T"
			
					# To FT, we want the oscillation zero'd along the y-axis
					fftdata = ft_df['Delta Res. Mean (ohm-cm)'].values
			
					# Perform the FFT, where yf is the amplitudes and xf are the frequencies
					yf = rfft(fftdata, n= len(fftdata), norm='ortho')
					xf = rfftfreq(len(fftdata), 1/len(fftdata))
			
					# Package the results
					freq_df = pd.DataFrame({'freqs (cycles/rot)':xf,
											'amps':yf,
											'mag (ohm-cm)':np.abs(yf),
											'phase':np.angle(yf)
										   })

					# freq_df.sort_values(by='mag (ohm-cm)', ascending=False, inplace=True)

					freq_df = self._package_ft(freq_df)
					# Amplitudes relative to the strongest
					freq_df['amp_ratio'] = freq_df['mag (ohm-cm)']/freq_df['mag (ohm-cm)'].max()
					freq_df['freqs (cycles/rot)'] = freq_df['freqs (cycles/rot)'].astype(int)


					# Force positive phase values
					if force_pos_phase:
						freq_df['phase'] = np.select(
							freq_df['phase']<0,
							freq_df['phase']+2*np.pi,
							freq_df['phase']
						)
					
					# freq_df = freq_df.reset_index(drop=True)
					
					# Add additional labelling information
					freq_df['ACT_str'] = act_label 
					freq_df['ACT'] = float(act_label.replace("ACTRot",""))
					freq_df['T'] = t 
					freq_df['H'] = h 
					freq_df['geo'] = geo_label 
			
					# Truncate to get the desired number of frequencies
					#TODO: separate out, should be called from main script

			
			
					# FT_output = FFTAMROPlot(ft_df, n=10) # Want n strongest amplitudes
					
					self.FT_results_df = pd.concat([FT_results_df, strongest_freqs], ignore_index=True)#.reset_index(drop=True)


				# except KeyError as e:
				# 	print(e)
				# 	print('No data for: '+act_label+'. Skipping...')
					
				# Save the results of the FT
				self.FT_results_df.to_csv(self.save_name, sep=',',index=False)
				print('Results saved to: {}'.format(self.save_name))
			return

	def GetNStrongest(self, n:int):
		'''
			Queries the n strongest contributions for each experiment in the data set.
		''' 
		# self.get_n_strongest()
		# strongest_df = self.FT_results_df.query("`freqs (cycles/rot)`<{}".format(max_sym))
		# # freq_df = freq_df.reset_index(drop=True)
		print(self.FT_results_df.head())#.columns)
		strongest_df = self.FT_results_df.groupby(['ACT','H','T'])
		strongest_freqs =  strongest_df.sort_values(by='mag (ohm-cm)', ascending=False).head(n)

		return strongest_freqs

	def PlotNStrongest(self, n:int):
		'''
		Plots the n-strongest. 

		If n=0, then plots all available contributions.
		'''
		return



# class FitAMRO:




if __name__ == "__main__":
	import sys
	load = LoadAMRO(sys.argv[1],sys.argv[2])
	_ = load.combineAMRO()
	