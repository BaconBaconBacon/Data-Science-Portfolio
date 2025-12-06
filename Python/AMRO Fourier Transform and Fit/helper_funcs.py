import os
import pandas as pd



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

				self.AMRO = pd.read_csv(self.file_path)
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
		self.META_DATA[act_label][H_label][T_label] = {}
		
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

		act_label=str(df['ACT'].values[0])
		H_label=str(df['H'].values[0])
		T_label=str(df['T'].values[0])
		print(act_label, H_label, T_label)
		
		# Calc for additional columns as needed
		mean_res = df['Res. (ohm-cm)'].mean()
		zero_deg_res= df.loc[df['Sample Position (deg)'].idxmin(), 'Res. (ohm-cm)']
		
		# Store additional meta data
		this_meta_data = self.META_DATA[act_label][H_label][T_label]
		this_meta_data['res. units']['mean res (ohm-cm)'] = mean_res
		this_meta_data['res. units']['0deg res (ohm-cm)'] = zero_deg_res
		
		return

	def _createAltResistanceUnits(self, df:pd.DataFrame)->pd.DataFrame:
		'''
			Calculates alternative resistivity units based on the new meta data
		'''
		act_label=df['ACT'].values[0]
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
			df[new_labels[0]] =  df['Res. (ohm-cm)'] - mean_res
			df[new_labels[1]] =  df[new_labels[0]] / mean_res
			df[new_labels[2]] =  df[new_labels[1]] * 100

		# uohms
		for col in df.columns:
			if 'Res' in col:
				new_col = col.replace('ohm','uohm')
				df[new_col] = df[col]*10**6

		return df

if __name__ == "__main__":
	import sys
	load = LoadAMRO(sys.argv[1],sys.argv[2])
	_ = load.combineAMRO()
	quit()