import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd


DESIRED_COLS = [
	'LATITUDE', # Center of nominal 375 m fire pixel
	'LONGITUDE', # Center of nominal 375 m fire pixel
	# 'BRIGHTNESS', 
	# 'SCAN', 
	# 'TRACK', 
	'ACQ_DATE',
	# 'ACQ_TIME', 
	'SATELLITE',  # N21 = NOAA-21, N=SNPP
	# 'INSTRUMENT', 
	'CONFIDENCE', # It is intended to help users gauge the quality of individual hotspot/fire pixels. 
	# 'VERSION',
	'BRIGHT_T31',  # T31 Channel brightness temperature of the fire pixel measured in Kelvin
	'FRP',  # FRP depicts the pixel-integrated fire radiative power in MW (megawatts)
	# 'DAYNIGHT',
	'TYPE',   # Inferred hot spot type: 0 is presumed vegetation fire
	'geometry'
]

class LoadGIS():



	def __init__(self, filepaths:list):

		# TODO: If sql db doesn't exist, load data
		raw_data = self.extract(filepaths)
		clean_data = self.transform(raw_data)
		self.load(clean_data)
		# Else: just load sql db 
		return clean_data

	def extract(self, filepaths:list):
		"""
			Extracts GIS data from VIIRS sources downloaded manually and fixes data type issues.

			TODO: Query the data directly from a website API.
		"""
		#########

		# Extract VIIRS NOAA-20 Point Data

		fp=os.path.join("Data","VIIRS NOAA20 375m","fire_archive_J1V-C2_633153.shp")
		noaa20 = gpd.read_file(fp)[desired_cols]

		noaa20['ACQ_DATE'] =pd.to_datetime(noaa20['ACQ_DATE'])

		noaa20['year'] = noaa20['ACQ_DATE'].dt.year

		# 0 is 'probable wildfire', h is high confidence
		noaa20 = noaa20.query('TYPE==0 & CONFIDENCE=="h"')

		#########

		# Extract VIIRS S-NPP Data
		fp=os.path.join("Data","VIIRS SNPP 375m","fire_archive_SV-C2_633155.shp")
		snpp= gpd.read_file(fp)

		snpp['ACQ_DATE'] = pd.to_datetime(snpp['ACQ_DATE'])
		snpp['ACQ_TIME'] = pd.to_datetime(snpp['ACQ_DATE'])

		snpp['year'] = snpp['ACQ_DATE'].dt.year

		# 0 is probably wildfire, h is high confidence
		snpp = snpp.query('TYPE==0 & CONFIDENCE=="h"')
		#########
		# ^ This can be simplified



		return

	def transform(self, raw_data):
		# Need to make sure we're not double counting the VIIRS data
		return

	def load(self, clean_data):
		# Store it in a local SQL database

		return


