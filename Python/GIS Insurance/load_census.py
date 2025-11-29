import censusdata
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd


class LoadCensus():
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

	def __init__(self, filepaths:list):

		# TODO: If sql db doesn't exist, load data
		raw_data = self.extract(filepaths)
		clean_data = self.transform(raw_data)
		self.load(clean_data)
		# Else: just load sql db 
		return clean_data

	def extract(self, filepaths:list): -> gpd.GeoDataFrame
		for fp in filepaths:
			pass
		return

	def transform(self, raw_data):
		return

	def load(self, clean_data):
		return


