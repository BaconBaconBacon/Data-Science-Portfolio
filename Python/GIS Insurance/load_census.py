import censusdata
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd


class CensusData():
	DESIRED_COLS = [
	
	]

	# Set the coordinate system
	CRS = 5070

	self.DATA = None
	self.SQL_ENGINE = None
	self.SQL_CONN = None

	def __init__(self, tracts:list|str):

		# TODO: If sql db doesn't exist, load data
		raw_data = self._extract(tracts)
		self.DATA = self._transform(raw_data)
		# Save in SQL db
		self._load(clean_data)

		# Else: just load sql db 
		return self

	def extract_from_tracts(self, tracts:list|str): -> gpd.GeoDataFrame
		'''
			Takes in a region or list of regions for which to pull (USA) census data.
		'''
		data = gpd.GeoDataFrame()

		if isinstance(filepaths, str):
			if filepaths.endswith(".csv") or filepaths.endswith(".shp"):
				return
			else :
				temp_list = []
				for file in os.listdir():
					if filepaths.endswith(".csv") or filepaths.endswith(".shp"):
						temp_list.append(file)
				filepaths = temp_list

		for fp in filepaths:
			pass
		return data

	def extract_from_regions(self, tracts:list|str): -> gpd.GeoDataFrame
		'''
			Can add this functionality at another time.
		'''
		return
	def _transform(self, raw_data: gpd.GeoDataFrame): -> gpd.GeoDataFrame
		return

	def _load(self, clean_data: gpd.GeoDataFrame):
		'''
			Save the data as a SQL db.
		'''
		return

	def visualize_data(self):
		return


if __name__ == "__main__":
	test_obj = LoadCensus("Data")
	test_obj.visualize_data()
