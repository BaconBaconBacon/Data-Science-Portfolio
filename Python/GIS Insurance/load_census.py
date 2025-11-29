import census
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



	def __init__(self, tracts:list|str, year :int):

		self.C = census.Census(
				os.environ.get('US_CENSUS_API_KEY'),
				year = year
			).acs5


		# TODO: If sql db doesn't exist, load data
		self.SQL_ENGINE = None
		self.SQL_CONN = None

		raw_data = self._extract_from_tracts(tracts, year)
		self.DATA = self._transform(raw_data)
		# Save in SQL db
		self._load(self.DATA)

		# Else: just load sql db 
		return self

	def _extract_from_tracts(self, tracts:list|str) -> gpd.GeoDataFrame:
		'''
			Takes in a list of tracts to get from the US Census API.
		'''
		

		tmp_list = []
		for t in tracts:
				tmp_list.append()
		data = gpd.GeoDataFrame(tmp_list, crs=CRS)

		return data

	def extract_from_regions(self, tracts:list|str) -> gpd.GeoDataFrame:
		'''
			Can add this functionality at another time.
		'''
		return
	def _transform(self, raw_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
	print(test_obj.C.tables())
	test_obj.visualize_data()
