import census
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pytidycensus as ptc
import sqlalchemy as sql


class CensusData():

    '''
        Based on the properties of interest, we need to pull ACS5
        features relevant to each property. There are over 60,000
        possible features to choose, so will need to find guess
        relevant ones.

        FOR ACS5, some ariables are at block group precision, and
        some are stored at tract and higher.

        Feature choices suggested by an LLM:
            Income :
            Population 
            Housing :
            Ethnicity :
            Economy :
            
    '''
	DESIRED_COLS = [
	
	]

	# Set the coordinate system
	CRS = 5070



	def __init__(self, year :int, sql_engine, sql_conn):

		# self.census = census.Census(
		# 		os.environ.get('US_CENSUS_API_KEY'),
		# 		year = year
		# 	).acs5

        ptc.set_census_api_key(os.environ.get('US_CENSUS_API_KEY'))

		# TODO: If sql db doesn't exist, load data
		self.sql_engine = sql_engine
		self.sql_conn = sql_conn

		raw_data = self._extract_from_tracts(tracts, year)
		self.data = self._transform(raw_data)
		# Save in SQL db
		self._load(self.DATA)

        self.census_variables_list = []

		# Else: just load sql db 
		return self

	def _extract_from_geoid(self, geo_id:list|str) -> gpd.GeoDataFrame:
		'''
			Takes in a list of tracts to get from the US Census API.
		'''
		

		tmp_list = []
		for t in geo_id:
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

    def add_census_variables(self, var_list:list|str)-> None:

        if isinstance(var, list):
            for var in var_list:
                pass
        elif isinstance(var, str):
            pass
        else:
            raise TypeError
        return


	def visualize_data(self):
		return


if __name__ == "__main__":
	test_obj = LoadCensus("Data")
	print(test_obj.C.tables())
	test_obj.visualize_data()
