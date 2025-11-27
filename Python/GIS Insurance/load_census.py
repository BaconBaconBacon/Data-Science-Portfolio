import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd


class LoadCensus():


	def __init__(self, filepaths:list):

		# TODO: If sql db doesn't exist, load data
		raw_data = self.extract(filepaths)
		clean_data = self.transform(raw_data)
		self.load(clean_data)
		# Else: just load sql db 
		return clean_data

	def extract(self, filepaths:list):
		for fp in filepaths:
			pass
		return

	def transform(self, raw_data):
		return

	def load(self, clean_data):
		return


