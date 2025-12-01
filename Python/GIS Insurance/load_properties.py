import census
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd


class Properties():

    # Set the coordinate system
    CRS = 5070
    
    
    def __init__(self, filepath:list|str):
        
        self.SQL_ENGINE = None
        self.SQL_CONN = None
        self.NUM_PROPERTIES = -1
        
        # If .shp is populated, load it
        self.properties_gpd = gpd.GeoDataFrame()
        
        return self

    def add_random_properties(self, quantity:int)->None:
        """
         Generate 'quantity' many new random addresses
        """
        # TODO: Drop duplicates. Floating point rounding errors in the coords may affect this.
        # Could keep a hash of the address, for privacy?
    
        # If properties dropped, add more.
        
        return
    
    def remove_random_properties(self, quantity:int)->None:
        """
            Remove "quantity" many addresses from the db, at random.
        """
        return
    
    def pull_random_subset(self,quantity:int)->gpd.GeoDataFrame:
    
        return
    
    def pull_state_subset(self, state:str)->gpd.GeoDataFrame:
        return

    def propert_count(self) ->int:
        '''
            Returns how many properties are in the list.
        '''
        return

    def load_property_list(self, filepaths->str)->None:
        return

        
if __name__ == "__main__":
	test_obj = Properties()
	# test_obj.visualize_data()
