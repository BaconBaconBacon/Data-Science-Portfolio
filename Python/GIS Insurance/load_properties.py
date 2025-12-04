# import census
import censusgeocode as cg
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd

from random_address import real_random_address


class Properties():
    '''
        For each property, we want lat/long, and state/county/tract/block 
        group information so that we can later query the ACS5.
    '''
    # Set the coordinate system
    DEFAULT_CRS = 5070
    
    def __init__(self, filepath:list|str, sql_engine, sql_conn):
        
        self.sql_engine = sql_engine
        self.sql_conn = sql_conn

        
        self.file_path = filepath

        self.num_properties = -1
        self.properties_gpd = gpd.GeoDataFrame()


        
        if os.path.exists(self.file_path) and filepath.endswith('.shp'):
            self.load_property_list(self.file_path)
        else:
            print('Existing .shp file not found at filepath. Call \
                    add_random_properties to populate new .shp file')
        return self
    
    def add_random_properties(self, quantity:int)->None:
        """
            Add 'quantity' many new random addresses
        """
        # TODO: Drop duplicates. Floating point rounding errors in the coords may affect this.
        # Could keep a hash of the address, for privacy?
        
        # If properties dropped, add more.
        start = time.now()

        temp_lst = [None]*quantity
        for i in range(quantity):
        
            coords = real_random_address()['coordinates']
            
            lat = coords['lat']
            long = coords['lng']
            block = cg.coordinates(x=long, y=lat)['2020 Census Blocks'][0]
            
            temp_lst[i] = {
                # 'index': i,
                'geometry' :   Point(long, lat),
                'geoid': int(block['GEOID']),
                'block_id' : int(block['BLOCK']),
                'block_grp': int(block["BLKGRP"]),
                'tract_id': int(block['TRACT']),
                'county_id':int(block['COUNTY']),
                'state_id':int(block['STATE'])
            }
        
               
        self.properties_gpd = gpd.GeoDataFrame(temp_lst, crs=CRS)
        self.properties_gpd.to_file(filepath, mode='a')
        print(time.now()-start)
        return
    
    def delete_at_random(self, quantity:int)->None:
        """
        Remove "quantity" many addresses from the db, at random.
        """
        return
    
    def get_random_subset(self, quantity:int)->gpd.GeoDataFrame:
        return
    
    def get_state_subset(self, state:str)->gpd.GeoDataFrame:
        return
    
    # def propert_count(self) ->int:
    #     '''
    #         Returns how many properties are in the list.
    #     '''
    #     return
    
    def read_property_list(self, filepath:str)->None:
        
        if os.path.exists(filepath) and filepath.endswith('.shp'):
            self.properties_gpd = gpd.read_file(self.file_path).to_crs(DEFAULT_CRS)
            print('.shp file found and loaded.')
        else:
            print('No .shp file found at {filepath}. Creating empty GeoDataFrame.')
            self.properties_gpd = gpd.GeoDataFrame()
        return
        
    
if __name__ == "__main__":
    test_obj = Properties()
    # test_obj.visualize_data()
