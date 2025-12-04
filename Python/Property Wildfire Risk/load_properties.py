# import census
import censusgeocode as cg
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd

from random_address import real_random_address
from shapely.geometry import Point
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


class Properties():
    '''
        For each property, we want lat/long, and state/county/tract/block 
        group information so that we can later query the ACS5.
    '''
    # Set the coordinate system
    DEFAULT_CRS = 5070
    TABLE_NAME = 'properties'
    LABELS_KEYS_MAP = {
        'geoid':  'GEOID',
        'block_id' :  'BLOCK',
        'block_grp':  "BLKGRP",
        'tract_id':  'TRACT',
        'county_id': 'COUNTY',
        'state_id': 'STATE'
    }
    
    def __init__(self, sql_engine, sql_conn):
        
        self.sql_engine = sql_engine
        self.sql_conn = sql_conn
        self.Session = sessionmaker(bind=self.sql_engine)

        self._connect_to_sql()
        # self.num_properties = self.properties_gpd.shape[0]


    def add_random_properties(self, quantity:int)->None:
        """
            Add 'quantity' many new random addresses
        """
        # TODO: Drop duplicates. Floating point rounding errors in the coords may affect this.
        # Could keep a hash of the address, for privacy?
        # self.session = self.Session()
        print('Adding {} more properties'.format(quantity))

        # TODO: Turn this into a dictionary, should be faster
        temp_lst = [None]*quantity
        for i in range(quantity):
        
            coords = real_random_address()['coordinates']
            
            lat = coords['lat']
            long = coords['lng']
            block = cg.coordinates(x=long, y=lat)['2020 Census Blocks'][0]
            self.LABELS_KEYS_MAP
            temp_lst[i] = { key :  int(block[self.LABELS_KEYS_MAP[key]]) for key in self.LABELS_KEYS_MAP.keys()}
            temp_lst[i]['geom'] = Point(long, lat)
            
        tmp = gpd.GeoDataFrame(temp_lst, geometry='geom', crs=self.DEFAULT_CRS)
        
        self.properties_gpd = pd.concat([self.properties_gpd,tmp])

        # TODO: should find a way to just append
        self.properties_gpd.to_postgis(self.TABLE_NAME, con=self.sql_conn, if_exists='replace')
        self._commit_database_changes()
        # self.session.commit()
        # self.properties_gpd.to_file(filepath, mode='a')
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
    
    # def property_count(self) ->int:
    #     '''
    #         Returns how many properties are in the list.
    #     '''
    #     return
    def _connect_to_sql(self)->None:
            
        # check if properties table exists, and connect
        if self.sql_engine.dialect.has_table(self.sql_conn, self.TABLE_NAME):
            print("'properties' table found. Loading...")
            q = 'SELECT * FROM {}'.format(self.TABLE_NAME)

            # TODO: needs to convert to gpd
            self.properties_gpd  = gpd.read_postgis(q, con=self.sql_conn,  geom_col='geom')
        else:
            print("Creating new 'properties' table with 10 entries.")
            q = 'CREATE TABLE {} ('.format(self.TABLE_NAME)
            for key in self.LABELS_KEYS_MAP.keys():
                q+= '{} INTEGER,'.format(key)
            q+= 'geom geometry); '.format(self.DEFAULT_CRS)
            self.sql_conn.execute(text(q))

            
            q = 'SELECT * FROM {}'.format(self.TABLE_NAME)
            self.properties_gpd  = gpd.read_postgis(q, con=self.sql_conn,  geom_col='geom')
            self.add_random_properties(10)
            
        return


        
    def _commit_database_changes(self)->None:
        self.sql_conn.execute(text('COMMIT;'))
        
    
if __name__ == "__main__":
    import sys
    engine = sql.create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/wildfire_risk_project")

    conn = engine.connect() 
    props = Properties(engine, conn)
    props.add_random_properties(int(sys.argv[1]))
