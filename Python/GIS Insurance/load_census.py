import census
import geopandas as gpd
import json
import numpy as np
import os
import pandas as pd
import pytidycensus as tc
import sqlalchemy as sql


class CensusData():
    
    '''
    Based on the properties of interest, we need to pull ACS5
    features relevant to each property. There are over 60,000
    possible features to choose, so will need to find guess
    relevant ones.
    
    FOR ACS5, some ariables are at block group precision, and
    some are stored at tract and higher.
    
    To prep for machine learning, it's best to get each feature
    into a percentage of total in their respective census universe.
    
    Universes (can extract this programmatically from pytidycensus responses):
        Households
        Housing Units
        Civ. employed pop 16 years and over
        Occupied housing units
        
    Feature choices suggested by LLM (each will need unique cleaning/binning):
        Income :
            Median household income : B19013
            Household income in last year : B19001
            Public Assistance income : B19058
            
        Population :
            Sex by age: B01001
            Total population : B01003
        Housing :
            Housing unit count : B25001
            Household type : B11001
            Household type by relatives & nonrelatives : B11002
            Tenure by age of householder by occupants per room :B25015
            Tenure by plumbing facilities by occupants per room : B25016
            Single-parent householes : B11005
            Group Quarters Population : B26001
            Multigenerational households : B11017
            Units in structure (detached, MF, mobile homes) : B25024
            Year structure built : B25034
            Occupied vs vacant : B25002
            Owner vs renter occupied : B25003
            Rooms : B25017
            Mobile homes vs. conventional structures : B25024
            Total housing units (derive housing density): B25001
            Vacancy status : B25002
            Seasonal / recreational / occasional-use units : B25004
            House heating fuel : B25040
            Tenure by Occupants per Room: B25014
        Ethnicity :
            Language isolation : B16002
        Economy :
            Poverty rate : B17001
            Rent burden : B25070
            Mortgage burden: B25091
            Energy costs (selected monthly costs) : B25031
            Gini index : B19083
            
        Employment:
            Sex of workers by place of work: B08008
            Unemployment : B23025
            Sex by Occupation for Civ employed >=16 years old :C24010
            Worker class (pvt/gov/self): C24060
        Education : 
            Educational attainment : B15003
        Transport : 
            Household Size by Vehicles Available : B08201
            Number of Workers in Household by Vehicles Available : B08203
            Means of Transportation to Work : B08301
            Travel time to work : B08303
            Sex of workers by Means of Transportation to Work : B08006
        Disability :
            Disability by age : C18108
    
    
            
    '''
    
    # Set the coordinate system
    DEFAULT_CRS = 5070

    DEFAULT_FEATURES = [
            'B25014', 'B25015', 'B25016', 'B25017', 'B25024', 'B25032', 
            'B25033', 'B25034', 'B25035', 'B25040', 'B25041', 'B25047', 
            'B25063', 'B25070', 'B25091', 'B01001', 'B01002', 'B01003', 
            'B01005', 'B01009', 'B02001', 'B03001', 'B05001', 'B05002', 
            'B06001', 'B19001', 'B19013', 'B19019', 'B19020', 'B19083', 
            'B19301', 'B19326', 'B23025', 'B23032', 'B24010', 'B24050', 
            'B08201', 'B08202', 'B08203', 'B08204', 'B25031'
        ]
        
    
    
    def __init__(self, sql_engine, sql_conn,  year :int):

        self.year = year
        
        self.census = census.Census(
        		os.environ.get('US_CENSUS_API_KEY'),
        		year = self.year
        	).acs5
        self.metadata_filepath = os.path.join('Data', "acs5_variables_{}.txt".format(self.year))

        if not os.path.exists(self.metadata_filepath):
            self._generate_variable_metadata()

        # TODO: IF EXISTS, CONNECT. IF NOT EXISTS, CREATE EMPTY
        self.sql_engine = sql_engine
        self.sql_conn = sql_conn
        
        # Chosen variables for model (query these from the sql table, if they don't exist, generate it
        self.census_variables_list = []
        
        return self
    
    def _extract_from_geoid(self, geo_id:list|str) -> gpd.GeoDataFrame:
        '''
            Takes in regional identfier to get census information.
        '''
            
        tmp_list = []
        for t in geo_id:
                tmp_list.append()
        data = gpd.GeoDataFrame(tmp_list, crs=DEFAULT_CRS)
        
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
    def _generate_variable_metadata(self)-> None:
        
        acs5_dict = {}
        for item in self.census.tables():
            keys = [x for x in item.keys() if 'name' not in x and 'variables' not in x]
            acs5_dict[item['name']] = {key : item[key] for key in keys}
        
        with open(self.metadata_filepath, "w") as f:
            json.dump(acs5_dict, f, indent=4)
        return

    def get_variable_universe(self, variable:str|list)->str:
        meta_fp = os.path.join('Data', self.metadata_filepath)
        with open("acs_variables.txt", "r") as f:
            loaded_dict = json.load(f)
        if isinstance(variable, str):
            return loaded_dict[variable]['universe']
        elif isinstance(variable, list):
            return [loaded_dict[var]['universe'] for var in variable]
        else:
            raise TypeError("'variable' must be a string or list of strings.")
        
    def add_census_variables(self, var_list:list|str)-> None:
        
        if isinstance(var, list):
            for var in var_list:
                pass
        elif isinstance(var, str):
            pass
        else:
            raise TypeError("'var_list' must be a string of list of strings")
        return
        
        
    def visualize_data(self):
        return
    
    
if __name__ == "__main__":
    test_obj = LoadCensus("Data")
    print(test_obj.C.tables())
    test_obj.visualize_data()
