import census
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
import geopandas as gpd


class GISData():
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
    
    # Set the coordinate system
    DEFAULT_CRS = 5070

    def __init__(self,  sql_engine, sql_conn):
        
        # TODO: If sql db doesn't exist, load data
        self.sql_engine = None
        self.sql_conn = None
        
        raw_data = self._extract(filepaths)
        self.data = self._transform(raw_data)
        
        # Save in SQL db
        self._load(self.data)
        
        # Else: just load sql db 
        return self

    def _extract(self, filepaths:list|str) -> gpd.GeoDataFrame:
        '''
            Takes as input a string or a list of strings consisting
            of filenames or filepaths to wildfire GIS data.

            TODO: It would be neat to query regions and years, and
            use a gov't API to get that data. 
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
            sat_name = fp.split('.')[0]

            sat_data = gpd.read_file(fp)[DESIRED_COLS]
            sat_data['SAT_ID']  = sat_name

            # 0 is 'probable wildfire', h is high confidence
            sat_data = sat_data.query('TYPE==0 & CONFIDENCE=="h"')

            # Set coordinate system
            sat_data = sat_data.to_crs(self.DEFAULT_CRS)
            data.append(sat_data)

        return data

    def _transform(self, raw_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        
        raw_data['ACQ_DATE'] =pd.to_datetime(sat_data['ACQ_DATE'])
        raw_data['year'] = raw_data['ACQ_DATE'].dt.year

        return raw_data

    def _load(self, clean_data: gpd.GeoDataFrame):
        '''
            Save the data as a SQL db.
        '''
        return

    def visualize_data(self):
        '''
            Helper function for visualizing the data using Folium.

            Needs adaptation.
        '''

        #     legend_html = '''
        #         <div style="position: fixed; 
        #              bottom: 50px; left: 50px; width: 200px; height: 150px; 
        #              border:2px solid grey; z-index:9999; font-size:14px;
        #              background-color:white; opacity: 0.85;">
        #              &nbsp; <b>Legend</b> <br>
        #              &nbsp; NOAA-20 &nbsp; <i class="fa fa-circle" style="color:red"></i><br>
        #              &nbsp; S-NPP &nbsp; <i class="fa fa-circle" style="color:purple"></i><br>
        #              &nbsp; MTBS &nbsp; <i class="fa fa-square" style="color:red"></i><br>
        #              &nbsp; MADIS &nbsp; <i class="fa fa-square" style="color:orange"></i><br>
        #              &nbsp; WFIGS &nbsp; <i class="fa fa-square" style="color:blue"></i><br>
        #         </div>
        #     '''


        #     centre = [pred_df['latitude'].mean(), pred_df['longitude'].mean()]
        #     m=folium.Map(centre, zoom_start=5)

        #     # Perimeters
        #     i=0
        #     for df in perims_lst:
        #         for _, r in df.iterrows():
        #             fill_color = per_color_lst[i]  # Issue with Python closures
        #             sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
        #             geo_j = sim_geo.to_json()
        #             geo_j = folium.GeoJson(
        #                 data=geo_j, 
        #                 style_function=make_style(fill_color)
        #             )
        #             geo_j.add_to(m)
        #         i+=1
                
        #     # Points
        #     i=0
        #     for df in points_lst:
        #         for _, r in df.iterrows():
        #             folium.CircleMarker(
        #                 location=[r['LATITUDE'], r['LONGITUDE']],
        #                 radius=3,
        #                 fill=True,
        #                 fill_opacity=0.7,
        #                 weight=1,
        #                 fill_color=pts_color_lst[i],
        #                 color=pts_color_lst[i]
             
        #             ).add_to(m)

        #         i+=1
        #     # State map
        #     geo_j = folium.GeoJson(data=usa_map.to_json()) 
        #     geo_j.add_to(m)

        #     # add legend
        #     m.get_root().html.add_child(folium.Element(legend_html))


        # IFrame(src=save_path, width=1000, height=600)

        return


if __name__ == "__main__":
    test_obj = LoadGIS("Data")
    test_obj.visualize_data()
