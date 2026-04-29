import numpy as np
import geopandas as gpd
import pandas as pd
from helper_funs import convert_location
import csv


def create_utm_loc_file_from_geojson(file_path):
    df = gpd.read_file(file_path)

    x_diff = np.diff(df.geometry.x)
    y_diff = np.diff(df.geometry.y)

    az = np.degrees(np.arctan2(x_diff,y_diff))

    crs = str(df.crs)[-5:]
    northern = 'NORTHERN' if crs[:3] == '326' else 'SOUTHERN'
    zone_num = int(crs[-2:])

    out = gpd.GeoDataFrame({
    "ID": range(len(x_diff)),
    "easting": df.geometry.x[:-1],
    "northing": df.geometry.y[:-1],
    "Azimuth": az,
    "Zone Num": zone_num,
    "Zone Let": northern 
    })

    out.to_csv("Turkey_Swath_Az_file.csv", index=False)


def create_utm_loc_file_from_csv(file_path):
    df = pd.read_csv(file_path, names=['ID', 'X', 'Y'], header=0)

    df['ID'] = df['ID'].astype(int)
    for col in ['X', 'Y']:
        df[col] = df[col].astype(float)

    x_diff = np.diff(df['X'])
    y_diff = np.diff(df['Y'])

    az = np.degrees(np.arctan2(x_diff,y_diff))

    crs = '32647'
    northern = 'NORTHERN' if crs[:3] == '326' else 'SOUTHERN'
    zone_num = int(crs[-2:])

    out = pd.DataFrame({
    "ID": df["ID"][:-1],
    "easting": df['X'][:-1],
    "northing": df['Y'][:-1],
    "Azimuth": az,
    "Zone Num": zone_num,
    "Zone Let": northern 
    })

    out.to_csv("Burma_Swath_Az_file.csv", index=False)


create_utm_loc_file_from_csv("PFDHA Processed Data - Burma Profile Cntpts (1).csv")

def create_export_file():
    locations_path = "PFDHA Processed Data - Burma Loc.csv"
    coords_path = "Burma_Swath_Az_file.csv"

    locations = pd.read_csv(locations_path)
    coords = pd.read_csv(coords_path)


    UTM_results = []
    LATLON_results = []

    for row in locations.itertuples():
        coords_row = coords.loc[coords['ID'] == row.ID].iloc[0]
        coords_input = [coords_row['easting'],
                        coords_row['northing'],
                        coords_row['Azimuth'],
                        coords_row['Zone Num'],
                        coords_row['Zone Let']]
        
        ouput_coords = convert_location(row.Loc, coords_input, input_format="UTM")
        ouput_coords["UTM"]["Width"] = row.Width
        ouput_coords["UTM"]["Disp"] = row.Disp
        ouput_coords["LATLON"]["Width"] = row.Width
        ouput_coords["LATLON"]["Disp"] = row.Disp

        UTM_results.append(ouput_coords["UTM"])
        LATLON_results.append(ouput_coords["LATLON"])

    with open("Turkey_UTM_3xScaled_Results_1.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UTM_results[0].keys())
        writer.writeheader()
        writer.writerows(UTM_results)

    f.close()

    # with open("Turkey_LATLON_3xScaled_Results.csv", "w", newline="") as f2:
    #     writer = csv.DictWriter(f2, fieldnames=LATLON_results[0].keys())
    #     writer.writeheader()
    #     writer.writerows(LATLON_results)

    # f2.close()

create_export_file()