import pandas as pd
import requests
from bs4 import BeautifulSoup
import gzip
import time
import numpy as np


def read_lite(path):
    '''Reads a ISD Lite file and returns the df version
    ISD Lite specificatios:
    https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.pdf

    path => path to file

    Return
    df => df containing ISD every value is an int or float'''
    data = []
    with open(path) as f:
        temps = []
        n = 0
        for line in f:
            row = {}
            row['Observation Year'] = int(line[0:4])
            row['Observation Month'] = int(line[5:7])
            row['Observation Day'] = int(line[8:11])
            row['Observation Hour'] = int(line[11:13])
            row['Air Temperature'] = int(line[13:19]) / 10
            row['Dew Point Temperature'] = int(line[19:25]) / 10
            row['Sea Level Pressure'] = int(line[25:31]) / 10
            row['Wind Direction'] = int(line[31:37])
            row['Wind Speed Rate'] = int(line[37:43]) / 10
            # 9 is code for cannot get a reading.  Test how this works
            sky = int(line[43:49]) if int(line[43:49]) != -9999 else 9
            row['Sky Condition Total Coverage Code'] = sky
            row['Liquid Precipitation Depth - One Hour'] = int(line[49:55])/10

            # Almost always -9999 (invalid)
            # row['Liquid Precipitation Depth - Six Hour']= int(line[55:61])/10

            # get 24 highs and lows
            row['24hr Air Temp High'] = np.nan
            row['24hr Air Temp Low'] = np.nan
            if n > 47:
                data[n-48]['24hr Air Temp High'] = max(temps)
                data[n-48]['24hr Air Temp Low'] = min(temps)

            temps.append(row['Air Temperature'])
            temps = temps[-24:]

            data.append(row)
            n += 1

    df = pd.DataFrame(data)
    df = df.replace(-9999, np.nan)
    df = df.replace(-999.9, np.nan)
    df['USAF ID'] = path[5:11]
    df = df.set_index(['USAF ID',
                       'Observation Year',
                       'Observation Month',
                       'Observation Day',
                       'Observation Hour'])
    return df


def valid_file(name, fil):
    '''Checks if the file is a gz file and is a
    station in the filter

    name => a tag from soup
    fil => USAF id

    Return
    True if the file is .gz and for a station in fil'''
    return name['href'].endswith('.gz') and name['href'][:6] in fil


def get_files(url, fil):
    '''Gets list of files at url which pertain to the
    stations in the filter

    url => link with files
    fil => list of station numbers (USAF iden)

    Return
    list of urls with files in the filter'''
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    files = [a['href'] for a in soup.find_all('a', href=True)
             if valid_file(a, fil)]
    print(files)
    return files


def download_file(url, save_path):
    """Download a file from a URL and save it to a specified path."""
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def unzip_gz_file(in_path, out_path):
    '''Unzips gz file'''
    with gzip.open(in_path, 'rt') as f:
        content = f.read()

    with open(out_path, 'w') as f:
        f.write(content)

    return content


def download_data(download_url, fil):
    '''Downloads every file at the given url which has a station in the filter
    Returns the location of every output'''
    files = get_files(download_url, fil)

    locations = []
    for file_url in files:
        compressed = f'data\\compressed\\{file_url}'
        uncompressed = f'data\\{file_url[:-3]}'
        download_file(download_url + file_url, compressed)
        unzip_gz_file(compressed, uncompressed)

        locations.append(uncompressed)

    return locations


def download_years(base_url, years, fil):
    '''Download from base url for each year'''
    locations = []
    for year in years:
        print(year)
        locations += download_data(f'{base_url}/{year}/', fil)
        time.sleep(10)

    return locations


def create_df(locations):
    '''Saves all the ISD Lite data into one df

    locations => path to each file

    Return
    df => dataframe with all data combined'''
    dfs = []
    for location in locations:
        dfs.append(read_lite(location))

    df = pd.concat(dfs)
    return df


def download_csv(years, fil, base_url, out_path):
    '''Download data for each station/year and save to a csv

    years => list of years to search
    fil => list of USAF station IDs
    base_url => url to find the data without year included
    out_path => path to save the csv

    Return
    df => dataframe with all data combined'''

    locs = download_years(base_url, years, fil)
    df = create_df(locs)
    df.to_csv(out_path)

    return df

# df = read_lite('725030-14732-2024')
# df = read_lite('744864-54787-2024')
# print(df)
