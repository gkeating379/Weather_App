from data_processing import process, read, verify
import pandas as pd

start_year, end_year = 2010, 2024
# Port Authority, JFK, LGA, EWR, Central Park
USAFs_train = ['720553', '744860', '725030', '725020', '725053'] 
USAFs_test = ['744864']
years = [year for year in range(start_year, end_year+1)]
lite_base_url = 'https://www.ncei.noaa.gov/pub/data/noaa/isd-lite'

read.download_csv(years, USAFs_train, lite_base_url, 'processed_data\\train.csv')
read.download_csv(years, USAFs_test, lite_base_url, 'processed_data\\test.csv')

#############################################################################

df_train = pd.read_csv('processed_data/train.csv')
df_train = df_train.set_index(['USAF ID',
                               'Observation Year',
                               'Observation Month',
                               'Observation Day',
                               'Observation Hour'])

df_test = pd.read_csv('processed_data/test.csv')
df_test = df_test.set_index(['USAF ID',
                             'Observation Year',
                             'Observation Month',
                             'Observation Day',
                             'Observation Hour'])

verify.compare_stats(df_train, df_test, 'compare.csv')

# x_train, y_train = process.df_to_np(df_train, USAFs_train, start_year, end_year)
# x_test, y_test = process.df_to_np(df_test, USAFs_test, start_year, end_year)

# print(len(x_train))
# print(len(y_train))
