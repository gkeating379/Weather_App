# Table of Contents
- [Description](#description)
- [Results](#results)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvments)

# Description
This project looks to build models that can accurately predict the high and low temperatures for the next 24 hours given a set of previous weather readings.  The readings are taken from the [NOAA ISD Lite dataset](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/).  This dataset includes 7 common readings taken every hour (although in practice one of these readings is almost always left blank and thus was not used here).  The dataset includes readings from as early as 1901 and across thousands of stations.  Futher details can be found [here](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-technical-document.pdf).  For this project, 6 stations were used in the NYC area to build models.  Each model takes the past two days of readings (48 sets of 6 readings) and uses them to predict the high and low temperature in the next 24 hour window. 

About 25% of the data was invalid readings.  Two different types of imputation were tested to resolve this.  Listwise deletion was used where any reading with an invalid was removed.  This was compared with last observation carried forward (LOCF) where any invalid was replaced with the last valid reading.  Data was also normalized using MinMax scaling in order to enhance training with several different variables that had vastly different ranges.

Two different models were also compared to evaluate which would be better suited to the task.  A stacked RNN was used to take advantage of the sequential nature of the data and the possibility that recent data should be much more relevant to the next temperature reading.  A multi-headed attention model was also made to allow the model to learn for itself when to look at data.  This was chosen under the hypothesis that some readings may have longer time dependencies than others.

Models were also made with seq-to-seq and direct targets.  The seq-to-seq took the past 48 hourly readings and predicted the next hour’s 6 readings, then used this to predict the 24 hour high and low.  The direct target took the past 48 hourly readings and predicted the high and low directly with no other output information.

# Results

The values below are the MSE of the predicted high and low temperature values for USAF ID 744864 which is located in Farmingdale Republic Airport.  In bold is th best loss for each target type

| Direct Models         | 10 Epochs     | 20 Epochs     | 30 Epochs     |
|:-:                    |:-:            |:-:            |:-:            |
| LSTM listwise         | 16.97805214   |**13.33794212**| 14.23445034   |
| Attention listwise    | 14.73112583   | 15.26622105   | 15.33016777   |
| LSTM LOCF             | 15.76882076   | 14.95208168   | 13.74439907   |
| Attention LOCF        | 14.78586006   | 14.59226894   | 14.81693459   |


| Seq-to-Seq Models     | 10 Epochs     | 20 Epochs     | 30 Epochs     |
|:-:                    |:-:            |:-:            |:-:            |
| LSTM listwise         | 34.31060873   | 27.38061697   | 27.38061697   |
| Attention listwise    | 41.44368002   | 29.25171358   | 29.25171358   |
| LSTM LOCF             | 29.18684518   | 30.85417642   |**26.71249242**|
| Attention LOCF        | 69.76342095   | 34.68203168   | 47.2019048    |

First, comparing Seq-to-Seq with direct targets finds that in every case the direct target performs much better.  This makes sense as the model gets to train explicitly on the task it is evaluated on.  Interesting follow ups could be to have Seq-to-Seq predict the next 24 hourly readings instead of one in order to better predict the temperatures in that range.  A model could also be made with both targets as they seem to both have advantages.

Next is the imputations.  In the Seq-to-Seq targets, listwise deletion performs better for Multi-headed Attention and about the same as LOCF for LSTM.  For the direct target, interestingly LOCF was the clear winner for both models.  This would present an interesting challenge to a dual target approach because it is unclear if either method has a clear leg up or why one method works better than the other for a specific case.

Finally, let's compare LSTM to Multi-headed Attention.  In the direct target, with either imputation method, LSTM starts with higher loss but eventually wins.  In Seq-to-Seq targets, LSTM is always the winner.  This indicates that there is a strong sequential aspect to the data and there is no need to build the more complex attention relations.


# Features

- Download ISD Lite data for any number of stations and any number of years
- 4 different models with included performance results
- Preprocess data with either listwise deletion or last observation carried forward imputation.

# Installation

```
pip install -r requirements.txt
```

# Usage
- Models can be trained by running any of the attached jupyter notebook files in the training folder.  The notebooks expect files called train.csv and test.csv both in the proccessed_data folder.
- Downloading data can be done as follows.  Calling download_csv() in data_processing/read.py with the parameters of 
    - A list of years to pull from
    - The USAF IDs for the stations to go in the training data
    - The USAF IDs for the stations to go in the testing data
    - The base url 'https://www.ncei.noaa.gov/pub/data/noaa/isd-lite'
    - The output path
This will create a csv for your test and training data with all stations appended together. An example can be found in main.py.  

- Stations can be identified using this NOAA resource [here](https://www.ncei.noaa.gov/access/search/data-search/global-hourly?pageNum=1).  Remeber they are downloaded based on their USAF IDs

