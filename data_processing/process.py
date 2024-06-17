import numpy as np
import pandas as pd


def df_to_np(df, ids, start_year, end_year):
    '''Converts a dataframe to an np array ready
    to be passed for training/testing'''
    blocks = []
    targets = []
    for id in ids:
        id = int(id)
        year = start_year
        month = 1
        day = 1
        hour = 0
        block = []
        while year <= end_year:
            if (id, year, month, day, hour) in df.index:
                cur = df.loc[(id, year, month, day, hour)]
                if cur['24hr Air Temp High'] != -9999:
                    block.append(cur.values)
                if len(block) == 48:
                    tmp = np.array(block)
                    targets.append(tmp[-1, -2:])
                    blocks.append(tmp[:, :-2])
                    block = []
            year, month, day, hour = next_time(year, month, day, hour)

    return blocks, targets


def next_time(year, month, day, hour):

    hour += 1
    if hour > 23:
        hour -= 24
        day += 1

    if day > 31:
        day -= 31
        month += 1

    if month > 12:
        month -= 12
        year += 1

    return year, month, day, hour
