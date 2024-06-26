import numpy as np
import pandas as pd
from datetime import datetime, timedelta

df = df.replace('', np.nan)
df = df.fillna(method='ffill')
df = df.dropna()


def df_to_np_carry(df, ids, start_year, end_year, context_window):
    '''Converts a dataframe to an np array ready
    to be passed for training/testing

    df => dataframe with ISD Lite
    ids => List[str] of station ids
    start_year => int start of time range
    end_year => int end of time range
    context_window => int, number of previous results used to predict next'''

    blocks = []
    targets = []

    df.index = pd.MultiIndex.from_tuples([(id, datetime(year, month, day, hour))
                                          for id, year, month, day, hour in df.index])

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31, 23)

    for i, id in enumerate(ids):
        print(f'{i+1}/{len(ids)}')
        id = int(id)
        current_date = start_date
        block = []
        while current_date <= end_date:
            if (id, current_date) in df.index:
                cur = df.loc[(id, current_date)]
                block.append(cur.values)

                if len(block) == context_window:
                    tmp = np.array(block)
                    targets.append(tmp[-1, -2:])
                    blocks.append(tmp[:, :-2])
                    # no overlaps
                    # block = []
                    # overlaps
                    block = block[1:]

            current_date += timedelta(hours=1)

    return np.array(blocks), np.array(targets)
