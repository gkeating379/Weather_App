import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def df_to_np_hl(df, ids, start_year, end_year, context_window, overlap=False):
    '''Converts a dataframe to an np array ready
    to be passed for training/testing
    FOR High/Low Targeting

    df => dataframe with ISD Lite
    ids => List[str] of station ids
    start_year => int start of time range
    end_year => int end of time range
    context_window => int, number of previous results used to predict next
    overlap => bool, if a block is cw away from the last block or 1'''

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
                    if not overlap:
                        block = []
                    else:
                        block = block[1:]

            current_date += timedelta(hours=1)

    return np.array(blocks), np.array(targets)


def df_to_np_seq(df, ids, start_year, end_year, context_window, overlap=False):
    '''Converts a dataframe to an np array ready
    to be passed for training/testing
    FOR Sequence to Sequence Targetting

    df => dataframe with ISD Lite
    ids => List[str] of station ids
    start_year => int start of time range
    end_year => int end of time range
    context_window => int, number of previous results used to predict next
    overlap => bool, if a block is cw away from the last block or 1'''

    blocks = []
    targets = []
    hi_lows = []

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

                if len(block) == context_window + 1:
                    tmp = np.array(block)
                    targets.append(tmp[1:, :-2])
                    blocks.append(tmp[:-1, :-2])
                    hi_lows.append(tmp[context_window-1, -2:])
                    if not overlap:
                        block = []
                    else:
                        block = block[1:]

            current_date += timedelta(hours=1)

    return np.array(blocks), np.array(targets), np.array(hi_lows)
