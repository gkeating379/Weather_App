import pandas as pd


def compare_stats(df_train, df_test, out_path):
    '''Creates a comparission of the train
    and test data summary statistics
    
    Returns the test summary stat - train summary stat'''
    train_sum = df_train.describe()
    test_sum = df_test.describe()

    dif = test_sum - train_sum

    dif.to_csv(out_path)
