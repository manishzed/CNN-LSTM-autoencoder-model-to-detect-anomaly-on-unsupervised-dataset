from sklearn import datasets, preprocessing
import pandas as pd
import numpy as np
import math

import options

def load_drillpress(sequence_lenght):

    # read training data 
    train_df = pd.read_csv('./DrillPressData/base.csv', sep=",", header=0)
    train_df.drop(train_df.columns[[0]], axis=1, inplace=True)

    # We keep 80% of the normal events to train the network
    n = 80
    remaining_normal_events = train_df.iloc[::-1].head(int(len(train_df)*((100-n)/100)))
    normal_events = train_df.head(int(len(train_df)*(n/100)))

    if (normal_events.shape[0] - math.floor(normal_events.shape[0] / sequence_lenght) * sequence_lenght) != 0:
        normal_events = normal_events.head(- (normal_events.shape[0] - math.floor(normal_events.shape[0] / sequence_lenght) * sequence_lenght))
    if (remaining_normal_events.shape[0] - math.floor(remaining_normal_events.shape[0] / sequence_lenght) * sequence_lenght) != 0:
        remaining_normal_events = remaining_normal_events.head(- (remaining_normal_events.shape[0] - math.floor(remaining_normal_events.shape[0] / sequence_lenght) * sequence_lenght))

    x_train = np.asarray(normal_events)
    # Normalize dataset
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)

    # reshape the dataset
    x_train = x_train.reshape(int(x_train.shape[0] / sequence_lenght), sequence_lenght , x_train.shape[1])
    y_train = np.zeros(x_train.shape)
    x_train = np.asarray(x_train) 

    # read test data 
    # dry run / imbalance 1 / imbalance 2
    test_df_dryrun = pd.read_csv('./DrillPressData/dry run.csv', sep=",", header=0)
    test_df_imbalance1 = pd.read_csv('./DrillPressData/dry run.csv', sep=",", header=0)
    test_df_imbalance2 = pd.read_csv('./DrillPressData/dry run.csv', sep=",", header=0)
    test_df = pd.concat([test_df_dryrun, test_df_imbalance1, test_df_imbalance2])

    test_df.drop(test_df.columns[[0]], axis=1, inplace=True)
    if (test_df.shape[0] - math.floor(test_df.shape[0] / sequence_lenght) * sequence_lenght) != 0:
        test_df = test_df.head(- (test_df.shape[0] - math.floor(test_df.shape[0] / sequence_lenght) * sequence_lenght))

    # Normalize data
    remaining_normal_events = scaler.transform(remaining_normal_events)
    test_df = scaler.transform(test_df)

    # reshape the dataset
    remaining_normal_events = np.asarray(remaining_normal_events).reshape(int(remaining_normal_events.shape[0] / sequence_lenght), sequence_lenght , remaining_normal_events.shape[1])
    test_df = np.asarray(test_df).reshape(int(test_df.shape[0] / sequence_lenght), sequence_lenght , test_df.shape[1])

    # We test with 20% of the normal events and 100% of the abnormal events 
    x_test = np.concatenate([remaining_normal_events, test_df])
   
    y_test = np.concatenate([np.zeros(remaining_normal_events.shape), np.ones(test_df.shape)])

    return x_train, y_train, x_test, y_test, remaining_normal_events.shape[0]
