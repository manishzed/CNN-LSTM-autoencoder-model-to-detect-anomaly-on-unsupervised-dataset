from sklearn import datasets, preprocessing
import pandas as pd
import numpy as np
import math
import os

WALK=["35_01","35_02","35_03","35_04","35_05","35_06","35_07","35_08","35_09","35_10",
      "35_11","35_12","35_13","35_14","35_15","35_16"]
RUN=["35_17","35_18","35_19","35_20","35_21","35_22","35_23","35_24","35_25","35_26"]

def read_mocap_file(file_path):
    timeseries=[]
    with open(file_path,"r") as f:
        for line in f.readlines():
            x=line.strip().split(" ")
            timeseries.append([float(xx) for xx in x])
    timeseries=np.array(timeseries)
    return timeseries

def load_motion(sequence_lenght):
    data_dir = 'MotionData'
    train_x = None
    test_x = None

    for walk in WALK[:-2]:
        ts=read_mocap_file(os.path.join(data_dir,"walk",walk+".amc.4d"))
        if train_x is None:
            train_x = ts
        else:
            train_x=np.concatenate([train_x,ts])
    
    train_df = pd.DataFrame(train_x)

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

    for run in RUN[:]:
        ts = read_mocap_file(os.path.join(data_dir, "run", run + ".amc.4d"))
        if test_x is None:
            test_x = ts
        else:
            test_x = np.concatenate([test_x, ts])

    # add jump test data for experiment
    ts = read_mocap_file(os.path.join(data_dir,"other","49_02.amc.4d"))
    test_x = np.concatenate([test_x, ts])

    test_df = pd.DataFrame(test_x)

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