from sklearn import datasets, preprocessing
import pandas as pd
import numpy as np
import math

import options

def load_skearn_blobs(sequence_lenght):
    x, y = datasets.make_blobs(
        n_samples=1000,
        n_features=4,
        centers=2,
        center_box=(-4.0, 4.0),
        cluster_std=1.75,
        random_state=42)

    df = pd.concat([pd.DataFrame(x), pd.DataFrame({'anomaly': y})], axis=1)
    normal_events = df[df['anomaly'] == 0]
    abnormal_events = df[df['anomaly'] == 1]

    normal_events = normal_events.loc[:, normal_events.columns != 'anomaly']
    abnormal_events = abnormal_events.loc[:, abnormal_events.columns != 'anomaly']

    # We keep 80% of the normal events to train the network
    n = 80
    remaining_normal_events = normal_events.iloc[::-1].head(int(len(normal_events)*((100-n)/100)))
    normal_events = normal_events.head(int(len(normal_events)*(n/100)))

    # Normalize the dataset
    scaler = preprocessing.MinMaxScaler()
    normal_events = scaler.fit_transform(normal_events)
    remaining_normal_events = scaler.transform(remaining_normal_events)
    abnormal_events = scaler.transform(abnormal_events)

    # We reorganize the data depending of the sequence lenght

    normal_events = pd.DataFrame(normal_events)
    remaining_normal_events = pd.DataFrame(remaining_normal_events)
    abnormal_events = pd.DataFrame(abnormal_events)

    if normal_events.shape[0] - math.floor(normal_events.shape[0] / sequence_lenght) * sequence_lenght != 0:
        normal_events = normal_events.head(- (normal_events.shape[0] - math.floor(normal_events.shape[0] / sequence_lenght) * sequence_lenght))
    if remaining_normal_events.shape[0] - math.floor(remaining_normal_events.shape[0] / sequence_lenght) * sequence_lenght != 0:
        remaining_normal_events = remaining_normal_events.head(- (remaining_normal_events.shape[0] - math.floor(remaining_normal_events.shape[0] / sequence_lenght) * sequence_lenght))
    if abnormal_events.shape[0] - math.floor(abnormal_events.shape[0] / sequence_lenght) * sequence_lenght != 0:
        abnormal_events = abnormal_events.head(- (abnormal_events.shape[0] - math.floor(abnormal_events.shape[0] / sequence_lenght) * sequence_lenght))
    
    normal_events = np.asarray(normal_events)
    abnormal_events = np.asarray(abnormal_events)
    remaining_normal_events = np.asarray(remaining_normal_events)

    x_train = normal_events.reshape(int(normal_events.shape[0] / sequence_lenght), sequence_lenght , normal_events.shape[1])
    abnormal_events = abnormal_events.reshape(int(abnormal_events.shape[0] / sequence_lenght), sequence_lenght , abnormal_events.shape[1])
    remaining_normal_events = remaining_normal_events.reshape(int(remaining_normal_events.shape[0] / sequence_lenght), sequence_lenght , remaining_normal_events.shape[1])

    # We test with 20% of the normal events and 100% of the abnormal events 
    x_test = np.concatenate([remaining_normal_events, abnormal_events], axis=0)

    # We generate the ground truth
    y_train = np.concatenate([np.zeros(x_train.shape)])
    y_test = np.concatenate([np.zeros(remaining_normal_events.shape), np.ones(abnormal_events.shape)])

    return x_train, y_train, x_test, y_test, remaining_normal_events.shape[0]

