from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
import scipy as sp
from scipy import signal

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging

import options

if options.DATASET_SKLEARN_BLOBS == True:
    import data_sklearn_blob as data
elif options.DATASET_MOTION == True:
    import data_motion as data
elif options.DATASET_DRILLPRESS == True:
    import data_drillpress as data

import models
import report
#import tools
import visualization
import datetime

# Set the different seeds for reproducibility
seed(10)
set_random_seed(10)

# Setup the logging
LOG_LEVEL = logging.INFO

logging.basicConfig(level=LOG_LEVEL, 
                    format='%(asctime)s %(levelname)-8s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Check if the results directory exists, if not create it
if not os.path.isdir(options.PATH_RESULTS):
    os.mkdir(options.PATH_RESULTS)

# Read the dataset
if options.DATASET_SKLEARN_BLOBS == True:
    epochs = 100
    if options.MODEL_KMEANS == True:
        sequence_lenght = 1
    else:
        sequence_lenght = 2
    # In percentage (0 to 100%), how many samples are displayed if explainable AI is used
    # (train, test no anomaly, test anomaly)
    explainable_ai_size = (100, 100, 100)
    x_train, y_train, x_test, y_test, test_split = data.load_skearn_blobs(sequence_lenght)
    dataset_name = "SKLEARN_BLOBS"
elif options.DATASET_MOTION == True:
    epochs = 100
    if options.MODEL_KMEANS == True:
        sequence_lenght = 1
    else:
        sequence_lenght = 16
    # In percentage (0 to 100%), how many samples are displayed if explainable AI is used
    # (train, test no anomaly, test anomaly)
    explainable_ai_size = (100, 100, 100)
    x_train, y_train, x_test, y_test, test_split = data.load_motion(sequence_lenght)
    dataset_name = "MOTION"
elif options.DATASET_DRILLPRESS == True:
    epochs = 100
    if options.MODEL_KMEANS == True:
        sequence_lenght = 1
    else:
        sequence_lenght = 64
    # In percentage (0 to 100%), how many samples are displayed if explainable AI is used
    # (train, test no anomaly, test anomaly)
    explainable_ai_size = (100, 100, 100)
    x_train, y_train, x_test, y_test, test_split = data.load_drillpress(sequence_lenght)
    dataset_name = "DRILLPRESS"
print(dataset_name)
print(x_train[0], y_train[0], x_test[0], y_test[0], test_split)
# Select the model
if options.MODEL_SOFTWEB_CNN == True:
    model = models.autoencoder_CNN_Softweb(x_train)
    network_name = "SOFTWEB_CNN"
elif options.MODEL_SOFTWEB_LTSM == True:
    model = models.autoencoder_LTSM_Softweb(x_train)
    network_name = "SOFTWEB_LTSM"
elif options.MODEL_SOFTWEB_CNN_LSTM_LQ == True:
    model = models.autoencoder_CNN_LSTM_LQ_Softweb(x_train)
    network_name = "SOFTWEB_CNN_LTSM_LQ"
    
elif options.MODEL_SOFTWEB_CNN_LSTM == True:
    model = models.autoencoder_CNN_LTSM_Softweb(x_train)
    network_name = "SOFTWEB_CNN_LTSM"
    
elif options.MODEL_WITEKIO_CNN == True:
    model = models.autoencoder_CNN_witekio(x_train)
    network_name = "WITEKIO_CNN"
elif options.MODEL_KMEANS == True:
    x_train = np.average(x_train, axis=1)
    x_test = np.average(x_test, axis=1)
    y_train = np.average(y_train, axis=1)
    y_test = np.average(y_test, axis=1)
    model = KMeans(n_clusters=2, random_state=0)
    network_name = "OCTONION_KMEANS"

fh = logging.FileHandler(r'' + options.PATH_RESULTS + "/report_" + network_name + "_" + dataset_name + "_seqlenght_" + str(sequence_lenght) + ".txt")
logger.addHandler(fh)

logger.info("Start training")

if (options.MODEL_KMEANS != True):
    # Setup the optimizer
    optimizer=tf.keras.optimizers.Adam(0.001)

    # Compile the model (loss function mae)
    model.compile(optimizer=optimizer, loss='mae')

    # Display the architecture of the selected network
    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard()
    
    #time calculate
    t_ini = datetime.datetime.now()

    # Train the network
    model.fit(x_train, x_train, epochs=epochs, batch_size=10, validation_split=0.05, callbacks=[tensorboard_callback]).history
    
    t_fin = datetime.datetime.now()
    print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))

else:
    #time calculate
    t_ini = datetime.datetime.now()
    model.fit(x_train)
    
    t_fin = datetime.datetime.now()
    print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))


if (options.MODEL_KMEANS != True):
    # Predict with x_train
    x_pred_train = model.predict(x_train)

    # Compute the loss function (mae)
    loss_train = abs(x_pred_train - x_train)

    # Compute the average value per sequence lenght
    loss_train = np.average(loss_train, axis=1)
    # Compute the average value for all sensors
    loss_train = np.average(loss_train, axis=1)

    # Compute what is the maximum loss value reached with the training dataset
    if options.DATASET_DRILLPRESS == True:
        # Remove outliers
        loss_train = sp.signal.medfilt(loss_train,3)

    train_max_loss = np.max(loss_train)

# Predict with x_test
x_pred_test = model.predict(x_test)

# Create a vector set to no anomaly at the beginning of error report
no_error = np.full((x_train.shape[0]), False, dtype=bool)

if (options.MODEL_KMEANS != True):
    # Compute the loss function (mae)
    loss_test = abs(x_pred_test - x_test)

    # Compute the average error for a sequence lenght
    loss_test = np.average(loss_test, axis=1)
    # Compute the average value for all sensors
    loss_test = np.average(loss_test, axis=1)

    # Compare the test error with the maximum error reached with the training dataset
    x_pred_error = np.greater(loss_test, train_max_loss)

    error_report = np.concatenate([no_error, x_pred_error])
    error_report = 1*error_report
    x_pred = x_pred_error
else:
    error_report = np.concatenate([no_error, x_pred_test])
    x_pred = x_pred_test

# Concatenate the data for the different visalutizations/reports
if (options.MODEL_KMEANS != True):
    x = np.concatenate([np.average(x_train, axis=1), np.average(x_test, axis=1)])
    x_reconstructed = np.concatenate([np.average(x_pred_train, axis=1), np.average(x_pred_test, axis=1)])
    loss = np.concatenate([loss_train, loss_test]) 
    y = np.average(np.concatenate([np.average(y_train, axis=1), np.average(y_test, axis=1)]), axis=1)
    x_train_range = x_train.shape[0]
else:
    x = np.concatenate([x_train, x_test])
    y = np.average(np.concatenate([y_train, y_test]), axis=1)

# Visualize the dataset (after dimension reduction) as a scatter plot
visualization.scatter_plot(x, y, error_report, network_name, dataset_name, sequence_lenght) 
print(y, error_report)
# Save a report with the different metrics of the network
report.tofile(logger, y, error_report, network_name, dataset_name)  

# Visualize the dataset (sensor original and reconstructed) + the loss function + the error trigger
if (options.MODEL_KMEANS != True):
    if options.EXPLAINABLE_AI == True:

        x_train = x_train[:int(x_train.shape[0]*explainable_ai_size[0]/100)]
        x_test_noanomaly = x_test[:int(test_split*explainable_ai_size[1]/100)]
        x_test_anomaly = x_test[test_split:int(x_test.shape[0]*explainable_ai_size[2]/100)]
        x_test = np.concatenate([x_test_noanomaly, x_test_anomaly])

        x_pred_train = x_pred_train[:int(x_pred_train.shape[0]*explainable_ai_size[0]/100)]
        x_pred_test_noanomaly = x_pred_test[:int(test_split*explainable_ai_size[1]/100)]
        x_pred_test_anomaly = x_pred_test[test_split:int(x_pred_test.shape[0]*explainable_ai_size[2]/100)]
        x_pred_test = np.concatenate([x_pred_test_noanomaly, x_pred_test_anomaly])

        y_train = y_train[:int(y_train.shape[0]*explainable_ai_size[0]/100)]
        y_test_noanomaly = y_test[:int(test_split*explainable_ai_size[1]/100)]
        y_test_anomaly = y_test[test_split:int(y_test.shape[0]*explainable_ai_size[2]/100)]
        y_test = np.concatenate([y_test_noanomaly, y_test_anomaly])

        loss_train = loss_train[:int(loss_train.shape[0]*explainable_ai_size[0]/100)]
        loss_test_noanomaly = loss_test[:int(test_split*explainable_ai_size[1]/100)]
        loss_test_anomaly = loss_test[test_split:int(loss_test.shape[0]*explainable_ai_size[2]/100)]
        loss = np.concatenate([loss_train, loss_test_noanomaly, loss_test_anomaly])
        loss = loss.repeat(sequence_lenght)

        x = np.concatenate([x_train, x_test])
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        x_reconstructed = np.concatenate([x_pred_train, x_pred_test])
        x_reconstructed = x_reconstructed.reshape(x_reconstructed.shape[0]*x_reconstructed.shape[1], x_reconstructed.shape[2])
        y = np.concatenate([y_train, y_test])
        y = y.reshape(y.shape[0]*y.shape[1], y.shape[2])
        y = np.average(y, axis=1)
        x_train_range = x_train.shape[0] * x_train.shape[1]

        
    visualization.linear_plot(x, x_reconstructed, y, x_train_range, loss, train_max_loss, network_name, dataset_name, sequence_lenght)

  




