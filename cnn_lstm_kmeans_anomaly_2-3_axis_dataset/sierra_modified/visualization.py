from sklearn import decomposition

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.io import output_file, show,  save
import bokeh.layouts as bl
from bokeh.plotting import figure
import numpy as np

import options

def scatter_plot(x, y, error_report, network_name, dataset_name, sequence_lenght):

    pca = decomposition.PCA(n_components=2)
    pca_result = pca.fit_transform(x)

    pca_df = pd.DataFrame(data=pca_result, columns=['pc_1', 'pc_2'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'label': y})], axis=1)

    sns.scatterplot(x=pca_df.pc_1, y=pca_df.pc_2, hue=error_report, style=y, legend="full", data=pca_df)
    plt.savefig(options.PATH_RESULTS + "/scatter_plot_" + network_name + "_" + dataset_name + "_seqlenght_" + str(sequence_lenght) + ".png")

def linear_plot(sensor_real, sensor_reconstructed, y, x_train_range, loss, error_trigger, network_name, dataset_name, sequence_lenght):
    
    sensor_real = np.transpose(sensor_real, (1, 0))
    sensor_reconstructed = np.transpose(sensor_reconstructed, (1, 0))
    
    nbr_sensors = sensor_real.shape[0]
    x_range = (0, sensor_real.shape[1])
    p_sensor = [figure() for f in range(nbr_sensors)]
    plots = []
    TimeAxis = np.arange(sensor_real.shape[1])

    p_loss = figure(plot_width=800, plot_height=400, x_axis_type="linear", x_range=x_range, title='Anomaly detection')
    p_loss.circle(TimeAxis[loss<=error_trigger], loss[loss<=error_trigger], color="green", alpha=0.5,  legend_label="loss_noanomaly")
    p_loss.circle(TimeAxis[loss>error_trigger], loss[loss>error_trigger], color="red", alpha=0.5,  legend_label="loss_anomaly")
    line_error = np.repeat(error_trigger, loss.shape[0])
    p_loss.line(TimeAxis, line_error, color="blue", line_width=2,  legend_label="error limit")
    p_loss.quad(top=[np.max(loss)], bottom=[0], left=[x_train_range], right=[np.unique(y, return_counts=True)[1][0]], fill_alpha=0.2, color="#B3DE69")
    p_loss.quad(top=[np.max(loss)], bottom=[0], left=[np.unique(y, return_counts=True)[1][0]], right=[y.size], fill_alpha=0.2, color="red")
    plots.append(p_loss)

    for i in range(nbr_sensors):
        p_sensor[i] = figure(plot_width=800, plot_height=400, x_axis_type="linear", x_range=p_loss.x_range, title='Sensor ' + str(i))
        p_sensor[i].line(TimeAxis, sensor_real[i], color="blue", line_width=2,  alpha=0.5, legend_label="real")
        p_sensor[i].line(TimeAxis, sensor_reconstructed[i], color="red", line_width=2, alpha=0.5, legend_label="reconstructed")
        p_sensor[i].quad(top=[np.max([np.max(sensor_real[i]), np.max(sensor_reconstructed[i])])], bottom=[np.min([np.min(sensor_real[i]), np.min(sensor_reconstructed[i])])], left=[x_train_range], right=[np.unique(y, return_counts=True)[1][0]], fill_alpha=0.2, color="#B3DE69")
        p_sensor[i].quad(top=[np.max([np.max(sensor_real[i]), np.max(sensor_reconstructed[i])])], bottom=[np.min([np.min(sensor_real[i]), np.min(sensor_reconstructed[i])])], left=[np.unique(y, return_counts=True)[1][0]], right=[y.size], fill_alpha=0.2, color="red")
        plots.append(p_sensor[i])

    layout = bl.column(*plots)
    output_file(options.PATH_RESULTS + "/linear_plot_" + network_name + "_" + dataset_name + "_seqlenght_" + str(sequence_lenght) + ".html")
    save(layout)
