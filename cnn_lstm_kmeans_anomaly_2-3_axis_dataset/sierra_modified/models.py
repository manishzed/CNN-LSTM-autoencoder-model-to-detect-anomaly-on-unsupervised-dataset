import tensorflow as tf
#import larq as lq
import tensorflow_model_optimization as tfmot

import tensorflow as tf
import layers
import numpy as np
import uuid
def autoencoder_CNN_LSTM_LQ_Softweb(X):


    input_dim = X.shape[1]

    training =True
    inp = tf.placeholder(shape=[None,input_dim], dtype=tf.float32)
    fc1 = layers.binaryDense(inp, 16, activation=None, name="binarydense1", binarize_input=False)
    bn1 = tf.layers.batch_normalization(fc1, training=training)
    ac1 = tf.clip_by_value(bn1, -1, 1)
    fc2 = layers.binaryDense(ac1, 4, activation=None, name="binarydense2")
    bn2 = tf.layers.batch_normalization(fc2, training=training)
    ac2 = tf.clip_by_value(bn2, -1, 1)
    fc3 = layers.binaryDense(ac2, 4, activation=None, name="binarydense3")
    bn3 = tf.layers.batch_normalization(fc3, training=training)
    ac3 = tf.clip_by_value(bn3, -1, 1)
    fc4 = layers.binaryDense(ac3, 16, activation=None, name="binarydense4")
    bn4 = tf.layers.batch_normalization(fc4, training=training)
    ac4 = tf.clip_by_value(bn4, -1, 1)
    fc5 = layers.binaryDense(ac4, input_dim, activation=None, name="binarydense5")
    output =  tf.layers.batch_normalization(fc5, training=training)
        

    #cost = tf.reduce_mean((output - inp)**2)
    #optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

    learning_rate = 0.01
    n_epochs = 1000
    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(output - inp, 2))
    #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        uniq_id = "tensorboard-layers-api/" + uuid.uuid1().__str__()[:6]
        summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
        x_vals = np.random.normal(0, 1, (10000, input_dim))
        
        for step in range(1, n_epochs + 1):
            _, val, summary = sess.run([optimizer, loss, merged_summary_op],
                                       feed_dict={inp: x_vals})
            
            print("epoch: {}, loss: {}".format(step, val))
            summary_writer.add_summary(summary, step)
        print('Done writing the summaries')
    
    """
    initializer = 'he_uniform'
    
    input_layer = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    conv1 = lq.layers.QuantConv1D(filters=16,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(input_layer)
    lstm1 = tf.keras.layers.LSTM(16, return_sequences=True)(conv1)
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[2]))(lstm1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model
    """
    


def autoencoder_CNN_LTSM_Softweb(X):
    input_layer = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    conv1 = tf.keras.layers.Conv1D(filters=8,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(input_layer)
    conv2 = tf.keras.layers.Conv1D(16, activation='relu', kernel_size=4, padding='same')(conv1)
    conv3 = tf.keras.layers.Conv1D(16, activation='relu', kernel_size=4, padding='same')(conv2)
    conv4 = tf.keras.layers.Conv1D(32, activation='relu', kernel_size=4, padding='same')(conv3)

    L1 = tf.keras.layers.LSTM(8, activation='relu', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(conv4)
    L2 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=True)(L1)
    #L2 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=False)(L1)
    #L3 = tf.keras.layers.RepeatVector(X.shape[1])(L2)
    L3 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=True)(L2)
    L4 = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)(L3)

    D1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8, activation='relu'))(L4)
    D1 = tf.keras.layers.Dropout(0.5)(D1)
    D2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16, activation='relu'))(D1)
    D2 = tf.keras.layers.Dropout(0.5)(D2)
    D3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16, activation='relu'))(D2)
    D3 = tf.keras.layers.Dropout(0.5)(D3)
    D4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(D3)
    D4 = tf.keras.layers.Dropout(0.5)(D4)

    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[2]))(D4)
    model =  tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def autoencoder_LTSM_Softweb(X):
    
    initializer = 'he_uniform'
    input_layer = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    
    lstm1 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=tf.keras.regularizers.l2(0.00))(input_layer)
    
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[2]))(lstm1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model
    
    """
    inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    L1 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=tf.keras.regularizers.l2(0.00))(inputs)
    L2 = tf.keras.layers.LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = tf.keras.layers.RepeatVector(X.shape[1])(L2)
    L4 = tf.keras.layers.LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = tf.keras.layers.LSTM(16, activation='sigmoid', return_sequences=True)(L4)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[2]))(L5)    
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
    """
    
def autoencoder_CNN_Softweb(X):
    initializer = 'he_uniform'
    Quant = lq.quantizers.SteSign(clip_value=1.)
    Weight = lq.constraints.WeightClip(clip_value=1.)
    kwargs = dict(kernel_quantizer=Quant,
    kernel_constraint=Weight,
    input_quantizer=Quant,
    use_bias=True)
    """
    inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    L1 = tf.keras.layers.Conv1D(16, 4, activation='relu', padding='same')(inputs)
    L2 = tf.keras.layers.Conv1D(4, 4, activation='relu', padding='same')(L1)
    L3 = tf.keras.layers.Conv1D(4, 4, activation='relu', padding='same')(L2)
    L4 = tf.keras.layers.Conv1D(16, 4, activation='relu', padding='same')(L3)
    output = tf.keras.layers.Dense(X.shape[2])(L4)   

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
    """
    inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    L1 = lq.layers.QuantConv1D(16, activation='relu', kernel_size=4, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.00), **kwargs )(inputs)
    d1 = tf.keras.layers.SpatialDropout1D(0.01)(L1)
    L2 = lq.layers.QuantConv1D(4, activation='relu', kernel_size=4, padding='same', **kwargs)(d1)
    d2 = tf.keras.layers.SpatialDropout1D(0.01)(L2)
    L4 = lq.layers.QuantConv1D(4, activation='relu', kernel_size=4, padding='same', **kwargs)(d2)
    d3 = tf.keras.layers.SpatialDropout1D(0.01)(L4)
    L5 = lq.layers.QuantConv1D(16, activation='relu', kernel_size=4, padding='same', **kwargs)(d3)
    d4 = tf.keras.layers.SpatialDropout1D(0.01)(L5)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[2]))(d4)    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def autoencoder_CNN_witekio(X):
    initializer = 'he_uniform'
    filter_size = 4
    nbr_filter = 16
    dropout = 0.01
    Quant = lq.quantizers.SteSign(clip_value=1.)
    Weight = lq.constraints.WeightClip(clip_value=1.)
    kwargs = dict(kernel_quantizer=Quant,
    kernel_constraint=Weight,
    input_quantizer=Quant,
    use_bias=True)

    inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2]))
    x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv1D(nbr_filter, filter_size, activation='relu', padding='causal', dilation_rate = 1,
                kernel_initializer=initializer))(inputs)
    shortcut = x

    x = lq.layers.QuantConv1D(nbr_filter, filter_size, activation='relu', padding='causal', dilation_rate = 2, kernel_initializer=initializer, **kwargs)(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.SpatialDropout1D(dropout)(x)
    shortcut = x

    x = lq.layers.QuantConv1D(nbr_filter, filter_size, activation='relu', padding='causal', dilation_rate = 4,
                kernel_initializer=initializer, **kwargs)(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.SpatialDropout1D(dropout)(x)
    shortcut = x

    x = lq.layers.QuantConv1D(nbr_filter, filter_size, activation='relu', padding='causal', dilation_rate = 8,
                kernel_initializer=initializer, **kwargs)(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.SpatialDropout1D(dropout)(x)

    decoded = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv1D(X.shape[2], filter_size, activation='sigmoid', padding='causal', 
                kernel_initializer=initializer))(x)

    model = tf.keras.Model(inputs=inputs, outputs=decoded)
    return model