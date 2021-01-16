#!/usr/bin/env python
# coding: utf-8

# Imports
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.models import Sequential

# Don't run this block of code if you aren't using Tensorflow-GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def normalize(values):
    """This function takes as input a set of values and outputs the normalized values, their mean and their standard
    deviation.

    :param values: Integer or float values that we want to normalize.
    :returns values: Normalized values.
             mean: Mean of the values.
             std: Standard deviation of the values."""

    mean = np.mean(values)
    values -= mean
    std = np.std(values)
    values /= std

    return values, mean, std


def create_sequences(values, time_steps):
    """This function takes as input a set of values and outputs the values in a time-series format.

    :param values: Integer or float values that we want to convert to time-series format.
    :param time_steps: Integer indicating the length of the time-series.
    :return output: An array of time-series values."""

    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i: (i + time_steps)])

    return np.expand_dims(output, axis=2)


def get_anomaly_indices(threshold, train_loss):
    """This function returns the indices of the anomalies.
    :param threshold: Integer or float indicating the threshold we above which we would consider data points to be
                      anomalies.
    :param train_loss: The training loss.
    :return: indices: The indices of the anomalies."""

    anomalies = (train_loss > threshold).tolist()
    flat_list = []
    for sublist in anomalies:
        for item in sublist:
            flat_list.append(item)
    anomaly_indices = [i for i, x in enumerate(flat_list) if x == True]

    return anomaly_indices


def compare(dataset, anomaly_indices):
    """This function plots the comparison of the anomalies with the original dataset.

    :param dataset: The dataset on which we want to make a comparison.
    :param anomaly_indices: The indices of the anomalies in the dataset."""

    anomalies = dataset.iloc[anomaly_indices]
    dataset_copy = dataset.copy()
    dataset_copy['value'] = None
    combine = anomalies.combine_first(dataset_copy)
    dates = dataset["timestamp"].to_list()
    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
    values = dataset["value"].to_list()
    plt.figure(figsize=(15, 7))
    plt.plot(dates, values, label="test data")  # Change
    dates = combine["timestamp"].to_list()
    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
    values = combine["value"].to_list()
    plt.plot(dates, values, label="anomalies", color="r")  # Change
    plt.legend()
    plt.show()


def dense(x__train):
    """This function creates the Dense auto-encoder model.

    :param x__train: The training data. It is used to set the input shape.
    :return autoencoder: The autoencoder model."""

    inputs = Input(shape=(x__train.shape[1], x__train.shape[2]))

    # Encoding
    x = Flatten()(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(128, activation='relu')(x)

    # Decoding
    x = Dense(128, activation='relu')(encoded)
    x = Dense(64, activation='relu')(x)
    x = Dense(32)(x)
    decoded = Reshape((32, 1))(x)

    autoencoder = Model(inputs, decoded)

    return autoencoder


def lstm_modify(threshold1, threshold2, train_loss):
    """This function returns the anomaly indices for a certain range of MAE loss.

    :param threshold1: Integer or float representing the first threshold value.
    :param threshold2: Integer or float representing the second threshold value.
    :param train_loss: Training loss.
    :return anomaly_indices: Indices of anomalies in the dataset."""

    anomalies = (threshold1 < train_loss).tolist()
    flat_list = []

    for sublist in anomalies:
        for item in sublist:
            flat_list.append(item)
    indices1 = [i for i, x in enumerate(flat_list) if x == True]

    anomalies2 = (train_loss < threshold2).tolist()
    flat_list2 = []

    for sublist2 in anomalies2:
        for item2 in sublist2:
            flat_list2.append(item2)
    indices2 = [i for i, x in enumerate(flat_list2) if x == True]

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    anomaly_indices = intersection(indices1, indices2)

    return anomaly_indices


cpu_utilization = pd.read_csv('ec2_cpu_utilization_5f5533.csv')
# Statistics about the original dataset.
print('\nCPU Utilization Dataset Shape:', cpu_utilization.shape)
print('\nFirst 5 rows of the CPU Utilization Dataset:\n', cpu_utilization.head())
print('\nCPU Utilization Dataset description:\n', cpu_utilization.describe())

# Visualize the data.
plt.figure(figsize=(50, 10))
plt.plot(cpu_utilization['value'].values)
plt.xlabel('Timestamps (Each value represents a 5 minute interval.)', fontsize=28)
plt.ylabel('CPU Utilization', fontsize=28)
plt.title('CPU Utilization Over a Period of Time', fontsize=32)
plt.show()

cpu_utilization_values = cpu_utilization.value.to_list()
# print(cpu_utilization_values)

normalized_cpu_utilization_values, training_mean, training_std = normalize(cpu_utilization_values)

x_train = create_sequences(normalized_cpu_utilization_values, time_steps=32)
print("Training data shape: ", x_train.shape)

# Conv1D based auto-encoder model.
model = keras.Sequential([layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
                          layers.Conv1D(filters=18, kernel_size=7, padding="same", strides=2, activation="relu"),
                          layers.Dropout(rate=0.2),
                          layers.Conv1D(filters=9, kernel_size=7, padding="same", strides=2, activation="relu"),
                          layers.Conv1DTranspose(filters=9, kernel_size=7, padding="same", strides=2,
                                                 activation="relu"),
                          layers.Dropout(rate=0.2),
                          layers.Conv1DTranspose(filters=18, kernel_size=7, padding="same", strides=2,
                                                 activation="relu"),
                          layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same")])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")])

# Plotting the training and validation losses.
plt.figure(figsize=(15, 7))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("Number of Samples")
plt.show()

# Reconstructing the first window to see how well the model performs.
plt.plot(x_train[0], 'red')
plt.plot(x_train_pred[0], 'blue')
plt.show()


# Plotting the comparison of the dataset with the anomalies.
anomalies_index = get_anomaly_indices(threshold=0.245, train_loss=train_mae_loss)
compare(cpu_utilization, anomalies_index)


# LSTM based auto-encoder model.
model2 = Sequential()
model2.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model2.add(LSTM(64, activation='relu', return_sequences=False))
model2.add(RepeatVector(x_train.shape[1]))
model2.add(LSTM(64, activation='relu', return_sequences=True))
model2.add(LSTM(128, activation='relu', return_sequences=True))
model2.add(TimeDistributed(Dense(x_train.shape[2])))
model2.compile(optimizer='adam', loss='mse')
model2.summary()


history2 = model2.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.1).history


# Plotting the training and validation losses.
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history2['loss'], 'b', label='Train', linewidth=2)
ax.plot(history2['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


# Get train MAE loss.
x_train_pred2 = model2.predict(x_train)
train_mae_loss2 = np.mean(np.abs(x_train_pred2 - x_train), axis=1)

plt.hist(train_mae_loss2, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("Number of Samples")
plt.show()

# We select the threshold which filters most of smaller MAE loss and it doesn't perform as well as Conv1D model.
anomalies_index2 = get_anomaly_indices(threshold=0.6, train_loss=train_mae_loss2)
compare(cpu_utilization, anomalies_index2)

# We select only the second bell curve and find that the second curve represents the range of dataset which has higher
# values. 0.2 is the demarcation point between first and second bell curve.
anomalies_index3_1 = get_anomaly_indices(threshold=0.2, train_loss=train_mae_loss2)
compare(cpu_utilization, anomalies_index3_1)

# We pick the first bell curve (0.1~0.19) and find that it represents the range of lower values
# (which is actually cpu utilization) in original dataset.
anomalies_index3_2 = lstm_modify(threshold1=0.1, threshold2=0.19, train_loss=train_mae_loss2)
compare(cpu_utilization, anomalies_index3_2)


# Dense autoencoder model.
model3 = dense(x_train)
model3.compile(optimizer='adam', loss='mae')
model3.summary()

history3 = model3.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.1).history


# Plotting the training and validation losses.
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history3['loss'], 'b', label='Train', linewidth=2)
ax.plot(history3['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


x_train_pred3 = model3.predict(x_train)
train_mae_loss3 = np.mean(np.abs(x_train_pred3 - x_train), axis=1)

plt.hist(train_mae_loss3, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("Number of Samples")
plt.show()


# Plotting the comparison of the anomalies and the original dataset.
anomalies_index4 = get_anomaly_indices(threshold=0.28, train_loss=train_mae_loss3)
compare(cpu_utilization, anomalies_index4)
