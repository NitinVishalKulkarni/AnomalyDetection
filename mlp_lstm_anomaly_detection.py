#!/usr/bin/env python
# coding: utf-8

# Imports
import itertools
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, LSTM
from keras.models import Sequential
from keras.regularizers import l1, l1_l2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Don't run this block of code if you aren't using Tensorflow-GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function plots the confusion matrix."""

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.

    for a, b in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(b, a, cm[a, b],
                 horizontalalignment="center",
                 color="black" if cm[a, b] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Reading the dataset.
cpu_utilization = pd.read_csv('ec2_cpu_utilization_5f5533.csv')
cpu_utilization.set_index('timestamp', inplace=True)

# Statistics about the original dataset.
print('\nCPU Utilization Dataset Shape:', cpu_utilization.shape)
print('\nFirst 5 rows of the CPU Utilization Dataset:\n', cpu_utilization.head())
print('\nCPU Utilization Dataset description:\n', cpu_utilization.describe())

# Visualizing the original data.
plt.figure(figsize=(50, 10))
plt.plot(cpu_utilization['value'].values)
plt.xlabel('Timestamps (Each value represents a 5 minute interval.)', fontsize=28)
plt.ylabel('CPU Utilization', fontsize=28)
plt.title('CPU Utilization Over a Period of Time', fontsize=32)
plt.show()

# When CPU Utilization goes over 60 % AWS stars a new machine to prevent putting too much load on a single machine.
# Thus, we are treating loads above 60 % as being anomalies.


# Replacing the anomalies with the median value and creating a dataset with normal values.
normal_cpu_utilization = cpu_utilization.copy()
for i in range(len(cpu_utilization['value'])):
    if cpu_utilization['value'].iloc[i] > 60.0:
        normal_cpu_utilization['value'].iloc[i] = cpu_utilization['value'].median()

# Statistics about the normal dataset.
print('\nNormal CPU Utilization Dataset Shape:', normal_cpu_utilization.shape)
print('\nFirst 5 rows of the Normal CPU Utilization Dataset:\n', normal_cpu_utilization.head())
print('\nNormal CPU Utilization Dataset description:\n', normal_cpu_utilization.describe())

# Visualizing the normal data.
plt.figure(figsize=(50, 10))
plt.plot(normal_cpu_utilization['value'].values)
plt.xlabel('Timestamps (Each value represents a 5 minute interval.)', fontsize=28)
plt.ylabel('CPU Utilization', fontsize=28)
plt.title('Normal CPU Utilization Over a Period of Time', fontsize=32)
plt.show()

# Normalizing the data.
normal_cpu_utilization = normal_cpu_utilization.values
scaler = MinMaxScaler()
normal_cpu_utilization = scaler.fit_transform(normal_cpu_utilization)

# Creating the labels for plotting the confusion matrix.
cpu_utilization.insert(1, 'Label', '')
for i in range(len(cpu_utilization)):
    if cpu_utilization['value'][i] > 60.0:  # Remember we have assumed valued over 60 to be anomalous.
        cpu_utilization['Label'][i] = 'anomaly'
    else:
        cpu_utilization['Label'][i] = 'normal'

print('First 5 rows of the dataset after creating the labels:\n', cpu_utilization.head())

# # MLP Model #1
print("\n\nMLP Model 1\n\n")

# Specifying the window and prediction parameters.
window = 10
predict = 5

# Creating the lists to store the data and the labels.
x = []
y = []

for i in range(window, len(normal_cpu_utilization) + 1):
    x.append(normal_cpu_utilization[i - window:i, 0])

for i in range(predict, len(normal_cpu_utilization) + 1):
    y.append(normal_cpu_utilization[i - predict:i, 0])

x = x[:-predict]
y = y[window:]

# Printing some statistics about the data.
print('Shape X:', len(x))
print('Shape Y:', len(y))
print('Maximum Data Value:', np.max(x))

# Converting to array.
x = np.array(x)
y = np.array(y)

# Creating the train and test dataset.
x_train = x[:int(0.7 * len(x))]
y_train = y[:int(0.7 * len(y))]

x_test = x[int(0.7 * len(x)):]
y_test = y[int(0.7 * len(y)):]

# Printing some statistics about the data.
print('\nShape of the training dataset:', x_train.shape)
print('\n2 First two rows of the training data:\n', x_train[:2])
print('\n2 Last two rows of the training data:\n', x_train[-2:])

print('\nShape of the testing dataset:', x_test.shape)
print('\n2 First two rows of the testing data:\n', x_test[:2])
print('\n2 Last two rows of the testing data:\n', x_test[-2:])

print('\nShape of training data labels:', y_train.shape)
print('\n2 First two rows of the training data labels:\n', y_train[:2])
print('\n2 Last two rows of the training data labels:\n', y_train[-2:])

print('\nShape of testing data labels:', y_test.shape)
print('\n2 First two rows of the testing data labels:\n', y_test[:2])
print('\n2 Last two rows of the testing data labels:\n', y_test[-2:])

# MLP Model #1 with the loss as 'Mean Absolute Error'.
model = Sequential()
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(196))
model.add(Dropout(0.2))
model.add(Dense(predict))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# Fitting the MLP model on the training data.
history = model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(x_test, y_test))

# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()

# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='best')
plt.show()

# Making the predictions.
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

print('Predictions Shape:', predictions.shape)
print('\nFirst 10 Predictions:\n', predictions[:10])
print('\nFirst 10 Actual Values:\n', scaler.inverse_transform(y_test[:10]))
print('\nLast 10 Predictions:\n', predictions[-10:])
print('\nLast 10 Actual Values:\n', scaler.inverse_transform(y_test[-10:]))

# Getting the average value for all the predictions.
average_predictions = [predictions[0][0], (predictions[0][1] + predictions[1][0]) / 2,
                       (predictions[0][2] + predictions[1][1] + predictions[2][0]) / 3,
                       (predictions[0][3] + predictions[1][2] + predictions[2][1] + predictions[3][0]) / 4]

for i in range(len(predictions) - 4):
    average_predictions.append(
        (predictions[i][4] + predictions[i + 1][3] + predictions[i + 2][2] + predictions[i + 3][1] +
         predictions[i + 4][0]) / 5)

average_predictions.append((predictions[-1][1] + predictions[-2][2] + predictions[-3][3] + predictions[-4][4]) / 4)
average_predictions.append((predictions[-1][2] + predictions[-2][3] + predictions[-3][4]) / 3)
average_predictions.append((predictions[-1][3] + predictions[-2][4]) / 2)
average_predictions.append(predictions[-1][4])

print('\nAverage Predictions Length:', len(average_predictions))
# print('\nAverage Predictions:\n', average_predictions)


# Calculating the errors. # Here we identify the anomalies using Absolute Error.

# Creating the comparison dataset.
comparison_cpu_utilization = cpu_utilization['value'].values[len(x_train) + window:]

print('Length of the comparison dataset:', len(comparison_cpu_utilization))

absolute_errors = []
for i in range(len(average_predictions)):
    absolute_errors.append(abs(comparison_cpu_utilization[i] - average_predictions[i]))

# Printing some statistics about the errors.
print('\nMaximum Absolute Error Value:', max(absolute_errors))
print('\nMinimum Absolute Error Value', min(absolute_errors))
print('\nMean Absolute Error Value:', np.mean(absolute_errors))
print('\nMedian Absolute Error Value:', np.median(absolute_errors))
print('\nStandard Deviation of Absolute Error Values:', np.std(absolute_errors))

# Calculating the threshold.
threshold = np.mean(absolute_errors) + 2 * np.std(absolute_errors)
print('\nThreshold:', threshold)

# Plotting Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(average_predictions, label='Predicted CPU Utilization', color='red')
plt.legend()
plt.title('MLP Model #1: Actual CPU Utilization Vs Predicted CPU Utilization')
plt.show()

# Getting the anomaly indices.
anomalies = []
for absolute_error in absolute_errors:
    if absolute_error > threshold:
        anomalies.append(absolute_error)

print('Length of anomalies:', len(anomalies))

anomaly_indices = []
for anomaly in anomalies:
    anomaly_indices.append(absolute_errors.index(anomaly))
print('\nAnomaly Indices:', anomaly_indices)

# Creating the anomalies dataset.
anomalies_dataset = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        anomalies_dataset.append(comparison_cpu_utilization[i])
    else:
        anomalies_dataset.append(None)

# print((anomalies_dataset))


# Plotting Actual CPU Utilization vs Predicted Anomalies.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(anomalies_dataset, label='Anomalies', color='red')
plt.legend()
plt.title('MLP Model #1: Actual CPU Utilization Vs Predicted Anomalies')
plt.show()

# Getting the predicted classes for plotting the confusion matrix.
normal_anomaly_predictions = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        normal_anomaly_predictions.append('anomaly')
    else:
        normal_anomaly_predictions.append('normal')

# Getting the hypothesized classes for plotting the confusion matrix.
comparison_cpu_utilization_label = cpu_utilization['Label'].values[len(x_train) + window:]

# Plotting the confusion matrix
confusion_mtx = confusion_matrix(comparison_cpu_utilization_label, normal_anomaly_predictions)
plt.figure(figsize=(20, 20))
sns.set(font_scale=2.0)
plot_confusion_matrix(confusion_mtx, classes=['anomaly', 'normal'])

# # LSTM Model #1
print("\n\nLSTM Model 1\n\n")

# Reshaping.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# LSTM model #1 with the loss as 'Mean Absolute Error'.
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=196, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(predict))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# Fitting the model on the training data.
history = model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(x_test, y_test))

# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()

# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='best')
plt.show()

# Making the predictions.
predictions = (model.predict(x_test))
predictions = scaler.inverse_transform(predictions)

print('Predictions Shape:', predictions.shape)
print('\nFirst 10 Predictions:\n', predictions[:10])
print('\nFirst 10 Actual Values:\n', scaler.inverse_transform(y_test[:10]))
print('\nLast 10 Predictions:\n', predictions[-10:])
print('\nLast 10 Actual Values:\n', scaler.inverse_transform(y_test[-10:]))

# Getting the average value for all the predictions.
average_predictions = [predictions[0][0], (predictions[0][1] + predictions[1][0]) / 2,
                       (predictions[0][2] + predictions[1][1] + predictions[2][0]) / 3,
                       (predictions[0][3] + predictions[1][2] + predictions[2][1] + predictions[3][0]) / 4]

for i in range(len(predictions) - 4):
    average_predictions.append(
        (predictions[i][4] + predictions[i + 1][3] + predictions[i + 2][2] + predictions[i + 3][1] +
         predictions[i + 4][0]) / 5)

average_predictions.append((predictions[-1][1] + predictions[-2][2] + predictions[-3][3] + predictions[-4][4]) / 4)
average_predictions.append((predictions[-1][2] + predictions[-2][3] + predictions[-3][4]) / 3)
average_predictions.append((predictions[-1][3] + predictions[-2][4]) / 2)
average_predictions.append(predictions[-1][4])

print('\nAverage Predictions Length:', len(average_predictions))
# print('\nAverage Predictions:\n', average_predictions)


# Calculating the errors. # Here we identify the anomalies using Absolute Error.

# Creating the comparison dataset.
comparison_cpu_utilization = cpu_utilization['value'].values[len(x_train) + window:]
print('Length of the comparison dataset:', len(comparison_cpu_utilization))

absolute_errors = []
for i in range(len(average_predictions)):
    absolute_errors.append(abs(comparison_cpu_utilization[i] - average_predictions[i]))

print('\nMaximum Absolute Error Value:', max(absolute_errors))
print('\nMinimum Absolute Error Value', min(absolute_errors))
print('\nMean Absolute Error Value:', np.mean(absolute_errors))
print('\nMedian Absolute Error Value:', np.median(absolute_errors))
print('\nStandard Deviation of Absolute Error Values:', np.std(absolute_errors))

# Calculating the threshold.
threshold = np.mean(absolute_errors) + 2 * np.std(absolute_errors)
print('\nThreshold:', threshold)

# Plotting the Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(average_predictions, label='Predicted CPU Utilization', color='red')
plt.legend()
plt.title('LSTM Model #1: Actual CPU Utilization Vs Predicted CPU Utilization')
plt.show()

# Getting the anomaly indices.
anomalies = []
for absolute_error in absolute_errors:
    if absolute_error > threshold:
        anomalies.append(absolute_error)

print('Length of anomalies:', len(anomalies))

anomaly_indices = []
for anomaly in anomalies:
    anomaly_indices.append(absolute_errors.index(anomaly))
print('\nAnomaly Indices:', anomaly_indices)

# Creating the anomalies dataset.
anomalies_dataset = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        anomalies_dataset.append(comparison_cpu_utilization[i])
    else:
        anomalies_dataset.append(None)

# print((anomalies_dataset))


# Plotting Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(anomalies_dataset, label='Anomalies', color='red')
plt.legend()
plt.title('LSTM Model #1: Actual CPU Utilization Vs Predicted Anomalies')
plt.show()

# Getting the predicted classes for plotting the confusion matrix.
normal_anomaly_predictions = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        normal_anomaly_predictions.append('anomaly')
    else:
        normal_anomaly_predictions.append('normal')

# Getting the hypothesized classes for plotting the confusion matrix.
comparison_cpu_utilization_label = cpu_utilization['Label'].values[len(x_train) + window:]

# Plotting the confusion matrix.
confusion_mtx = confusion_matrix(comparison_cpu_utilization_label, normal_anomaly_predictions)
plt.figure(figsize=(20, 20))
sns.set(font_scale=2.0)
plot_confusion_matrix(confusion_mtx, classes=['anomaly', 'normal'])

# # MLP Model #2
print("\n\nMLP Model 2\n\n")

# Specifying the window and prediction parameters.
window = 144
predict = 5


# Creating the lists to store the data and the labels.
x = []
y = []
for i in range(window, len(normal_cpu_utilization) + 1):
    x.append(normal_cpu_utilization[i - window:i, 0])

for i in range(predict, len(normal_cpu_utilization) + 1):
    y.append(normal_cpu_utilization[i - predict:i, 0])

x = x[:-predict]
y = y[window:]


# Printing some statistics about the data.
print('Shape X:', len(x))
print('Shape Y:', len(y))
print('Maximum Data Value:', np.max(x))


# Converting to array.
x = np.array(x)
y = np.array(y)


# Creating the train and test dataset.
x_train = x[:int(0.7 * len(x))]
y_train = y[:int(0.7 * len(y))]

x_test = x[int(0.7 * len(x)):]
y_test = y[int(0.7 * len(y)):]


# Printing some statistics about the data.
print('\nShape of the training dataset:', x_train.shape)
print('\n2 First two rows of the training data:\n', x_train[:2])
print('\n2 Last two rows of the training data:\n', x_train[-2:])

print('\nShape of the testing dataset:', x_test.shape)
print('\n2 First two rows of the testing data:\n', x_test[:2])
print('\n2 Last two rows of the testing data:\n', x_test[-2:])

print('\nShape of training data labels:', y_train.shape)
print('\n2 First two rows of the training data labels:\n', y_train[:2])
print('\n2 Last two rows of the training data labels:\n', y_train[-2:])

print('\nShape of testing data labels:', y_test.shape)
print('\n2 First two rows of the testing data labels:\n', y_test[:2])
print('\n2 Last two rows of the testing data labels:\n', y_test[-2:])


# MLP Model #2 with the loss as 'Mean Squared Error'.
model = Sequential()
model.add(Dense(384, activity_regularizer=l1(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(256, activity_regularizer=l1(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(192, activity_regularizer=l1(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(predict))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# Fitting the MLP model on the training data.
history = model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=1, validation_data=(x_test, y_test))


# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()


# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='best')
plt.show()


# Making the predictions.
predictions = (model.predict(x_test))
predictions = scaler.inverse_transform(predictions)

print('Predictions Shape:', predictions.shape)
print('\nFirst 10 Predictions:\n', predictions[:10])
print('\nFirst 10 Actual Values:\n', scaler.inverse_transform(y_test[:10]))
print('\nLast 10 Predictions:\n', predictions[-10:])
print('\nLast 10 Actual Values:\n', scaler.inverse_transform(y_test[-10:]))


# Getting the average value for all the predictions.
average_predictions = [predictions[0][0], (predictions[0][1] + predictions[1][0]) / 2,
                       (predictions[0][2] + predictions[1][1] + predictions[2][0]) / 3,
                       (predictions[0][3] + predictions[1][2] + predictions[2][1] + predictions[3][0]) / 4]

for i in range(len(predictions) - 4):
    average_predictions.append((predictions[i][4] + predictions[i + 1][3] + predictions[i + 2][2] + predictions[i + 3][
        1] + predictions[i + 4][0]) / 5)

average_predictions.append((predictions[-1][1] + predictions[-2][2] + predictions[-3][3] + predictions[-4][4]) / 4)
average_predictions.append((predictions[-1][2] + predictions[-2][3] + predictions[-3][4]) / 3)
average_predictions.append((predictions[-1][3] + predictions[-2][4]) / 2)
average_predictions.append(predictions[-1][4])

print('\nAverage Predictions Length:', len(average_predictions))
# print('\nAverage Predictions:\n', average_predictions)


# Calculating the errors. # Here we identify the anomalies using Squared Error.

# Creating the comparison dataset.
comparison_cpu_utilization = cpu_utilization['value'].values[len(x_train) + window:]

print('Length of the comparison dataset:', len(comparison_cpu_utilization))

squared_errors = []
for i in range(len(average_predictions)):
    squared_errors.append((comparison_cpu_utilization[i] - average_predictions[i]) ** 2)

# Printing some statistics about the errors.
print('\nMaximum Squared Error Value:', max(squared_errors))
print('\nMinimum Squared Error Value', min(squared_errors))
print('\nMean Squared Error Value:', np.mean(squared_errors))
print('\nMedian Squared Error Value:', np.median(squared_errors))
print('\nStandard Deviation of Squared Error Values:', np.std(squared_errors))

# Calculating the threshold.
threshold = np.mean(squared_errors) + 0.5 * np.std(squared_errors)
print('\nThreshold:', threshold)


# Plotting Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(average_predictions, label='Predicted CPU Utilization', color='red')
plt.legend()
plt.title('MLP Model #2: Actual CPU Utilization Vs Predicted CPU Utilization')
plt.show()


# Getting the anomaly indices.
anomalies = []
for squared_error in squared_errors:
    if squared_error > threshold:
        anomalies.append(squared_error)

print('Length of anomalies:', len(anomalies))

anomaly_indices = []
for anomaly in anomalies:
    anomaly_indices.append(squared_errors.index(anomaly))
print('\nAnomaly Indices:', anomaly_indices)


# Creating the anomalies dataset.
anomalies_dataset = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        anomalies_dataset.append(comparison_cpu_utilization[i])
    else:
        anomalies_dataset.append(None)

# print((anomalies_dataset))


# Plotting Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(anomalies_dataset, label='Anomalies', color='red')
plt.legend()
plt.title('MLP Model #2: Actual CPU Utilization Vs Predicted Anomalies')
plt.show()


# Getting the predicted classes for plotting the confusion matrix.
normal_anomaly_predictions = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        normal_anomaly_predictions.append('anomaly')
    else:
        normal_anomaly_predictions.append('normal')

# Getting the hypothesized classes for plotting the confusion matrix.
comparison_cpu_utilization_label = cpu_utilization['Label'].values[len(x_train) + window:]


# Plotting the confusion matrix
confusion_mtx = confusion_matrix(comparison_cpu_utilization_label, normal_anomaly_predictions)
plt.figure(figsize=(20, 20))
sns.set(font_scale=2.0)
plot_confusion_matrix(confusion_mtx, classes=['anomaly', 'normal'])


# # LSTM Model #2
print("\n\nLSTM Model 2\n\n")


# Reshaping.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# LSTM Model #2 with the loss as 'Mean Squared Error'.
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=192, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(predict))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# Fitting the model on the training data.
history = model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_test, y_test))


# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()


# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='best')
plt.show()


# Making the predictions.
predictions = (model.predict(x_test))
predictions = scaler.inverse_transform(predictions)

print('Predictions Shape:', predictions.shape)
print('\nFirst 10 Predictions:\n', predictions[:10])
print('\nFirst 10 Actual Values:\n', scaler.inverse_transform(y_test[:10]))
print('\nLast 10 Predictions:\n', predictions[-10:])
print('\nLast 10 Actual Values:\n', scaler.inverse_transform(y_test[-10:]))


# Getting the average value for all the predictions.
average_predictions = [predictions[0][0], (predictions[0][1] + predictions[1][0]) / 2,
                       (predictions[0][2] + predictions[1][1] + predictions[2][0]) / 3,
                       (predictions[0][3] + predictions[1][2] + predictions[2][1] + predictions[3][0]) / 4]

for i in range(len(predictions) - 4):
    average_predictions.append((predictions[i][4] + predictions[i + 1][3] + predictions[i + 2][2] + predictions[i + 3][
        1] + predictions[i + 4][0]) / 5)

average_predictions.append((predictions[-1][1] + predictions[-2][2] + predictions[-3][3] + predictions[-4][4]) / 4)
average_predictions.append((predictions[-1][2] + predictions[-2][3] + predictions[-3][4]) / 3)
average_predictions.append((predictions[-1][3] + predictions[-2][4]) / 2)
average_predictions.append(predictions[-1][4])

print('\nAverage Predictions Length:', len(average_predictions))
# print('\nAverage Predictions:\n', average_predictions)


# Calculating the errors. # Here we identify the anomalies using Squared Error.

# Creating the comparison dataset.
comparison_cpu_utilization = cpu_utilization['value'].values[len(x_train) + window:]
print('Length of the comparison dataset:', len(comparison_cpu_utilization))

squared_errors = []
for i in range(len(average_predictions)):
    squared_errors.append((comparison_cpu_utilization[i] - average_predictions[i]) ** 2)

print('\nMaximum Squared Error Value:', max(squared_errors))
print('\nMinimum Squared Error Value', min(squared_errors))
print('\nMean Squared Error Value:', np.mean(squared_errors))
print('\nMedian Squared Error Value:', np.median(squared_errors))
print('\nStandard Deviation of Squared Error Values:', np.std(squared_errors))

# Calculating the threshold.
threshold = np.mean(squared_errors) + np.std(squared_errors)
print('\nThreshold:', threshold)


# Plotting Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(average_predictions, label='Predicted CPU Utilization', color='red')
plt.legend()
plt.title('LSTM Model #2: Actual CPU Utilization Vs Predicted CPU Utilization')
plt.show()


# Getting the anomaly indices.
anomalies = []
for squared_error in squared_errors:
    if squared_error > threshold:
        anomalies.append(squared_error)

print('Length of anomalies:', len(anomalies))

anomaly_indices = []
for anomaly in anomalies:
    anomaly_indices.append(squared_errors.index(anomaly))
print('\nAnomaly Indices:', anomaly_indices)


# Creating the anomalies dataset.
anomalies_dataset = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        anomalies_dataset.append(comparison_cpu_utilization[i])
    else:
        anomalies_dataset.append(None)

# print((anomalies_dataset))


# Plotting Actual CPU Utilization vs Predicted Anomalies.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(anomalies_dataset, label='Anomalies', color='red')
plt.legend()
plt.title('LSTM Model #2: Actual CPU Utilization Vs Predicted Anomalies')
plt.show()


# Getting the predicted classes for plotting the confusion matrix.
normal_anomaly_predictions = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        normal_anomaly_predictions.append('anomaly')
    else:
        normal_anomaly_predictions.append('normal')

# Getting the hypothesized classes for plotting the confusion matrix.
comparison_cpu_utilization_label = cpu_utilization['Label'].values[len(x_train) + window:]

# Plotting the confusion matrix
confusion_mtx = confusion_matrix(comparison_cpu_utilization_label, normal_anomaly_predictions)
plt.figure(figsize=(20, 20))
sns.set(font_scale=2.0)
plot_confusion_matrix(confusion_mtx, classes=['anomaly', 'normal'])


# # MLP Model #3
print("\n\nMLP Model 3\n\n")


# Specifying the window and prediction parameters.
window = 288
predict = 5


# Creating the lists to store the data and the labels.
x = []
y = []
for i in range(window, len(normal_cpu_utilization) + 1):
    x.append(normal_cpu_utilization[i - window:i, 0])

for i in range(predict, len(normal_cpu_utilization) + 1):
    y.append(normal_cpu_utilization[i - predict:i, 0])

x = x[:-predict]
y = y[window:]


# Printing some statistics about the data.
print('Shape X:', len(x))
print('Shape Y:', len(y))
print('Maximum Data Value:', np.max(x))


# Converting to array.
x = np.array(x)
y = np.array(y)


# Creating the train and test dataset.
x_train = x[:int(0.7 * len(x))]
y_train = y[:int(0.7 * len(y))]

x_test = x[int(0.7 * len(x)):]
y_test = y[int(0.7 * len(y)):]


# Printing some statistics about the data.
print('\nShape of the training dataset:', x_train.shape)
print('\n2 First two rows of the training data:\n', x_train[:2])
print('\n2 Last two rows of the training data:\n', x_train[-2:])

print('\nShape of the testing dataset:', x_test.shape)
print('\n2 First two rows of the testing data:\n', x_test[:2])
print('\n2 Last two rows of the testing data:\n', x_test[-2:])

print('\nShape of training data labels:', y_train.shape)
print('\n2 First two rows of the training data labels:\n', y_train[:2])
print('\n2 Last two rows of the training data labels:\n', y_train[-2:])

print('\nShape of testing data labels:', y_test.shape)
print('\n2 First two rows of the testing data labels:\n', y_test[:2])
print('\n2 Last two rows of the testing data labels:\n', y_test[-2:])


# MLP Model #3 with the loss as 'Mean Absolute Percentage Error'.
model = Sequential()
model.add(Dense(256, activity_regularizer=l1_l2(0.000001)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(196, activity_regularizer=l1_l2(0.000001)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, activity_regularizer=l1_l2(0.000001)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(predict))
model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])


# Fitting the MLP model on the training data.
history = model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(x_test, y_test))


# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()


# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='best')
plt.show()


# Making the predictions.
predictions = (model.predict(x_test))
predictions = scaler.inverse_transform(predictions)

print('Predictions Shape:', predictions.shape)
print('\nFirst 10 Predictions:\n', predictions[:10])
print('\nFirst 10 Actual Values:\n', scaler.inverse_transform(y_test[:10]))
print('\nLast 10 Predictions:\n', predictions[-10:])
print('\nLast 10 Actual Values:\n', scaler.inverse_transform(y_test[-10:]))


# Getting the average value for all the predictions.
average_predictions = [predictions[0][0], (predictions[0][1] + predictions[1][0]) / 2,
                       (predictions[0][2] + predictions[1][1] + predictions[2][0]) / 3,
                       (predictions[0][3] + predictions[1][2] + predictions[2][1] + predictions[3][0]) / 4]

for i in range(len(predictions) - 4):
    average_predictions.append((predictions[i][4] + predictions[i + 1][3] + predictions[i + 2][2] + predictions[i + 3][
        1] + predictions[i + 4][0]) / 5)

average_predictions.append((predictions[-1][1] + predictions[-2][2] + predictions[-3][3] + predictions[-4][4]) / 4)
average_predictions.append((predictions[-1][2] + predictions[-2][3] + predictions[-3][4]) / 3)
average_predictions.append((predictions[-1][3] + predictions[-2][4]) / 2)
average_predictions.append(predictions[-1][4])

print('\nAverage Predictions Length:', len(average_predictions))
# print('\nAverage Predictions:\n', average_predictions)


# Calculating the errors. # Here we identify the anomalies using Absolute Percentage Error.

# Creating the comparison dataset.
comparison_cpu_utilization = cpu_utilization['value'].values[len(x_train) + window:]

print('Length of the comparison dataset:', len(comparison_cpu_utilization))

absolute_percentage_errors = []
for i in range(len(average_predictions)):
    absolute_percentage_errors.append(
        abs((comparison_cpu_utilization[i] - average_predictions[i]) / comparison_cpu_utilization[i]) * 100)

# Printing some statistics about the errors.
print('\nMaximum Absolute Percentage Error Value:', max(absolute_percentage_errors))
print('\nMinimum Absolute Percentage Error Value', min(absolute_percentage_errors))
print('\nMean Absolute Percentage Error Value:', np.mean(absolute_percentage_errors))
print('\nMedian Absolute Percentage Error Value:', np.median(absolute_percentage_errors))
print('\nStandard Deviation of Absolute Percentage Error Values:', np.std(absolute_percentage_errors))

# Calculating the threshold.
threshold = np.mean(absolute_percentage_errors) + 2 * np.std(absolute_percentage_errors)
print('\nThreshold:', threshold)


# Plotting the Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(average_predictions, label='Predicted CPU Utilization', color='red')
plt.legend()
plt.title('MLP Model #3: Actual CPU Utilization Vs Predicted CPU Utilization')
plt.show()


# Getting the anomaly indices.
anomalies = []
for absolute_percentage_error in absolute_percentage_errors:
    if absolute_percentage_error > threshold:
        anomalies.append(absolute_percentage_error)

print('Length of anomalies:', len(anomalies))

anomaly_indices = []
for anomaly in anomalies:
    anomaly_indices.append(absolute_percentage_errors.index(anomaly))
print('\nAnomaly Indices:', anomaly_indices)


# Creating the anomalies dataset.
anomalies_dataset = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        anomalies_dataset.append(comparison_cpu_utilization[i])
    else:
        anomalies_dataset.append(None)

# print((anomalies_dataset))


# Plotting the Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(anomalies_dataset, label='Anomalies', color='red')
plt.legend()
plt.title('MLP Model #3: Actual CPU Utilization Vs Anomalies')
plt.show()


# Getting the predicted classes for plotting the confusion matrix.
normal_anomaly_predictions = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        normal_anomaly_predictions.append('anomaly')
    else:
        normal_anomaly_predictions.append('normal')

# Getting the hypothesized classes for plotting the confusion matrix.
comparison_cpu_utilization_label = cpu_utilization['Label'].values[len(x_train) + window:]

# Plotting the confusion matrix.
confusion_mtx = confusion_matrix(comparison_cpu_utilization_label, normal_anomaly_predictions)
plt.figure(figsize=(20, 20))
sns.set(font_scale=2.0)
plot_confusion_matrix(confusion_mtx, classes=['anomaly', 'normal'])

# # LSTM Model #3
print("\n\nLSTM Model 3\n\n")


# Reshaping.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# LSTM Model #3 with the loss as 'Mean Absolute Percentage Error'.
model = Sequential()
model.add(LSTM(units=384, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=192, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(predict))
model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])


# Fitting the model on the training data.
history = model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_test, y_test))


# Plot Accuracy.
plt.figure(figsize=(15, 7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='best')
plt.show()


# Plot Loss.
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='best')
plt.show()


# Making the predictions.
predictions = (model.predict(x_test))
predictions = scaler.inverse_transform(predictions)

print('Predictions Shape:', predictions.shape)
print('\nFirst 10 Predictions:\n', predictions[:10])
print('\nFirst 10 Actual Values:\n', scaler.inverse_transform(y_test[:10]))
print('\nLast 10 Predictions:\n', predictions[-10:])
print('\nLast 10 Actual Values:\n', scaler.inverse_transform(y_test[-10:]))


# Getting the average value for all the predictions.
average_predictions = [predictions[0][0], (predictions[0][1] + predictions[1][0]) / 2,
                       (predictions[0][2] + predictions[1][1] + predictions[2][0]) / 3,
                       (predictions[0][3] + predictions[1][2] + predictions[2][1] + predictions[3][0]) / 4]

for i in range(len(predictions) - 4):
    average_predictions.append((predictions[i][4] + predictions[i + 1][3] + predictions[i + 2][2] + predictions[i + 3][
        1] + predictions[i + 4][0]) / 5)

average_predictions.append((predictions[-1][1] + predictions[-2][2] + predictions[-3][3] + predictions[-4][4]) / 4)
average_predictions.append((predictions[-1][2] + predictions[-2][3] + predictions[-3][4]) / 3)
average_predictions.append((predictions[-1][3] + predictions[-2][4]) / 2)
average_predictions.append(predictions[-1][4])

print('\nAverage Predictions Length:', len(average_predictions))
# print('\nAverage Predictions:\n', average_predictions)


# Calculating the errors. # Here we identify the anomalies using Absolute Percentage Error.

# Creating the comparison dataset.
comparison_cpu_utilization = cpu_utilization['value'].values[len(x_train) + window:]

print('Length of the comparison dataset:', len(comparison_cpu_utilization))

absolute_percentage_errors = []
for i in range(len(average_predictions)):
    absolute_percentage_errors.append(
        abs((comparison_cpu_utilization[i] - average_predictions[i]) / comparison_cpu_utilization[i]) * 100)

# Printing some statistics about the errors.
print('\nMaximum Absolute Percentage Error Value:', max(absolute_percentage_errors))
print('\nMinimum Absolute Percentage Error Value', min(absolute_percentage_errors))
print('\nMean Absolute Percentage Error Value:', np.mean(absolute_percentage_errors))
print('\nMedian Absolute Percentage Error Value:', np.median(absolute_percentage_errors))
print('\nStandard Deviation of Absolute Percentage Error Values:', np.std(absolute_percentage_errors))

# Calculating the threshold.
threshold = np.mean(absolute_percentage_errors) + 2 * np.std(absolute_percentage_errors)
print('\nThreshold:', threshold)


# Plotting the Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(average_predictions, label='Predicted CPU Utilization', color='red')
plt.legend()
plt.title('LSTM Model #3: Actual CPU Utilization Vs Predicted CPU Utilization')
plt.show()


# Getting the anomaly indices.
anomalies = []
for absolute_percentage_error in absolute_percentage_errors:
    if absolute_percentage_error > threshold:
        anomalies.append(absolute_percentage_error)

print('Length of anomalies:', len(anomalies))

anomaly_indices = []
for anomaly in anomalies:
    anomaly_indices.append(absolute_percentage_errors.index(anomaly))
print('\nAnomaly Indices:', anomaly_indices)


# Creating the anomalies dataset.
anomalies_dataset = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        anomalies_dataset.append(comparison_cpu_utilization[i])
    else:
        anomalies_dataset.append(None)

# print((anomalies_dataset))


# Plotting the Actual CPU Utilization vs Predicted CPU Utilization.
plt.figure(figsize=(25, 10))
plt.plot(comparison_cpu_utilization, label='Actual CPU Utilization')
plt.plot(anomalies_dataset, label='Anomalies', color='red')
plt.legend()
plt.title('LSTM Model #3: Actual CPU Utilization Vs Anomalies')
plt.show()


# Getting the predicted classes for plotting the confusion matrix.
normal_anomaly_predictions = []
for i in range(len(x_test) + predict - 1):
    if i in anomaly_indices:
        normal_anomaly_predictions.append('anomaly')
    else:
        normal_anomaly_predictions.append('normal')

# Getting the hypothesized classes for plotting the confusion matrix.
comparison_cpu_utilization_label = cpu_utilization['Label'].values[len(x_train) + window:]

# Plotting the confusion matrix.
confusion_mtx = confusion_matrix(comparison_cpu_utilization_label, normal_anomaly_predictions)
plt.figure(figsize=(20, 20))
sns.set(font_scale=2.0)
plot_confusion_matrix(confusion_mtx, classes=['anomaly', 'normal'])

# # Comment on Results:
# We can see that the MLP models make better predictions than the LSTM models. The loss values for MLP models are lower
# than that of LSTM models. Also, LSTM doesn't perform as well with smaller window sizes. The LSTM model makes smooth
# predictions if we use three layers instead of two.
