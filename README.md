# AnomalyDetection
MLP, LSTM, and Autoencoder models to detect anomalies on AWS CPU utilization.

AWS spins up a new machine each time the CPU utilization on the machine goes above 60 %, thus sustained loads above 60 % can be considered as anomalies. 

For the MLP and LSTM model we create a normalized dataset and create the labels for the anomalies, then train our model on the normalized dataset and get the predictions. We then set a threshold based on the error values between the prediction and the actual values and identify the anomalies.

For the autoencoder model, we reconstruct the original dataset from the latent space. Finally, we plot the distribution of the reconstructions errors and find the highest ones. They are essentially where the anomalies exist.
