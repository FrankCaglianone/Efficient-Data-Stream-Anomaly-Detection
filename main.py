import time
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque







# Data stream generator function
def data_stream_generator(max_data_points):
    count = 0  # Counter to stop after max_data_points

    while count < max_data_points:
        # Simulate a real-time data point (normal distribution with occasional spikes)
        data_point = random.gauss(0, 1)  # Mean = 0, Standard Deviation = 1

        # Introduce random anomalies occasionally
        if random.random() < 0.1:  # 0.1 = 10% chance of anomaly
            data_point += random.uniform(15, 20)  # Spike anomaly
            print(f"Anomaly introduced at {count} --> {data_point}")
        # else:
        #     print(f"Normal point at {count} --> {data_point}")

        yield data_point
        
        # Simulate real-time delay
        time.sleep(0.9)  # 900ms delay between data points (adjust as needed)
        
        count += 1






def rolling_z_score_anomaly_detection(data_stream, window_size, z_threshold):
    window = deque(maxlen=window_size)  # Fixed-size window to store data points
    data_points = []
    z_scores = []
    anomalies = []
    rolling_means = []
    rolling_stds = []
    
    for i, data_point in enumerate(data_stream):
        # Append new data point to the window
        window.append(data_point)
        data_points.append(data_point)

        
        if len(window) == window_size:
            # Calculate mean and standard deviation of the window
            mean = np.mean(window)
            std_dev = np.std(window)
            rolling_means.append(mean)
            rolling_stds.append(std_dev)
            
            # Avoid division by zero
            if std_dev == 0:
                z_score = 0
            else:
                # Calculate z-score
                z_score = (data_point - mean) / std_dev
            
            z_scores.append(z_score)
            # print(f"Index: {i}, Data point: {data_point}, Rolling mean: {mean}, Std dev: {std_dev}, Z-score: {z_score}")

            # Check if z-score exceeds the threshold (absolute value)
            if abs(z_score) > z_threshold:
                # print(f"\n Anomaly detected at index {i}: z_score = {z_score}\n")
                anomalies.append(i)  # Index of the anomaly for plotting
            
            # print('\n')

        else:
            z_scores.append(None)  # Not enough data points yet
    
    return data_points, z_scores, anomalies, rolling_means, rolling_stds











def isolation_forest_anomaly_detection(data_stream, buffer_size, step_size):
    anomaly_indices = []
    data_buffer = []

    # Initialize Isolation Forest model
    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    for index, data_point in enumerate(data_stream):  # Simulate 100 points
        data_buffer.append([data_point])  # Add new data point to the buffer

        # Check if buffer is full
        if len(data_buffer) >= buffer_size:
            # Apply Isolation Forest to detect anomalies in the buffer
            iso_forest.fit(data_buffer)  # Fit the model to the buffer
            anomaly_labels = iso_forest.predict(data_buffer)  # Predict anomalies

            # Find indices of anomalies in the current buffer
            anomalies = np.where(anomaly_labels == -1)[0]
            anomaly_indices.extend(anomalies + (index - buffer_size + 1))  # Adjust indices

            # Clear the buffer after processing
            data_buffer = data_buffer[step_size:]

    return anomaly_indices


























# Create the data stream generator
data_stream = data_stream_generator(50)



# Detect anomalies with Rolling Z-Score
data_points, z_scores, z_score_anomalies, rolling_means, rolling_stds = rolling_z_score_anomaly_detection(data_stream=data_stream, window_size=10, z_threshold=2)
print("Anomalies found with Rolling Z-Score at indices:", z_score_anomalies)



# Detect anomalies with Isolation Forest
iso_anomalies = isolation_forest_anomaly_detection(data_stream=data_stream, buffer_size=50, step_size=10)
print("Anomalies found at indices:", iso_anomalies)





# Plot the results
# plot_anomalies(data_points, anomalies)




#* NoteSliding Window: Instead of clearing the entire buffer after each detection round, we now implement a sliding window approach by keeping most of the buffer (removing only step_size points) to preserve the recent history of the data stream.














