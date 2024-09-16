import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
import time
import random
from collections import deque





# TODO: delete return count in data stream generator


# Data stream generator function
def data_stream_generator(max_data_points):
    """
    A generator function that continuously yields real-time floating-point numbers.
    Simulates a real-time data stream. Stops after 'max_data_points' are generated.
    """
    n = 0  # Count of anomalies
    count = 0  # Counter to stop after max_data_points

    while count < max_data_points:
        # Simulate a real-time data point (normal distribution with occasional spikes)
        data_point = random.gauss(0, 1)  # Mean = 0, Standard Deviation = 1

        # Introduce random anomalies occasionally
        if random.random() < 0.1:  # 0.1 = 10% chance of anomaly
            data_point += random.uniform(15, 20)  # Spike anomaly
            print(f"Anomaly introduced at {count} --> {data_point}")
            n += 1
        # else:
        #     print(f"Normal point at {count} --> {data_point}")

        yield data_point
        
        # Simulate real-time delay
        time.sleep(0.9)  # 900ms delay between data points (adjust as needed)
        
        count += 1

    return n  # Return number of anomalies detected when finished






def rolling_z_score(data_stream, window_size, z_threshold, num_points):
    """
    Calculate the rolling z-score for a real-time data stream.
    
    Args:
        data_stream: A generator that yields floating-point numbers.
        window_size: The size of the rolling window.
    
    Yields:
        A tuple of (data_point, z_score) for each new data point.
    """
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
            print(f"Index: {i}, Data point: {data_point}, Rolling mean: {mean}, Std dev: {std_dev}, Z-score: {z_score}")

            # Check if z-score exceeds the threshold (absolute value)
            if abs(z_score) > z_threshold:
                print(f"\n Anomaly detected at index {i}: z_score = {z_score}\n")
                anomalies.append(i)  # Index of the anomaly for plotting
            
            # print('\n')

        else:
            z_scores.append(None)  # Not enough data points yet


        # Stop after processing the desired number of points
        if i >= num_points - 1:
            break
    
    return data_points, z_scores, anomalies, rolling_means, rolling_stds









# # Function to process data in chunks using Isolation Forest
# def isolation_forest_detection(data_stream, chunk_size, contamination=0.02):
#     """
#     Processes the data stream in chunks and applies Isolation Forest for anomaly detection.
    
#     :param max_data_points: Total number of data points to process.
#     :param chunk_size: Size of each chunk of data to fit the Isolation Forest model.
#     :param contamination: Proportion of the dataset to be considered as outliers.
#     """
#     # Create Isolation Forest model
#     model = IsolationForest(contamination=contamination, random_state=42)
    
#     # Initialize data storage
#     data_chunk = []

#     # Counters for anomalies detected
#     total_anomalies_detected = 0

#     for data_point in data_stream:
#         # Append the new data point to the chunk
#         data_chunk.append([data_point])

#         # When the chunk is full, fit the model and detect anomalies
#         if len(data_chunk) == chunk_size:
#             # Convert to DataFrame
#             df = pd.DataFrame(data_chunk)
            
#             # Fit the model and predict
#             model.fit(df)
#             df['anomaly'] = model.predict(df)
            
#             # Count anomalies
#             anomalies_in_chunk = df[df['anomaly'] == -1]
#             normal_data_in_chunk = df[df['anomaly'] == 1]

#             total_anomalies_detected += len(anomalies_in_chunk)

#             print(f"Processed {chunk_size} data points: {len(anomalies_in_chunk)} anomalies detected in this chunk")

#             # Reset the chunk for the next set of data points
#             data_chunk = []

#     print(f"Total anomalies detected: {total_anomalies_detected}")






























# Create the data stream generator
data_stream = data_stream_generator(500)



# Detect anomalies with Rolling Z-Score
# window_size = 10  # You can adjust the window size here
# z_threshold = 2  # Z-score threshold for anomaly detection
# num_points = 50  # Number of points to process and plot
# data_points, z_scores, anomalies, rolling_means, rolling_stds = rolling_z_score(stream, window_size, z_threshold, num_points)
# print(anomalies)














def detect_anomalies_in_stream(data_stream, buffer_size, step_size):
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









# Call the function to detect anomalies in the stream
#* NoteSliding Window: Instead of clearing the entire buffer after each detection round, we now implement a sliding window approach by keeping most of the buffer (removing only step_size points) to preserve the recent history of the data stream.
anomaly_indices = detect_anomalies_in_stream(data_stream=data_stream, buffer_size=50, step_size=10)


print('\n\n\n')
print("Anomalies found at indices:", anomaly_indices)





























# Run the data stream processing with Isolation Forest
# isolation_forest_detection(stream, chunk_size=10)




# # Plot the results
# plot_anomalies(data_points, anomalies)



















