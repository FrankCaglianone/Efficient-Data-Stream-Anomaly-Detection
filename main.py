import numpy as np
import time
import random
from collections import deque




# Data stream generator function
def data_stream_generator():
    """
    A generator function that continuously yields real-time floating-point numbers.
    Simulates a real-time data stream.
    """
    while True:
        # Simulate a real-time data point (normal distribution with occasional spikes)
        data_point = random.gauss(0, 1)  # Mean = 0, Standard Deviation = 1

        # Introduce random anomalies occasionally
        if random.random() < 0.5:  # 0.01 = 1% chance of anomaly
            data_point += random.uniform(5, 10)  # Spike anomaly

        print(data_point)
        yield data_point
        
        # Simulate real-time delay
        time.sleep(0.9)  # 100ms delay between data points (adjust as needed)







def rolling_z_score(data_stream, window_size, z_threshold):
    """
    Calculate the rolling z-score for a real-time data stream.
    
    Args:
        data_stream: A generator that yields floating-point numbers.
        window_size: The size of the rolling window.
    
    Yields:
        A tuple of (data_point, z_score) for each new data point.
    """
    window = deque(maxlen=window_size)  # Fixed-size window to store data points
    
    for data_point in data_stream:
        # Append new data point to the window
        window.append(data_point)
        
        if len(window) == window_size:
            # Calculate mean and standard deviation of the window
            mean = np.mean(window)
            std_dev = np.std(window)
            
            # Avoid division by zero
            if std_dev == 0:
                z_score = 0
            else:
                # Calculate z-score
                z_score = (data_point - mean) / std_dev

            # Check if z-score exceeds the threshold (absolute value)
            is_anomaly = abs(z_score) > z_threshold

            yield data_point, z_score, is_anomaly
        else:
            # Not enough data points to compute z-score, return None for z-score and anomaly flag
            yield data_point, None, False












window_size = 10  # Define your rolling window size
z_threshold = 3   # Set z-score threshold for anomaly detection (default: 3 for 99.7% confidence)

stream = data_stream_generator()  # Assuming you have a real-time data generator
anomaly_detector = rolling_z_score(stream, window_size, z_threshold)

# Process and print anomalies in the data stream
for _ in range(20):  # You can adjust the number of points to process
    data_point, z_score, is_anomaly = next(anomaly_detector)
    if is_anomaly:
        print(f"Anomaly Detected! Data Point: {data_point}, Z-Score: {z_score}")
    else:
        print(f"Data Point: {data_point}, Z-Score: {z_score}")









# data = [1, 2, 2, 2, 3, 1, 1, 15, 2, 2, 2, 3, 1, 1, 2]
# mean = np.mean(data)
# std = np.std(data)
# print('mean of the dataset is', mean)
# print('std. deviation is', std)




# threshold = 3
# outlier = []
# for i in data:
# 	z = (i-mean)/std
# 	print("z-score", z)
# 	if z > threshold:
# 		outlier.append(i)

# print('\n')
# print('outlier in dataset is', outlier)
















