import numpy as np
import time
import random
from collections import deque
import matplotlib.pyplot as plt



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
            print(f"Anomaly introduced: {data_point}")
            n += 1
        else:
            print(f"Normal point: {data_point}")

        yield data_point
        
        # Simulate real-time delay
        time.sleep(0.9)  # 900ms delay between data points (adjust as needed)
        
        count += 1

    return n  # Return number of anomalies detected when finished




# def data_stream_generator():
#     """
#     A generator function that continuously yields real-time floating-point numbers.
#     Simulates a real-time data stream.
#     """

#     n = 0
#     while True:
#         # Simulate a real-time data point (normal distribution with occasional spikes)
#         data_point = random.gauss(0, 1)  # Mean = 0, Standard Deviation = 1

#         # Introduce random anomalies occasionally
#         if random.random() < 0.1:  # 0.01 = 1% chance of anomaly
#             data_point += random.uniform(15, 20)  # Spike anomaly (old = 5, 10)
#             print(f"Anomaly introduced: {data_point}")
#             n += 1
#         else:
#             print(f"Normal point: {data_point}")

#         yield data_point
        
#         # Simulate real-time delay
#         time.sleep(0.9)  # 100ms delay between data points (adjust as needed)







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
            # print(f"Index: {i}, Data point: {data_point}, Rolling mean: {mean}, Std dev: {std_dev}, Z-score: {z_score}")

            # Check if z-score exceeds the threshold (absolute value)
            if abs(z_score) > z_threshold:
                # print(f"Anomaly detected at index {i}: z_score = {z_score}")
                anomalies.append(i)  # Index of the anomaly for plotting
            
            # print('\n')

        else:
            z_scores.append(None)  # Not enough data points yet


        # Stop after processing the desired number of points
        if i >= num_points - 1:
            break
    
    return data_points, z_scores, anomalies, rolling_means, rolling_stds










# Function to plot the data points and anomalies
def plot_anomalies(data_points, anomalies):
    """
    Plots data points and highlights anomalies based on z-scores.
    
    Args:
        data_points: List of data points.
        z_scores: List of rolling z-scores (use None for incomplete rolling windows).
        anomalies: List of indices where anomalies were detected.
        z_threshold: Z-score threshold used for anomaly detection (default is 3).
    """
    z_threshold = 3

    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.plot(data_points, label='Data Points', color='blue', alpha=0.7)
    
    # Plot anomalies (if any)
    if anomalies:
        anomaly_points = [data_points[i] for i in anomalies]
        plt.scatter(anomalies, anomaly_points, color='red', label='Anomalies', zorder=5)
    
    # Plot z-score threshold lines
    plt.axhline(y=z_threshold, color='green', linestyle='--', label=f'Z-Score Threshold (+{z_threshold})')
    plt.axhline(y=-z_threshold, color='green', linestyle='--', label=f'Z-Score Threshold (-{z_threshold})')
        

    # Add labels and title
    plt.title(f'Anomaly Detection using Rolling Z-Score (Z-Score Threshold: {z_threshold})')
    plt.xlabel('Data Points')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Display the plot
    plt.show()














# Example Usage
window_size = 10  # You can adjust the window size here
z_threshold = 2  # Z-score threshold for anomaly detection
num_points = 50  # Number of points to process and plot

# Create the data stream generator
stream = data_stream_generator(num_points)
print(stream)

# Detect anomalies and calculate z-scores
data_points, z_scores, anomalies, rolling_means, rolling_stds = rolling_z_score(stream, window_size, z_threshold, num_points)




print(anomalies)

# Plot the results
plot_anomalies(data_points, anomalies)






















