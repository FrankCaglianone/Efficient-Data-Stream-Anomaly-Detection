import time
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import matplotlib.pyplot as plt







def data_stream_generator(max_data_points):
    '''
    * TODO
    '''

    count = 0

    while count < max_data_points:
        # Simulate a real-time data point (normal distribution with occasional spikes)
        data_point = random.gauss(0, 1)  # Mean = 0, Standard Deviation = 1

        # Introduce random anomalies occasionally
        if count >= 10 and random.random() < 0.05:  # 5% chance of anomaly
            data_point += random.uniform(15, 20)  # Spike anomaly
            print(f"Anomaly introduced at {count} --> {data_point}")
        # else:
        #     print(f"Normal point at {count} --> {data_point}")

        yield data_point
        
        # Simulate real-time delay
        time.sleep(0.1)
        
        count += 1














def initialize_real_time_plot(window_size=50):
    """
    Initializes a real-time plot with a specific window size.
    """
    plt.ion()  # Enable interactive mode
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axis
    data_window = deque([0] * window_size, maxlen=window_size)  # Data window
    x_data = np.linspace(0, window_size - 1, window_size)  # X-axis for data points
    line, = ax.plot(x_data, data_window, color='blue')  # Line object

    ax.set_ylim((-25, 25))  # Set y-axis limits

    return fig, ax, line, data_window, x_data


def update_plot(data_window, color_window, x_data, line, scatter):
    """
    Updates the real-time plot with the new data value and color.
    """
    # Update the line plot
    line.set_ydata(data_window)
    
    # Update the scatter plot for anomalies
    scatter.set_offsets(np.c_[x_data, data_window])
    scatter.set_color(color_window)
    
    plt.draw()
    plt.pause(0.01)























'''
* TODO
'''
def rolling_z_score_anomaly_detection(data_point, window, window_size, z_threshold):    #index
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
            z_score = (data_point - mean) / std_dev   # Calculate z-score
        
        # print(f"Index: {i}, Data point: {data_point}, Rolling mean: {mean}, Std dev: {std_dev}, Z-score: {z_score}")

        # Check if z-score exceeds the threshold (absolute value)
        if abs(z_score) > z_threshold:
            # print(f"\n Anomaly detected at index {index}: z_score = {z_score}\n")
            # return index
            return True
    
    return False
    # return None
    # return data_points, z_scores, anomalies, rolling_means, rolling_stds

















'''
* TODO
'''
def isolation_forest_anomaly_detection(iso_forest, data_point, data_buffer, buffer_size):  #* step_size, index
    anomaly_indices = []

    data_buffer.append([data_point])  # Add new data point to the buffer

    # Check if buffer is full
    if len(data_buffer) >= buffer_size:
        # Apply Isolation Forest to detect anomalies in the buffer
        iso_forest.fit(data_buffer)  # Fit the model to the buffer
        anomaly_labels = iso_forest.predict(data_buffer)  # Predict anomalies

        if anomaly_labels[-1] == -1:  # Check if the latest data point is an anomaly
            return True

        # # Find indices of anomalies in the current buffer
        # anomalies = np.where(anomaly_labels == -1)[0]
        # anomaly_indices.extend(anomalies + (index - buffer_size + 1))  # Adjust indices

        # # Clear the buffer after processing
        # data_buffer = data_buffer[step_size:]

    # return anomaly_indices, data_buffer
    return False
















def parallel_anomaly_detection(data_stream, window_size, buffer_size=50):
    """
    Runs both anomaly detection algorithms in parallel and updates the plot in real time.
    """
    # Initialize variables
    window = deque(maxlen=window_size)
    data_buffer = []
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    z_threshold = 2

    # Initialize plot
    fig, ax, line, data_window, x_data = initialize_real_time_plot(window_size=window_size)
    color_window = ['blue'] * window_size  # Color window for dynamic color updates
    scatter = ax.scatter(x_data, data_window, color=color_window)  # Scatter plot for color changes

    # Anomalies
    anomalies = []

    for index, data_point in enumerate(data_stream):
        # Update the rolling window with the new data point
        data_window.append(data_point)

        # Anomaly detection using rolling Z-score
        z_anomaly = rolling_z_score_anomaly_detection(data_point, window, window_size, z_threshold)

        # Anomaly detection using Isolation Forest
        iso_anomaly = isolation_forest_anomaly_detection(iso_forest, data_point, data_buffer, buffer_size)

        # Determine if it's an anomaly (detected by either method)
        is_anomaly = z_anomaly or iso_anomaly

        # TODO
        if is_anomaly:
            anomalies.append(index)

        # Update color window: red for anomaly, blue for normal
        color_window.append('red' if is_anomaly else 'blue')
        if len(color_window) > window_size:
            color_window.pop(0)  # Keep the color window in sync with data

        # Update the plot with new data and color window
        update_plot(data_window, color_window, x_data, line, scatter)

    
    print("Filtered combined anomalies:", anomalies)

    



def main():
    # Create the data stream generator
    data_stream = data_stream_generator(500)

    # Run the parallel anomaly detection and real-time plot update
    parallel_anomaly_detection(data_stream, window_size=200)

    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Display the final plot

















































# def parallel_anomaly_detection(data_stream, visualization_window, visualization_line):
#     # Variables
#     buffer_size = 50
#     window_size = 10

#     # Create fixed-size window for rolling Z-Score
#     window = deque(maxlen=window_size)

#     # Buffer for Isolation Forest
#     data_buffer = []
#     iso_forest = IsolationForest(contamination=0.08, random_state=42)

#     # Anomalies
#     z_score_anomalies = []
#     iso_anomalies = []

#     # Save Data Points
#     data_points = []

#     for index, data_point in enumerate(data_stream):
#         # TODO: !!!!!! Update the real-time plot with the new data point
#         update_plot(data_point, visualization_window, visualization_line)

#         # Run both anomaly detection algorithms in parallel
#         z_tmp = rolling_z_score_anomaly_detection(data_point=data_point, index=index, window=window, window_size=window_size, z_threshold=2)
#         if z_tmp is not None:
#             z_score_anomalies.append(z_tmp)

        
#         iso_tmp, data_buffer = isolation_forest_anomaly_detection(iso_forest=iso_forest, data_point=data_point, index =index, data_buffer=data_buffer, buffer_size=buffer_size, step_size=10)
#         if iso_tmp:
#             iso_anomalies.extend(iso_tmp)
#         # Fit the Isolation Forest model periodically
#         if len(data_buffer) == buffer_size:
#             iso_forest.fit(data_buffer)

#         data_points.append(data_point)


#     print('\n\n')
#     # print("Anomalies found with Rolling Z-Score at indices:", z_score_anomalies)
#     # print("Anomalies found with Forest Isolation at indices:", sorted(list(set(iso_anomalies))))

#     # combined_anomalies = set(z_score_anomalies) & set(iso_anomalies)
#     # print("All detected anomalies:", combined_anomalies)

#     # Combining anomalies using union instead of intersection
#     combined_anomalies = set(z_score_anomalies) | set(iso_anomalies)

#     # Additional filter: If the anomaly is only detected by one method, check if it is near another anomaly
#     # TODO: Anomaly Clustering
#     final_anomalies = set()
#     for anomaly in combined_anomalies:
#         # If an anomaly is detected by both methods, add immediately.
#         if anomaly in z_score_anomalies and anomaly in iso_anomalies:
#             final_anomalies.add(anomaly)
#         elif any(abs(anomaly - a) <= 5 for a in combined_anomalies if a != anomaly):
#             final_anomalies.add(anomaly)

#     print("Filtered combined anomalies:", final_anomalies)


#     return data_points, combined_anomalies










# def main():
#     # Create the data stream generator
#     data_stream = data_stream_generator(500)


#     # Initialize the real-time plot
#     fig, ax, line, data_window, x_data = initialize_real_time_plot(window_size=200)


#     # Run the parallel anomaly detection
#     data_points, anomalies_detected = parallel_anomaly_detection(data_stream, data_window, line)


#     plt.ioff()  # Turn off interactive mode when done
#     plt.show()  # Display the final plot





main()









#* Note Sliding Window: Instead of clearing the entire buffer after each detection round, we now implement a sliding window approach by keeping most of the buffer (removing only step_size points) to preserve the recent history of the data stream.














