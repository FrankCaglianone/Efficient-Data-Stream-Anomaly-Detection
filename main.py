import time
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import matplotlib.pyplot as plt







def data_stream_generator(max_data_points):
    '''
     TODO
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



def initialize_real_time_plot(window_size):
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



def rolling_z_score_anomaly_detection(data_point, window, window_size, z_threshold):    #* index
    '''
     TODO
    '''

    window.append(data_point)

    if len(window) == window_size:
        mean = np.mean(window)  # Calculate mean of the window
        std_dev = np.std(window)    # Calculate standard deviation of the window
        
        # Avoid division by zero
        if std_dev == 0:
            z_score = 0
        else:
            z_score = (data_point - mean) / std_dev   # Calculate z-score
        # print(f"Index: {i}, Data point: {data_point}, Rolling mean: {mean}, Std dev: {std_dev}, Z-score: {z_score}")

        if abs(z_score) > z_threshold:
            # print(f"\n Anomaly detected at index {index}: z_score = {z_score}\n")
            return True
    
    return False



def isolation_forest_anomaly_detection(iso_forest, data_point, data_buffer, buffer_size):  #* index
    '''
     TODO
    '''
    
    data_buffer.append([data_point])  # Add new data point to the buffer

    if len(data_buffer) >= buffer_size:    
        anomaly_labels = iso_forest.fit_predict(data_buffer)  # Fit the model to the buffer & predict anomalies

        if anomaly_labels[-1] == -1:  # Check if the latest data point is an anomaly
            return True
        
    return False












# TODO: 
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

    print("Detected anomalies:", anomalies)

    



def main():
    # Create the data stream generator
    data_stream = data_stream_generator(500)

    # Run the parallel anomaly detection and real-time plot update
    parallel_anomaly_detection(data_stream, window_size=200)

    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Display the final plot












def tmp(data_stream, visualization_window, visualization_line):
    # Variables
    buffer_size = 50
    window_size = 10

    # Create fixed-size window for rolling Z-Score
    window = deque(maxlen=window_size)



    
    z_score_anomalies = []
    iso_anomalies = []
    data_points = []


    # print("Anomalies found with Rolling Z-Score at indices:", z_score_anomalies)
    # print("Anomalies found with Forest Isolation at indices:", sorted(list(set(iso_anomalies))))

    # combined_anomalies = set(z_score_anomalies) & set(iso_anomalies)
    # print("All detected anomalies:", combined_anomalies)

    # Combining anomalies using union instead of intersection
    combined_anomalies = set(z_score_anomalies) | set(iso_anomalies)
    print("Filtered combined anomalies:", combined_anomalies)










main()









#* Note Sliding Window: Instead of clearing the entire buffer after each detection round, we now implement a sliding window approach by keeping most of the buffer (removing only step_size points) to preserve the recent history of the data stream.














