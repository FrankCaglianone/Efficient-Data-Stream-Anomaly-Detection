import time
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import matplotlib.pyplot as plt




def data_stream_simulation(max_points=500, amplitude=1, frequency=1, noise_scale=0.1, seasonal_amplitude=0.5, seasonal_frequency=0.1, sleep_time=0.1):    
    t = 0
    count = 0
    while count <= max_points: # TODO: True:
        # Generate the regular pattern (sine wave)
        regular_pattern = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add the seasonal component (cosine wave)
        seasonal_component = seasonal_amplitude * np.cos(2 * np.pi * seasonal_frequency * t)
        
        # Add random noise (Gaussian distribution)
        noise = np.random.normal(scale=noise_scale)
        
        # Combine all components to create the data point
        data_point = regular_pattern + seasonal_component + noise
        
        # Inject an anomaly every 'anomaly_interval' data points
        # Randomly inject an anomaly with probability anomaly_chance
        if count > 30 and random.random() < 0.05:
            anomaly_magnitude = random.uniform(15, 20)
            data_point += anomaly_magnitude * random.choice([-1, 1])  # Randomly inject positive or negative anomalies
            print(f"Anomaly introduced at {count} --> {data_point}")
        
        # Yield the generated data point
        yield data_point
        
        # Increment time and counter for the next data point
        t += 0.01
        count += 1
        
        # Simulate real-time streaming
        time.sleep(sleep_time)



def initialize_real_time_plot(window_size):
    """
     Initializes a real-time plot with a specific window size.
    """
    plt.ion()  # Enable interactive mode
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axis
    data_window = deque([0] * window_size, maxlen=window_size)  # Data window
    x_data = np.linspace(0, window_size - 1, window_size)  # X-axis for data points
    line, = ax.plot(x_data, data_window, color='blue', zorder=1)  # Line object

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



def rolling_z_score_anomaly_detection(data_point, window, window_size, z_threshold):
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



def isolation_forest_anomaly_detection(iso_forest, data_point, data_buffer, buffer_size):
    '''
     TODO
    '''
    
    data_buffer.append([data_point])  # Add new data point to the buffer

    if len(data_buffer) >= buffer_size:    
        anomaly_labels = iso_forest.fit_predict(data_buffer)  # Fit the model to the buffer & predict anomalies

        if anomaly_labels[-1] == -1:  # Check if the latest data point is an anomaly
            return True
        
    return False



def parallel_anomaly_detection(data_stream):
    """
     Runs both anomaly detection algorithms in parallel and updates the plot in real time.
    """
    # Initialize variables
    plot_window_size=500
    rolling_window_size=50
    buffer_size = 50
    z_threshold = 2
    window = deque(maxlen=rolling_window_size)
    data_buffer = []
    iso_forest = IsolationForest(contamination=0.03, n_estimators=150, random_state=42)
 
    # Initialize plot
    fig, ax, line, data_window, x_data = initialize_real_time_plot(window_size=plot_window_size)
    color_window = ['blue'] * plot_window_size  # Color window for dynamic color updates
    scatter = ax.scatter(x_data, data_window, color=color_window, zorder=2)  # Scatter plot for color changes

    # TODO: Anomalies Comment
    z_score_anomalies = []
    iso_anomalies = []
    all_anomalies = []

    for index, data_point in enumerate(data_stream):
        # Update the rolling window with the new data point
        data_window.append(data_point)

        # Anomaly detection using rolling Z-score
        z_anomaly = rolling_z_score_anomaly_detection(data_point, window, rolling_window_size, z_threshold)
        if z_anomaly: z_score_anomalies.append(index)

        # Anomaly detection using Isolation Forest
        iso_anomaly = isolation_forest_anomaly_detection(iso_forest, data_point, data_buffer, buffer_size)
        if iso_anomaly : iso_anomalies.append(index)

        # Determine if it's an anomaly (detected by either method)
        is_anomaly = z_anomaly or iso_anomaly
        if is_anomaly: all_anomalies.append(index)

        # Update color window: red for anomaly, blue for normal
        color_window.append('red' if is_anomaly else 'blue')
        if len(color_window) > plot_window_size:
            color_window.pop(0)  # Keep the color window in sync with data

        # Update the plot with new data and color window
        update_plot(data_window, color_window, x_data, line, scatter)

    print("Anomalies found with Rolling Z-Score at indices:", z_score_anomalies)
    print("Anomalies found with Forest Isolation at indices:", sorted(list(set(iso_anomalies))))
    print("Detected anomalies:", all_anomalies)





def main():
    # Create the data stream generator
    data_stream = data_stream_simulation()

    # Run the parallel anomaly detection and real-time plot update
    parallel_anomaly_detection(data_stream)

    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Display the final plot




if __name__ == "__main__":
    main()



#* Note Sliding Window: Instead of clearing the entire buffer after each detection round, we now implement a sliding window approach by keeping most of the buffer (removing only step_size points) to preserve the recent history of the data stream.


