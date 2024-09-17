import time
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import matplotlib.pyplot as plt




def data_stream_simulation(amplitude=1, frequency=1, noise_scale=0.1, seasonal_amplitude=0.5, seasonal_frequency=0.1, sleep_time=0.1):
    """
        Simulates a real-time data stream by generating data points with regular patterns, 
        seasonal component, and random noise. It occasionally injects anomalies.
        
        Args:
            amplitude (float): Amplitude of the regular sine wave pattern.
            frequency (float): Frequency of the regular sine wave pattern.
            noise_scale (float): Scale of random noise added to the data.
            seasonal_amplitude (float): Amplitude of the seasonal component (cosine wave).
            seasonal_frequency (float): Frequency of the seasonal component.
            sleep_time (float): Time to sleep between each data point to simulate real-time data streaming.
        
        Yields:
            float: A simulated data point with noise and occasional anomalies.
    """ 

    t = 0
    count = 0
    while True:
        # Generate the regular pattern using a sine wave
        regular_pattern = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add the seasonal component a cosine wave
        seasonal_component = seasonal_amplitude * np.cos(2 * np.pi * seasonal_frequency * t)
        
        # Add random noise (Gaussian distribution)
        noise = np.random.normal(scale=noise_scale)
        
        # Combine all components to create the data point
        data_point = regular_pattern + seasonal_component + noise
        
        # Randomly inject an anomaly with probability of 5%
        if count > 70 and random.random() < 0.05:
            anomaly_magnitude = random.uniform(10, 20)  # Randomly generate the magnitude of the anomaly between 10 and 20 units
            data_point += anomaly_magnitude * random.choice([-1, 1])  # Randomly inject positive or negative anomalies
        
        # Yield the generated data point
        yield data_point
        
        # Increment time and counter for the next data point
        t += 0.01
        count += 1
        
        # Simulate real-time streaming
        time.sleep(sleep_time)



def initialize_dynamic_plot():
    """
        Initializes a real-time dynamic plot.
        
        Returns:
            tuple: Figure, axis, line plot, data_window list, and initial x-axis data array.
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axis
    data_window = []  # Initialize an empty data window (grows over time)
    x_data = []  # Initialize an empty x-axis data array (grows over time)
    line, = ax.plot([], [], color='blue', zorder=1)  # Create an empty line plot with blue color and zorder=1 (sets its layering below scatter points)

    ax.set_ylim((-25, 25))  # Set initial y-axis limits
    ax.set_xlim((0, 200))  # Set initial x-axis limits (we'll update this later)

    return fig, ax, line, data_window, x_data


def update_dynamic_plot(data_window, color_window, x_data, line, scatter, ax):
    """
        Updates the real-time plot with new data and dynamically extends the x-axis when needed.
    
        Args:
            data_window (list): A list of data points to be plotted.
            color_window (list): A list of colors for each data point (red for anomaly, blue for normal).
            x_data (list): The x-axis data points, grows as more data is added.
            line (Line2D): The line object for the regular data plot.
            scatter (PathCollection): The scatter object for anomaly visualization.
            ax (Axes): The axis object to update the x-axis limits.
    """
    # Update the line plot with new data
    line.set_ydata(data_window)
    line.set_xdata(x_data)

    # Update the scatter plot with the same new data and colors
    scatter.set_offsets(np.c_[x_data, data_window])
    scatter.set_color(color_window)

    # Dynamically extend the x-axis when needed
    if len(x_data) > ax.get_xlim()[1]:
        ax.set_xlim(0, len(x_data) + 100)  # Extend the x-axis by 100 more data points

    plt.draw()
    plt.pause(0.01)



def rolling_z_score_anomaly_detection(data_point, window, window_size, z_threshold):
    """
        Detects anomalies using the rolling z-score method.
        
        Args:
            data_point (float): The current data point.
            window (deque): A rolling window of recent data points.
            window_size (int): The size of the rolling window.
            z_threshold (float): The z-score threshold for detecting anomalies.
        
        Returns:
            bool: True if the data point is detected as an anomaly, False otherwise.
    """

    window.append(data_point)

    if len(window) == window_size:
        mean = np.mean(window)  # Calculate mean of the window
        std_dev = np.std(window)    # Calculate standard deviation of the window

        # Avoid division by zero
        z_score = 0 if std_dev == 0 else (data_point - mean) / std_dev  # Calculate z-score

        if abs(z_score) > z_threshold:  # Check if the absolute value of the z-score exceeds the threshold
            return True
    
    return False



def isolation_forest_anomaly_detection(iso_forest, data_point, data_buffer, buffer_size):
    """
        Detects anomalies using the Isolation Forest method.
        
        Args:
            iso_forest (IsolationForest): An instance of the IsolationForest model.
            data_point (float): The current data point.
            data_buffer (deque): A buffer of recent data points.
            buffer_size (int): The size of the data buffer.
        
        Returns:
            bool: True if the data point is detected as an anomaly, False otherwise.
    """
    
    data_buffer.append([data_point])  # Add new data point to the buffer

    if len(data_buffer) >= buffer_size:    
        anomaly_labels = iso_forest.fit_predict(data_buffer)  # Fit the model to the buffer & predict anomalies
        if anomaly_labels[-1] == -1:  # Check if the latest data point is an anomaly
            return True
        
    return False



def parallel_anomaly_detection(data_stream):
    """
     TODO: Runs both anomaly detection algorithms in parallel and updates the plot in real time.
    """
    # Initialize variables
    plot_window_size=500
    rolling_window_size=70
    buffer_size = 100
    z_threshold = 3
    window = deque(maxlen=rolling_window_size)
    data_buffer = deque(maxlen=buffer_size)
    iso_forest = IsolationForest(contamination=0.05, n_estimators=150, random_state=42)
 
    # Initialize plot
    fig, ax, line, data_window, x_data = initialize_dynamic_plot()
    color_window = []  # Initialize an empty color window
    scatter = ax.scatter([], [], color=[], zorder=2)

    # TODO: Anomalies Comment
    z_score_anomalies = []
    iso_anomalies = []
    all_anomalies = []

    for index, data_point in enumerate(data_stream):
        # Update the rolling window with the new data point
        data_window.append(data_point)
        x_data.append(index)

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

        # Update the plot with new data and color window
        update_dynamic_plot(data_window, color_window, x_data, line, scatter, ax)


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


