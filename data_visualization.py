import matplotlib.pyplot as plt


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
