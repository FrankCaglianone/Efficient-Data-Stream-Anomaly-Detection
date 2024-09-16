import numpy as np
import pandas as pd
import time
import random







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






stream = data_stream_generator()


for _ in range(10):  # Adjust the range as needed
    next(stream)















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
















