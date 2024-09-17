# Efficient-Data-Stream-Anomaly-Detection
Cobblestone





Note Sliding Window: Instead of clearing the entire buffer after each detection round, we now implement a sliding window approach by keeping most of the buffer (removing only step_size points) to preserve the recent history of the data stream.
