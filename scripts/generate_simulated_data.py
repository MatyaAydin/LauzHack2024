import pandas as pd
import numpy as np
import os

# Define the parameters for simulation
files = [
    "29_11-18-24.csv", "1_12-6-12.csv", "1_12-12-18.csv",
    "1_12-18-24.csv", "1_12-24-6.csv", "29_11-6-12.csv",
    "29_11-12-18.csv", "29_11-24-6.csv", "30_11-6-12.csv",
    "30_11-12-18.csv", "30_11-18-24.csv", "30_11-24-6.csv"
]

columns = ["ID", "timestamp_in", "timestamp_out", "duration", "direction"]

# Create a directory for the CSV files
os.makedirs("simulated_boat_data", exist_ok=True)

for file_name in files:
    num_records = np.random.randint(5, 20)  # Random number of boat records per file
    data = []
    for i in range(num_records):
        boat_id = i + 1
        timestamp_in = np.round(np.random.uniform(0, 9000), 2)  # Random entry time within 3-hour window
        duration = np.round(np.random.uniform(30, 300), 2)  # Boat stays in the area between 30 and 300 seconds
        timestamp_out = timestamp_in + duration
        direction = np.random.choice(["up", "down"])  # Random direction of movement
        data.append([boat_id, timestamp_in, timestamp_out, duration, direction])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=columns)
    output_path = os.path.join("simulated_boat_data", file_name)
    df.to_csv(output_path, index=False)

print("Simulated data generated successfully!")
