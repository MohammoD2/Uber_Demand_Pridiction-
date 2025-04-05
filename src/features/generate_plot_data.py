import pandas as pd
from pathlib import Path

# Load the final data
final_data_path = Path(__file__).resolve().parents[2] / "data/interim/final_data.csv"
final_data = pd.read_csv(final_data_path, parse_dates=["tpep_pickup_datetime"])

# Select relevant columns
final_data = final_data[['pickup_longitude', 'pickup_latitude', 'region']]

# Randomly select 500 data points from each region
plot_data = final_data.groupby('region').apply(lambda x: x.sample(n=500, random_state=1)).reset_index(drop=True)

# Save the plot data to CSV
plot_data_path = Path(__file__).resolve().parents[2] / "data/processed/plot_data.csv"
plot_data.to_csv(plot_data_path, index=False)

print("plot_data.csv has been generated successfully.")
