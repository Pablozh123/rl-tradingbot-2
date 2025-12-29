import pandas as pd
import os

file_path = 'data/btc_2024_5m.csv'
if os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path, usecols=['timestamp'])
        print(f"{file_path} Start: {df['timestamp'].iloc[0]}")
        print(f"{file_path} End:   {df['timestamp'].iloc[-1]}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("File not found")
