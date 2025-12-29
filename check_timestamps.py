import pandas as pd
import os

files_to_check = [
    'data/btc_2022_5m.csv',
    'data/btc_2023_5m.csv',
    'data/preprocessed_2023.csv',
    'data/preprocessed_train_22_23.csv',
    'data/preprocessed_test_2024.csv'
]

print(f"{'File':<35} | {'Start Date':<20} | {'End Date':<20} | {'Rows':<10}")
print("-" * 95)

for file_path in files_to_check:
    if os.path.exists(file_path):
        try:
            # Read only timestamp column to be fast
            df = pd.read_csv(file_path, usecols=['timestamp'])
            start_date = df['timestamp'].iloc[0]
            end_date = df['timestamp'].iloc[-1]
            rows = len(df)
            print(f"{file_path:<35} | {str(start_date):<20} | {str(end_date):<20} | {rows:<10}")
        except Exception as e:
            print(f"{file_path:<35} | Error: {e}")
    else:
        print(f"{file_path:<35} | NOT FOUND")
