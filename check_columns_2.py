import pandas as pd
import os

files_to_check = [
    'data/preprocessed_train_22_23.csv',
    'data/preprocessed_test_2024.csv'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, nrows=0)
            print(f"{file_path} columns: {list(df.columns)}")
        except Exception as e:
            print(f"{file_path} Error: {e}")
