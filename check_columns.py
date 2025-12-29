import pandas as pd

try:
    # Load only the header
    df = pd.read_csv('data/preprocessed_2023.csv', nrows=0)
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error: {e}")
