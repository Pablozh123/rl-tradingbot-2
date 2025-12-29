import pandas as pd

try:
    # Load the data
    df = pd.read_csv('data/preprocessed_2023.csv')

    # Get details for the Entry (Step 242066) and Exit (around Step 250801)
    entry_step = 242066
    exit_step = 250801

    print("--- ENTRY DETAILS (Step 242066) ---")
    print(df.iloc[entry_step][['timestamp', 'Open', 'High', 'Low', 'Close']])

    print("\n--- EXIT DETAILS (Step 250801) ---")
    print(df.iloc[exit_step][['timestamp', 'Open', 'High', 'Low', 'Close']])

except Exception as e:
    print(f"Error: {e}")
