import pandas as pd
import os

DATA_PATH = 'data/preprocessed_2022.csv' # Using 2022 as we saw GP trades there in the report

def find_gp_trades():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure Date is datetime
    if 'Date' not in df.columns and 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'])
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    # We need to simulate the logic briefly to find the trades, 
    # or just look at the signals if they are pre-calculated columns.
    # The validation script simulates execution. Let's look for the signal columns first.
    
    # In signal_validation.py, GP trades are triggered when:
    # if i in gp_longs: ...
    # gp_longs = np.where(df['dist_to_gp_long'] < 0.01)[0]
    
    # Let's find where these signals happen
    gp_long_indices = df[df['dist_to_gp_long'] < 0.01].index
    gp_short_indices = df[df['dist_to_gp_short'] < 0.01].index
    
    print(f"Found {len(gp_long_indices)} GP Long signals and {len(gp_short_indices)} GP Short signals.")
    
    # Group by week to find a dense week
    df['Week'] = df['Date'].dt.to_period('W')
    
    gp_signals = pd.concat([
        df.iloc[gp_long_indices].assign(Type='GP Long'),
        df.iloc[gp_short_indices].assign(Type='GP Short')
    ])
    
    weekly_counts = gp_signals.groupby('Week').size().sort_values(ascending=False)
    print("\nTop 5 Weeks with most GP signals:")
    print(weekly_counts.head(5))
    
    # Let's pick the top week and print the start/end date
    if not weekly_counts.empty:
        top_week = weekly_counts.index[0]
        print(f"\nTop Week: {top_week}")
        print(f"Start Date: {top_week.start_time}")
        print(f"End Date: {top_week.end_time}")

if __name__ == "__main__":
    find_gp_trades()
