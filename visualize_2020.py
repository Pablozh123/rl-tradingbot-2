import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load Data
EVENTS_FILE = 'trade_events_2020.csv'
LOG_FILE = 'trade_log_detailed_2020.csv'
DATA_FILE = 'data/preprocessed_2020.csv'

def visualize_2020_results():
    # 1. Load Trade Log & Data
    if not os.path.exists(EVENTS_FILE) or not os.path.exists(DATA_FILE):
        print(f"Files not found. Please run evaluate_2020.py first. Checking {EVENTS_FILE} and {DATA_FILE}")
        return

    df_events = pd.read_csv(EVENTS_FILE)
    df_data = pd.read_csv(DATA_FILE)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    
    # 2. Reconstruct Equity Curve (Optional, but good to have)
    if os.path.exists(LOG_FILE):
        df_trades = pd.read_csv(LOG_FILE)
        equity = [10000]
        for pnl in df_trades['pnl']:
            equity.append(equity[-1] + pnl)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity, color='green', linewidth=2)
        plt.title('Equity Curve 2020 (Out-of-Sample)', fontsize=14)
        plt.xlabel('Number of Trades')
        plt.ylabel('Account Balance ($)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.legend()
        plt.savefig('equity_curve_2020.png')
        print("Saved equity_curve_2020.png")

    # 3. Visualize ALL Trades on Chart
    print("Generating full year chart with all trades...")
    
    plt.figure(figsize=(24, 10)) # Wide figure
    
    # Plot Price
    plt.plot(df_data['timestamp'], df_data['Close'], label='BTC Price', color='gray', alpha=0.5, linewidth=1)
    
    # Filter Events
    entries_long = df_events[df_events['type'] == 'entry_long']
    entries_short = df_events[df_events['type'] == 'entry_short']
    exits_win = df_events[df_events['type'] == 'exit_win']
    exits_loss = df_events[df_events['type'] == 'exit_loss']
    
    # Map steps to timestamps
    # We need to ensure the step indices are valid
    valid_indices = df_data.index
    
    # Helper to get data for plotting
    def get_plot_data(events_df):
        valid_events = events_df[events_df['step'].isin(valid_indices)]
        timestamps = df_data.loc[valid_events['step'], 'timestamp']
        prices = valid_events['price']
        return timestamps, prices

    # Plot Entries
    t_long, p_long = get_plot_data(entries_long)
    plt.scatter(t_long, p_long, color='green', s=100, marker='^', label='Long Entry', zorder=5)
    
    t_short, p_short = get_plot_data(entries_short)
    plt.scatter(t_short, p_short, color='red', s=100, marker='v', label='Short Entry', zorder=5)
    
    # Plot Exits
    t_win, p_win = get_plot_data(exits_win)
    plt.scatter(t_win, p_win, color='gold', s=80, marker='*', label='Exit Win', zorder=6)
    
    t_loss, p_loss = get_plot_data(exits_loss)
    plt.scatter(t_loss, p_loss, color='black', s=50, marker='x', label='Exit Loss', zorder=6)

    plt.title('BTC 2020 Price Action & All Trades', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    output_file = 'all_trades_2020_chart.png'
    plt.savefig(output_file, dpi=300) # High DPI for zoom
    print(f"Saved {output_file}")

if __name__ == "__main__":
    visualize_2020_results()
