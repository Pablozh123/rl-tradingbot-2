import pandas as pd
import numpy as np

LOG_FILE = 'trade_log_detailed_2020.csv'
OUTPUT_FILE = 'Results_2020_Detailed.txt'

def calculate_metrics(df, name="All Trades"):
    if df.empty:
        return f"\n--- {name} ---\nNo trades found.\n"

    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades) * 100
    
    total_pnl = df['pnl'].sum()
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    avg_pnl = df['pnl'].mean()
    
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    # Max Drawdown Calculation (Equity Curve based on this subset of trades)
    # Note: This is an approximation as it assumes these trades happened in sequence without others in between
    # For "All Trades" it is accurate. For subsets, it shows the performance of that strategy in isolation.
    equity = [10000]
    for pnl in df['pnl']:
        equity.append(equity[-1] + pnl)
    
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown_pct = drawdown.max() * 100
    
    report = f"\n--- {name} ---\n"
    report += f"Total Trades:       {total_trades}\n"
    report += f"Net Profit:         ${total_pnl:.2f}\n"
    report += f"Win Rate:           {win_rate:.2f}%\n"
    report += f"Profit Factor:      {profit_factor:.2f}\n"
    report += f"Avg Trade:          ${avg_pnl:.2f}\n"
    report += f"Avg Win:            ${avg_win:.2f}\n"
    report += f"Avg Loss:           ${avg_loss:.2f}\n"
    report += f"Risk/Reward Ratio:  {rr_ratio:.2f}\n"
    report += f"Max Drawdown:       {max_drawdown_pct:.2f}%\n"
    
    return report

def main():
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print(f"Error: {LOG_FILE} not found. Run evaluate_2020.py first.")
        return

    # Separate Strategies
    # In evaluate_2020.py:
    # Swing trades (SFP) are logged as 'swing_exit'
    # Scalp trades (Grind) are logged as 'scalp_exit'
    
    df_swing = df[df['reason'] == 'swing_exit']
    df_scalp = df[df['reason'] == 'scalp_exit']
    
    output = "==================================================\n"
    output += "       DETAILED PERFORMANCE ANALYSIS 2020\n"
    output += "==================================================\n"
    
    # 1. Overall Performance
    output += calculate_metrics(df, "OVERALL PERFORMANCE")
    
    # 2. Strategy Breakdown
    output += "\n" + "="*50 + "\n"
    output += "           STRATEGY BREAKDOWN\n"
    output += "="*50 + "\n"
    
    output += calculate_metrics(df_swing, "STRATEGY 1: SFP (Swing)")
    output += calculate_metrics(df_scalp, "STRATEGY 2: GRIND (Scalp)")
    
    # 3. Conclusion
    output += "\n" + "="*50 + "\n"
    output += "               CONCLUSION\n"
    output += "="*50 + "\n"
    
    best_strategy = "SFP" if df_swing['pnl'].sum() > df_scalp['pnl'].sum() else "Grind"
    output += f"Best Performing Strategy: {best_strategy}\n"
    
    print(output)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(output)
    print(f"\nAnalysis saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
