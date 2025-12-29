import pandas as pd
import numpy as np
from envs.futures_lstm_env import CryptoLstmEnv
from sb3_contrib import RecurrentPPO
import os

# 1. Load the Data
DATA_PATH = 'data/preprocessed_2023.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

# Slice to second half for validation (~52,000 candles onwards)
print("Using full 2023 data for validation...")
# data = data.iloc[52000:].copy()

# 2. Initialize the Environment
# We use the same window_size as during training
env = CryptoLstmEnv(data=data, window_size=50, initial_balance=10_000)

# 3. Load the Trained Model
# The .zip extension is added automatically by save(), but load() handles it.
MODEL_PATH = "models/ppo_lstm_parallel_1765930806"
print(f"Loading model from {MODEL_PATH}...")

try:
    model = RecurrentPPO.load(MODEL_PATH, env=env)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 4. Run a Test Loop (Backtest)
obs, _ = env.reset()
done = False

# Statistics Tracking
balance_history = [env.balance]
trades = []
wins = 0
losses = 0

# Visualization Data
trade_events = [] # Stores (step, price, type, description)
detailed_trade_log = [] # Stores full trade details for analysis

print("\nStarting Backtest...")
print("-" * 50)

# We need to track the offset because env.reset() starts at window_size
start_step = env.current_step 
active_swing_trade = None
active_scalp_trade = None

while not done:
    prev_balance = env.balance
    step_idx = env.current_step - start_step # Relative step for plotting
    
    # Predict action
    # Use deterministic=False to see the model's "creative" side (like in training)
    # Use deterministic=True for the strict "best" behavior
    action, _states = model.predict(obs, deterministic=False)
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    
    # --- 1. Show Entry Details (The "Why") ---
    # Check for Swing Entry
    if 'swing_entry' in info:
        entry_data = env.swing_trade # Get current state directly
        idx = env.current_step - 1
        reason = entry_data['reason']
        
        if entry_data['side'] is not None:
            print(f"\n[OPEN SWING {entry_data['side'].upper()}] Step {idx}")
            print(f"  Price: {env.prices[idx]:.2f} | SL: {entry_data['sl']:.2f}")
            print(f"  Reason: {reason}")
            
            trade_events.append({
                'step': step_idx,
                'price': env.prices[idx],
                'type': f"entry_{entry_data['side']}",
                'desc': f"Swing: {reason}"
            })
            
            active_swing_trade = {
                'entry_step': idx,
                'entry_price': env.prices[idx],
                'side': entry_data['side'],
                'reason': reason,
                'sl': entry_data['sl'],
                'type': 'swing'
            }

    # Check for Scalp Entry
    if 'scalp_entry' in info:
        entry_data = env.scalp_trade
        idx = env.current_step - 1
        reason = entry_data['reason']
        
        if entry_data['side'] is not None:
            print(f"\n[OPEN SCALP {entry_data['side'].upper()}] Step {idx}")
            print(f"  Price: {env.prices[idx]:.2f} | SL: {entry_data['sl']:.2f}")
            print(f"  Reason: {reason}")
            
            trade_events.append({
                'step': step_idx,
                'price': env.prices[idx],
                'type': f"entry_{entry_data['side']}",
                'desc': f"Scalp: {reason}"
            })
            
            active_scalp_trade = {
                'entry_step': idx,
                'entry_price': env.prices[idx],
                'side': entry_data['side'],
                'reason': reason,
                'sl': entry_data['sl'],
                'type': 'scalp'
            }

    # --- 2. Show Exit Details ---
    current_balance = env.balance
    balance_history.append(current_balance)
    
    pnl = current_balance - prev_balance
    
    if pnl != 0:
        trades.append(pnl)
        exit_price = env.prices[env.current_step - 1]
        
        if pnl > 0:
            wins += 1
            print(f"  [CLOSE WIN] +${pnl:6.2f} | Balance: ${current_balance:.2f}")
            trade_events.append({'step': step_idx, 'price': exit_price, 'type': 'exit_win', 'desc': f"+{pnl:.2f}"})
        else:
            losses += 1
            print(f"  [CLOSE LOSS] -${abs(pnl):6.2f} | Balance: ${current_balance:.2f}")
            trade_events.append({'step': step_idx, 'price': exit_price, 'type': 'exit_loss', 'desc': f"{pnl:.2f}"})
            
        # Determine which trade closed based on info flags
        swing_closed = 'swing_sl_hit' in info or 'swing_reversal_exit' in info
        scalp_closed = 'scalp_manual_close' in info or 'scalp_sl_hit' in info or 'scalp_trend_exit' in info
        scalp_partial = 'scalp_partial_tp' in info
        
        if swing_closed and not scalp_closed and active_swing_trade:
             detailed_trade_log.append({
                 'pnl': pnl,
                 'reason': active_swing_trade['reason']
             })
             active_swing_trade = None
        elif (scalp_closed or scalp_partial) and not swing_closed and active_scalp_trade:
             detailed_trade_log.append({
                 'pnl': pnl,
                 'reason': active_scalp_trade['reason']
             })
             if scalp_closed:
                 active_scalp_trade = None
        elif swing_closed and (scalp_closed or scalp_partial):
             # Both closed. We can't split PnL easily without env support.
             # Log as 'mixed'
             detailed_trade_log.append({
                 'pnl': pnl,
                 'reason': 'mixed_exit'
             })
             active_swing_trade = None
             if scalp_closed:
                 active_scalp_trade = None
        else:
             # Fallback: Trade closed but logic didn't catch which one.
             # This ensures the PnL is at least recorded in the table.
             detailed_trade_log.append({
                 'pnl': pnl,
                 'reason': 'unattributed_exit'
             })
             # Reset both to be safe, though this might kill a valid active trade tracking
             # active_swing_trade = None 
             # active_scalp_trade = None

# 5. Calculate Statistics
balance_history = np.array(balance_history)
total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
total_return_pct = ((env.balance - 10_000) / 10_000) * 100

# Max Drawdown
peak = np.maximum.accumulate(balance_history)
drawdown = (peak - balance_history) / peak
max_drawdown = drawdown.max() * 100

print("-" * 50)
print("BACKTEST RESULTS (2023)")
print("-" * 50)
print(f"Final Balance:      ${env.balance:.2f}")
print(f"Total Return:       {total_return_pct:.2f}%")
print(f"Total Trades:       {total_trades}")
print(f"  - Longs:          {len([t for t in trade_events if t['type'] == 'entry_long'])}")
print(f"  - Shorts:         {len([t for t in trade_events if t['type'] == 'entry_short'])}")
print(f"Win Rate:           {win_rate:.2f}%")
print(f"Max Drawdown:       {max_drawdown:.2f}%")
print(f"Best Trade:         ${max(trades):.2f}" if trades else "Best Trade: N/A")
print(f"Worst Trade:        ${min(trades):.2f}" if trades else "Worst Trade: N/A")
print("-" * 50)

# --- 6. Detailed Performance Analysis ---
if detailed_trade_log:
    df_trades = pd.DataFrame(detailed_trade_log)
    
    # --- Advanced Metrics Calculation ---
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
    avg_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].mean())
    avg_rr = avg_win / avg_loss if avg_loss != 0 else 0
    
    # Sortino Ratio (based on step-wise balance history)
    returns = pd.Series(balance_history).pct_change().dropna()
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        # Annualization for 5m candles (approx 105120 candles/year)
        annual_factor = np.sqrt(365 * 24 * 12)
        downside_deviation = downside_returns.std() * annual_factor
        annualized_return = returns.mean() * (365 * 24 * 12)
        sortino = annualized_return / downside_deviation if downside_deviation != 0 else 0
    else:
        sortino = 0

    print("\nADVANCED METRICS")
    print("-" * 30)
    print(f"Profit Factor:      {profit_factor:.2f}")
    print(f"Avg Risk:Reward:    1:{avg_rr:.2f}")
    print(f"Sortino Ratio:      {sortino:.2f}")
    print("-" * 30)

    print("\nPERFORMANCE BY STRATEGY")
    print("-" * 60)
    print(f"{'Strategy':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Total PnL':<10} | {'Avg PnL':<8}")
    print("-" * 60)
    
    # Group by 'reason' (Strategy)
    # We might need to simplify reasons if they are complex combinations
    # For now, let's just use the raw reason string
    strategies = df_trades.groupby('reason')
    
    for name, group in strategies:
        count = len(group)
        wins_strat = len(group[group['pnl'] > 0])
        wr = (wins_strat / count) * 100
        total_pnl = group['pnl'].sum()
        avg_pnl = group['pnl'].mean()
        
        print(f"{name[:20]:<20} | {count:<6} | {wr:6.1f}% | ${total_pnl:9.2f} | ${avg_pnl:7.2f}")
        
    print("-" * 60)
    
    # Save to CSV
    df_trades.to_csv('trade_log_detailed_2023.csv', index=False)
    print("Detailed trade log saved to 'trade_log_detailed_2023.csv'")

# --- 7. Advanced Visualization ---
try:
    import matplotlib.pyplot as plt
    
    # Setup Data for Plotting
    # Get price data corresponding to the backtest period
    price_data = env.prices[start_step : env.current_step]
    steps = range(len(price_data))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Price Action & Trades
    ax1.plot(steps, price_data, label='Price', color='gray', alpha=0.6, linewidth=1)
    
    # --- NEW: Plot Signals (SFP/GP/Grind) ---
    # We need to get the signal arrays for the same slice
    sfp_long_mask = env.is_sfp_long[start_step : env.current_step]
    sfp_short_mask = env.is_sfp_short[start_step : env.current_step]
    gp_long_mask = env.dist_to_gp_long[start_step : env.current_step] < 0.01
    gp_short_mask = env.dist_to_gp_short[start_step : env.current_step] < 0.01
    
    # Grind Signals
    slope_slice = env.trend_slope[start_step : env.current_step]
    r2_slice = env.trend_r2[start_step : env.current_step]
    grind_long_mask = (slope_slice > 0) & (r2_slice > 0.85)
    grind_short_mask = (slope_slice < 0) & (r2_slice > 0.85)

    # Plot SFP Signals (Small dots)
    sfp_long_indices = np.where(sfp_long_mask)[0]
    if len(sfp_long_indices) > 0:
        ax1.scatter(sfp_long_indices, price_data[sfp_long_indices], marker='.', color='lime', s=20, alpha=0.5, label='SFP Long Signal')
        
    sfp_short_indices = np.where(sfp_short_mask)[0]
    if len(sfp_short_indices) > 0:
        ax1.scatter(sfp_short_indices, price_data[sfp_short_indices], marker='.', color='orange', s=20, alpha=0.5, label='SFP Short Signal')

    # Plot GP Signals (Small dots)
    gp_long_indices = np.where(gp_long_mask)[0]
    if len(gp_long_indices) > 0:
        ax1.scatter(gp_long_indices, price_data[gp_long_indices], marker='.', color='cyan', s=20, alpha=0.5, label='GP Long Signal')
        
    gp_short_indices = np.where(gp_short_mask)[0]
    if len(gp_short_indices) > 0:
        ax1.scatter(gp_short_indices, price_data[gp_short_indices], marker='.', color='magenta', s=20, alpha=0.5, label='GP Short Signal')

    # Plot Grind Signals (Small dots)
    grind_long_indices = np.where(grind_long_mask)[0]
    if len(grind_long_indices) > 0:
        ax1.scatter(grind_long_indices, price_data[grind_long_indices], marker='.', color='blue', s=15, alpha=0.3, label='Grind Long')

    grind_short_indices = np.where(grind_short_mask)[0]
    if len(grind_short_indices) > 0:
        ax1.scatter(grind_short_indices, price_data[grind_short_indices], marker='.', color='purple', s=15, alpha=0.3, label='Grind Short')

    # Plot Markers
    for event in trade_events:
        if event['type'] == 'entry_long':
            ax1.scatter(event['step'], event['price'], marker='^', color='green', s=100, label='Long' if 'Long' not in ax1.get_legend_handles_labels()[1] else "")
        elif event['type'] == 'entry_short':
            ax1.scatter(event['step'], event['price'], marker='v', color='red', s=100, label='Short' if 'Short' not in ax1.get_legend_handles_labels()[1] else "")
        elif event['type'] == 'exit_win':
            ax1.scatter(event['step'], event['price'], marker='o', color='blue', s=50)
        elif event['type'] == 'exit_loss':
            ax1.scatter(event['step'], event['price'], marker='x', color='black', s=50)

    ax1.set_title('Price Action with Trade Entries & Exits (2023)')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity Curve
    ax2.plot(balance_history, label='Account Balance', color='blue', linewidth=2)
    ax2.fill_between(range(len(balance_history)), balance_history, 10000, alpha=0.1, color='blue')
    ax2.set_title('Account Balance Growth (2023)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Balance ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_dashboard_2023.png')
    print("Advanced dashboard saved to 'backtest_dashboard_2023.png'")
    
except Exception as e:
    print(f"Could not save plot: {e}")
