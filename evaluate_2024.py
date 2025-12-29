import pandas as pd
import numpy as np
from envs.futures_lstm_env import CryptoLstmEnv
from sb3_contrib import RecurrentPPO
import os

# 1. Load the Data
DATA_PATH = 'data/preprocessed_test_2024.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

print("Using full 2024 data for validation...")

# 2. Initialize the Environment
# We use the same window_size as during training
env = CryptoLstmEnv(data=data, window_size=50, initial_balance=10_000)

# 3. Load the Trained Model
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

    # --- 2. Show Exit Details (The Result) ---
    # Check if balance changed (Trade Closed)
    if env.balance != prev_balance:
        pnl = env.balance - prev_balance
        idx = env.current_step - 1
        
        # Determine which trade closed (Swing or Scalp)
        # Note: This is a simplification. Ideally, the env should tell us which one closed.
        # But since we only have one of each max, we can infer or just log the PnL.
        
        trade_type = "Unknown"
        trade_info = {}
        
        # Check if Swing closed
        if active_swing_trade and env.swing_trade['size'] == 0:
            trade_type = "Swing"
            trade_info = active_swing_trade
            active_swing_trade = None
        elif active_scalp_trade and env.scalp_trade['size'] == 0:
            trade_type = "Scalp"
            trade_info = active_scalp_trade
            active_scalp_trade = None
            
        if pnl > 0:
            print(f"  [WIN] {trade_type} PnL: +${pnl:.2f} | New Balance: ${env.balance:.2f}")
            wins += 1
            trade_events.append({'step': step_idx, 'price': env.prices[idx], 'type': 'exit_win', 'desc': f"Win: +{pnl:.2f}"})
        else:
            print(f"  [LOSS] {trade_type} PnL: -${abs(pnl):.2f} | New Balance: ${env.balance:.2f}")
            losses += 1
            trade_events.append({'step': step_idx, 'price': env.prices[idx], 'type': 'exit_loss', 'desc': f"Loss: {pnl:.2f}"})
            
        trades.append(pnl)
        
        # Log Detailed Trade
        if trade_info:
            detailed_trade_log.append({
                'entry_step': trade_info['entry_step'],
                'exit_step': idx,
                'type': trade_info['type'],
                'side': trade_info['side'],
                'reason': trade_info['reason'],
                'entry_price': trade_info['entry_price'],
                'exit_price': env.prices[idx],
                'pnl': pnl,
                'balance': env.balance
            })

    balance_history.append(env.balance)

# 5. Summary
print("\n" + "=" * 30)
print("       FINAL RESULTS 2024       ")
print("=" * 30)
print(f"Final Balance: ${env.balance:.2f}")
print(f"Total Return:  {((env.balance - 10000) / 10000) * 100:.2f}%")
print(f"Total Trades:  {len(trades)}")
print(f"Win Rate:      {(wins / len(trades)) * 100 if len(trades) > 0 else 0:.2f}%")
print("=" * 30)

# 6. Detailed Analysis by Strategy
if len(detailed_trade_log) > 0:
    df_trades = pd.DataFrame(detailed_trade_log)
    
    print("\n--- Performance by Strategy ---")
    print(f"{'Strategy':<20} | {'Count':<6} | {'Win%':<6} | {'Total PnL':<9} | {'Avg PnL':<7}")
    print("-" * 60)
    
    for name, group in df_trades.groupby('reason'):
        count = len(group)
        wins_strat = len(group[group['pnl'] > 0])
        wr = (wins_strat / count) * 100
        total_pnl = group['pnl'].sum()
        avg_pnl = group['pnl'].mean()
        
        print(f"{name[:20]:<20} | {count:<6} | {wr:6.1f}% | ${total_pnl:9.2f} | ${avg_pnl:7.2f}")
        
    print("-" * 60)
    
    # Save to CSV
    df_trades.to_csv('trade_log_detailed_2024.csv', index=False)
    print("Detailed trade log saved to 'trade_log_detailed_2024.csv'")

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

    ax1.set_title('Price Action with Trade Entries & Exits (2024)')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity Curve
    ax2.plot(balance_history, label='Account Balance', color='blue', linewidth=2)
    ax2.fill_between(range(len(balance_history)), balance_history, 10000, alpha=0.1, color='blue')
    ax2.set_title('Account Balance Growth (2024)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Balance ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_dashboard_2024.png')
    print("Advanced dashboard saved to 'backtest_dashboard_2024.png'")
    
except Exception as e:
    print(f"Could not save plot: {e}")
