import pandas as pd
import numpy as np
from envs.futures_lstm_env import CryptoLstmEnv
from sb3_contrib import RecurrentPPO
import os
import matplotlib.pyplot as plt

# 1. Load the Data
DATA_PATH = 'data/preprocessed_2021.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

print("Using full 2021 data for validation...")

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

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    balance_history.append(env.balance)
    
    # Log Trades
    if 'swing_entry' in info:
        side = env.swing_trade['side']
        if side:
            trade_type = 'entry_long' if side == 'long' else 'entry_short'
            print(f"\n[OPEN SWING {side.upper()}] Step {env.current_step}")
            print(f"  Price: {env.prices[env.current_step]:.2f} | SL: {env.swing_trade['sl']:.2f}")
            print(f"  Reason: {env.swing_trade['reason']}")
            trade_events.append({'step': env.current_step, 'price': env.prices[env.current_step], 'type': trade_type, 'desc': env.swing_trade['reason']})
        
    if 'scalp_entry' in info:
        side = env.scalp_trade['side']
        if side:
            trade_type = 'entry_long' if side == 'long' else 'entry_short'
            print(f"\n[OPEN SCALP {side.upper()}] Step {env.current_step}")
            print(f"  Price: {env.prices[env.current_step]:.2f} | SL: {env.scalp_trade['sl']:.2f}")
            print(f"  Reason: {env.scalp_trade['reason']}")
            trade_events.append({'step': env.current_step, 'price': env.prices[env.current_step], 'type': trade_type, 'desc': env.scalp_trade['reason']})

    # Check for Exits (Swing)
    # We can check if the swing trade was closed in this step by checking if it's now empty but wasn't before?
    # Easier: The env returns info['swing_pnl'] if a trade closed.
    
    # Note: The env logic for info keys needs to be checked. 
    # Looking at env code: 
    # if pnl != 0: info['swing_exit_pnl'] = pnl
    
    # Let's just check if balance changed significantly or use the info keys if available
    # But we don't have easy access to "just closed" state without modifying env or tracking state here.
    # We'll rely on the print statements in the loop for now, or better:
    # The env prints are not captured here. We should add our own logging based on state changes.
    
    # Actually, let's just look at the info dict returned
    if 'swing_manual_close' in info or 'swing_sl_hit' in info or 'swing_tp_hit' in info:
        # We don't get the exact PnL in info easily unless we modified env to pass it.
        # But we can infer it from balance change (approx)
        pnl = balance_history[-1] - balance_history[-2]
        result = "WIN" if pnl > 0 else "LOSS"
        if pnl != 0: # Filter out noise
            print(f"  [CLOSE {result}] Swing PnL: ${pnl:.2f} | New Balance: ${env.balance:.2f}")
            trade_events.append({'step': env.current_step, 'price': env.prices[env.current_step], 'type': 'exit_win' if pnl > 0 else 'exit_loss', 'desc': f"Swing {result}"})
            detailed_trade_log.append({'pnl': pnl, 'reason': 'swing_exit'})
            if pnl > 0: wins += 1
            else: losses += 1

    if 'scalp_manual_close' in info or 'scalp_sl_hit' in info or 'scalp_tp_hit' in info:
        pnl = balance_history[-1] - balance_history[-2]
        # If both closed same step, this might be mixed, but rare.
        result = "WIN" if pnl > 0 else "LOSS"
        if pnl != 0:
            print(f"  [CLOSE {result}] Scalp PnL: ${pnl:.2f} | New Balance: ${env.balance:.2f}")
            trade_events.append({'step': env.current_step, 'price': env.prices[env.current_step], 'type': 'exit_win' if pnl > 0 else 'exit_loss', 'desc': f"Scalp {result}"})
            detailed_trade_log.append({'pnl': pnl, 'reason': 'scalp_exit'})
            if pnl > 0: wins += 1
            else: losses += 1

# 5. Results Analysis
total_return = (env.balance - 10_000) / 10_000 * 100
print("\n" + "="*30)
print(f"       FINAL RESULTS 2021")
print("="*30)
print(f"Final Balance: ${env.balance:.2f}")
print(f"Total Return:  {total_return:.2f}%")
print(f"Total Trades:  {wins + losses}")
print(f"Win Rate:      {wins / (wins + losses) * 100 if (wins+losses) > 0 else 0:.2f}%")
print("="*30)

# Save detailed log
df_trades = pd.DataFrame(detailed_trade_log)
df_trades.to_csv('trade_log_detailed_2021.csv', index=False)
print("Detailed trade log saved to 'trade_log_detailed_2021.csv'")

# --- 7. Advanced Visualization ---
try:
    # Setup Data for Plotting
    # Get price data corresponding to the backtest period
    price_data = env.prices[start_step : env.current_step]
    steps = range(len(price_data))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Price Action & Trades
    ax1.plot(steps, price_data, label='Price', color='gray', alpha=0.6, linewidth=1)
    
    # Plot Markers
    for event in trade_events:
        # Adjust step to be relative to start_step
        rel_step = event['step'] - start_step
        if rel_step < 0 or rel_step >= len(steps): continue
        
        if event['type'] == 'entry_long':
            ax1.scatter(rel_step, event['price'], marker='^', color='green', s=100, label='Long' if 'Long' not in ax1.get_legend_handles_labels()[1] else "")
        elif event['type'] == 'entry_short':
            ax1.scatter(rel_step, event['price'], marker='v', color='red', s=100, label='Short' if 'Short' not in ax1.get_legend_handles_labels()[1] else "")
        elif event['type'] == 'exit_win':
            ax1.scatter(rel_step, event['price'], marker='o', color='blue', s=50)
        elif event['type'] == 'exit_loss':
            ax1.scatter(rel_step, event['price'], marker='x', color='black', s=50)

    ax1.set_title('Price Action with Trade Entries & Exits (2021)')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity Curve
    ax2.plot(balance_history, label='Account Balance', color='blue', linewidth=2)
    ax2.fill_between(range(len(balance_history)), balance_history, 10000, alpha=0.1, color='blue')
    ax2.set_title('Account Balance Growth (2021)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Balance ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_dashboard_2021.png')
    print("Advanced dashboard saved to 'backtest_dashboard_2021.png'")
    
except Exception as e:
    print(f"Could not save plot: {e}")
