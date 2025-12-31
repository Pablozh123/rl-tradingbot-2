import pandas as pd
import numpy as np
from envs.futures_lstm_env import CryptoLstmEnv
from sb3_contrib import RecurrentPPO
import os
import matplotlib.pyplot as plt

# 1. Load the Data
DATA_PATH = 'data/preprocessed_2020.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

print("Using full 2020 data for validation...")

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

# FORCE START AT BEGINNING (Fix for random start in Env)
env.current_step = env.window_size
obs = env._get_observation() # Update observation to match start
print(f"Forced start at step {env.current_step} to test full year.")

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
    action, _states = model.predict(obs, deterministic=True)
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
    if 'swing_manual_close' in info or 'swing_sl_hit' in info or 'swing_reversal_exit' in info:
        pnl = balance_history[-1] - balance_history[-2]
        result = "WIN" if pnl > 0 else "LOSS"
        if pnl != 0: # Filter out noise
            print(f"  [CLOSE {result}] Swing PnL: ${pnl:.2f} | New Balance: ${env.balance:.2f}")
            trade_events.append({'step': env.current_step, 'price': env.prices[env.current_step], 'type': 'exit_win' if pnl > 0 else 'exit_loss', 'desc': f"Swing {result}"})
            detailed_trade_log.append({'pnl': pnl, 'reason': 'swing_exit'})
            if pnl > 0: wins += 1
            else: losses += 1

    # Check for Exits (Scalp)
    if 'scalp_manual_close' in info or 'scalp_sl_hit' in info or 'scalp_trend_exit' in info or 'scalp_partial_tp' in info:
        pnl = balance_history[-1] - balance_history[-2]
        # Partial TP is always a win, but we need to be careful not to double count trades
        # For simplicity, we count every PnL event as a "trade outcome" here
        result = "WIN" if pnl > 0 else "LOSS"
        if pnl != 0:
            print(f"  [CLOSE {result}] Scalp PnL: ${pnl:.2f} | New Balance: ${env.balance:.2f}")
            trade_events.append({'step': env.current_step, 'price': env.prices[env.current_step], 'type': 'exit_win' if pnl > 0 else 'exit_loss', 'desc': f"Scalp {result}"})
            detailed_trade_log.append({'pnl': pnl, 'reason': 'scalp_exit'})
            if pnl > 0: wins += 1
            else: losses += 1

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
print("BACKTEST RESULTS (2020)")
print("-" * 50)
print(f"Final Balance:      ${env.balance:.2f}")
print(f"Total Return:       {total_return_pct:.2f}%")
print(f"Total Trades:       {total_trades}")
print(f"Win Rate:           {win_rate:.2f}%")
print(f"Max Drawdown:       {max_drawdown:.2f}%")
print("-" * 50)

# 6. Plotting
plt.figure(figsize=(12, 6))
plt.plot(balance_history, label='Account Balance')
plt.title(f'Backtest 2020 - Return: {total_return_pct:.2f}%')
plt.xlabel('Steps')
plt.ylabel('Balance ($)')
plt.legend()
plt.grid(True)
plt.savefig('backtest_dashboard_2020.png')
print("Saved plot to backtest_dashboard_2020.png")

# Save detailed log
df_log = pd.DataFrame(detailed_trade_log)
df_log.to_csv('trade_log_detailed_2020.csv', index=False)
print("Saved detailed trade log to trade_log_detailed_2020.csv")

# Save trade events for visualization
df_events = pd.DataFrame(trade_events)
df_events.to_csv('trade_events_2020.csv', index=False)
print("Saved trade events to trade_events_2020.csv")
