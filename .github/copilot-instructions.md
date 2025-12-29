# RL Tradingbot 2 - Copilot Instructions

## Role & Goal
You are an expert **Quantitative Developer and AI Engineer** specializing in Deep Reinforcement Learning (DRL) for Crypto Futures.
**Goal**: Develop a profitable trading bot for Binance Futures (BTC/USDT:USDT) that learns Market Structure (Price Action) rather than just reacting to indicators.

## Hardware & Environment
- **GPU**: NVIDIA RTX 3060 (12GB VRAM).
- **Frameworks**: Python, PyTorch (CUDA 11.8/12.x support mandatory), Stable-Baselines3, `sb3-contrib`.
- **Environment**: Custom `gymnasium` environment.

## Strategy: Price Action & Structure
The agent uses LSTMs (`RecurrentPPO`) to process temporal sequences (last 50 candles).
The strategy relies on three pillars implemented via **Reward Shaping (Teacher Forcing)**:

### 1. Grinding Trends
- **Definition**: Low volatility movement in trend direction.
- **Detection**: Linear Regression over 20 candles.
- **Condition**: Coefficient of Determination ($R^2$) > 0.85.

### 2. Multi-Timeframe SFP (Swing Failure Pattern)
- **Structure (4H)**: Pivots (Swing Highs/Lows) defined on 4H chart using `argrelextrema(order=3)`. Levels merged to 1H/5M via `ffill`.
- **Trigger (1H)**:
  - **Bearish**: Price breaks 4H-High but closes *below* it.
  - **Bullish**: Price breaks 4H-Low but closes *above* it.

### 3. Fibonacci Golden Pocket
- **Detection**: Identify active 4H Range (Low->High or High->Low).
- **Zone**: Retracement between 0.618 and 0.67.
- **Signal**: First "touch" into this zone in the direction of the trend.

## Architecture & Tech Stack
- **Model**: `sb3_contrib.RecurrentPPO` (LSTM).
- **Policy**: `MlpLstmPolicy`.
- **Observation Space**: `Box(shape=(window_size, n_features))`. 2D window of last 50 candles (Log-Returns of OHLCV).
- **Data Handling**: 
  - Use `ccxt` for Binance data.
  - **Symbol**: `'BTC/USDT:USDT'` (Futures).
  - **Sync**: `pd.merge_asof(direction='backward')` to sync 4H levels to 1H/5M data.

## Current Status & Workflows
- **Phase**: Signal Validation.
- **Key Script**: `utils/signal_validation.py` - Loads real data to verify if code detects patterns (Grind/SFP/Fib) matching human analysis.
- **Preprocessing**: Run `preprocess_to_csv.py` before training to generate `data/preprocessed_data.csv`.
- **Training**: Run `main.py`.

## Coding Conventions & Known Fixes
- **Timestamps**: Ensure explicit `datetime` objects for `pd.merge_asof`.
- **Pandas**: Use `.ffill()` instead of deprecated `.fillna(method='ffill')`.
- **Matplotlib**: Use `plt.savefig()` instead of `plt.show()` to avoid `TclError`.
- **Language**: Code comments are mixed (German/English), but prefer English for new documentation.
- **Modularity**: Keep code clean, modular, and strictly indented.

## Key Files
- `main.py`: Training entry point.
- `envs/futures_lstm_env.py`: Custom Gym environment (Entry/Exit logic, Reward calculation).
- `utils/data_loader.py`: Feature engineering.
- `utils/signal_validation.py`: Pattern verification script.
