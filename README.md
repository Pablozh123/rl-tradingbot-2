# 🤖 RL Tradingbot 2 - Deep Reinforcement Learning for Crypto Futures

## 📌 Project Overview
**RL Tradingbot 2** is a sophisticated **Regime-Aware Trading System** that combines **Deep Reinforcement Learning (DRL)** with classical **Price Action Analysis**. 

Unlike traditional bots that blindly follow indicators, this agent acts as a "Senior Trader" managing a "Junior Analyst" (Hardcoded Logic). It evaluates signals based on market context (Regime) to decide on entries, position sizing, and exits.

### 🚀 Key Features
*   **Hybrid Architecture**: Combines Hardcoded Signal Detection (SFP, Grind, Golden Pocket) with an LSTM-based RL Agent.
*   **Algorithm**: `RecurrentPPO` (Proximal Policy Optimization with LSTM) from `sb3-contrib`.
*   **Market**: Binance Futures (BTC/USDT).
*   **Timeframe**: 5-Minute Candles (Agent looks back 50 candles / ~4 hours).
*   **Risk Management**: Dynamic Position Sizing (1% or 2% Risk) and Regime Filtering.

---

## 🧠 Architecture & Strategy

### 1. The "Junior Analyst" (Signal Detection)
The environment mathematically identifies high-probability setups:
*   **SFP (Swing Failure Pattern)**: Liquidity grabs at key 4H Swing Highs/Lows.
*   **Grinding**: Strong trend-following setups with low volatility ($R^2 > 0.85$).
*   **Golden Pocket**: Retracements to the 0.618 - 0.65 Fibonacci zone.

### 2. The "Senior Trader" (RL Agent)
The AI evaluates these signals using an **LSTM Network** that processes the last 50 candles.
*   **Observation Space**: `(50, 6)` - Log Returns of OHLCV + Technical Features.
*   **Action Space**: `Discrete(6)`
    *   `0`: Hold / Ignore Signal
    *   `1`: Close Scalp
    *   `2`: Long (Low Risk - 1%)
    *   `3`: Long (High Risk - 2%)
    *   `4`: Short (Low Risk - 1%)
    *   `5`: Short (High Risk - 2%)

### 3. Regime Awareness (Filters)
*   **Asymmetric Filter**: 
    *   **Shorts**: BLOCKED if Price > EMA 200 (Don't fight the trend).
    *   **Longs**: ALLOWED always (Buy the dip).
*   **Long-Only Bias**: Based on extensive backtesting (2021-2024), the system performs best when focusing purely on Long opportunities in the crypto market.

---

## 📊 Performance (Out-of-Sample)

The model was validated on unseen data to ensure robustness:

| Year | Market Condition | Return | Win Rate | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **2024** | Bull / Chop | **+4.69%** | 40.5% | Profitable in difficult chop. Filter saved capital. |
| **2021** | Volatile / Crash | **+16.84%** | ~4% | High R:R strategy. Survived major crashes. |
| **2023** | Bull Run | **+153%*** | 30.3% | *In-Sample / Potential (Shows learning capacity).* |

*> Note: 2023 results were part of the training set/extended dataset and show the theoretical ceiling of the strategy.*

---

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.10+
*   CUDA-capable GPU (Recommended for training)

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rl-tradingbot-2.git
cd rl-tradingbot-2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Download and preprocess Binance Futures data:
```bash
python prepare_new_datasets.py
```

### 3. Training
Train the PPO LSTM agent:
```bash
python main.py
```
*Models are saved to `models/` and logs to `logs/tensorboard/`.*

### 4. Evaluation / Backtesting
Test the trained model on specific years:
```bash
python evaluate_2024.py
```

### 5. Live Trading
Run the bot with real money (API Keys required in `.env`):
```bash
python live_trader.py
```

---

## 📂 Project Structure
```
├── data/                   # Raw and preprocessed CSV data
├── envs/                   # Custom Gymnasium Environment (futures_lstm_env.py)
├── models/                 # Trained PPO models
├── utils/                  # Helper scripts (Data loader, Indicators)
├── main.py                 # Training Entry Point
├── evaluate_2024.py        # Backtesting Script
├── live_trader.py          # Live Execution Script
└── requirements.txt        # Dependencies
```

## ⚠️ Disclaimer
*This software is for educational purposes only. Do not trade with money you cannot afford to lose. The authors are not responsible for any financial losses.*
