import os
import time
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# Import your custom environment
from envs.futures_lstm_env import CryptoLstmEnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # In a vectorized environment, 'infos' is a list of dictionaries (one per env)
        infos = self.locals.get("infos", [])
        
        # We aggregate metrics from all environments
        total_longs = 0
        total_shorts = 0
        
        for info in infos:
            if "long_trades" in info:
                total_longs += info["long_trades"]
            if "short_trades" in info:
                total_shorts += info["short_trades"]
        
        # Record the sum of trades happening in this step across all environments
        self.logger.record("custom/long_trades", total_longs)
        self.logger.record("custom/short_trades", total_shorts)
            
        return True

def main():
    # 1. Load Data
    DATA_PATH = 'data/preprocessed_train_22_23.csv'
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run preprocessing first!")
        return

    print("Loading data...")
    # We don't need to load the dataframe here if we pass the path to env_kwargs
    # data = pd.read_csv(DATA_PATH) 
    # print(f"Data loaded: {len(data)} rows")

    # 2. Parallelization Configuration
    N_ENVS = 4  # Number of parallel environments (Adjust based on CPU cores)
    
    # Arguments for the environment
    # OPTIMIZATION: Pass data_path instead of the dataframe to avoid pickling overhead in SubprocVecEnv
    env_kwargs = {
        'data_path': DATA_PATH, 
        'window_size': 50, 
        'initial_balance': 10000
    }

    print(f"Starting {N_ENVS} parallel environments...")
    # Create the vectorized environment
    # SubprocVecEnv is crucial for parallel execution on Windows
    vec_env = make_vec_env(
        CryptoLstmEnv, 
        n_envs=N_ENVS, 
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    # 3. Model Setup
    run_name = f"specialist_grind_parallel_{int(time.time())}"
    log_path = os.path.join("logs", "tensorboard", run_name)
    
    new_logger = configure(log_path, ["stdout", "tensorboard"])
    print(f"TensorBoard Log Path: {log_path}")

    # CREATE NEW MODEL
    print("Initializing new RecurrentPPO model...")
    
    policy_kwargs = dict(
        enable_critic_lstm=True,
        lstm_hidden_size=256,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512, 
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_path
    )
    
    model.set_logger(new_logger)

    # 4. Training
    print("Starting training on combined 2022-2023 dataset...")
    TOTAL_TIMESTEPS = 1_000_000
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TensorboardCallback())

    # 5. Save Model
    save_path = os.path.join("models", f"ppo_lstm_parallel_{int(time.time())}")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    vec_env.close()

if __name__ == "__main__":
    main()
