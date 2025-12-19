import gymnasium as gym       
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from trading_env import TradingEnv  # Make sure this is the updated scalper env (no Hold)

# Load and preprocess 1-minute data
df = pd.read_csv("data/btcusdt.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)

df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
}, inplace=True)

# Filter recent data if needed (optional)
df = df[df.index >= "2025-08-01"]


# Split into train/test (80/20)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Create environments
train_env = DummyVecEnv([lambda: TradingEnv(df=train_df, window_size=30)])
eval_env = DummyVecEnv([lambda: TradingEnv(df=test_df, window_size=30)])

# Callbacks
os.makedirs("models", exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_trading"
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,       # Collect this many steps before update
    batch_size=64,
    n_epochs=10,        # More epochs per update for stable learning
    gamma=0.99,
    ent_coef=0.01,      # Encourage exploration
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

print("\nStarting PPO training...")
model.learn(
    total_timesteps=1_000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

model.save("models/ppo_trading_model_final")
print("\nTraining complete!")

# Evaluate on test set
print("\nEvaluating on test set...")
obs = eval_env.reset()
done = False
episode_reward = 0
actions_taken = {0: 0, 1: 0}  # Short, Long

while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_taken[action[0]] += 1
    obs, reward, done, info = eval_env.step(action)
    episode_reward += reward[0]

print(f"\nTest Episode Reward: {episode_reward:.2f}")
print(f"Final Balance: ${info[0]['balance']:.2f}")
print(f"\nAction Distribution:")
print(f"  Short (0): {actions_taken[0]} ({actions_taken[0]/sum(actions_taken.values())*100:.1f}%)")
print(f"  Long (1): {actions_taken[1]} ({actions_taken[1]/sum(actions_taken.values())*100:.1f}%)")

# Export trades
env_unwrapped = eval_env.envs[0].unwrapped
trades_df = env_unwrapped.export_trades()
if len(trades_df) > 0:
    trades_df.to_csv("trades_evaluation.csv", index=False)
    print(f"\nTotal trades executed: {len(trades_df)}")
    print(f"Trades saved to 'trades_evaluation.csv'")
else:
    print("\nNo trades were executed!")
