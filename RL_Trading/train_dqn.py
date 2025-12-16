import os
import gymnasium as gym
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from trading_env import TradingEnv

# Load and preprocess data
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

# Resample to 1h
df_1h = pd.DataFrame()
df_1h["Open"] = df["Open"].resample("1h").first()
df_1h["High"] = df["High"].resample("1h").max()
df_1h["Low"] = df["Low"].resample("1h").min()
df_1h["Close"] = df["Close"].resample("1h").last()
df_1h["Volume"] = df["Volume"].resample("1h").sum()
df_1h.dropna(inplace=True)

# Split into train/test (80/20)
split_idx = int(len(df_1h) * 0.8)
train_df = df_1h.iloc[:split_idx]
test_df = df_1h.iloc[split_idx:]

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Create environments
train_env = DummyVecEnv([lambda: TradingEnv(df=train_df, window_size=30)])
eval_env = DummyVecEnv([lambda: TradingEnv(df=test_df, window_size=30)])

# Callbacks for saving best model
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
    name_prefix="dqn_trading"
)

# Create model with better hyperparameters
model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,  # Higher learning rate
    buffer_size=100000,
    batch_size=64,  # Larger batch
    tau=1.0,
    gamma=0.99,
    train_freq=4,  # Train every 4 steps
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.3,  # Explore for 30% of training
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,  # Maintain some exploration
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

print("\nStarting training...")
model.learn(
    total_timesteps=500000,  # More training steps
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

model.save("models/dqn_trading_model_final")
print("\nTraining complete!")

# Evaluate on test set
print("\nEvaluating on test set...")
obs = eval_env.reset()
done = False
episode_reward = 0
actions_taken = {0: 0, 1: 0, 2: 0}  # Short, Hold, Long

while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_taken[action[0]] += 1
    obs, reward, done, info = eval_env.step(action)
    episode_reward += reward[0]

print(f"\nTest Episode Reward: {episode_reward:.2f}")
print(f"Final Balance: ${info[0]['balance']:.2f}")
print(f"\nAction Distribution:")
print(f"  Short (0): {actions_taken[0]} ({actions_taken[0]/sum(actions_taken.values())*100:.1f}%)")
print(f"  Hold (1): {actions_taken[1]} ({actions_taken[1]/sum(actions_taken.values())*100:.1f}%)")
print(f"  Long (2): {actions_taken[2]} ({actions_taken[2]/sum(actions_taken.values())*100:.1f}%)")

# Export trades
env_unwrapped = eval_env.envs[0].unwrapped
trades_df = env_unwrapped.export_trades()
if len(trades_df) > 0:
    trades_df.to_csv("trades_evaluation.csv", index=False)
    print(f"\nTotal trades executed: {len(trades_df)}")
    print(f"\nTrades saved to 'trades_evaluation.csv'")
else:
    print("\nNo trades were executed!")