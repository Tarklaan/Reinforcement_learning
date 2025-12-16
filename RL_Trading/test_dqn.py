import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv

df = pd.read_csv("data/btcusdt.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)
df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)

df = df[df.index >= "2025-01-01"]

df_1h = pd.DataFrame()
df_1h["Open"] = df["Open"].resample("1h").first()
df_1h["High"] = df["High"].resample("1h").max()
df_1h["Low"] = df["Low"].resample("1h").min()
df_1h["Close"] = df["Close"].resample("1h").last()
df_1h["Volume"] = df["Volume"].resample("1h").sum()
df_1h.dropna(inplace=True)

env_fn = lambda: TradingEnv(df=df_1h, window_size=30)
env = DummyVecEnv([env_fn])

model = DQN.load("models/dqn_trading_model")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

ledger = env.envs[0].export_trades()

os.makedirs("results", exist_ok=True)
ledger.to_csv("results/ledger.csv", index=False)

if len(ledger) == 0:
    print("No trades executed.")
else:
    closed = ledger[ledger["event"].str.contains("CLOSE")]
    pnl_list = closed["pnl"].tolist()
    colors = ["g" if x > 0 else "r" for x in pnl_list]

    plt.figure(figsize=(14,5))
    plt.bar(range(len(pnl_list)), pnl_list, color=colors)
    plt.title("Individual Closed Trade PnL")
    plt.xlabel("Trade #")
    plt.ylabel("PnL")
    plt.show()
