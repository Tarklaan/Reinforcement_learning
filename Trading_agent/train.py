import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('btcusdt.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
df_1h = pd.DataFrame()
df_1h['Open'] = df['Open'].resample('1H').first()
df_1h['High'] = df['High'].resample('1H').max()
df_1h['Low']  = df['Low'].resample('1H').min()
df_1h['Close'] = df['Close'].resample('1H').last()
df_1h['Volume'] = df['Volume'].resample('1H').sum()

env = gym.make(
    'stocks-v0', 
    df=df_1h, 
    frame_bound=(30, len(df_1h)), 
    window_size=30, 
    # render_mode='human' 
)

#==================building env and training model===================
env_maker = lambda: gym.make('stocks-v0', df=df_1h, frame_bound=(10, len(df_1h)), window_size=10)
env = DummyVecEnv([env_maker])

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("dqn_btc_1h_model")
print("Model saved as 'dqn_btc_1h_model.zip'")

