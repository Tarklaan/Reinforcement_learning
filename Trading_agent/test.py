import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
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

df = df[df.index >= '2025-09-09']

df_1h = pd.DataFrame()
df_1h['Open'] = df['Open'].resample('1H').first()
df_1h['High'] = df['High'].resample('1H').max()
df_1h['Low'] = df['Low'].resample('1H').min()
df_1h['Close'] = df['Close'].resample('1H').last()
df_1h['Volume'] = df['Volume'].resample('1H').sum()
df_1h.dropna(inplace=True)

def make_env_human():
    return gym.make(
        'stocks-v0',
        df=df_1h,
        frame_bound=(10, len(df_1h)),
        window_size=10,
        render_mode='human' 
    )
env = make_env_human()
if 'render_fps' in env.unwrapped.metadata:
    env.unwrapped.metadata['render_fps'] = 2 
model = DQN.load("dqn_btc_1h_model", env=env)
state, info = env.reset()
done = False

while not done:
    action, _ = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("Action taken:", action, "Current position info:", info['position'])


print("Testing finished. Info:", info)
env.close()