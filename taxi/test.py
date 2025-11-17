import gymnasium as gym
from stable_baselines3 import DQN

model = DQN.load("dqn_cartpole_best")
env = gym.make('Taxi-v3', render_mode="human")
observation, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(observation, deterministic=True)
    action = int(action)  # ✅ Convert from numpy array to int
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    env.render()

print(f"Episode finished! Total reward: {total_reward}")
env.close()
