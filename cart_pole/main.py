import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    # done = terminated or truncated
    if terminated or truncated:
        obs, info = env.reset()
        print("reward:", total_reward)
        total_reward = 0
    env.render()

print(f"Episode finished! Total reward: {total_reward}")
env.close()
