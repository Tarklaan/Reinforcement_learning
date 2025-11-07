import gymnasium as gym
import optuna
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def optimize_dqn(trial):
    env = gym.make("LunarLander-v3")
    env = Monitor(env)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_int("buffer_size", 50000, 200000, step=50000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_uniform("gamma", 0.90, 0.999)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.1, 0.4)
    target_update_interval = trial.suggest_int("target_update_interval", 100, 1000, step=100)
    train_freq = trial.suggest_int("train_freq", 1, 8)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 8)

    model = DQN(
        "MlpPolicy", env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=0,
    )

    model.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    env.close()
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(optimize_dqn, n_trials=10)
best_params = study.best_params

env = gym.make("LunarLander-v3")
env = Monitor(env)
best_model = DQN("MlpPolicy", env, **best_params, verbose=1)
best_model.learn(total_timesteps=200000)
best_model.save("dqn_lunar_lander_best")
env.close()

df = study.trials_dataframe()
df.to_csv("dqn_lunar_lander_optuna_results.csv", index=False)

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = best_model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    env.render()

print(f"Episode finished! Total reward: {total_reward}")
env.close()
