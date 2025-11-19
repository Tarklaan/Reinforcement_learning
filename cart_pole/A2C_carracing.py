# #### train
# import optuna
# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
# import gymnasium as gym
# import numpy as np


# # -----------------------------
# # ENV CREATION
# # -----------------------------
# def make_env(mode="rgb_array"):
#     return gym.make(
#         "CarRacing-v3",
#         render_mode=mode,
#         lap_complete_percent=0.95,
#         domain_randomize=False,
#         continuous=False
#     )


# # -----------------------------
# # OBJECTIVE FUNCTION FOR OPTUNA
# # -----------------------------
# def objective(trial):

#     # ---- Hyperparameter Search Space ----
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     gamma = trial.suggest_float("gamma", 0.90, 0.999)
#     gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
#     vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
#     ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.01)

#     # VecEnv for training
#     env = make_vec_env(lambda: make_env("rgb_array"), n_envs=1)

#     # ---- Build model ----
#     model = A2C(
#         policy="MlpPolicy",
#         env=env,
#         learning_rate=learning_rate,
#         gamma=gamma,
#         gae_lambda=gae_lambda,
#         vf_coef=vf_coef,
#         ent_coef=ent_coef,
#         verbose=0,
#     )

#     # Train for short duration (Optuna trial)
#     model.learn(total_timesteps=30000)

#     # Evaluate on fresh non-vectorized env
#     eval_env = make_env("rgb_array")
#     mean_reward, _ = evaluate_policy(
#         model,
#         eval_env,
#         n_eval_episodes=3,
#         deterministic=True
#     )

#     eval_env.close()
#     env.close()

#     return mean_reward


# # -----------------------------
# # RUN OPTUNA STUDY
# # -----------------------------
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)

# print("Best trial:")
# print(study.best_trial.params)

# # -----------------------------
# # TRAIN FINAL MODEL WITH BEST HP
# # -----------------------------
# best_params = study.best_trial.params

# print("\nTraining final model using best HP...")

# env = make_vec_env(lambda: make_env("rgb_array"), n_envs=1)

# model = A2C(
#     policy="MlpPolicy",
#     env=env,
#     learning_rate=best_params["learning_rate"],
#     gamma=best_params["gamma"],
#     gae_lambda=best_params["gae_lambda"],
#     vf_coef=best_params["vf_coef"],
#     ent_coef=best_params["ent_coef"],
#     verbose=1,
# )

# model.learn(total_timesteps=100000)
# model.save("a2c_carracing_optuna_best")

# print("Saved best model as a2c_carracing_optuna_best.zip")



# -----------------------------test
from stable_baselines3 import A2C
import gymnasium as gym

def make_env(mode="human"):
    return gym.make(
        "CarRacing-v3",
        render_mode=mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )

model = A2C.load("a2c_carracing_optuna_best")

env = make_env("human")
obs, info = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
