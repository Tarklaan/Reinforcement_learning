import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import time

class CompetitiveGridWorld(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        super(CompetitiveGridWorld, self).__init__()
        
        self.height = 5
        self.width = 5
        self.render_mode = render_mode
        
        self.bomb_location = (1, 3)
        self.gold_location = (0, 3)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, 3), dtype=np.float32)
        
        self.agent_positions = {
            'q_learning': None,
            'dqn': None
        }
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        available_positions = [(4, i) for i in range(5)]
        positions = self.np_random.choice(len(available_positions), size=2, replace=False)
        
        self.agent_positions['q_learning'] = available_positions[positions[0]]
        self.agent_positions['dqn'] = available_positions[positions[1]]
        
        self.done = {'q_learning': False, 'dqn': False}
        self.rewards = {'q_learning': 0, 'dqn': 0}
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        obs[self.bomb_location[0], self.bomb_location[1], 0] = 1.0
        obs[self.gold_location[0], self.gold_location[1], 1] = 1.0
        
        if self.agent_positions['q_learning']:
            obs[self.agent_positions['q_learning'][0], self.agent_positions['q_learning'][1], 2] = 0.5
        if self.agent_positions['dqn']:
            obs[self.agent_positions['dqn'][0], self.agent_positions['dqn'][1], 2] = 1.0
        
        return obs
    
    def step_agent(self, agent_name, action):
        if self.done[agent_name]:
            return 0, True
        
        current_pos = self.agent_positions[agent_name]
        new_pos = list(current_pos)
        
        if action == 0:
            new_pos[0] = max(0, current_pos[0] - 1)
        elif action == 1:
            new_pos[0] = min(self.height - 1, current_pos[0] + 1)
        elif action == 2:
            new_pos[1] = max(0, current_pos[1] - 1)
        elif action == 3:
            new_pos[1] = min(self.width - 1, current_pos[1] + 1)
        
        new_pos = tuple(new_pos)
        self.agent_positions[agent_name] = new_pos
        
        reward = -0.1
        
        if new_pos == self.bomb_location:
            reward = -10
            self.done[agent_name] = True
        elif new_pos == self.gold_location:
            reward = 10
            self.done[agent_name] = True
        
        self.rewards[agent_name] += reward
        
        return reward, self.done[agent_name]
    
    def step(self, action):
        reward, done = self.step_agent('dqn', action)
        obs = self._get_obs()
        
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode != 'human':
            return
        
        grid_display = np.ones((self.height, self.width, 3))
        
        grid_display[self.bomb_location[0], self.bomb_location[1]] = [1, 0, 0]
        grid_display[self.gold_location[0], self.gold_location[1]] = [1, 0.84, 0]
        
        if self.agent_positions['q_learning']:
            pos = self.agent_positions['q_learning']
            if not self.done['q_learning']:
                grid_display[pos[0], pos[1]] = [0, 0, 1]
        
        if self.agent_positions['dqn']:
            pos = self.agent_positions['dqn']
            if not self.done['dqn']:
                if self.agent_positions['q_learning'] == pos:
                    grid_display[pos[0], pos[1]] = [0.5, 0, 0.5]
                else:
                    grid_display[pos[0], pos[1]] = [0, 1, 0]
        
        plt.clf()
        plt.imshow(grid_display, interpolation='nearest')
        plt.title(f'Q-Learning (Blue): {self.rewards["q_learning"]:.1f} | DQN (Green): {self.rewards["dqn"]:.1f}')
        
        for i in range(self.height + 1):
            plt.axhline(y=i - 0.5, color='black', linewidth=1)
        for i in range(self.width + 1):
            plt.axvline(x=i - 0.5, color='black', linewidth=1)
        
        plt.text(self.gold_location[1], self.gold_location[0], 'GOLD', 
                ha='center', va='center', fontsize=8, weight='bold')
        plt.text(self.bomb_location[1], self.bomb_location[0], 'BOMB', 
                ha='center', va='center', fontsize=8, weight='bold')
        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.pause(0.001)


class QAgent:
    def __init__(self, height, width, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.q_table = {}
        for x in range(height):
            for y in range(width):
                self.q_table[(x, y)] = [0, 0, 0, 0]
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(4)
        else:
            q_values = self.q_table[state]
            max_value = max(q_values)
            best_actions = [i for i, v in enumerate(q_values) if v == max_value]
            return np.random.choice(best_actions)
    
    def learn(self, old_state, action, reward, new_state):
        max_q_new = max(self.q_table[new_state])
        current_q = self.q_table[old_state][action]
        
        self.q_table[old_state][action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_q_new)


def train_q_agent(q_agent, env, episodes=1000):
    print("Training Q-Learning Agent...")
    rewards_history = []
    
    for episode in range(episodes):
        env.reset()
        cumulative_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            state = env.agent_positions['q_learning']
            action = q_agent.choose_action(state)
            reward, done = env.step_agent('q_learning', action)
            new_state = env.agent_positions['q_learning']
            
            q_agent.learn(state, action, reward, new_state)
            cumulative_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards_history.append(cumulative_reward)
        
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    return rewards_history


def train_dqn_agent(env, total_timesteps=50000):
    print("\nTraining DQN Agent...")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    return model


def compete(q_agent, dqn_model, env, num_games=10, visualize=True):
    print("\n" + "="*60)
    print("COMPETITION STARTING")
    print("="*60)
    
    q_wins = 0
    dqn_wins = 0
    draws = 0
    
    if visualize:
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
    
    for game in range(num_games):
        print(f"\n--- Game {game + 1}/{num_games} ---")
        obs, _ = env.reset()
        
        q_agent.epsilon = 0.0
        
        steps = 0
        max_steps = 100
        
        if visualize:
            env.render()
            time.sleep(1)
        
        while steps < max_steps:
            if not env.done['q_learning']:
                q_state = env.agent_positions['q_learning']
                q_action = q_agent.choose_action(q_state)
                env.step_agent('q_learning', q_action)
            
            if not env.done['dqn']:
                dqn_action, _ = dqn_model.predict(obs, deterministic=True)
                obs, _, _, _, _ = env.step(dqn_action)
            
            if visualize:
                env.render()
                time.sleep(0.3)
            
            steps += 1
            
            if env.done['q_learning'] or env.done['dqn']:
                if visualize:
                    env.render()
                    time.sleep(0.5)
                break
        
        print(f"Q-Learning: {env.rewards['q_learning']:.1f}, DQN: {env.rewards['dqn']:.1f}")
        
        if env.rewards['q_learning'] > env.rewards['dqn']:
            q_wins += 1
            print("Winner: Q-Learning!")
        elif env.rewards['dqn'] > env.rewards['q_learning']:
            dqn_wins += 1
            print("Winner: DQN!")
        else:
            draws += 1
            print("Draw!")
        
        if visualize:
            time.sleep(1)
    
    if visualize:
        plt.ioff()
        plt.close()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Q-Learning Wins: {q_wins}")
    print(f"DQN Wins: {dqn_wins}")
    print(f"Draws: {draws}")
    print("="*60)
    
    return q_wins, dqn_wins, draws


if __name__ == "__main__":
    train_env = CompetitiveGridWorld()
    
    q_agent = QAgent(train_env.height, train_env.width)
    q_rewards = train_q_agent(q_agent, train_env, episodes=1000)
    
    dqn_model = train_dqn_agent(train_env, total_timesteps=50000)
    
    test_env = CompetitiveGridWorld(render_mode='human')
    
    compete(q_agent, dqn_model, test_env, num_games=5, visualize=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(q_rewards)
    plt.title('Q-Learning Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q_learning_training.png')
    plt.show()