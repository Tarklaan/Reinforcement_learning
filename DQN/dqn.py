import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

class GridWorld(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=10, render_mode=None):
        super(GridWorld, self).__init__()
        
        self.height = grid_size
        self.width = grid_size
        self.render_mode = render_mode
        
        self.bomb_location = (2, grid_size // 2)
        self.gold_location = (0, grid_size // 2)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.height * self.width,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_position = (self.height - 1, self.np_random.integers(0, self.width))
        self.done = False
        self.cumulative_reward = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        state = np.zeros((self.height, self.width), dtype=np.float32)
        state[self.agent_position[0], self.agent_position[1]] = 1.0
        return state.flatten()
    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
        
        current_pos = self.agent_position
        new_pos = list(current_pos)
        
        if action == 0:
            new_pos[0] = max(0, current_pos[0] - 1)                   #up
        elif action == 1:
            new_pos[0] = min(self.height - 1, current_pos[0] + 1)     #down
        elif action == 2:
            new_pos[1] = max(0, current_pos[1] - 1)                   #left
        elif action == 3:
            new_pos[1] = min(self.width - 1, current_pos[1] + 1)      #right
        
        new_pos = tuple(new_pos)
        self.agent_position = new_pos
        
        reward = -0.1
        
        if new_pos == self.bomb_location:
            reward = -10
            self.done = True
        elif new_pos == self.gold_location:
            reward = 10
            self.done = True
        
        self.cumulative_reward += reward
        
        return self._get_obs(), reward, self.done, False, {}
    
    def render(self):
        if self.render_mode != 'human':
            return
        
        grid_display = np.ones((self.height, self.width, 3))
        
        grid_display[self.bomb_location[0], self.bomb_location[1]] = [1, 0, 0]
        grid_display[self.gold_location[0], self.gold_location[1]] = [1, 0.84, 0]
        
        if not self.done:
            pos = self.agent_position
            grid_display[pos[0], pos[1]] = [0, 0, 1]
        
        plt.clf()
        plt.imshow(grid_display, interpolation='nearest')
        plt.title(f'DQN Agent | Reward: {self.cumulative_reward:.1f}')
        
        for i in range(self.height + 1):
            plt.axhline(y=i - 0.5, color='black', linewidth=0.5)
        for i in range(self.width + 1):
            plt.axvline(x=i - 0.5, color='black', linewidth=0.5)
        
        plt.text(self.gold_location[1], self.gold_location[0], 'GOLD', 
                ha='center', va='center', fontsize=8, weight='bold', color='black')
        plt.text(self.bomb_location[1], self.bomb_location[0], 'BOMB', 
                ha='center', va='center', fontsize=8, weight='bold', color='white')
        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.pause(0.001)

#======================= DQN
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size

        self.policy_net = DQNNetwork(state_size, action_size)
        self.target_net = DQNNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.memory = deque(maxlen=50000)

        self.gamma = 0.99
        self.batch_size = 64
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.update_target_every = 100

        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, deterministic=False):
        if not deterministic and random.random() < self.eps:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps += 1

    def save(self, filepath):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'eps': self.eps,
            'steps': self.steps
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.eps = checkpoint['eps']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {filepath}")


def train_dqn(env, agent, episodes=1000, max_steps=200):
    print(f"Training DQN Agent on {env.height}x{env.width} GridWorld...")
    print("="*60)
    
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.eps:.3f}")
    
    print("="*60)
    print("Training Complete!")
    return rewards_history


def test_dqn(env, agent, num_episodes=5, visualize=True, delay=0.3):
    print("\n" + "="*60)
    print("TESTING DQN AGENT")
    print("="*60)
    
    agent.eps = 0.0
    
    if visualize:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
    
    test_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n--- Test Episode {episode + 1}/{num_episodes} ---")
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        if visualize:
            env.render()
            time.sleep(1)
        
        while steps < 200:
            action = agent.act(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if visualize:
                env.render()
                time.sleep(delay)
            
            if done:
                if visualize:
                    env.render()
                    time.sleep(1)
                break
        
        test_rewards.append(total_reward)
        print(f"Total Reward: {total_reward:.2f} | Steps: {steps}")
    
    if visualize:
        plt.ioff()
        plt.close()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Best Reward: {np.max(test_rewards):.2f}")
    print(f"Worst Reward: {np.min(test_rewards):.2f}")
    print("="*60)
    
    return test_rewards


if __name__ == "__main__":
    GRID_SIZE = 10
    TRAIN_EPISODES = 2000
    TEST_EPISODES = 5
    MODEL_PATH = "dqn_gridworld_model.pth"
    
    train_env = GridWorld(grid_size=GRID_SIZE)
    
    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    rewards = train_dqn(train_env, agent, episodes=TRAIN_EPISODES)
    
    agent.save(MODEL_PATH)
    test_env = GridWorld(grid_size=GRID_SIZE, render_mode='human')
    test_rewards = test_dqn(test_env, agent, num_episodes=TEST_EPISODES, visualize=True, delay=0.3)
    
    print("\n✓ Training and testing complete!")
    print(f"✓ Model saved to: {MODEL_PATH}")