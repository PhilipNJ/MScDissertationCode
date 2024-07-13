import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the trading environment
class TradingEnvironment:
    def __init__(self, data, window_size=12):
        self.data = data
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.index = self.window_size
        self.positions = []
        self.current_position = None
        return self._get_state()

    def step(self, action):
        reward = 0
        done = False

        if action == 0:  # Buy
            if self.current_position is None:
                self.current_position = self.data.iloc[self.index]
                self.positions.append(self.current_position)
        elif action == 1:  # Sell
            if self.current_position is not None:
                reward = self.data.iloc[self.index]['Close'] - self.current_position['Close']
                self.current_position = None
        # Action 2 is Hold, do nothing

        self.index += 1
        if self.index >= len(self.data) - 1:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        return self.data.iloc[self.index-self.window_size:self.index].values.flatten()

    @property
    def state_size(self):
        return self.window_size * len(self.data.columns)
    
# Implement the DQN algorithm
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state).detach())
            target_f = self.model(state)
            target_f[0, action] = target
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Load and preprocess the data
data = pd.read_pickle('data/SP500.pkl')  # Replace with your data file
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Initialize the environment and agent
env = TradingEnvironment(train_data)
state_size = env.window_size * 5  # 5 features: Open, High, Low, Close, Volume
action_size = 3  # Buy, Sell, Hold
agent = DQNAgent(state_size, action_size)

# Training
episodes = 100
batch_size = 32

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if e % 10 == 0:
        agent.update_target_model()

    print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Testing
env = TradingEnvironment(test_data)
state = env.reset()
done = False
total_reward = 0
actions = []

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    state = next_state
    total_reward += reward
    actions.append(action)

print(f"Test Total Reward: {total_reward}")

# Generate buy, sell, and hold signals
signals = pd.DataFrame(index=test_data.index)
signals['Signal'] = actions
signals['Signal'] = signals['Signal'].map({0: 'Buy', 1: 'Sell', 2: 'Hold'})

print(signals)