import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import optuna
from tqdm import tqdm


def create_states(df, window_size=9):
    states = []
    for i in range(window_size, len(df)):
        state = df.iloc[i-window_size:i].values
        states.append(state)
    return np.array(states)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvDQN(nn.Module):
    def __init__(self, input_dim, output_dim, window_size):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * window_size, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change the shape to (batch_size, num_features, window_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output from the conv layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.loss_history = []  # To store loss values

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        total_loss = 0  # Track total loss for the batch
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            total_loss += loss.item()  # Aggregate loss
            loss.backward()
            self.optimizer.step()
        self.loss_history.append(total_loss / batch_size)  # Store average loss for the batch
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_loss_per_episode(self, loss_per_episode):
        plt.plot(loss_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Loss Per Episode')
        plt.show()

def log_model_parameters(agent, episode):
    print(f"Model parameters at episode {episode}:")
    for name, param in agent.model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")


# Define the objective function
def objective(trial, states):
    state_size = states.shape[1] * states.shape[2]
    action_size = 3

    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.999)
    epsilon = trial.suggest_uniform('epsilon', 0.8, 1.0)
    epsilon_min = trial.suggest_uniform('epsilon_min', 0.01, 0.1)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.9, 0.999)

    # Initialize the agent with suggested hyperparameters
    agent = DQNAgent(state_size, action_size, lr, gamma, epsilon, epsilon_min, epsilon_decay)

    # Train the agent on the training set
    episodes = 100
    batch_size = 32
    total_rewards = []

    for e in range(episodes):
        state = states[0]
        total_reward = 0
        last_action = 'Hold'
        days_since_last_buy = 0
        has_open_position = False

        for time in range(1, len(states)):
            action = agent.act(state)

            # Ensure sell only after buy and max 11 days between buy and sell
            if has_open_position:
                days_since_last_buy += 1

            if action == 0:  # Sell
                if not has_open_position:  # Can't sell if no open position
                    action = 2  # Change to Hold if not previously bought
                else:
                    has_open_position = False  # Sell the open position
                    days_since_last_buy = 0  # Reset days since last buy

            if action == 1:  # Buy
                if has_open_position:  # Can't buy if already have an open position
                    action = 2  # Change to Hold if already bought
                else:
                    has_open_position = True  # Buy, opening a new position
                    days_since_last_buy = 0  # Reset days since last buy

            # Force a sell if more than 11 days since last buy
            if has_open_position and days_since_last_buy > 11:
                action = 0  # Force a sell

            # Update last action and reset days since last buy if sell or buy happens
            if action == 1:  # Buy
                days_since_last_buy = 0
                has_open_position = True
            if action == 0:  # Sell
                has_open_position = False

            next_state = states[time]
            reward = next_state[-1][3] - state[-1][3]  # Example: Reward based on price difference
            agent.remember(state, action, reward, next_state, done=(time == len(states) - 1))
            state = next_state
            total_reward += reward

            if time == len(states) - 1:
                total_rewards.append(total_reward)
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"Iteration {e+1}/{episodes}, Total Reward: {total_reward}")

    return np.mean(total_rewards)

def train_agent(agent, states, episodes, batch_size):
    Episode = []
    Time = []
    Reward = []
    Total_Reward = []
    Action = []
    next_price = []
    prev_price = []
    loss_per_episode = []

    for e in tqdm(range(episodes), desc='Training Progress'):
        state = states[0]
        total_reward = 0
        last_action = 'Hold'
        days_since_last_buy = 0
        has_open_position = False  # Track if there's an open position
        
        for time in range(1, len(states)):
            action = agent.act(state)

            # Ensure sell only after buy and max 11 days between buy and sell
            if has_open_position:
                days_since_last_buy += 1

            if action == 0:  # Sell
                if not has_open_position:  # Can't sell if no open position
                    action = 2  # Change to Hold if not previously bought
                else:
                    has_open_position = False  # Sell the open position
                    days_since_last_buy = 0  # Reset days since last buy

            if action == 1:  # Buy
                if has_open_position:  # Can't buy if already have an open position
                    action = 2  # Change to Hold if already bought
                else:
                    has_open_position = True  # Buy, opening a new position
                    days_since_last_buy = 0  # Reset days since last buy

            # Force a sell if more than 11 days since last buy
            if has_open_position and days_since_last_buy > 11:
                action = 0  # Force a sell

            # Update last action and reset days since last buy if sell or buy happens
            if action == 1:  # Buy
                days_since_last_buy = 0
                has_open_position = True
            if action == 0:  # Sell
                has_open_position = False

            next_state = states[time]
            reward = next_state[-1][3] - state[-1][3]  # Example: Reward based on price difference
            next_price.append(next_state[-1][3])
            prev_price.append(state[-1][3])
            done = time == len(states) - 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            Episode.append(e+1)
            Time.append(time)
            Reward.append(reward)
            Total_Reward.append(total_reward)
            Action.append(action)
            if done:
                print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Calculate the loss for the current episode and append it
        loss_per_episode.append(agent.loss_history[-1])

    log_train = pd.DataFrame({'Episode': Episode, 'Time': Time, 'Reward': Reward, 'Total_Reward': Total_Reward, 'Action': Action, 'Next_Price': next_price, 'Prev_Price': prev_price})
    log_train['Action'] = log_train['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})
    log_train['Action'].value_counts()

    # Plot loss per episode at the end of training
    agent.plot_loss_per_episode(loss_per_episode)
    return log_train
    
def evaluate_agent(agent, states):
    state = states[0]
    total_reward = 0
    actions = []
    days_since_last_buy = 0
    has_open_position = False
    last_action = 'Hold'
    
    for time in tqdm((range(1, len(states))), desc='Evaluation Progress'):
        action = agent.act(state)

        # Ensure sell only after buy and max 11 days between buy and sell
        if has_open_position:
            days_since_last_buy += 1

        if action == 0:  # Sell
            if not has_open_position:  # Can't sell if no open position
                action = 2  # Change to Hold if not previously bought
            else:
                has_open_position = False  # Sell the open position
                days_since_last_buy = 0  # Reset days since last buy

        if action == 1:  # Buy
            if has_open_position:  # Can't buy if already have an open position
                action = 2  # Change to Hold if already bought
            else:
                has_open_position = True  # Buy, opening a new position
                days_since_last_buy = 0  # Reset days since last buy

        # Force a sell if more than 11 days since last buy
        if has_open_position and days_since_last_buy > 11:
            action = 0  # Force a sell

        # Update last action and reset days since last buy if sell or buy happens
        if action == 1:  # Buy
            days_since_last_buy = 0
            has_open_position = True
        if action == 0:  # Sell
            has_open_position = False

        next_state = states[time]
        reward = next_state[-1][3] - state[-1][3]  # Example: Reward based on price difference
        state = next_state
        total_reward += reward
        actions.append(action)

    return total_reward, actions