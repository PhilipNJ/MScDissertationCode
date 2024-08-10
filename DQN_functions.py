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
import matplotlib.pyplot as plt

def create_states(df, window_size=9):
    states = []
    for i in range(window_size, len(df)):
        state = df.iloc[i-window_size:i].values
        states.append(state)
    return np.array(states)

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
    def __init__(self, state_size, action_size, window_size,model, lr=0.00001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(50000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.loss_history = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        # Added no_grad to avoid gradient calculation for inference
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    # Optimized replay method with batch processing
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        actions = torch.LongTensor(actions)

        # Compute target values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_loss_per_episode(self, loss_per_episode):
        plt.plot(loss_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Loss Per Episode')
        plt.show()


def train_agent(agent, states, episodes, batch_size):
    Episode = []
    Time = []
    Reward = []
    Action = []
    prev_price = []
    loss_per_episode = []

    for e in range(episodes):
        state = states[0]
        reward = 0
        has_open_position = False
        price_at_buy = 0
        days_since_last_buy = 0

        for time in range(1, len(states)):
            action = agent.act(state)

            if has_open_position:
                days_since_last_buy += 1

            if action == 1: #buy
                if has_open_position:
                    action = 2
                else:
                    has_open_position = True
                    days_since_last_buy = 0
                    price_at_buy = state[-1][3]

            if action == 0: #sell
                if not has_open_position:
                    action = 2
                else:
                    has_open_position = False
                    days_since_last_buy = 0
                    profit = state[-1][3] - price_at_buy
                    reward =+ profit

            if has_open_position and days_since_last_buy > 11:
                action = 0
                has_open_position = False
                days_since_last_buy = 0
                reward = state[-1][3] - price_at_buy
                reward =+ profit
            
            if action == 2 and not has_open_position:
                reward = reward * (1-0.1)

            next_state = states[time]
            prev_price.append(state[-1][3])
            done = time == len(states) - 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            Episode.append(e + 1)
            Time.append(time)
            Reward.append(reward)
            Action.append(action)
            if done:
                if (e + 1) % 1 == 0:
                    print(f"Episode {e + 1}/{episodes}, Total Reward: {reward}, Loss: {np.log(agent.loss_history[-1])}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        loss_per_episode.append(agent.loss_history[-1])

    log_train = pd.DataFrame({'Episode': Episode, 'Time': Time, 'Reward': Reward, 'Action': Action, 'Price': prev_price})
    log_train['Action'] = log_train['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})
    log_train['Action'].value_counts()
    log_train['Loss'] = log_train['Episode'].map({index: element for index, element in enumerate(loss_per_episode, start=1)})

    agent.plot_loss_per_episode(loss_per_episode)
    return log_train


# def evaluate_agent(agent, states):
#     Episode = []
#     Time = []
#     Reward = []
#     Total_Reward = []
#     Action = []
#     next_price = []
#     prev_price = []

#     state = states[0]
#     reward = 0
#     last_action = 'Hold'
#     days_since_last_buy = 0
#     has_open_position = False  # Track if there's an open position

#     for time in range(1, len(states)):
#         action = agent.act(state)

#         # Ensure sell only after buy and max 11 days between buy and sell
#         if has_open_position:
#             days_since_last_buy += 1

#         if action == 0:  # Sell
#             if not has_open_position:  # Can't sell if no open position
#                 action = 2  # Change to Hold if not previously bought
#             else:
#                 has_open_position = False  # Sell the open position
#                 days_since_last_buy = 0  # Reset days since last buy
#                 reward += state[-1][3]  # Example: Reward based on price difference

#         if action == 1:  # Buy
#             if has_open_position:  # Can't buy if already have an open position
#                 action = 2  # Change to Hold if already bought
#             else:
#                 has_open_position = True  # Buy, opening a new position
#                 days_since_last_buy = 0  # Reset days since last buy
#                 reward -= state[-1][3]  # Example: Reward based on price difference

#         # Force a sell if more than 11 days since last buy
#         if has_open_position and days_since_last_buy > 11:
#             action = 0  # Force a sell
#             has_open_position = False
#             days_since_last_buy = 0
#             reward += state[-1][3]  # Example: Reward based on price difference

#         # Update last action and reset days since last buy if sell or buy happens
#         if action == 1:  # Buy
#             days_since_last_buy = 0
#             has_open_position = True
#         if action == 0:  # Sell
#             has_open_position = False
#         if action == 2:  # Hold
#             reward = reward  # Example: Reward based on price difference

#         next_state = states[time]
#         prev_price.append(state[-1][3])
#         done = time == len(states) - 1
#         state = next_state
#         Episode.append(1)
#         Time.append(time)
#         Reward.append(reward)
#         Action.append(action)
#         if done:
#             break

#     log_evaluate = pd.DataFrame({'Episode': Episode, 'Time': Time, 'Reward': Reward, 'Total_Reward': reward, 'Action': Action, 'Price': prev_price})
#     log_evaluate['Action'] = log_evaluate['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})
#     log_evaluate['Action'].value_counts()

#     return log_evaluate

def evaluate_agent(agent, states):
    Time = []
    Action = []
    prev_price = []

    has_open_position = False
    price_at_buy = 0
    days_since_last_buy = 0

    for time in range(1, len(states)):
        state = states[time - 1]
        action = agent.act(state)  # Assuming agent has a 'training' flag to distinguish between train and evaluation mode

        if has_open_position:
            days_since_last_buy += 1

        if action == 1:  # Buy
            if has_open_position:
                action = 2  # Invalid action, change to Hold
            else:
                has_open_position = True
                days_since_last_buy = 0
                price_at_buy = state[-1][3]

        if action == 0:  # Sell
            if not has_open_position:
                action = 2  # Invalid action, change to Hold
            else:
                has_open_position = False
                days_since_last_buy = 0

        if has_open_position and days_since_last_buy > 11:
            action = 0  # Force sell after 11 days
            has_open_position = False
            days_since_last_buy = 0

        prev_price.append(state[-1][3])
        Time.append(time)
        Action.append(action)

    log_evaluation = pd.DataFrame({'Time': Time, 'Action': Action, 'Price': prev_price})
    log_evaluation['Action'] = log_evaluation['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})

    return log_evaluation

def create_action_episode_df(log_train):

    # Get the unique episodes
    unique_episodes = log_train['Episode'].unique()
    
    # Create a dictionary to store actions for each episode
    action_dict = {f'action_ep{ep}': [] for ep in unique_episodes}
    
    # Iterate over each episode and store the actions
    for ep in unique_episodes:
        actions = log_train[log_train['Episode'] == ep]['Action'].tolist()
        action_dict[f'action_ep{ep}'] = actions
    
    # Find the maximum length of the actions list to pad shorter lists
    max_length = max(len(actions) for actions in action_dict.values())
    
    # Pad shorter lists with None to ensure all columns have the same length
    for key in action_dict:
        action_dict[key] += [None] * (max_length - len(action_dict[key]))
    
    # Create the output DataFrame
    action_episode_df = pd.DataFrame(action_dict)
    
    return action_episode_df


def plot_training(log_train):
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the figsize to make the graph wider

    # Plotting the Reward series
    ax1.plot(log_train.groupby('Episode').sum()['Reward'], color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Creating a second y-axis
    ax2 = ax1.twinx()

    # Plotting the Loss series
    ax2.plot(log_train.groupby('Episode').mean()['Loss'], color='red')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Reward and Loss vs Episode')
    plt.show()


# def train_agent(agent, states, episodes, batch_size, price_history, reward_window=12, patience=10, min_delta=0.5,penalty=0.01):
#     Episode = []
#     Time = []
#     Reward = []
#     Action = []
#     prev_price = []
#     loss_per_episode = []

#     close_prices = price_history['Close'].values

#     best_loss = float('inf')
#     patience_counter = 0

#     for e in range(episodes):
#         state = states[0]
#         reward = 0
#         has_open_position = False
#         days_since_last_buy = 0
#         reward_per_episode = 0

#         for time in range(1, len(states)):
#             action = agent.act(state)

#             if has_open_position:
#                 days_since_last_buy += 1
#                 if days_since_last_buy > 11:
#                     action = 2  # Force Sell

#             # Precomputed prices
#             self_price = close_prices[time]
#             self_prev_price = close_prices[time - 1]
#             if has_open_position:
#                 self_init_price = close_prices[time - days_since_last_buy]
#             else:
#                 self_init_price = close_prices[time - reward_window]
            

#             if action == 2:  # Sell
#                 if has_open_position:
#                     has_open_position = False
                    # reward = (1 + action * (self_price - self_prev_price) / self_prev_price)
                    # reward = (reward * self_prev_price / self_init_price)-1
#                     days_since_last_buy = 0
#                 else:
#                     action = 1  # Hold

#             if action == 0:  # Buy
#                 if not has_open_position:
#                     has_open_position = True
#                     reward = (1 + action * (self_price - self_prev_price) / self_prev_price)
#                     reward = (reward * self_prev_price / self_init_price)-1
#                 else:
#                     action = 1  # Hold
            
#             if action == 1:  # Hold
#                 if not has_open_position:
#                     reward = reward * penalty

#             reward_per_episode += reward
#             next_state = states[time]
#             prev_price.append(self_prev_price)
#             done = time == len(states) - 1
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             Episode.append(e + 1)
#             Time.append(time)
#             Reward.append(reward)
#             Action.append(action)
#             if done:
#                 current_loss = np.log(agent.loss_history[-1])
#                 print(f"Episode {e + 1}/{episodes}, Total Reward: {reward_per_episode}, Loss: {current_loss}")

#                 # Early stopping check
#                 if best_loss - current_loss > min_delta:
#                     best_loss = current_loss
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1

#                 if patience_counter > patience:
#                     print("Early stopping triggered.")
#                     break

#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)

#         loss_per_episode.append(agent.loss_history[-1])
#         if patience_counter > patience:
#             break

#     log_train = pd.DataFrame({
#         'Episode': Episode,
#         'Time': Time,
#         'Reward': Reward,
#         'Action': Action,
#         'Price': prev_price
#     })
#     log_train['Action'] = log_train['Action'].map({0: 'Buy', 1: 'Hold', 2: 'Sell'})
#     log_train['Loss'] = log_train['Episode'].map({index: element for index, element in enumerate(loss_per_episode, start=1)})

#     agent.plot_loss_per_episode(loss_per_episode)
#     return log_train
