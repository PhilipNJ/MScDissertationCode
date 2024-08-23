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
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Baseline_functions import calculate_macd_signals

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

def plot_dual_axis(all_states_eval):
    fig, ax1 = plt.subplots(figsize=(10,6))
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Close', color=color)
    ax1.plot(all_states_eval['Time'], all_states_eval['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Capital', color=color)
    ax2.plot(all_states_eval['Time'], all_states_eval['Capital'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()
    plt.show()

def train_agent_hold(agent, states, episodes, batch_size):
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

def reward_filter(reward):
    reward_df = pd.DataFrame(reward, columns=['Reward'])
    reward_df['Episode'] = reward_df.index+1

    plt.figure(figsize=(10,6))
    plt.plot(reward_df['Episode'], reward_df['Reward'], label='Reward', color='blue')
    plt.plot(reward_df['Episode'], savgol_filter(reward_df['Reward'], 51, 3), label='Reward 51', color='red')
    plt.plot(reward_df['Episode'], savgol_filter(reward_df['Reward'], 101, 3), label='Reward 101', color='green')
    plt.plot(reward_df['Episode'], savgol_filter(reward_df['Reward'], 201, 3), label='Reward 201', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward')
    plt.legend()
    plt.show()

def generate_trade_summary(df,Action,Price):
    # Initialize an empty list to store trade summaries
    trades = []

    # Initialize variables for tracking trades
    buy_price = None
    buy_date = None

    # Iterate through the DataFrame to find buy and sell actions
    for index, row in df.iterrows():
        if row[Action] == 'Buy':
            buy_price = row[Price]
            buy_date = row['Date']
        elif row[Action] == 'Sell' and buy_price is not None:
            sell_price = row[Price]
            sell_date = row['Date']
            profit_loss = sell_price - buy_price
            trades.append({
                "Trade": len(trades) + 1,
                "Buy Date": buy_date,
                "Sell Date": sell_date,
                "Buy Price": buy_price,
                "Sell Price": sell_price,
                "Profit/Loss": profit_loss
            })
            # Reset buy_price and buy_date for next trade
            buy_price = None
            buy_date = None

    # Create a DataFrame from the trades list
    trade_summary_df = pd.DataFrame(trades)
    trade_summary_df['Profit/Loss_percent'] = trade_summary_df['Profit/Loss']/trade_summary_df['Buy Price']
    trade_summary_df['Profit_Loss'] = trade_summary_df['Profit/Loss'].apply(lambda x: 'Profit' if x>0 else 'Loss')
    trade_summary_df= trade_summary_df.sort_values(by='Profit/Loss', ascending=False)
    return trade_summary_df


def plot_test_trades(df):
    # Create the subplot figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the Close price trace
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')),
        secondary_y=False,
    )

    # Add the Capital_MACD trace
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Capital_MACD'], mode='lines', name='Capital MACD', line=dict(color='green')),
        secondary_y=True,
    )

    # Add the Capital_DQN trace
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Capital_DQN'], mode='lines', name='Capital DQN', line=dict(color='orange')),
        secondary_y=True,
    )

    # Add Buy/Sell markers for MACD_Trades
    for i in range(len(df)):
        if df['MACD_Trades'][i] == 'Buy':
            fig.add_trace(
                go.Scatter(x=[df['Date'][i]], y=[df['Capital_MACD'][i]], mode='markers',
                        marker=dict( size=3, color='black'),showlegend=False,hoverinfo='none'),
                secondary_y=True,)

    # Add Buy/Sell markers for DQN_Trades
    for i in range(len(df)):
        if df['DQN_Trades'][i] == 'Buy':
            fig.add_trace(
                go.Scatter(x=[df['Date'][i]], y=[df['Capital_DQN'][i]], mode='markers',
                        marker=dict(size=3, color='black'),showlegend=False,hoverinfo='none'),
                secondary_y=True,)

    # Update layout for better readability and interactivity
    fig.update_layout(
        title='Close Price and Capital Over Time with Buy/Sell Markers',
        xaxis_title='Date',
        yaxis_title='Close Price',
        yaxis2_title='Capital',
        hovermode='x unified'
    )

    # Show the figure
    fig.show()

def test_trades_summary(df_test, evaluation_log):
    test_trades_combined = df_test[['Close']]
    test_trades_combined = test_trades_combined.join(calculate_macd_signals(df_test)[['MACD_Trades','Capital_MACD']])
    test_trades_combined['MACD_Trades'] = test_trades_combined['MACD_Trades'].fillna('Hold')
    test_trades_combined['Capital_MACD'] = test_trades_combined['Capital_MACD'].fillna(100)
    test_trades_combined = test_trades_combined.reset_index()
    test_trades_combined = test_trades_combined.rename(columns={'index':'Date'})
    test_trades_combined = test_trades_combined.merge(evaluation_log[['Date','Action','Capital']], on='Date', how='left')
    test_trades_combined = test_trades_combined.rename(columns={'Action':'DQN_Trades','Capital':'Capital_DQN'})
    test_trades_combined['DQN_Trades'] = test_trades_combined['DQN_Trades'].fillna('Hold')
    test_trades_combined['Capital_DQN'] = test_trades_combined['Capital_DQN'].fillna(100)
    #print empty line
    print()
    print("--- Test Trades Summary ---")
    print(f"Return without DQN: {round((test_trades_combined['Close'].iloc[-1] - test_trades_combined['Close'].iloc[0])/test_trades_combined['Close'].iloc[0], 3)}%")
    print(f"Return with DQN: {round((test_trades_combined['Capital_DQN'].iloc[-1] - 100), 3)}%")
    print(f"Return with MACD: {round((test_trades_combined['Capital_MACD'].iloc[-1] - 100), 3)}%")
    print(f"Return with Combined Capital: {round(test_trades_combined['Capital_MACD'].iloc[-1] - 100 + test_trades_combined['Capital_DQN'].iloc[-1] - 100, 3)}%")
    plot_test_trades(test_trades_combined)
    return test_trades_combined

def train_agent_long(agent, states, episodes, batch_size):
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

        for time in range(1, len(states)):
            action = agent.act(state)

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
