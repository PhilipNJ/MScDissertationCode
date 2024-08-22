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

# Function to create states from a DataFrame
def create_states(df, window_size=9):
    """
    Create sequences of states from the data with a given window size.
    Each state is a sequence of rows from the DataFrame.
    
    Parameters:
    df (DataFrame): Input data
    window_size (int): Number of consecutive rows to form a state
    
    Returns:
    np.array: Array of states
    """
    states = []
    for i in range(window_size, len(df)):
        state = df.iloc[i-window_size:i].values
        states.append(state)
    return np.array(states)

# Deep Q-Network model with convolutional layers
class ConvDQN(nn.Module):
    def __init__(self, input_dim, output_dim, window_size):
        """
        Initialize the ConvDQN model.
        
        Parameters:
        input_dim (int): Number of input features
        output_dim (int): Number of possible actions
        window_size (int): Size of the input sequence (window size)
        """
        super(ConvDQN, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Define the second convolutional layer
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(64 * window_size, 128)
        # Output layer
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
        x (Tensor): Input tensor
        
        Returns:
        Tensor: Output tensor with action values
        """
        # Permute the input to match the expected shape for Conv1d (batch_size, num_features, window_size)
        x = x.permute(0, 2, 1)
        # Apply the first convolutional layer and ReLU activation
        x = F.relu(self.conv1(x))
        # Apply the second convolutional layer and ReLU activation
        x = F.relu(self.conv2(x))
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        # Apply the first fully connected layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the output layer
        x = self.fc2(x)
        return x

# Replay memory to store experiences
class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialize the ReplayMemory with a given capacity.
        
        Parameters:
        capacity (int): Maximum number of experiences to store
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the memory.
        
        Parameters:
        state (array): Current state
        action (int): Action taken
        reward (float): Reward received
        next_state (array): Next state
        done (bool): Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from memory.
        
        Parameters:
        batch_size (int): Number of experiences to sample
        
        Returns:
        list: List of sampled experiences
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return the current size of memory.
        
        Returns:
        int: Number of experiences in memory
        """
        return len(self.memory)

# DQN Agent for interacting with the environment
class DQNAgent:
    def __init__(self, state_size, action_size, window_size, model, lr=0.00001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        """
        Initialize the DQN Agent.
        
        Parameters:
        state_size (int): Dimension of the state space
        action_size (int): Number of possible actions
        window_size (int): Size of the input sequence (window size)
        model (nn.Module): DQN model
        lr (float): Learning rate for the optimizer
        gamma (float): Discount factor for future rewards
        epsilon (float): Initial exploration rate
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for epsilon
        """
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
        """
        Store an experience in the replay memory.
        
        Parameters:
        state (array): Current state
        action (int): Action taken
        reward (float): Reward received
        next_state (array): Next state
        done (bool): Whether the episode is done
        """
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        """
        Choose an action based on the current state.
        
        Parameters:
        state (array): Current state
        
        Returns:
        int: Selected action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        # Avoid gradient calculation during inference
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        """
        Train the model by replaying experiences.
        
        Parameters:
        batch_size (int): Number of experiences to sample and train on
        """
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        actions = torch.LongTensor(actions)

        # Compute target Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_loss_per_episode(self, loss_per_episode):
        """
        Plot the loss per episode.
        
        Parameters:
        loss_per_episode (list): List of loss values for each episode
        """
        plt.plot(loss_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Loss Per Episode')
        plt.show()

# Function to train the agent
def train_agent(agent, states, episodes, batch_size):
    """
    Train the DQN agent over a specified number of episodes.
    
    Parameters:
    agent (DQNAgent): The agent to be trained
    states (array): Array of states
    episodes (int): Number of episodes to train
    batch_size (int): Batch size for replaying experiences
    
    Returns:
    DataFrame: Log of the training process
    """
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

            # Buy action
            if action == 1:
                if has_open_position:
                    action = 2
                else:
                    has_open_position = True
                    days_since_last_buy = 0
                    price_at_buy = state[-1][3]

            # Sell action
            if action == 0:
                if not has_open_position:
                    action = 2
                else:
                    has_open_position = False
                    days_since_last_buy = 0
                    profit = state[-1][3] - price_at_buy
                    reward =+ profit

            # Force sell after holding for too long
            if has_open_position and days_since_last_buy > 11:
                action = 0
                has_open_position = False
                days_since_last_buy = 0
                reward = state[-1][3] - price_at_buy
                reward =+ profit
            
            # Penalize holding with a cost
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

    # Create a log DataFrame for training data
    log_train = pd.DataFrame({'Episode': Episode, 'Time': Time, 'Reward': Reward, 'Action': Action, 'Price': prev_price})
    log_train['Action'] = log_train['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})
    log_train['Action'].value_counts()
    log_train['Loss'] = log_train['Episode'].map({index: element for index, element in enumerate(loss_per_episode, start=1)})

    agent.plot_loss_per_episode(loss_per_episode)
    return log_train

# Function to evaluate the trained agent
def evaluate_agent(agent, states):
    """
    Evaluate the DQN agent on the provided states.
    
    Parameters:
    agent (DQNAgent): The agent to be evaluated
    states (array): Array of states
    
    Returns:
    DataFrame: Log of the evaluation process
    """
    Time = []
    Action = []
    prev_price = []

    has_open_position = False
    price_at_buy = 0
    days_since_last_buy = 0

    for time in range(1, len(states)):
        state = states[time - 1]
        action = agent.act(state)

        if has_open_position:
            days_since_last_buy += 1

        # Buy action
        if action == 1:
            if has_open_position:
                action = 2
            else:
                has_open_position = True
                days_since_last_buy = 0
                price_at_buy = state[-1][3]

        # Sell action
        if action == 0:
            if not has_open_position:
                action = 2
            else:
                has_open_position = False
                days_since_last_buy = 0

        # Force sell after holding for too long
        if has_open_position and days_since_last_buy > 11:
            action = 0
            has_open_position = False
            days_since_last_buy = 0

        prev_price.append(state[-1][3])
        Time.append(time)
        Action.append(action)

    # Create a log DataFrame for evaluation data
    log_evaluation = pd.DataFrame({'Time': Time, 'Action': Action, 'Price': prev_price})
    log_evaluation['Action'] = log_evaluation['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})

    return log_evaluation

# Function to create a DataFrame of actions per episode
def create_action_episode_df(log_train):
    """
    Create a DataFrame where each column corresponds to actions taken in a specific episode.
    
    Parameters:
    log_train (DataFrame): Log of the training process
    
    Returns:
    DataFrame: DataFrame with actions per episode
    """
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

# Function to plot training metrics (reward and loss)
def plot_training(log_train):
    """
    Plot the reward and loss metrics over the episodes.
    
    Parameters:
    log_train (DataFrame): Log of the training process
    """
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

# Function to plot dual axis graph for close price and capital
def plot_dual_axis(all_states_eval):
    """
    Plot a dual-axis graph for Close price and Capital over time.
    
    Parameters:
    all_states_eval (DataFrame): DataFrame containing Close and Capital values over time
    """
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

# Alternative training function with different reward structure
def train_agent_hold(agent, states, episodes, batch_size):
    """
    Train the DQN agent over a specified number of episodes, with a focus on holding positions.
    
    Parameters:
    agent (DQNAgent): The agent to be trained
    states (array): Array of states
    episodes (int): Number of episodes to train
    batch_size (int): Batch size for replaying experiences
    
    Returns:
    DataFrame: Log of the training process
    """
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

            # Buy action
            if action == 1:
                if has_open_position:
                    action = 2
                else:
                    has_open_position = True
                    days_since_last_buy = 0
                    price_at_buy = state[-1][3]

            # Sell action
            if action == 0:
                if not has_open_position:
                    action = 2
                else:
                    has_open_position = False
                    days_since_last_buy = 0
                    profit = state[-1][3] - price_at_buy
                    reward =+ profit

            # Force sell after holding for too long
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

    # Create a log DataFrame for training data
    log_train = pd.DataFrame({'Episode': Episode, 'Time': Time, 'Reward': Reward, 'Action': Action, 'Price': prev_price})
    log_train['Action'] = log_train['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})
    log_train['Action'].value_counts()
    log_train['Loss'] = log_train['Episode'].map({index: element for index, element in enumerate(loss_per_episode, start=1)})

    agent.plot_loss_per_episode(loss_per_episode)
    return log_train

# Function to apply Savitzky-Golay filter to smooth reward data
def reward_filter(reward):
    """
    Plot the reward with different levels of smoothing using Savitzky-Golay filter.
    
    Parameters:
    reward (list): List of rewards per episode
    """
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

# Function to generate a summary of trades
def generate_trade_summary(df,Action,Price):
    """
    Generate a summary of buy/sell trades.
    
    Parameters:
    df (DataFrame): Input DataFrame with trading data
    Action (str): Column name for actions
    Price (str): Column name for prices
    
    Returns:
    DataFrame: Summary of trades with profit/loss calculations
    """
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

# Function to plot the trades on a dual-axis graph
def plot_test_trades(df):
    """
    Plot Close price and Capital over time with Buy/Sell markers.
    
    Parameters:
    df (DataFrame): Input DataFrame with trading data
    """
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

# Function to summarize test trades and plot results
def test_trades_summary(df_test, evaluation_log):
    """
    Summarize and compare test trades using MACD and DQN strategies.
    
    Parameters:
    df_test (DataFrame): Test DataFrame with close prices
    evaluation_log (DataFrame): Log of the evaluation process
    
    Returns:
    DataFrame: Summary of test trades
    """
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

    print()  # Print empty line
    print("--- Test Trades Summary ---")
    print(f"Return without DQN: {round((test_trades_combined['Close'].iloc[-1] - test_trades_combined['Close'].iloc[0])/test_trades_combined['Close'].iloc[0], 3)}%")
    print(f"Return with DQN: {round((test_trades_combined['Capital_DQN'].iloc[-1] - 100), 3)}%")
    print(f"Return with MACD: {round((test_trades_combined['Capital_MACD'].iloc[-1] - 100), 3)}%")
    print(f"Return with Combined Capital: {round(test_trades_combined['Capital_MACD'].iloc[-1] - 100 + test_trades_combined['Capital_DQN'].iloc[-1] - 100, 3)}%")
    plot_test_trades(test_trades_combined)
    return test_trades_combined

# Alternative training function with a different approach
def train_agent_long(agent, states, episodes, batch_size):
    """
    Train the DQN agent over a specified number of episodes, with a focus on holding positions.
    
    Parameters:
    agent (DQNAgent): The agent to be trained
    states (array): Array of states
    episodes (int): Number of episodes to train
    batch_size (int): Batch size for replaying experiences
    
    Returns:
    DataFrame: Log of the training process
    """
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

            # Buy action
            if action == 1:
                if has_open_position:
                    action = 2
                else:
                    has_open_position = True
                    days_since_last_buy = 0
                    price_at_buy = state[-1][3]

            # Sell action
            if action == 0:
                if not has_open_position:
                    action = 2
                else:
                    has_open_position = False
                    days_since_last_buy = 0
                    profit = state[-1][3] - price_at_buy
                    reward =+ profit
            
            # Penalize holding with a cost
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

    # Create a log DataFrame for training data
    log_train = pd.DataFrame({'Episode': Episode, 'Time': Time, 'Reward': Reward, 'Action': Action, 'Price': prev_price})
    log_train['Action'] = log_train['Action'].map({0: 'Sell', 1: 'Buy', 2: 'Hold'})
    log_train['Action'].value_counts()
    log_train['Loss'] = log_train['Episode'].map({index: element for index, element in enumerate(loss_per_episode, start=1)})

    agent.plot_loss_per_episode(loss_per_episode)
    return log_train
