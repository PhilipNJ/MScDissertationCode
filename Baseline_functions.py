#import the necessary libraries
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go


def capital_calculation(df,col):
    # Calculate the capital
    current_capital = 100
    buy_price = None
    df['Capital'] = current_capital

    for index, row in df.iterrows():
        if row[col] == 'Buy':
            buy_price = row['Close']
        elif row[col] == 'Sell':
            if buy_price is not None:
                trade_return = (row['Close'] - buy_price) / buy_price
                current_capital += current_capital * trade_return
                buy_price = None
        df.at[index, 'Capital'] = current_capital
    
    return df.Capital

def calculate_macd_signals(df):
    # Remove first row
    df = df.iloc[1:].copy()
    
    # Calculate EMAs
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD and Signal Line
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Initialize trade signals
    df['MACD_Trades'] = 'Hold'
    
    position = 0  # 1 for holding a stock, 0 for no position
    buy_date = None  # Track the date of the buy signal
    
    for i, row in df.iterrows():
        if row['MACD'] > row['Signal'] and position == 0:
            df.at[i, 'MACD_Trades'] = 'Buy'
            position = 1
            buy_date = i  # Record the index (date) of the buy signal
        elif row['MACD'] < row['Signal'] and position == 1:
            df.at[i, 'MACD_Trades'] = 'Sell'
            position = 0
            buy_date = None  # Reset the buy date
        elif position == 1 and buy_date is not None and (i - buy_date).days >= 11:
            # If holding for 11 days, force a sell
            df.at[i, 'MACD_Trades'] = 'Sell'
            position = 0
            buy_date = None  # Reset the buy date
    
    # Calculate number of days and trades
    num_days = len(df)
    num_trades = len(df[df['MACD_Trades'] != 'Hold'])
    trades_ratio = num_trades / num_days
    df['Capital_MACD'] = capital_calculation(df, 'MACD_Trades')
    df.drop(['EMA_12', 'EMA_26', 'MACD', 'Signal', 'Capital'], axis=1, inplace=True)

    print(f"Final Capital: {df.Capital_MACD.iloc[-1]}")
    print(f"Overall Return: {df.Capital_MACD.iloc[-1] - 100}")
    print(f"Overall Return %: {(df.Capital_MACD.iloc[-1] - 100)/100*100}")
    print(f"Number of Days: {num_days}")
    print(f"Number of Trades: {num_trades}")
    print("Ratio of Trades to Days: ", trades_ratio)

    return df

def profit_trades(df):
    df['Trades_Profit'] = 'Hold'
    df.sort_index(inplace=True, ascending=False)
    i = 0
    while i < len(df):
        # Select the highest value in Close in the next 12 rows for a Sell
        max_idx = df.iloc[i:i+12]['Close'].idxmax()
        df.at[max_idx, 'Trades_Profit'] = 'Sell'
        i = df.index.get_loc(max_idx) + 1
        
        if i >= len(df):
            break
        
        # Select the lowest value in Close in the next 11 rows for a Buy
        min_idx = df.iloc[i:i+11]['Close'].idxmin()
        df.at[min_idx, 'Trades_Profit'] = 'Buy'
        i = df.index.get_loc(min_idx) + 1

    df.sort_index(inplace=True, ascending=True)

    df['Capital_Profit'] = capital_calculation(df, 'Trades_Profit')
    df.drop(['Capital'], axis=1, inplace=True)

    print(f"Final Capital: {df.Capital_Profit.iloc[-1]}")
    print(f"Overall Return: {df.Capital_Profit.iloc[-1] - 100}")
    print(f"Overall Return %: {(df.Capital_Profit.iloc[-1] - 100)/100*100}")
    print(f"Number of Days: {df.shape[0]}")
    print(f"Number of Trades: {df[df['Trades_Profit'] != 'Hold'].shape[0]}")
    print("Ratio of Trades to Days: ", df[df['Trades_Profit'] != 'Hold'].shape[0]/df.shape[0])
    return df

def loss_trades(df):
    df['Trades_Loss'] = 'Hold'
    df.sort_index(inplace=True, ascending=False)
    i = 0
    while i < len(df):
        # Select the highest value in Close in the next 12 rows for a Sell
        min_idx = df.iloc[i:i+12]['Close'].idxmin()
        df.at[min_idx, 'Trades_Loss'] = 'Sell'
        i = df.index.get_loc(min_idx) + 1
        
        if i >= len(df):
            break
        
        # Select the lowest value in Close in the next 11 rows for a Buy
        max_idx = df.iloc[i:i+11]['Close'].idxmax()
        df.at[max_idx, 'Trades_Loss'] = 'Buy'
        i = df.index.get_loc(max_idx) + 1

    df.sort_index(inplace=True, ascending=True)

    df['Capital_Loss'] = capital_calculation(df, 'Trades_Loss')
    df.drop(['Capital'], axis=1, inplace=True)

    print(f"Final Capital: {df.Capital_Loss.iloc[-1]}")
    print(f"Overall Return: {df.Capital_Loss.iloc[-1] - 100}")
    print(f"Overall Return %: {(df.Capital_Loss.iloc[-1] - 100)/100*100}")
    print(f"Number of Days: {df.shape[0]}")
    print(f"Number of Trades: {df[df['Trades_Loss'] != 'Hold'].shape[0]}")
    print("Ratio of Trades to Days: ", df[df['Trades_Loss'] != 'Hold'].shape[0]/df.shape[0])
    return df

def plot_trades(df, trades_column):
    df_filtered = df[df[trades_column].isin(['Buy', 'Sell'])]
    # Map trades to colors for visualization
    color_map = {'Buy': 'green', 'Sell': 'red'}
    # Create figure
    fig = go.Figure()
    # Add scatter trace for Buys and Sells
    for trade_type in ['Buy', 'Sell']:
        trade_data = df_filtered[df_filtered[trades_column] == trade_type]
        fig.add_trace(go.Scatter(
            x=trade_data.index,
            y=[1] * len(trade_data),  # Dummy y-value for scatter plot
            mode='markers',
            marker=dict(color=color_map[trade_type], size=10, symbol='triangle-up' if trade_type == 'Buy' else 'triangle-down'),
            name=trade_type
        ))
    # Update layout
    fig.update_layout(
        title='Buy and Sell Signals Over Time',
        xaxis_title='Date',
        yaxis_title='Trading Signal',
        yaxis=dict(
            tickvals=[1],
            ticktext=['Trading Signal'],
            showticklabels=True
        ),
        showlegend=True
    )
    # Show plot
    fig.show()
