def calculate_macd_signals(df, initial_capital=100):
    #remove first 1 row
    df = df.iloc[1:]
    # Calculate EMAs
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD and Signal Line
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Generate Buy, Sell, Hold signals
    df['Buy'] = np.where(df['MACD'] > df['Signal'], 1, 0)
    df['Sell'] = np.where(df['MACD'] < df['Signal'], 1, 0)
    df['Hold'] = np.where(df['MACD'] == df['Signal'], 1, 0)
    
    # Initialize trade flags and counter
    buy_flag = False
    hold_flag = False
    sell_flag = True
    days_count = 0
    
    df['Trades'] = 'Hold'
    
    # Initialize capital variables
    current_capital = initial_capital
    buy_price = None
    
    df['Capital'] = current_capital
    
    # Calculate trades and returns in the same loop
    for index, row in df.iterrows():
        if row['Buy'] == 1:
            if sell_flag:
                buy_flag = True
                hold_flag = False
                sell_flag = False
                days_count = 0
                df.at[index, 'Trades'] = 'Buy'
                buy_price = row['Close']
        elif buy_flag:
            if row['Sell'] == 1 or days_count == 11:
                buy_flag = False
                hold_flag = False
                sell_flag = True
                df.at[index, 'Trades'] = 'Sell'
                if buy_price is not None:
                    trade_return = (row['Close'] - buy_price) / buy_price
                    current_capital += current_capital * trade_return
                    buy_price = None
            else:
                hold_flag = True
                df.at[index, 'Trades'] = 'Hold'
                days_count += 1
        else:
            df.at[index, 'Trades'] = 'Hold'
        
        # Update the capital in the DataFrame
        df.at[index, 'Capital'] = current_capital
    
    # Print final results
    final_capital = df.iloc[-1]['Capital']
    overall_return = final_capital - initial_capital
    
    print(f"Final Capital: {final_capital}")
    print(f"Overall Return: {overall_return}")
    print(f"Overall Return %: {overall_return/initial_capital*100}")
    #print number of days
    print(f"Number of Days: {df.shape[0]}")
    #print number of trades
    print(f"Number of Trades: {df[df['Trades'] != 'Hold'].shape[0]}")
    print("Ratio of Trades to Days: ", df[df['Trades'] != 'Hold'].shape[0]/df.shape[0])
    print("----------------------------------------"+'\n')
    
    df.rename(columns={'Trades': 'Trades_MACD', 'Capital':'Capital_MACD'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume','Trades_MACD', 'Capital_MACD']]


def capital_return(df,col,initial_capital=100):
    # Initialize trade flags and counter
    buy_flag = False
    hold_flag = False
    sell_flag = True
    days_count = 0
    
    # Initialize capital variables
    current_capital = initial_capital
    buy_price = None
    
    df['Capital'] = current_capital
    
    # Calculate trades and returns in the same loop
    for index, row in df.iterrows():
        if row[col] == 'Sell':
            if sell_flag:
                buy_flag = True
                hold_flag = False
                sell_flag = False
                days_count = 0
                df.at[index, 'Trades'] = 'Buy'
                buy_price = row['Close']
        elif buy_flag:
            if row['Sell'] == 1 or days_count == 11:
                buy_flag = False
                hold_flag = False
                sell_flag = True
                df.at[index, 'Trades'] = 'Sell'
                if buy_price is not None:
                    trade_return = (row['Close'] - buy_price) / buy_price
                    current_capital += current_capital * trade_return
                    buy_price = None
            else:
                hold_flag = True
                df.at[index, 'Trades'] = 'Hold'
                days_count += 1
        else:
            df.at[index, 'Trades'] = 'Hold'
        
        # Update the capital in the DataFrame
        df.at[index, 'Capital'] = current_capital
    
    # Print final results
    final_capital = df.iloc[-1]['Capital']
    overall_return = final_capital - initial_capital
    


# Function to find Sell and Buy points
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


    # Calculate the capital
    current_capital = 100
    buy_price = None
    df['Capital_Profit'] = current_capital

    for index, row in df.iterrows():
        if row['Trades_Profit'] == 'Buy':
            buy_price = row['Close']
        elif row['Trades_Profit'] == 'Sell':
            if buy_price is not None:
                trade_return = (row['Close'] - buy_price) / buy_price
                current_capital += current_capital * trade_return
                buy_price = None
        df.at[index, 'Capital_Profit'] = current_capital

    print(f"Final Capital: {current_capital}")
    print(f"Overall Return: {current_capital - 100}")
    print(f"Overall Return %: {(current_capital - 100)/100*100}")
    print(f"Number of Days: {df.shape[0]}")
    print(f"Number of Trades: {df[df['Trades_Profit'] != 'Hold'].shape[0]}")
    print("Ratio of Trades to Days: ", df[df['Trades_Profit'] != 'Hold'].shape[0]/df.shape[0])
    print("----------------------------------------"+'\n')


    return df

# Function to find Sell and Buy points
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


    # Calculate the capital
    current_capital = 100
    buy_price = None
    df['Capital_Loss'] = current_capital

    for index, row in df.iterrows():
        if row['Trades_Loss'] == 'Buy':
            buy_price = row['Close']
        elif row['Trades_Loss'] == 'Sell':
            if buy_price is not None:
                trade_return = (row['Close'] - buy_price) / buy_price
                current_capital += current_capital * trade_return
                buy_price = None
        df.at[index, 'Capital_Loss'] = current_capital

    print(f"Final Capital: {current_capital}")
    print(f"Overall Return: {current_capital - 100}")
    print(f"Overall Return %: {(current_capital - 100)/100*100}")
    print(f"Number of Days: {df.shape[0]}")
    print(f"Number of Trades: {df[df['Trades_Loss'] != 'Hold'].shape[0]}")
    print("Ratio of Trades to Days: ", df[df['Trades_Loss'] != 'Hold'].shape[0]/df.shape[0])
    print("----------------------------------------"+'\n')

    return df


def plot_trades(df, trades_column):
    """
    Plot Buy and Sell signals over time using plotly.

    Args:
    - df (pd.DataFrame): DataFrame containing Date index and 'trades' column.
    - trades_column (str): Name of the column containing 'Buy', 'Sell', or 'Hold' signals.

    Returns:
    - None (displays the plot interactively).
    """
    # Filter out 'Hold' trades
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
