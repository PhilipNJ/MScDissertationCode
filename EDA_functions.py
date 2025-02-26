import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import jarque_bera
from scipy.stats import normaltest
import plotly.graph_objects as go
alt.data_transformers.enable('vegafusion')
alt.data_transformers.disable_max_rows()
import seaborn as sns
import matplotlib.pyplot as plt

def nifty_convert(df):
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    #rename price as Close, Vol. as Volume
    df.rename(columns={'Price':'Close'}, inplace=True)
    df = df[['Date','Open','High','Low','Close','Vol.']]
    df = df.ffill()
    df['Open'] = df['Open'].str.replace(',', '').astype(float)
    df['High'] = df['High'].str.replace(',', '').astype(float)
    df['Low'] = df['Low'].str.replace(',', '').astype(float)
    df['Close'] = df['Close'].str.replace(',', '').astype(float)
    
    # Function to convert the values
    def convert_volume(value):
        if 'M' in value:
            return float(value.replace('M', '').replace(',', '')) * 1_000_000
        elif 'B' in value:
            return float(value.replace('B', '').replace(',', '')) * 1_000_000_000
        else:
            return float(value)

    # Apply the conversion function to the 'Vol.' column
    df['Volume'] = df['Vol.'].apply(convert_volume)

    # Drop the 'Vol.' column
    df.drop('Vol.', axis=1, inplace=True)
    df = df.sort_values('Date')
    df = df.set_index('Date')
    
    return df

def EDA(df):
    df.index = pd.to_datetime(df.index)
    df['daily_return'] = df['Close'].pct_change().fillna(0)
    df['cumulative_return'] = (1 + df['daily_return']).cumprod().fillna(1)
    
    brush = alt.selection_interval(encodings=['x'])
    
    base = alt.Chart(df.reset_index()).encode(
        x='Date:T',
        tooltip=['Date:T', 'Open:Q', 'High:Q', 'Low:Q', 'Close:Q']
    ).properties(
        width=800,
        height=300,
        title='Candlestick Chart of Closing Prices'
    ).add_selection(
        brush
    ).interactive()
    
    rule = base.mark_rule().encode(
        y='Low:Q',
        y2='High:Q'
    )
    
    bar = base.mark_bar().encode(
        y='Open:Q',
        y2='Close:Q',
        color=alt.condition("datum.Open <= datum.Close",
                            alt.value("#06982d"),  # Green for increasing
                            alt.value("#ae1325"))  # Red for decreasing
    )
    
    candlestick_chart = rule + bar
    
    volume = alt.Chart(df.reset_index()).mark_line().encode(
        x='Date:T',
        y='Volume:Q',
        tooltip=['Date:T', 'Volume:Q']
    ).properties(
        width=800,
        height=200,
        title='Volume over Time'
    ).transform_filter(
        brush
    ).interactive()
    
    daily_return = alt.Chart(df.reset_index()).mark_line().encode(
        x='Date:T',
        y='daily_return:Q',
        tooltip=['Date:T', 'daily_return:Q']
    ).properties(
        width=800,
        height=200,
        title='Daily Returns'
    ).transform_filter(
        brush
    ).interactive()
    
    cumulative_return = alt.Chart(df.reset_index()).mark_line().encode(
        x='Date:T',
        y='cumulative_return:Q',
        tooltip=['Date:T', 'cumulative_return:Q']
    ).properties(
        width=800,
        height=200,
        title='Cumulative Returns'
    ).transform_filter(
        brush
    ).interactive()
    
    df_monthly = df['Close'].resample('M').mean().reset_index()
    monthly_close = alt.Chart(df_monthly).mark_line().encode(
        x='Date:T',
        y='Close:Q',
        tooltip=['Date:T', 'Close:Q']
    ).properties(
        width=800,
        height=200,
        title='Monthly Average Closing Prices'
    ).transform_filter(
        brush
    ).interactive()
    
    df_yearly = df['Close'].resample('Y').mean().reset_index()
    yearly_close = alt.Chart(df_yearly).mark_line().encode(
        x='Date:T',
        y='Close:Q',
        tooltip=['Date:T', 'Close:Q']
    ).properties(
        width=800,
        height=200,
        title='Yearly Average Closing Prices'
    ).transform_filter(
        brush
    ).interactive()
    
    dashboard = alt.vconcat(
        candlestick_chart,
        volume,
        daily_return,
        cumulative_return,
        monthly_close,
        yearly_close
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16
    ).properties(
        title='Exploratory Data Analysis Dashboard'
    )
    
    return dashboard

def decomposition_plot(df, difference=0):
    cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    fig, axs = plt.subplots(len(cols) * 4, 1, figsize=(10, 8 * len(cols)))
    fig.tight_layout(pad=5.0)
    
    for i, col in enumerate(cols):
        ts = df[col]
        if difference > 0:
            ts = ts.diff(difference).dropna()
        
        decomposition = seasonal_decompose(ts, period=255)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        axs[i*4].plot(np.log(ts), label='Original', color='blue')
        axs[i*4].legend(loc='best')
        axs[i*4].set_title(f'{col} - Original')
        
        axs[i*4 + 1].plot(trend, label='Trend', color='blue')
        axs[i*4 + 1].legend(loc='best')
        axs[i*4 + 1].set_title(f'{col} - Trend')
        
        axs[i*4 + 2].plot(seasonal, label='Seasonality', color='blue')
        axs[i*4 + 2].legend(loc='best')
        axs[i*4 + 2].set_title(f'{col} - Seasonality')
        
        axs[i*4 + 3].plot(residual, label='Residuals', color='blue')
        axs[i*4 + 3].legend(loc='best')
        axs[i*4 + 3].set_title(f'{col} - Residuals')
    
    plt.show()

def stationarity_check(df,difference=0):
    cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    fig, axs = plt.subplots(len(cols), 1, figsize=(12, 6 * len(cols)))
    fig.tight_layout(pad=5.0)
    
    for i, col in enumerate(cols):
        ts = df[col].dropna()
        if difference > 0:
            ts = ts.diff(difference).dropna()
        roll_mean = ts.rolling(window=8, center=False).mean()
        roll_std = ts.rolling(window=8, center=False).std()
        dftest = adfuller(ts) 

        axs[i].plot(ts, color='blue', label='Original')
        axs[i].plot(roll_mean, color='red', label='Rolling Mean')
        axs[i].plot(roll_std, color='green', label='Rolling Std')
        axs[i].legend(loc='best')
        axs[i].set_title(f'{col} - Rolling Mean & Standard Deviation')
    
        print(f'\nResults of Dickey-Fuller Test for {col}: \n')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput[f'Critical Value ({key})'] = value
        print(dfoutput)
    
    plt.show()

def normality_check(df,difference=0):
    cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    
    for col in cols:
        ts = df[col].dropna()
        if difference > 0:
            ts = ts.diff(difference).dropna()
        jb_test = jarque_bera(ts)
        norm_test = normaltest(ts)
        
        print(f'\nNormality Check for {col}:')
        print('Jarque-Bera Test ---- statistic: {}, p-value: {}'.format(jb_test[0], jb_test[1]))
        print('Normal Test ---- statistic: {}, p-value: {}'.format(norm_test[0], norm_test[1]))

def daily_returns(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    daily_returns = df.pct_change().dropna()

    fig, axes = plt.subplots(len(daily_returns.columns), 1, figsize=(12, 6 * len(daily_returns.columns)))
    fig.tight_layout(pad=3.0)
    
    for i, column in enumerate(daily_returns.columns):
        sns.boxplot(data=daily_returns[column], ax=axes[i], palette="Set3")

        # Calculate statistics
        mu = daily_returns[column].mean()
        sigma = daily_returns[column].std()
        percentiles = np.percentile(daily_returns[column], [90, 95, 99])

        # Plot mean plus/minus sigma lines
        axes[i].axhline(mu + sigma, color='blue', linestyle='--', label='Mean ± 1σ')
        axes[i].axhline(mu - sigma, color='blue', linestyle='--')
        axes[i].axhline(mu + 2 * sigma, color='blue', linestyle='--', label='Mean ± 2σ')
        axes[i].axhline(mu - 2 * sigma, color='blue', linestyle='--')
        axes[i].axhline(mu + 3 * sigma, color='blue', linestyle='--', label='Mean ± 3σ')
        axes[i].axhline(mu - 3 * sigma, color='blue', linestyle='--')

        # Plot percentile lines
        axes[i].axhline(percentiles[0], color='orange', linestyle=':', label='90th Percentile')
        axes[i].axhline(percentiles[1], color='red', linestyle=':', label='95th Percentile')
        axes[i].axhline(percentiles[2], color='green', linestyle=':', label='99th Percentile')

        # Avoid duplicate labels
        handles, labels = axes[i].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[i].legend(by_label.values(), by_label.keys(), loc='upper right')

        axes[i].set_title(f'Daily Returns for {column} with Statistical Lines')
        axes[i].set_ylabel('Daily Returns')
        axes[i].set_xlabel(column)

    plt.show()

def plot_fy_annual_returns(df, title):
    def calculate_fy_returns(df):
        df = df.copy()
        df['FY'] = df.index.to_period('A-MAR')  # Group by fiscal year ending in March
        fy_returns = df['Close'].resample('A-MAR').last().pct_change().dropna() * 100
        return fy_returns


    def plot_with_labels(ax, fy_returns, title, color):
        bars = ax.bar(fy_returns.index.year - 2, fy_returns, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel('Return (%)')
        ax.set_xlabel('Fiscal Year')
        ax.set_xticks(fy_returns.index.year - 1)

        # Add return labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Calculate fiscal year returns for the dataframe
    fy_returns = calculate_fy_returns(df)

    ten_year_return = (df[df.index >= '2014-04-01']['Close'][-1]-df[df.index >= '2014-04-01']['Close'][0])/df[df.index >= '2014-04-01']['Close'][0]
    overall_return = (df['Close'][-1]-df['Close'][0])/df['Close'][0]
    print(f'Ten year return: {round(ten_year_return,2)}%')
    print(f'Overall return: {round(overall_return,2)}%')

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_with_labels(ax, fy_returns, title, color='Red')

    plt.tight_layout()
    plt.show()