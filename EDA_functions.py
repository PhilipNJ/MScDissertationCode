import altair as alt
# Enable VegaFusion data transformer
alt.data_transformers.enable('vegafusion')
# Disable the default row limit data transformer
alt.data_transformers.disable_max_rows()
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import jarque_bera
from scipy.stats import normaltest



def EDA(df):
    # Ensure the 'Date' column is in datetime format
    df.index = pd.to_datetime(df.index)
    
    # Calculate daily return and cumulative return
    df['daily_return'] = df['Close'].pct_change().fillna(0)
    df['cumulative_return'] = (1 + df['daily_return']).cumprod().fillna(1)
    
    # Create a selection interval
    brush = alt.selection_interval(encodings=['x'])
    
    # Candlestick chart for OHLC data
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
    
    # Line plot for volume
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
    
    # Line plot for daily returns
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
    
    # Line plot for cumulative returns
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
    
    # Monthly resampled mean closing prices
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
    
    # Yearly resampled mean closing prices
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
    
    # Combine all charts into a single dashboard with an overall title
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


def decomposition_plot(ts):
# Apply seasonal_decompose 
    decomposition = seasonal_decompose(ts, period =255)
    
# Get trend, seasonality, and residuals
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

# Plotting
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(np.log(ts), label='Original', color='blue')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='blue')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality', color='blue')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals', color='blue')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def stationarity_check(ts):
            
    # Calculate rolling statistics
    roll_mean = ts.rolling(window=8, center=False).mean()
    roll_std = ts.rolling(window=8, center=False).std()

    # Perform the Dickey Fuller test
    dftest = adfuller(ts) 
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results

    print('\nResults of Dickey-Fuller Test: \n')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                             '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def normality_check(ts):
    # Normality test
    ts = ts.dropna()
    ts_1st_diff = ts.diff().dropna()
    jb_test = jarque_bera(ts)
    norm_test = normaltest(ts)
    jb_test_1st_diff = jarque_bera(ts_1st_diff)
    norm_test_1st_diff = normaltest(ts_1st_diff)
    
    print('Jarque-Bera Test ---- statistic: {}, p-value: {}'.format(jb_test[0], jb_test[1]))
    print('Normal Test ---- statistic: {}, p-value: {}'.format(norm_test[0], norm_test[1]))
    print('Jarque-Bera Test 1st diff ---- statistic: {}, p-value: {}'.format(jb_test_1st_diff[0], jb_test_1st_diff[1]))
    print('Normal Test 1st diff ---- statistic: {}, p-value: {}'.format(norm_test_1st_diff[0], norm_test_1st_diff[1]))