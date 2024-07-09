import altair as alt
# Enable VegaFusion data transformer
alt.data_transformers.enable('vegafusion')
# Disable the default row limit data transformer
alt.data_transformers.disable_max_rows()
import pandas as pd
from IPython.display import display



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