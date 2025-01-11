# import streamlit as st
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Hard-coded API key
# API_KEY = '0ZWV6OY7JZHCIVXL'
# BASE_URL = 'https://www.alphavantage.co/query'

# # Set page configuration
# st.set_page_config(page_title="StockBuddy Assistant",
#                    layout="wide",
#                    page_icon="💹")

# # Fetch stock data function
# def fetch_stock_data(symbol: str):
#     params = {
#         'function': 'TIME_SERIES_DAILY',
#         'symbol': symbol,
#         'apikey': API_KEY
#     }
#     response = requests.get(BASE_URL, params=params)
#     data = response.json()
#     return data

# # Process stock data into DataFrame
# def process_stock_data(data):
#     dates = []
#     open_prices = []
#     high_prices = []
#     close_prices = []
#     for date, stats in data['Time Series (Daily)'].items():
#         dates.append(date)
#         open_prices.append(float(stats['1. open']))
#         high_prices.append(float(stats['2. high']))
#         close_prices.append(float(stats['4. close']))
#     df = pd.DataFrame({
#         'Date': pd.to_datetime(dates),
#         'Open Price': open_prices,
#         'High Price': high_prices,
#         'Closing Price': close_prices
#     }).sort_values('Date')
#     return df

# # Plot stock data
# def plot_stock_data(df):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(df['Date'], df['Open Price'], label='Open Price', color='green', linestyle='--')
#     ax.plot(df['Date'], df['High Price'], label='High Price', color='red', linestyle='--')
#     ax.plot(df['Date'], df['Closing Price'], label='Closing Price', color='orange')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     ax.set_title('Stock Prices')
#     ax.legend()
#     st.pyplot(fig)

# # Forecast stock prices
# def forecast_stock_prices(df):
#     df['Days'] = np.arange(len(df))  

#     recent_data = df[-60:]

#     X = recent_data[['Days']]
#     y = recent_data['Closing Price']

#     # Train the model
#     model = LinearRegression()
#     model.fit(X, y)

#     # Forecast for the next 7 days
#     future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
#     forecast = model.predict(future_days)

#     # Append forecasted data to DataFrame
#     forecast_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
#     forecast_df = pd.DataFrame({
#         'Date': forecast_dates,
#         'Closing Price': forecast
#     })

#     # Combine historical and forecast data
#     combined_df = pd.concat([df[['Date', 'Closing Price']], forecast_df])

#     # Plot combined data
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(df['Date'], df['Closing Price'], label='Historical Prices', color='blue')
#     ax.plot(forecast_df['Date'], forecast_df['Closing Price'], label='Forecasted Prices', color='red', linestyle='--')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Closing Price')
#     ax.set_title('Stock Prices with 1-Week Forecast')
#     ax.legend()
#     st.pyplot(fig)

#     return forecast_df

# # Decision-making based on forecast
# def should_buy(forecast_df):
#     prices = forecast_df['Closing Price']
#     if prices.iloc[-1] > prices.iloc[0]:
#         st.success("Forecast suggests an upward trend! It may be a good time to BUY the stock.")
#     else:
#         st.warning("Forecast suggests a downward trend. It might not be the best time to buy the stock.")

# # Main app
# def main():
#     st.title('Stock Market Data Viewer with Forecast')

#     symbol = st.text_input('Enter Stock Symbol', 'AAPL')
#     if st.button('Get Data'):
#         if symbol:
#             data = fetch_stock_data(symbol)
#             if 'Time Series (Daily)' in data:
#                 df = process_stock_data(data)

#                 # Show the stock price chart first
#                 st.write("### Stock Price Chart")
#                 plot_stock_data(df)

#                 # Display the data table
#                 st.write("### Historical Stock Data")
#                 st.dataframe(df)

#                 # Show the forecast section
#                 st.write("### 1-Week Stock Price Forecast")
#                 forecast_df = forecast_stock_prices(df)

#                 # Display the forecasted data
#                 st.write("### Forecast Data")
#                 st.dataframe(forecast_df)

#                 # Display recommendation based on forecast
#                 st.write("### Recommendation")
#                 should_buy(forecast_df)
#             else:
#                 st.error("Error fetching data. Please check the stock symbol and try again.")
#         else:
#             st.warning("Please enter a stock symbol.")

# if __name__ == "__main__":
#     main()




import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Hard-coded API key
API_KEY = '0ZWV6OY7JZHCIVXL'
BASE_URL = 'https://www.alphavantage.co/query'

# Set page configuration
st.set_page_config(page_title="StockBuddy Assistant",
                   layout="wide",
                   page_icon="💹")

# Fetch stock data function
def fetch_stock_data(symbol: str):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data

# Process stock data into DataFrame
def process_stock_data(data):
    dates = []
    open_prices = []
    high_prices = []
    close_prices = []
    for date, stats in data['Time Series (Daily)'].items():
        dates.append(date)
        open_prices.append(float(stats['1. open']))
        high_prices.append(float(stats['2. high']))
        close_prices.append(float(stats['4. close']))
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Open Price': open_prices,
        'High Price': high_prices,
        'Closing Price': close_prices
    }).sort_values('Date')
    return df

# Plot stock data
def plot_stock_data(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df['Date'], df['Open Price'], label='Open Price', color='green', linestyle='--')
    ax.plot(df['Date'], df['High Price'], label='High Price', color='red', linestyle='--')
    ax.plot(df['Date'], df['Closing Price'], label='Closing Price', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Prices')
    ax.legend()
    st.pyplot(fig)

# Forecast stock prices (updated to exclude weekends)
def forecast_stock_prices(df):
    df['Days'] = np.arange(len(df))  

    recent_data = df[-60:]

    X = recent_data[['Days']]
    y = recent_data['Closing Price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast for the next 7 trading days (excluding weekends)
    future_dates = []
    last_date = df['Date'].iloc[-1]
    while len(future_dates) < 7:
        last_date += pd.Timedelta(days=1)
        if last_date.weekday() < 5:  # Weekday check (Mon-Fri)
            future_dates.append(last_date)

    future_days = np.arange(len(df), len(df) + len(future_dates)).reshape(-1, 1)
    forecast = model.predict(future_days)

    # Append forecasted data to DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Closing Price': forecast
    })

    # Combine historical and forecast data
    combined_df = pd.concat([df[['Date', 'Closing Price']], forecast_df])

    # Plot combined data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df['Date'], df['Closing Price'], label='Historical Prices', color='blue')
    ax.plot(forecast_df['Date'], forecast_df['Closing Price'], label='Forecasted Prices', color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Stock Prices with 1-Week Forecast')
    ax.legend()
    st.pyplot(fig)

    return forecast_df

# Decision-making based on forecast
def should_buy(forecast_df):
    prices = forecast_df['Closing Price']
    if prices.iloc[-1] > prices.iloc[0]:
        st.success("Forecast suggests an upward trend! It may be a good time to BUY the stock.")
    else:
        st.warning("Forecast suggests a downward trend. It might not be the best time to buy the stock.")

# Main app
def main():
    st.title('Stock Market Data Viewer with Forecast')

    symbol = st.text_input('Enter Stock Symbol', 'AAPL')
    if st.button('Get Data'):
        if symbol:
            data = fetch_stock_data(symbol)
            if 'Time Series (Daily)' in data:
                df = process_stock_data(data)

                # Show the stock price chart first
                st.write("### Stock Price Chart")
                plot_stock_data(df)

                # Display the data table
                st.write("### Historical Stock Data")
                st.dataframe(df)

                # Show the forecast section
                st.write("### 1-Week Stock Price Forecast")
                forecast_df = forecast_stock_prices(df)

                # Display the forecasted data
                st.write("### Forecast Data")
                st.dataframe(forecast_df)

                # Display recommendation based on forecast
                st.write("### Recommendation")
                should_buy(forecast_df)
            else:
                st.error("Error fetching data. Please check the stock symbol and try again.")
        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
