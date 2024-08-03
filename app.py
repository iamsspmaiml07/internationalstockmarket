import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Hard-coded API key
API_KEY = '0ZWV6OY7JZHCIVXL'
BASE_URL = 'https://www.alphavantage.co/query'

# Set page configuration
st.set_page_config(page_title="StockBuddy Assistant",
                   layout="wide",
                   page_icon="💹")

def fetch_stock_data(symbol: str):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data

def plot_stock_data(data):
    # Convert data to DataFrame
    dates = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    for date, stats in data['Time Series (Daily)'].items():
        dates.append(date)
        open_prices.append(float(stats['1. open']))
        high_prices.append(float(stats['2. high']))
        low_prices.append(float(stats['3. low']))
        close_prices.append(float(stats['4. close']))

    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Open Price': open_prices,
        'High Price': high_prices,
        'Low Price': low_prices,
        'Closing Price': close_prices
    }).sort_values('Date')

    # Plot data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df['Date'], df['Open Price'], label='Open Price', color='green', linestyle='--')
    ax.plot(df['Date'], df['High Price'], label='High Price', color='red', linestyle='--')
    ax.plot(df['Date'], df['Low Price'], label='Low Price', color='blue', linestyle='--')
    ax.plot(df['Date'], df['Closing Price'], label='Closing Price', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Prices')
    ax.legend()
    st.pyplot(fig)

    # Display data table
    st.write("### Stock Data Table")
    st.dataframe(df)

def main():
    st.title('Stock Market Data Viewer')

    symbol = st.text_input('Enter Stock Symbol', 'AAPL')
    if st.button('Get Data'):
        if symbol:
            data = fetch_stock_data(symbol)
            if 'Time Series (Daily)' in data:
                st.write(f"Data for {symbol}")
                plot_stock_data(data)
            else:
                st.error("Error fetching data. Please check the stock symbol and try again.")
        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
