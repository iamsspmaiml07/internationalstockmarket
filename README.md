# StockBuddy Assistant

**StockBuddy Assistant** is a powerful and intuitive web application built using **Streamlit** that enables users to view, analyze, and forecast stock market data. It fetches historical stock data from **Alpha Vantage API**, displays detailed stock trends, and provides a 7-day stock price forecast using a machine learning model. Designed for both novice and professional traders, this app helps users make informed investment decisions.

## Features

### Fetch Real-Time Stock Data
- Input any stock symbol (e.g., `AAPL` for Apple, `GOOGL` for Google).
- Fetches daily stock data using the **Alpha Vantage API**.

### Visualize Stock Trends
- Interactive line charts display historical prices including **Open**, **High**, and **Closing** prices.
- Visualizations make it easy to analyze price movements over time.

### Historical Data Table
- A clean, tabular representation of historical stock data for easier analysis.
- Data includes dates and key price points such as **Open**, **High**, and **Closing** prices.

### 1-Week Stock Price Forecast
- Utilizes a **Linear Regression** model to predict the stock's closing prices for the next 7 days.
- Forecasted prices are appended to historical data for better trend visualization.

### Investment Recommendation
- Based on forecasted trends, the app provides actionable recommendations:
  - **Buy**: If the stock shows an upward trend over the forecasted period.
  - **Hold**: If no significant upward trend is detected.

### User-Friendly Interface
- Intuitive input fields and buttons for streamlined data interaction.
- Fully responsive interface compatible with desktops and tablets.

## How to Use

### Run the Application
Ensure all required Python libraries are installed (see **Installation** section below). Start the app with the command:

```bash
streamlit run app.py
