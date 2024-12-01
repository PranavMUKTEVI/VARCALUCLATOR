import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import altair as alt

# VaR Class Definition
class VaR:
    def __init__(self, ticker, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.rolling = rolling_window
        self.conf_level = confidence_level
        self.portf_val = portfolio_val
        self.weights = np.array(weights)
        self.historical_var = None
        self.parametric_var = None

        self.data()
        
    def data(self):
        df = yf.download(self.ticker, self.start, self.end)
        self.adj_close_df = df["Adj Close"]
        self.log_returns_df = np.log(self.adj_close_df / self.adj_close_df.shift(1))
        self.log_returns_df = self.log_returns_df.dropna()

        # Portfolio returns with custom weights
        self.portfolio_returns = (self.log_returns_df * self.weights).sum(axis=1)
        self.rolling_returns = self.portfolio_returns.rolling(window=self.rolling).sum()
        self.rolling_returns = self.rolling_returns.dropna()

        self.historical_method()
        self.parametric_method()

    def historical_method(self):
        historical_VaR = -np.percentile(self.rolling_returns, 100 - (self.conf_level * 100)) * self.portf_val
        self.historical_var = historical_VaR

    def parametric_method(self):
        self.cov_matrix = self.log_returns_df.cov() * 252
        portfolio_std = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        parametric_VaR = portfolio_std * norm.ppf(self.conf_level) * np.sqrt(self.rolling / 252) * self.portf_val
        self.parametric_var = parametric_VaR

    def apply_custom_shocks(self, shocks):
        shocked_returns = self.log_returns_df + shocks
        shocked_portfolio_returns = (shocked_returns * self.weights).sum(axis=1)
        shocked_VaR = -np.percentile(shocked_portfolio_returns, 100 - (self.conf_level * 100)) * self.portf_val
        return shocked_VaR

    def plot_var_results(self, title, var_value, returns_dollar, conf_level):
        plt.figure(figsize=(12, 6))
        plt.hist(returns_dollar, bins=50, density=True)
        plt.xlabel(f'\n {title} VaR = ${var_value:.2f}')
        plt.ylabel('Frequency')
        plt.title(f"Distribution of Portfolio's {self.rolling}-Day Returns ({title} VaR)")
        plt.axvline(-var_value, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {conf_level:.0%} confidence level')
        plt.legend()
        plt.tight_layout()
        return plt

# Sidebar for User Inputs
with st.sidebar:
    st.title('📉 VaR Calculator')
    st.markdown(
    """
    <div style='display: flex; align-items: left;'>
        Created by  <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' alt='LinkedIn' style='width: 24px; margin-right: 10px;'>
        <a href='https://www.linkedin.com/in/pranav-muktevi' target='_blank'>Pranav Muktevi</a>
    </div>
    """,
    unsafe_allow_html=True
)
    tickers = st.text_input('Enter tickers separated by space', 'AAPL MSFT GOOG').split()
    start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('today'))
    rolling_window = st.slider('Rolling window', min_value=1, max_value=252, value=20)
    confidence_level = st.slider('Confidence level', min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    portfolio_val = st.number_input('Portfolio value', value=100000)

    st.markdown("### Portfolio Weights")
    default_weights = [1 / len(tickers)] * len(tickers)
    weights = [st.number_input(f"Weight for {ticker}", min_value=0.0, max_value=1.0, value=weight) for ticker, weight in zip(tickers, default_weights)]

    st.markdown("### Custom Shocks")
    shocks = [st.number_input(f"Custom Shock (%) for {ticker}", min_value=-100.0, max_value=100.0, value=0.0) / 100 for ticker in tickers]

    calculate_btn = st.button('Calculate VaR')

# Calculation and Display
def calculate_and_display_var(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights, shocks):
    var_instance = VaR(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights)
    
    # Layout for charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.info("Historical VaR Chart")
        historical_chart = var_instance.plot_var_results("Historical", var_instance.historical_var, var_instance.rolling_returns * var_instance.portf_val, confidence_level)
        st.pyplot(historical_chart)

    with chart_col2:
        st.info("Parametric VaR Chart")
        parametric_chart = var_instance.plot_var_results("Parametric", var_instance.parametric_var, var_instance.rolling_returns * var_instance.portf_val, confidence_level)
        st.pyplot(parametric_chart)

    # Stress Testing with Custom Shocks
    shocked_var = var_instance.apply_custom_shocks(np.array(shocks))
    st.markdown("### Custom Shock Results")
    st.write(f"VaR after applying custom shocks: **${shocked_var:,.2f}**")

    # Input Summary
    st.info("Input Summary")
    st.write(f"Tickers: {tickers}")
    st.write(f"Start Date: {start_date}")
    st.write(f"End Date: {end_date}")
    st.write(f"Rolling Window: {rolling_window} days")
    st.write(f"Confidence Level: {confidence_level:.2%}")
    st.write(f"Portfolio Value: ${portfolio_val:,.2f}")
    st.write(f"Portfolio Weights: {weights}")

    # Output Summary
    st.info("VaR Calculation Output")
    data = {
        "Method": ["Historical", "Parametric"],
        "VaR Value": [f"${var_instance.historical_var:,.2f}", f"${var_instance.parametric_var:,.2f}"]
    }
    df = pd.DataFrame(data)
    st.table(df)

# Display Results on Button Click
if calculate_btn:
    calculate_and_display_var(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights, shocks)
