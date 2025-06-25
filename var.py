import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Title
st.title("📉 Value at Risk (VaR) Calculator - Indian Stocks")

# User Inputs
tickers = st.text_input("Enter NSE stock tickers (comma separated, e.g., RELIANCE, INFY, TCS):")
weights_input = st.text_input("Enter corresponding weights in %, comma separated (e.g., 50, 30, 20):")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
rolling_window = st.slider("Rolling Window (in days)", min_value=5, max_value=60, value=20)
confidence_level = st.selectbox("Confidence Level", [90, 95, 99])
portfolio_value = st.number_input("Enter Portfolio Value (₹)", value=100000.0)

# Fetch NSE data using yfinance
def fetch_data(tickers, start, end):
    tickers = [t.strip().upper() + ".NS" for t in tickers.split(',')]
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

# Calculate daily portfolio returns using custom weights
def calculate_portfolio_returns(price_data, weights):
    daily_returns = price_data.pct_change().dropna()
    weights = np.array(weights) / 100  # convert from % to decimal
    if len(weights) != daily_returns.shape[1]:
        st.error("⚠️ Number of weights must match number of tickers.")
        st.stop()
    portfolio_returns = daily_returns.dot(weights)
    return portfolio_returns

# Historical VaR calculation
def historical_var(returns, window, confidence):
    rolling_returns = returns.rolling(window).apply(lambda x: (x + 1).prod() - 1).dropna()
    var_percentile = np.percentile(rolling_returns, 100 - confidence)
    return var_percentile, rolling_returns

# Parametric VaR (normal distribution)
def parametric_var(returns, window, confidence):
    mean_return = returns.mean()
    std_dev = returns.std()
    z_scores = {90: -1.28, 95: -1.65, 99: -2.33}
    z = z_scores[confidence]
    var_return = (mean_return + z * std_dev) * np.sqrt(window)
    return var_return

# Conditional VaR (CVaR) = average of worst losses beyond VaR
def conditional_var(rolling_returns, confidence):
    cutoff = np.percentile(rolling_returns, 100 - confidence)
    losses_beyond_var = rolling_returns[rolling_returns <= cutoff]
    cvar = losses_beyond_var.mean()
    return cvar

# Convert % return to ₹ amount
def calculate_var_amount(var_pct, portfolio_value):
    return round(var_pct * portfolio_value, 2)

# Plot histogram with VaR line
def plot_return_distribution(rolling_returns, var_value, confidence):
    fig, ax = plt.subplots()
    ax.hist(rolling_returns, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=var_value, color='red', linestyle='--', label=f'{confidence}% VaR')
    ax.set_title(f"Return Distribution with {confidence}% Historical VaR")
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# Export rolling returns to CSV
def export_csv(rolling_returns):
    df = pd.DataFrame(rolling_returns)
    df.columns = ['Rolling_Return']
    df.index.name = 'Date'
    return df.to_csv().encode('utf-8')

# Main logic
if tickers and weights_input and start_date and end_date:
    try:
        weights = list(map(float, weights_input.split(',')))
        if sum(weights) != 100:
            st.error("⚠️ Weights must sum to 100.")
            st.stop()
    except:
        st.error("⚠️ Invalid weights format. Enter numbers like: 50, 30, 20")
        st.stop()

    price_data = fetch_data(tickers, start_date, end_date)

    if not price_data.empty:
        returns = calculate_portfolio_returns(price_data, weights)

        # Historical VaR
        hist_var_pct, rolling_returns = historical_var(returns, rolling_window, confidence_level)
        hist_var_amount = calculate_var_amount(hist_var_pct, portfolio_value)

        st.subheader("📊 Historical VaR Result:")
        st.write(f"With {confidence_level}% confidence, your portfolio may lose up to **₹{hist_var_amount}** over the next {rolling_window} days.")

        # Parametric VaR
        para_var_pct = parametric_var(returns, rolling_window, confidence_level)
        para_var_amount = calculate_var_amount(para_var_pct, portfolio_value)

        st.subheader("📕 Parametric VaR Result:")
        st.write(f"Assuming normal distribution, your portfolio may lose up to **₹{para_var_amount}** over the next {rolling_window} days.")

        # Conditional VaR
        cvar_pct = conditional_var(rolling_returns, confidence_level)
        cvar_amount = calculate_var_amount(cvar_pct, portfolio_value)

        st.subheader("🔴 Conditional VaR (CVaR):")
        st.write(f"If losses exceed VaR, the average loss could be **₹{cvar_amount}**.")

        # Histogram plot
        st.subheader("📈 Return Distribution Plot")
        plot_return_distribution(rolling_returns, hist_var_pct, confidence_level)

        # CSV download
        st.subheader("📥 Download Rolling Returns CSV")
        csv = export_csv(rolling_returns)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rolling_returns_{confidence_level}pct_{rolling_window}d.csv",
            mime='text/csv'
        )
    else:
        st.warning("No data fetched. Please check your ticker symbols or date range.")
