import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("📉 Value at Risk (VaR) Calculator - Indian Stocks")

tickers = st.text_input("Enter NSE stock tickers (comma separated, e.g., RELIANCE, INFY, TCS):")
weights_input = st.text_input("Enter corresponding weights in %, comma separated (e.g., 50, 30, 20):")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
rolling_window = st.slider("Rolling Window (in days)", min_value=5, max_value=60, value=20)
confidence_level = st.selectbox("Confidence Level", [90, 95, 99])
portfolio_value = st.number_input("Enter Portfolio Value (₹)", value=100000.0)

def fetch_data(tickers, start, end):
    tickers = [t.strip().upper() + ".NS" for t in tickers.split(',')]
    data = yf.download(tickers, start=start, end=end)

    if data.empty:
        st.warning("⚠️ No data returned. Check tickers or date range.")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            return data['Adj Close']
        elif 'Close' in data.columns.levels[0]:
            st.warning("⚠️ 'Adj Close' not found. Using 'Close' instead.")
            return data['Close']
        else:
            st.error("❌ Neither 'Adj Close' nor 'Close' found.")
            return pd.DataFrame()
    elif 'Adj Close' in data.columns:
        return data[['Adj Close']]
    elif 'Close' in data.columns:
        st.warning("⚠️ 'Adj Close' not found. Using 'Close' instead.")
        return data[['Close']]
    else:
        st.error("❌ Neither 'Adj Close' nor 'Close' found.")
        return pd.DataFrame()

def calculate_portfolio_returns(price_data, weights):
    daily_returns = price_data.pct_change().dropna()
    weights = np.array(weights) / 100
    if len(weights) != daily_returns.shape[1]:
        st.error("⚠️ Number of weights must match number of tickers.")
        st.stop()
    return daily_returns.dot(weights)

def historical_var(returns, window, confidence):
    rolling_returns = returns.rolling(window).apply(lambda x: (x + 1).prod() - 1).dropna()

    if rolling_returns.empty:
        st.error(f"❌ Not enough data to compute rolling returns for window size = {window}.")
        return 0.0, pd.Series(dtype=float)

    var_percentile = np.percentile(rolling_returns, 100 - confidence)
    return var_percentile, rolling_returns

def parametric_var(returns, window, confidence):
    mean_return = returns.mean()
    std_dev = returns.std()
    z_scores = {90: -1.28, 95: -1.65, 99: -2.33}
    z = z_scores[confidence]
    return (mean_return + z * std_dev) * np.sqrt(window)

def conditional_var(rolling_returns, confidence):
    if rolling_returns.empty:
        return 0.0
    cutoff = np.percentile(rolling_returns, 100 - confidence)
    return rolling_returns[rolling_returns <= cutoff].mean()

def calculate_var_amount(var_pct, portfolio_value):
    return round(var_pct * portfolio_value, 2)

def plot_return_distribution(rolling_returns, var_value, confidence):
    fig, ax = plt.subplots()
    ax.hist(rolling_returns, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=var_value, color='red', linestyle='--', label=f'{confidence}% VaR')
    ax.set_title(f"Return Distribution with {confidence}% Historical VaR")
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def export_csv(rolling_returns):
    df = pd.DataFrame(rolling_returns)
    df.columns = ['Rolling_Return']
    df.index.name = 'Date'
    return df.to_csv().encode('utf-8')

if tickers and weights_input and start_date and end_date:
    try:
        weights = list(map(float, weights_input.split(',')))
        if sum(weights) != 100:
            st.error("⚠️ Weights must sum to 100.")
            st.stop()
    except:
        st.error("⚠️ Invalid weights format. Use comma-separated numbers like: 50, 30, 20")
        st.stop()

    price_data = fetch_data(tickers, start_date, end_date)

    if not price_data.empty:
        st.success("✅ Data fetched successfully:")
        st.dataframe(price_data.tail())
        st.info(f"📆 Available data points: {price_data.shape[0]} days")

        returns = calculate_portfolio_returns(price_data, weights)

        hist_var_pct, rolling_returns = historical_var(returns, rolling_window, confidence_level)
        hist_var_amount = calculate_var_amount(hist_var_pct, portfolio_value)
        st.subheader("📊 Historical VaR Result:")
        st.write(f"With {confidence_level}% confidence, your portfolio may lose up to **₹{hist_var_amount}** in {rolling_window} days.")

        para_var_pct = parametric_var(returns, rolling_window, confidence_level)
        para_var_amount = calculate_var_amount(para_var_pct, portfolio_value)
        st.subheader("📕 Parametric VaR Result:")
        st.write(f"Assuming normal distribution, you may lose up to **₹{para_var_amount}** in {rolling_window} days.")

        cvar_pct = conditional_var(rolling_returns, confidence_level)
        cvar_amount = calculate_var_amount(cvar_pct, portfolio_value)
        st.subheader("🔴 Conditional VaR (CVaR):")
        st.write(f"If losses exceed VaR, average loss could be **₹{cvar_amount}**.")

        st.subheader("📈 Return Distribution Plot")
        plot_return_distribution(rolling_returns, hist_var_pct, confidence_level)

        st.subheader("📥 Download Rolling Returns CSV")
        csv = export_csv(rolling_returns)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rolling_returns_{confidence_level}pct_{rolling_window}d.csv",
            mime='text/csv'
        )
    else:
        st.warning("⚠️ No data fetched. Try changing tickers or dates.")
