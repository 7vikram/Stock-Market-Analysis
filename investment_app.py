import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import newton
from pandas.tseries.offsets import BMonthBegin
import os

# Helper functions
def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0:
        return np.nan  # Return NaN if values are invalid
    return (end_value / start_value) ** (1 / years) - 1

def calculate_xirr(cash_flows, dates):
    def xirr_irr(rate):
        return sum(cf / (1 + rate) ** ((d - dates[0]).days / 365) for cf, d in zip(cash_flows, dates))

    try:
        return newton(xirr_irr, 0.1)
    except RuntimeError:
        return np.nan  # Return NaN if convergence fails

def find_nearest_date(dates, target_date):
    """Find the nearest date in the available dates."""
    nearest_date = min(dates, key=lambda x: abs(x - target_date))
    return nearest_date

# Function to load ETF tickers from file
def load_etf_tickers():
    etf_file = "etf_tickers.txt"  # Adjust the file name and path as needed
    if os.path.exists(etf_file):
        with open(etf_file, "r") as file:
            tickers = file.read().split(",")
        return tickers
    else:
        return ["^NSEI", "AAPL", "MSFT"]  # Default tickers if the file does not exist

# Streamlit UI
st.title('Investment Strategy Analysis')

# Tabs for different functionalities
tabs = st.tabs(["Investment Strategies", "Correlation Analysis"])

# Load ETF tickers from the file or use default
etf_tickers = load_etf_tickers()

# Tab 1: Investment Strategies
with tabs[0]:
    st.sidebar.header('Parameters')

    # User Inputs
    etf_ticker = st.sidebar.selectbox('Select ETF Ticker', etf_tickers)
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime("today") - pd.DateOffset(years=1))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime("today"))
    max_investment = st.sidebar.number_input('Max Investment (₹)', value=200000, step=1000)
    monthly_investment = st.sidebar.number_input('Increase Monthly Investment (Reactive) (₹)', value=0, step=1000)
    market_drop_threshold = st.sidebar.slider('Market Drop Threshold', min_value=1, max_value=20, value=3, step=1)
    market_drop_threshold = -market_drop_threshold / 100
    lookback_period_days = st.sidebar.number_input('Lookback Period (days)', value=60, step=1)

    # Calculate the number of months between start and end dates
    num_months = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 30  # Approximate months
    sip_monthly_investment = max_investment / num_months  # Divide the total investment by the number of months

    # When the user clicks the button, calculate and display the results
    if st.sidebar.button('Calculate Investment Strategies'):
        # Fetch ETF Data
        etf_data = yf.download(etf_ticker, start=start_date, end=end_date)
        if 'Adj Close' not in etf_data.columns:
            etf_data['Adj Close'] = etf_data['Close']

        etf_data['Returns'] = etf_data['Adj Close'].pct_change()
        etf_data['Max Drop'] = etf_data['Adj Close'].rolling(lookback_period_days).apply(
            lambda x: (x.iloc[-1] / x.max()) - 1
        )
        

        # Monthly SIP Strategy
        sip_dates = pd.date_range(start=start_date, end=end_date, freq='MS') + BMonthBegin(-1)
        sip_dates = sip_dates[sip_dates <= pd.to_datetime(end_date)]
        
        # Adjust SIP dates if they are not in the available ETF data
        available_dates = etf_data.index
        adjusted_sip_dates = [find_nearest_date(available_dates, sip_date) for sip_date in sip_dates][1:]
        sip_cash_flows = [-sip_monthly_investment] * len(adjusted_sip_dates)

        # SIP Cash Flow Calculation
        price_on_sip_dates = etf_data.loc[adjusted_sip_dates, 'Adj Close']
        sip_cash_flows = sip_cash_flows[:len(price_on_sip_dates)]
        sip_units = np.array(sip_cash_flows) / price_on_sip_dates.values
        sip_total_units = sip_units.sum()
        sip_final_value = -sip_total_units * etf_data['Adj Close'].iloc[-1]

        # Reactive Strategy Calculation
        reactive_dates = []
        reactive_cash_flows = []
        cumulative_reactive_investment = 0
        last_investment_date = None
        

        for date, row in etf_data.iterrows():
            max_drop = row['Max Drop'] if isinstance(row['Max Drop'], (float, int)) else row['Max Drop'].iloc[0]
            react_sip = sip_monthly_investment + monthly_investment
            if max_drop <= market_drop_threshold:
                if last_investment_date is None or (date - last_investment_date).days > lookback_period_days:
                    #if cumulative_reactive_investment + sip_monthly_investment <= max_investment:
                    reactive_dates.append(date)
                    reactive_cash_flows.append(-react_sip)
                    cumulative_reactive_investment += react_sip
                    last_investment_date = date

        # Reactive Portfolio Calculation
        price_on_reactive_dates = etf_data.loc[reactive_dates, 'Adj Close']
        reactive_units = np.array(reactive_cash_flows) / price_on_reactive_dates.values
        reactive_total_units = reactive_units.sum()
        reactive_final_value = -reactive_total_units * etf_data['Adj Close'].iloc[-1]

        # Calculate Metrics
        sip_cagr = calculate_cagr(-sum(sip_cash_flows), sip_final_value, 3)
        sip_xirr = calculate_xirr(sip_cash_flows + [sip_final_value], list(adjusted_sip_dates) + [etf_data.index[-1]])

        reactive_cagr = calculate_cagr(-sum(reactive_cash_flows), reactive_final_value, 3)
        reactive_xirr = calculate_xirr(reactive_cash_flows + [reactive_final_value], reactive_dates + [etf_data.index[-1]])

        # Display Results
        st.subheader('SIP Strategy:')
        st.write(f"Final Portfolio Value: ₹{sip_final_value:,.2f}")
        st.write(f"Monthly SIP Value: ₹{sip_monthly_investment:,.2f}")
        
        st.write(f"No of times invested(Months): {num_months}")
        st.write(f"CAGR: {sip_cagr:.2%}")
        st.write(f"XIRR: {sip_xirr:.2%}")
        st.write(f"Total Investment: ₹{-sum(sip_cash_flows):,.2f}")

        st.subheader('Reactive Strategy:')
        st.write(f"Final Portfolio Value: ₹{reactive_final_value:,.2f}")
        st.write(f"Monthly SIP Value: ₹{react_sip:,.2f}")
        st.write(f"No of times invested: {-sum(reactive_cash_flows)/react_sip:,.0f}")
        st.write(f"CAGR: {reactive_cagr:.2%}")
        st.write(f"XIRR: {reactive_xirr:.2%}")
        st.write(f"Total Investment: ₹{-sum(reactive_cash_flows):,.2f}")


        # Create Plotly Interactive Plot
        fig = go.Figure()

        # Add Nifty 50 Price Line
        fig.add_trace(go.Scatter(x=etf_data.index, y=etf_data['Adj Close'], mode='lines', name='Nifty 50 Price', line=dict(color='blue')))

        # Add SIP Investment Markers
        sip_prices = etf_data.loc[adjusted_sip_dates, 'Adj Close']
        fig.add_trace(go.Scatter(x=adjusted_sip_dates, y=sip_prices, mode='markers', name='SIP Investment', 
                                 marker=dict(symbol='triangle-up', color='green', size=10)))

        # Add Reactive Investment Markers
        reactive_prices = etf_data.loc[reactive_dates, 'Adj Close']
        fig.add_trace(go.Scatter(x=reactive_dates, y=reactive_prices, mode='markers', name='Reactive Investment', 
                                 marker=dict(symbol='circle', color='red', size=10)))

        # Update layout
        fig.update_layout(
            title=f"{etf_ticker} Price with SIP and Reactive Investment Markers",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode='closest'
        )

        st.plotly_chart(fig)

# Tab 2: Plot Chart of Two Tickers
with tabs[1]:
    st.header("Price Chart of Two Tickers")
    
    # User inputs for tickers and date range
    ticker1 = st.selectbox("Select First Ticker", etf_tickers, index=0)
    ticker2 = st.selectbox("Select Second Ticker", etf_tickers, index=1)
    chart_start_date = st.date_input("Chart Start Date", pd.to_datetime("today") - pd.DateOffset(years=1))
    chart_end_date = st.date_input("Chart End Date", pd.to_datetime("today"))

    if st.button("Plot Chart"):
        # Fetch data for both tickers
        data1 = yf.download(ticker1, start=chart_start_date, end=chart_end_date)
        data2 = yf.download(ticker2, start=chart_start_date, end=chart_end_date)

        # Drop the second level of columns if multi-level columns exist
        if isinstance(data1.columns, pd.MultiIndex):
            data1.columns = data1.columns.droplevel(level=1)
        if isinstance(data2.columns, pd.MultiIndex):
            data2.columns = data2.columns.droplevel(level=1)

        # Use 'Adj Close' if available, otherwise fallback to 'Close'
        if 'Adj Close' in data1.columns:
            data1 = data1['Adj Close']
        elif 'Close' in data1.columns:
            data1 = data1['Close']

        if 'Adj Close' in data2.columns:
            data2 = data2['Adj Close']
        elif 'Close' in data2.columns:
            data2 = data2['Close']

        # Combine data into a single DataFrame for plotting
        combined_data = pd.DataFrame({ticker1: data1, ticker2: data2}).dropna()

        # Plot line chart with two y-axes
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add first ticker to the primary y-axis
        fig.add_trace(
            go.Scatter(x=combined_data.index, y=combined_data[ticker1], mode='lines', name=ticker1),
            secondary_y=False
        )

        # Add second ticker to the secondary y-axis
        fig.add_trace(
            go.Scatter(x=combined_data.index, y=combined_data[ticker2], mode='lines', name=ticker2),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title=f"Price Chart of {ticker1} and {ticker2} with Separate Axes",
            xaxis_title="Date",
            yaxis_title=f"{ticker1} Price",
            yaxis2_title=f"{ticker2} Price",
            template="plotly_dark"
        )

        # Display the chart
        st.plotly_chart(fig)
        
# Add custom CSS for fixed footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100vw;  /* Use viewport width */
            background-color: #333;
            text-align: center;
            padding: 10px 0;
            color: white;
            box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.0);
        }
        .footer a {
            color: white;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Footer content with contact and social media links
st.markdown("""
    <div class="footer">
        <p>Contact us: <a href="mailto:purohitvikram77@gmail.com">purohitvikram77@gmail.com</a> | 
        <a href="https://github.com/7vikram" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/7vikram" target="_blank">LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)