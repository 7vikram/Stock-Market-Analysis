import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import newton
from pandas.tseries.offsets import BMonthBegin

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

# Streamlit UI
st.title('Investment Strategy Analysis')
st.sidebar.header('Parameters')

# User Inputs
etf_ticker = st.sidebar.text_input('ETF Ticker', "^NSEI")
start_date = st.sidebar.date_input('Start Date', pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input('End Date', pd.to_datetime("2025-01-01"))
max_investment = st.sidebar.number_input('Max Investment (₹)', value=200000, step=1000)
monthly_investment = st.sidebar.number_input('Monthly Investment (₹)', value=20000, step=1000)
market_drop_threshold = st.sidebar.slider('Market Drop Threshold', min_value=1, max_value=20, value=3, step=1)
market_drop_threshold = -market_drop_threshold/100
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
    etf_data = etf_data[['Adj Close', 'Returns', 'Max Drop']].dropna()

    # Monthly SIP Strategy
    sip_dates = pd.date_range(start=start_date, end=end_date, freq='MS') + BMonthBegin(5)
    sip_dates = sip_dates[sip_dates <= pd.to_datetime(end_date)]

    # Adjust SIP dates if they are not in the available ETF data
    available_dates = etf_data.index
    adjusted_sip_dates = []

    for sip_date in sip_dates:
        if sip_date in available_dates:
            adjusted_sip_dates.append(sip_date)
        else:
            # If SIP date is not available, shift to the nearest available date
            nearest_date = find_nearest_date(available_dates, sip_date)
            adjusted_sip_dates.append(nearest_date)
    
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

        if max_drop <= market_drop_threshold:
            if last_investment_date is None or (date - last_investment_date).days > lookback_period_days:
                if cumulative_reactive_investment + monthly_investment <= max_investment:
                    reactive_dates.append(date)
                    reactive_cash_flows.append(-monthly_investment)
                    cumulative_reactive_investment += monthly_investment
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
    st.write(f"CAGR: {sip_cagr:.2%}")
    st.write(f"XIRR: {sip_xirr:.2%}")
    st.write(f"Total Investment: ₹{-sum(sip_cash_flows):,.2f}")

    st.subheader('Reactive Strategy:')
    st.write(f"Final Portfolio Value: ₹{reactive_final_value:,.2f}")
    st.write(f"CAGR: {reactive_cagr:.2%}")
    st.write(f"XIRR: {reactive_xirr:.2%}")
    st.write(f"Total Investment: ₹{-sum(reactive_cash_flows):,.2f}")

    # Create Plotly Interactive Plot
    fig = go.Figure()

    # Add Nifty 50 Price Line
    fig.add_trace(go.Scatter(x=etf_data.index, y=etf_data['Adj Close'], mode='lines', name='Nifty 50 Price', line=dict(color='blue')))

    # Add SIP Investment Markers
    sip_dates_str = [date.strftime('%Y-%m-%d') for date in adjusted_sip_dates]
    sip_prices = etf_data.loc[adjusted_sip_dates, 'Adj Close']
    fig.add_trace(go.Scatter(x=adjusted_sip_dates, y=sip_prices, mode='markers', name='SIP Investment', 
                             marker=dict(symbol='triangle-up', color='green', size=10),
                             text=["SIP Investment on " + date for date in sip_dates_str],
                             hoverinfo='text'))

    # Add Reactive Investment Markers
    reactive_dates_str = [date.strftime('%Y-%m-%d') for date in reactive_dates]
    reactive_prices = etf_data.loc[reactive_dates, 'Adj Close']
    fig.add_trace(go.Scatter(x=reactive_dates, y=reactive_prices, mode='markers', name='Reactive Investment', 
                             marker=dict(symbol='circle', color='red', size=10),
                             text=["Reactive Investment on " + date for date in reactive_dates_str],
                             hoverinfo='text'))

    # Update layout
    fig.update_layout(
        title="Nifty 50 Price with SIP and Reactive Investment Markers",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode='closest'
    )

    st.plotly_chart(fig)
