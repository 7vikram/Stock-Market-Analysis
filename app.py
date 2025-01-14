import numpy as np
import pandas as pd
import plotly
import yfinance as yf
import json
import plotly.graph_objects as go
from scipy.optimize import newton
from flask import Flask, request, render_template

# Helper functions
def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1

def calculate_xirr(cash_flows, dates):
    def xirr_irr(rate):
        return sum(cf / (1 + rate) ** ((d - dates[0]).days / 365) for cf, d in zip(cash_flows, dates))

    try:
        return newton(xirr_irr, 0.1)
    except RuntimeError as e:
        print(f"Error calculating XIRR: {e}")
        return np.nan

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    print("Request received!")
    results = None
    graphJSON = None

    if request.method == 'POST':
        etf_ticker = request.form.get('etf_ticker', '^NSEI')
        start_date = request.form.get('start_date', '2022-01-01')
        end_date = request.form.get('end_date', '2025-01-01')
        max_investment = float(request.form.get('max_investment', 200000))
        monthly_investment_r = float(request.form.get('monthly_investment_reactive', 10000))
        default_lookback_period_days = int(request.form.get('lookback_period', 30))
        market_drop_threshold_range = float(request.form.get('market_drop_threshold', -0.05))

        print(f"ETF Ticker: {etf_ticker}, Start Date: {start_date}, Max Investment: {max_investment}, "
              f"Market Drop Threshold: {market_drop_threshold_range}, Lookback Period: {default_lookback_period_days}")

        # Fetch ETF data from Yahoo Finance
        etf_data = yf.download(etf_ticker, start=start_date, end=end_date)
        etf_data['Adj Close'] = etf_data.get('Adj Close', etf_data['Close'])
        etf_data['Returns'] = etf_data['Adj Close'].pct_change()
        etf_data = etf_data.dropna()

        # Prepare result tracking
        results = []

        # Test parameter combinations
        lookback_period_days = default_lookback_period_days
        market_drop_threshold = market_drop_threshold_range

        # Calculate 'Max Drop' for reactive investment
        etf_data['Max Drop'] = etf_data['Adj Close'].rolling(lookback_period_days).apply(
            lambda x: (x.iloc[-1] / x.max()) - 1
        ).dropna()

        # Reactive Investment Strategy
        reactive_dates = []
        reactive_cash_flows = []
        cumulative_reactive_investment = 0
        last_investment_date = None

        for date, row in etf_data.iterrows():
            max_drop = row['Max Drop'] if isinstance(row['Max Drop'], (float, int)) else row['Max Drop'].iloc[0]
    
            if max_drop <= market_drop_threshold:  # Check for market drop
                if last_investment_date is None or (date - last_investment_date).days > 60:  # 60 days = 2 months
                    if cumulative_reactive_investment + monthly_investment_r <= max_investment:  # Stay within investment cap
                        reactive_dates.append(date)
                        reactive_cash_flows.append(-monthly_investment_r)
                        cumulative_reactive_investment += monthly_investment_r
                        last_investment_date = date  # Update last investment date

        # Portfolio value and metrics
        if reactive_dates:
            price_on_reactive_dates = etf_data.loc[reactive_dates, 'Adj Close']
            reactive_units = np.array(reactive_cash_flows) / price_on_reactive_dates.values
            reactive_total_units = reactive_units.sum()
            reactive_final_value = -reactive_total_units * etf_data['Adj Close'].iloc[-1]
            reactive_cagr = calculate_cagr(-sum(reactive_cash_flows), reactive_final_value, 3)
            reactive_xirr = calculate_xirr(reactive_cash_flows + [reactive_final_value], reactive_dates + [etf_data.index[-1]])

            results.append({
                "Monthly Investment Reactive": monthly_investment_r,
                "Lookback Period Days": lookback_period_days,
                "Market Drop Threshold": market_drop_threshold,
                "CAGR": reactive_cagr,
                "XIRR": reactive_xirr,
                "Final Value": reactive_final_value,
                "Total Investment": -sum(reactive_cash_flows)
            })
        
        # Find the best configuration
        results_df = pd.DataFrame(results)
        best_config = results_df.loc[results_df['XIRR'].idxmax()]

        # Create plot
        best_dates = reactive_dates
        best_prices = etf_data.loc[best_dates, 'Adj Close']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=etf_data.index, y=etf_data['Adj Close'], mode='lines', name='ETF Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=best_dates, y=best_prices, mode='markers', name='Reactive Investment', 
                                 marker=dict(symbol='circle', color='red', size=10)))
        fig.update_layout(
            title="Reactive Investment Strategy for Best Configuration",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('index.html', results=best_config.to_dict(), graphJSON=graphJSON)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
