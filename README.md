# SIP vs Reactive Investment Strategy Tool 📈

A Python-based tool to compare **Systematic Investment Plan (SIP)** and **Reactive Investment Strategies** for evaluating portfolio performance over a given investment period. This application is perfect for investment enthusiasts, financial analysts, and anyone looking to optimize their investment strategy.

---

## 🚀 **Features**

- **ETF Data Analysis**: Automatically fetches ETF data using the Yahoo Finance API for historical price trends.
- **Dynamic SIP Calculations**: Handles missing SIP dates by adjusting to the nearest trading day.
- **Reactive Investments**: Identifies opportunities during market drops based on customizable thresholds.
- **Performance Metrics**: Calculates **CAGR (Compound Annual Growth Rate)** and **XIRR (Extended Internal Rate of Return)** for both strategies.
- **Interactive Visualization**: Displays investment markers and trends with an intuitive Plotly-based chart.
- **Customizable Parameters**: Users can set investment amounts, start and end dates, thresholds, and ETF tickers through a user-friendly interface.

---

## 🛠️ **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - `numpy`, `pandas`: Data manipulation and calculations
  - `yfinance`: Fetching historical market data
  - `plotly`: Interactive chart visualizations
  - `scipy`: Optimization for XIRR calculations
  - `streamlit`: Building an interactive web-based UI

---

## 🔧 **How It Works**

1. **Input Parameters**:
   - ETF Ticker (e.g., `^NSEI` for Nifty 50)
   - Start and End Dates for the investment period
   - Maximum Investment Amount
   - Monthly SIP Investment
   - Market Drop Threshold for reactive investments

2. **Data Fetching**:
   - Fetches historical ETF prices for the specified date range using the Yahoo Finance API.

3. **Strategy Calculations**:
   - **SIP Strategy**: Invests a fixed amount on the 5th trading day of each month (adjusted for holidays or missing data).
   - **Reactive Strategy**: Invests when a specified market drop threshold is met, limited to one investment per two-month period.

4. **Performance Metrics**:
   - Calculates the **Final Portfolio Value**, **CAGR**, and **XIRR** for both strategies.
   - Visualizes price trends and investment points on an interactive chart.

---

## 🖥️ **Installation & Usage**

### Prerequisites
Ensure you have Python 3.8+ installed, along with the following libraries:
```bash
pip install numpy pandas yfinance plotly scipy streamlit
