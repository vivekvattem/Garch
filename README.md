# 📈 S&P 500 GJR-GARCH Quantitative Trading Strategy

A full-stack quantitative trading research platform built using **Streamlit**, implementing a **multi-factor long-short strategy enhanced with GJR-GARCH volatility modeling**.

---

## 🚀 Overview

This project builds an institutional-grade quant pipeline that:

- 📊 Uses **technical + statistical factors**
- ⚖️ Applies **risk-aware position sizing**
- 🔴 Detects **market stress regimes**
- 🧠 Integrates **GJR-GARCH volatility modeling**
- 📉 Benchmarks against the **S&P 500 (SPY)**

---

## 🧠 Strategy Architecture

### Core Idea

> Inefficiency → Signal → Alpha → Risk Control → Profit

The system combines:

### 📐 Factors Used
- **RSI** → Momentum signal  
- **ATR** → Volatility normalization  
- **Bollinger Band Width** → Volatility expansion  
- **MACD Histogram** → Trend strength  
- **GARCH Volatility** → Conditional risk  

---

## 🔬 GARCH Enhancements (Key Innovation)

The model uses **GJR-GARCH (1,1)** in 3 ways:

1. 📊 **Factor Signal**
   - Low-volatility anomaly (stocks with lower predicted volatility outperform)

2. ⚖️ **Position Sizing**
   - Inverse volatility weighting (risk parity style)

3. 🔴 **Regime Filter**
   - Reduce exposure during high-volatility stress periods

---

## 🏗️ Pipeline Architecture


### Step-by-step:

1. Fetch S&P 500 tickers (Wikipedia)
2. Download historical OHLCV data (Yahoo Finance)
3. Clean & filter data
4. Compute technical indicators
5. Fit **GJR-GARCH per stock**
6. Convert to monthly data
7. Compute **Fama-French 5-factor betas**
8. Rank stocks by factors
9. Construct long-short portfolio
10. Backtest with:
   - Equal weight
   - GARCH inverse-vol weighting
   - Regime filtering

---

## 📊 Features

### 📌 Quant Features
- Multi-factor alpha generation
- Rolling regression (Fama-French)
- Cross-sectional ranking
- Long-short portfolio construction

### ⚙️ Risk Features
- GARCH volatility forecasting
- Stress regime detection
- Volatility scaling

### 📈 Analytics
- Sharpe Ratio
- Drawdown
- Rolling Sharpe
- Return distributions
- Factor performance ranking

### 📉 Visualization
- Cumulative returns
- Correlation heatmaps
- GARCH diagnostics
- Performance comparison charts

---

## 🛠️ Tech Stack

### 🧑‍💻 Backend / Quant
- `Python`
- `pandas`, `numpy`
- `statsmodels`
- `arch` (GARCH modeling)
- `pandas_ta` (technical indicators)

### 📡 Data
- `yfinance`
- `pandas_datareader`
- Wikipedia scraping

### 🎨 Frontend
- `Streamlit`
- `matplotlib`, `seaborn`

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/garch-quant-strategy.git
cd garch-quant-strategy
pip install -r requirements.txt
