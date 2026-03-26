# 📈 S&P 500 Quantitative Trading Strategy — GJR-GARCH Enhanced

A research-grade **multi-factor long/short trading system** integrating:

- 🧹 Robust Data Engineering  
- 📊 Technical Factor Modeling  
- 📈 GJR-GARCH Volatility Forecasting  
- 📉 Fama-French 5-Factor Exposure  
- ⚖️ Risk-Aware Portfolio Construction  

Built with **Python + Streamlit**, this project simulates an **institutional quant pipeline**.

---

## 🧠 Strategy Philosophy

> Inefficiency → Signal → Alpha → Risk Control → Profit

This system identifies cross-sectional inefficiencies and enhances them using **volatility-aware modeling**.

---

## 🔬 Core Innovations

### 🧩 1. Multi-Factor Alpha Model
Exactly **5 factors tested**:

- `rsi` → Momentum  
- `atr` → Volatility normalization  
- `bb_width` → Volatility expansion  
- `macd_hist` → Trend strength  
- `garch_vol` → Conditional volatility  

---

### 📈 2. GJR-GARCH(1,1) Volatility Modeling

Unlike standard GARCH, **GJR-GARCH captures leverage effects**:

- Negative returns → higher volatility than positive returns  
- Critical for realistic financial modeling  

---

### ⚙️ 3. Three GARCH Use-Cases

#### 📊 (a) Factor Signal
- Use volatility as alpha (Low-Vol Anomaly)

#### ⚖️ (b) Position Sizing
- Inverse volatility weighting  
- Risk parity style allocation  

#### 🔴 (c) Regime Filter
- Detect high-stress regimes  
- Reduce exposure dynamically  

---

## 🏗️ Full Pipeline


Raw Data → Cleaning → Feature Engineering → GARCH Modeling
→ Monthly Aggregation → Factor Ranking
→ Portfolio Construction → Backtesting → Diagnostics


---

## 🔄 Step-by-Step Workflow

### 📥 Data Layer
1. Fetch S&P 500 tickers (Wikipedia)
2. Download OHLCV data (Yahoo Finance)
3. Initial data cleaning & validation

---

### 🧹 Data Engineering
- Remove invalid OHLC values  
- Filter stocks with insufficient coverage  
- Ensure **minimal missing data**  

---

### 📊 Feature Engineering

#### Technical Indicators
- RSI (14)
- ATR (normalized)
- MACD Histogram (z-scored)
- Bollinger Band Width

#### GARCH Features
- Conditional volatility (`garch_vol`)
- Volatility ratio
- Leverage parameter (γ)

---

### 📆 Monthly Aggregation
- Convert daily → monthly
- Select **Top 150 liquid stocks**
- Liquidity = rolling dollar volume

---

### 📉 Returns Calculation
- Multi-horizon returns:
  - 1M, 2M, 3M, 6M, 9M, 12M
- Outlier clipping (robust stats)

---

### 🧠 Factor Exposure (Fama-French)

- Download **Fama-French 5 Factors**
- Compute **rolling betas (Rolling OLS)**
- Use as additional signals

---

### 🧪 Factor Testing

Each factor is tested via:

- Cross-sectional ranking
- Long top N stocks
- Short bottom N stocks
- Monthly rebalancing

---

### 🧠 Multi-Factor Strategy

- Convert factors → z-scores
- Combine into **composite score**
- Rank & construct portfolio

---

### ⚙️ GARCH Enhancements

#### 📊 Position Sizing

weight ∝ 1 / volatility


#### 🔴 Regime Filter
- Compute cross-sectional median volatility
- Identify stress periods
- Scale down exposure

---

## 📊 Backtesting Framework

### Strategies Compared

- Multi-Factor (baseline)
- Best Single Factor
- GARCH Inverse-Vol Strategy
- GARCH + Regime Filter
- S&P 500 Benchmark (SPY)

---

## 📈 Performance Metrics

- Total Return
- Sharpe Ratio
- Volatility
- Max Drawdown
- Win Rate
- Rolling Sharpe

---

## 🔬 GARCH Diagnostics

### Parameters Analyzed:
- α → ARCH effect  
- β → GARCH persistence  
- γ → Leverage effect  

### Persistence Check:

α + β + γ/2


- < 1 → Stationary  
- ≈ 1 → Highly persistent volatility  

---

## 📊 Visualizations

- Cumulative returns comparison  
- Rolling Sharpe ratio  
- Drawdown curves  
- Return distributions  
- Factor performance ranking  
- Correlation heatmaps  
- GARCH parameter distributions  

---

## 🛠️ Tech Stack

### Quant & Data
- Python
- pandas, numpy
- statsmodels
- arch (GARCH)
- pandas_ta

### Data Sources
- yfinance
- pandas_datareader
- Wikipedia scraping

### Visualization & UI
- Streamlit
- matplotlib
- seaborn

---

## ▶️ Running the Project

```bash id="h2v4w7"
pip install -r requirements.txt
streamlit run app.py
