"""
S&P 500 Quantitative Trading Strategy — GJR-GARCH Enhanced
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
import warnings
import io
warnings.filterwarnings('ignore')

from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import statsmodels.api as sm
from arch import arch_model

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S&P 500 GJR-GARCH Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #555; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1a73e8;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #888; }
    .metric-card p  { margin: 0; font-size: 1.5rem; font-weight: 700; color: #1a1a1a; }
    .section-title  { font-size: 1.25rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.5rem; color: #1a73e8; }
    .stProgress > div > div { background-color: #1a73e8; }
    .warning-box { background:#fff3cd; border-left:4px solid #ffc107; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem; }
    .success-box { background:#d4edda; border-left:4px solid #28a745; padding:0.8rem 1rem; border-radius:6px; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 S&P 500 GJR-GARCH Quantitative Strategy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">GJR-GARCH Enhanced | Fama-French 5-Factor | Multi-Signal Long/Short Backtest</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/S%26P_500_logo.svg/240px-S%26P_500_logo.svg.png", width=80)
    st.title("⚙️ Strategy Parameters")

    st.markdown("### 📅 Data Settings")
    end_date   = st.date_input("End Date", value=pd.to_datetime("2023-09-27"))
    lookback   = st.slider("Lookback (years)", 3, 10, 8)
    start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * lookback)

    st.markdown("### 🏗️ Universe")
    top_n_liquid = st.slider("Top N Liquid Stocks / Month", 50, 300, 150, step=25)
    min_coverage = st.slider("Min Data Coverage (%)", 50, 95, 80, step=5)

    st.markdown("### 📊 Portfolio Construction")
    n_long  = st.slider("Long Positions",  5, 50, 20)
    n_short = st.slider("Short Positions", 5, 50, 20)

    st.markdown("### 🔬 GARCH Settings")
    garch_min_obs   = st.slider("Min Obs for GARCH Fit", 100, 500, 252, step=50)
    stress_quantile = st.slider("Regime Filter Quantile (%)", 60, 95, 75, step=5)
    stress_scale    = st.slider("Stress Scale-down", 0.1, 0.9, 0.5, step=0.1)

    st.markdown("### 🎛️ Factors to Test")
    use_rsi      = st.checkbox("RSI",        value=True)
    use_atr      = st.checkbox("ATR",        value=True)
    use_bb_width = st.checkbox("BB Width",   value=True)
    use_macd     = st.checkbox("MACD Hist",  value=True)
    use_garch    = st.checkbox("GARCH Vol",  value=True)

    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

selected_factors = []
if use_rsi:      selected_factors.append("rsi")
if use_atr:      selected_factors.append("atr")
if use_bb_width: selected_factors.append("bb_width")
if use_macd:     selected_factors.append("macd_hist")
if use_garch:    selected_factors.append("garch_vol")

FF_FACTORS      = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
FF_BETA_COLS    = [f'beta_{f}' for f in FF_FACTORS]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def drawdown(cum):
    return (cum - cum.expanding().max()) / cum.expanding().max()

def rolling_sharpe(rets, w=12):
    return rets.rolling(w).mean() / rets.rolling(w).std() * np.sqrt(12)

def perf_stats(rets, name):
    cum = (1 + rets).cumprod()
    return {
        "Strategy":     name,
        "Total Return": f"{float((1 + rets).prod() - 1):.1%}",
        "Ann. Sharpe":  f"{float(rets.mean() / rets.std() * np.sqrt(12)):.2f}",
        "Ann. Vol":     f"{float(rets.std() * np.sqrt(12)):.1%}",
        "Max DD":       f"{float(drawdown(cum).min()):.1%}",
        "Win Rate":     f"{float((rets > 0).mean()):.1%}",
    }

@st.cache_data(show_spinner=False)
def get_sp500_tickers():
    url     = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    html    = requests.get(url, headers=headers).text
    sp500   = pd.read_html(html)[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)
    return sp500['Symbol'].unique().tolist()

@st.cache_data(show_spinner=False)
def download_price_data(symbols, start, end):
    df = yf.download(tickers=symbols, start=start, end=str(end), progress=False)
    df = df.stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    return df

@st.cache_data(show_spinner=False)
def clean_data(df, min_cov_pct):
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[(df['high'] >= df['low']) & (df['high'] >= df['close']) & (df['low'] <= df['close'])]
    total_dates  = df.index.get_level_values('date').nunique()
    min_required = int(total_dates * min_cov_pct / 100)
    counts       = df.groupby(level='ticker').size()
    valid        = counts[counts >= min_required].index
    df           = df[df.index.get_level_values('ticker').isin(valid)]
    return df.sort_index()

def add_features(stock):
    if len(stock) < 60:
        return stock
    stock = stock.copy()
    try:
        stock["rsi"] = ta.rsi(stock["close"], length=14)
        stock["dollar_vol"] = stock["close"] * stock["volume"]
        atr_raw = ta.atr(stock["high"], stock["low"], stock["close"], length=14)
        if atr_raw is not None:
            stock["atr"] = (atr_raw - atr_raw.mean()) / (atr_raw.std() + 1e-9)
        macd = ta.macd(stock["close"])
        if macd is not None and macd.shape[1] >= 3:
            m = macd.iloc[:, 2]
            stock["macd_hist"] = (m - m.mean()) / (m.std() + 1e-9)
        bb = ta.bbands(stock["close"], length=20)
        if bb is not None and bb.shape[1] >= 3:
            mid = bb.iloc[:, 1]
            stock["bb_width"] = np.where(mid != 0, (bb.iloc[:, 2] - bb.iloc[:, 0]) / mid, np.nan)
    except Exception:
        pass
    return stock

def add_garch(stock, min_obs):
    stock = stock.copy()
    if len(stock) < min_obs:
        stock['garch_vol'] = np.nan
        stock['garch_vol_ratio'] = np.nan
        stock['garch_leverage'] = np.nan
        return stock
    log_ret = np.log(stock['close'] / stock['close'].shift(1))
    r = log_ret.dropna() * 100
    try:
        am  = arch_model(r, vol='Garch', p=1, o=1, q=1, dist='Normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        stock['garch_vol'] = (res.conditional_volatility / 100).reindex(stock.index)
        stock['garch_leverage'] = float(res.params.get('gamma[1]', 0.0))
    except Exception:
        stock['garch_vol'] = log_ret.rolling(21).std()
        stock['garch_leverage'] = np.nan
    rolling_vol = log_ret.rolling(21).std()
    stock['garch_vol_ratio'] = stock['garch_vol'] / rolling_vol.replace(0, np.nan)
    return stock

def to_monthly(features_df, top_n):
    skip = {'dollar_vol', 'volume', 'open', 'high', 'low', 'close'}
    last_cols = [c for c in features_df.columns if c not in skip]

    dv = (features_df['dollar_vol'].unstack('ticker').resample('M').mean()
          .stack('ticker').rename('dollar_vol'))
    feat = (features_df[last_cols].unstack('ticker').resample('M').last()
            .stack('ticker'))
    cl = (features_df['close'].unstack('ticker').resample('M').last()
          .stack('ticker').rename('close'))

    mdf = pd.concat([dv, feat, cl], axis=1)
    mdf.index.names = ['date', 'ticker']
    mdf = mdf.sort_index()

    mdf['dollar_vol'] = (mdf['dollar_vol'].unstack('ticker')
                         .rolling(5 * 12, min_periods=12).mean().stack())
    mdf['dollar_vol_rank'] = mdf.groupby(level='date')['dollar_vol'].rank(ascending=False)
    mdf = mdf[mdf['dollar_vol_rank'] < top_n].drop(['dollar_vol', 'dollar_vol_rank'], axis=1)
    return mdf

def calc_returns(df):
    df = df.copy()
    for lag in [1, 2, 3, 6, 9, 12]:
        raw   = df['close'].pct_change(lag)
        valid = raw.dropna()
        if len(valid) >= 20:
            raw = raw.clip(lower=valid.quantile(0.005), upper=valid.quantile(0.995))
        df[f'return_{lag}m'] = raw.add(1).pow(1 / lag).sub(1)
    return df

def get_ff_betas(monthly_df):
    ff = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0]
    ff.index = ff.index.to_timestamp()
    ff = ff.drop('RF', axis=1).resample('M').last().div(100)
    ff.index.name = 'date'

    joined = ff.join(monthly_df['return_1m']).sort_index().dropna()
    obs    = joined.groupby(level=1).size()
    valid  = obs[obs >= 10].index
    joined = joined[joined.index.get_level_values('ticker').isin(valid)]

    betas = (joined.groupby(level=1, group_keys=False)
             .apply(lambda x: RollingOLS(
                 endog=x['return_1m'],
                 exog=sm.add_constant(x.drop('return_1m', axis=1)),
                 window=min(24, x.shape[0]),
                 min_nobs=len(x.columns) + 1
             ).fit(params_only=True).params.drop('const', axis=1)))
    betas.columns = FF_BETA_COLS
    return betas

def create_rankings(df, factor_col, n_long, n_short):
    df = df.copy()
    df['month'] = df.index.get_level_values('date').to_period('M')

    def rank_month(group):
        last_day  = group.index.get_level_values('date').max()
        mdata     = group.xs(last_day, level='date').dropna(subset=[factor_col])
        if len(mdata) < n_long + n_short:
            return pd.DataFrame()
        mdata['rank']     = mdata[factor_col].rank(ascending=False)
        mdata['position'] = 0
        mdata.loc[mdata['rank'] <= n_long, 'position'] = 1
        mdata.loc[mdata['rank'] > len(mdata) - n_short, 'position'] = -1
        return mdata[['position', 'rank']]

    return df.groupby('month', group_keys=False).apply(rank_month)

def backtest(df, rankings, return_col='return_fwd_1m'):
    dc = df[~df.index.duplicated(keep='last')]
    rc = rankings[~rankings.index.duplicated(keep='last')]
    bt = dc.join(rc[['position']], how='inner')
    bt['position_return'] = bt['position'] * bt[return_col]
    return bt[bt['position'] != 0].groupby(level='date')['position_return'].mean()

def backtest_garch_sized(df, rankings, high_stress_days, stress_sc, return_col='return_fwd_1m'):
    dc = df[~df.index.duplicated(keep='last')].copy()
    rc = rankings[~rankings.index.duplicated(keep='last')]
    bt = dc.join(rc[['position']], how='inner')
    bt = bt[bt['position'] != 0].copy()
    vol_col = 'garch_vol' if 'garch_vol' in bt.columns else 'atr'
    bt['inv_vol'] = 1.0 / bt[vol_col].replace(0, np.nan)
    bt['weight']  = (bt.groupby([bt.index.get_level_values('date'), 'position'])['inv_vol']
                     .transform(lambda x: x / x.sum()))
    bt['signed_weight'] = bt['position'] * bt['weight']
    if high_stress_days is not None:
        mask = bt.index.get_level_values('date').isin(high_stress_days)
        bt.loc[mask, 'signed_weight'] *= stress_sc
    bt['weighted_return'] = bt['signed_weight'] * bt[return_col]
    return bt.groupby(level='date')['weighted_return'].sum()

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB CONTENT (always visible)
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_pipeline, tab_results, tab_garch_diag = st.tabs(
    ["📋 Overview", "⚙️ Pipeline Progress", "📊 Results & Charts", "🔬 GARCH Diagnostics"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🧩 Strategy Architecture
        This app implements a **GJR-GARCH enhanced** quantitative long-short strategy on the S&P 500.

        **Three GARCH use-cases:**
        1. 📊 **Factor** — GARCH vol as a ranking signal (Low-Vol Anomaly)
        2. ⚖️ **Position Sizing** — Inverse-vol weighting (risk parity)
        3. 🔴 **Regime Filter** — Halve exposure in high-stress regimes
        """)
    with col2:
        st.markdown("""
        ### 📐 Five Core Factors
        | Factor | Description |
        |--------|-------------|
        | `rsi` | RSI(14) momentum |
        | `atr` | Normalised ATR |
        | `bb_width` | Bollinger Band width |
        | `macd_hist` | MACD Histogram |
        | `garch_vol` | GJR-GARCH conditional vol |
        """)
    with col3:
        st.markdown("""
        ### 🗺️ Pipeline Steps
        1. Download & clean S&P 500 data
        2. Calculate technical features
        3. Fit GJR-GARCH per ticker
        4. Aggregate to monthly + filter liquid stocks
        5. Fama-French rolling betas
        6. Factor testing
        7. GARCH-enhanced backtest
        8. Diagnostics & parameter analysis
        """)

    st.info("👈 Configure parameters in the sidebar, then click **🚀 Run Full Analysis** to start.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE — runs when button clicked
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not selected_factors:
        st.error("Please select at least one factor to test.")
        st.stop()

    # ── shared state placeholders ────────────────────────────────────────────
    results_store = {}

    with tab_pipeline:
        st.markdown("### ⚙️ Pipeline Execution Log")
        log_container = st.container()
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        TOTAL_STEPS   = 10

        def update(step, msg):
            progress_bar.progress(step / TOTAL_STEPS)
            status_text.info(f"**Step {step}/{TOTAL_STEPS}:** {msg}")
            log_container.markdown(f"✅ **[{step}/{TOTAL_STEPS}]** {msg}")

        # ── STEP 1: Tickers ───────────────────────────────────────────────
        update(1, "Fetching S&P 500 tickers from Wikipedia…")
        try:
            symbols = get_sp500_tickers()
            log_container.markdown(f"   → Found **{len(symbols)}** tickers")
        except Exception as e:
            st.error(f"Failed to fetch tickers: {e}")
            st.stop()

        # ── STEP 2: Download data ─────────────────────────────────────────
        update(2, f"Downloading price data ({start_date.date()} → {end_date})…")
        with st.spinner("Downloading (may take 2-3 min for full universe)…"):
            try:
                raw_df = download_price_data(symbols, start_date, str(end_date))
                log_container.markdown(f"   → Downloaded **{len(raw_df):,}** rows")
            except Exception as e:
                st.error(f"Download error: {e}")
                st.stop()

        # ── STEP 3: Clean data ────────────────────────────────────────────
        update(3, "Cleaning data (OHLC validation, coverage filter)…")
        df = clean_data(raw_df, min_coverage)
        log_container.markdown(f"   → After cleaning: **{len(df):,}** rows, "
                               f"**{df.index.get_level_values('ticker').nunique()}** tickers")

        # ── STEP 4: Technical features ────────────────────────────────────
        update(4, "Calculating technical features (RSI, ATR, MACD, BB)…")
        with st.spinner("Calculating features…"):
            features_df = df.groupby(level="ticker", group_keys=False).apply(add_features)
        log_container.markdown(f"   → Features shape: **{features_df.shape}**")

        # ── STEP 5: GJR-GARCH ────────────────────────────────────────────
        update(5, "Fitting GJR-GARCH(1,1) per ticker (this takes several minutes)…")
        with st.spinner("Fitting GARCH models…"):
            features_df = (features_df.groupby(level='ticker', group_keys=False)
                           .apply(lambda s: add_garch(s, garch_min_obs)))
        garch_missing = features_df['garch_vol'].isnull().mean() * 100
        log_container.markdown(f"   → GARCH vol missing: **{garch_missing:.1f}%**")

        # ── STEP 6: Monthly aggregation ───────────────────────────────────
        update(6, f"Aggregating to monthly, keeping top {top_n_liquid} liquid stocks…")
        monthly_df = to_monthly(features_df, top_n_liquid)
        monthly_df = monthly_df.groupby(level='ticker', group_keys=False).apply(calc_returns)
        monthly_df = monthly_df.dropna(subset=['return_1m'])
        log_container.markdown(f"   → Monthly df shape: **{monthly_df.shape}**")

        # ── STEP 7: Fama-French betas ─────────────────────────────────────
        update(7, "Downloading Fama-French 5-Factor data & calculating rolling betas…")
        with st.spinner("Rolling OLS betas (24-month window)…"):
            try:
                betas = get_ff_betas(monthly_df)
                monthly_df = monthly_df.join(betas.groupby('ticker').shift())
                monthly_df[FF_BETA_COLS] = (monthly_df.groupby('ticker', group_keys=False)[FF_BETA_COLS]
                                            .apply(lambda x: x.fillna(x.mean())))
                monthly_df = monthly_df.dropna(subset=FF_BETA_COLS)
                log_container.markdown(f"   → After FF join: **{monthly_df.shape}**")
            except Exception as e:
                log_container.markdown(f"   ⚠️ FF betas skipped: {e}")

        # ── STEP 7b: Advanced cleaning + forward returns ──────────────────
        update(8, "Advanced cleaning, forward returns, composite score…")

        # forward returns
        monthly_df['return_fwd_1m'] = monthly_df.groupby(level='ticker')['return_1m'].shift(-1)
        model_df = monthly_df.dropna(subset=['return_fwd_1m'])

        # ── STEP 8: Factor testing ────────────────────────────────────────
        update(9, f"Testing {len(selected_factors)} factors…")
        factor_results_list = []
        for col in selected_factors:
            if col not in model_df.columns:
                continue
            df_test = model_df.dropna(subset=[col, 'return_fwd_1m'])
            try:
                ranks   = create_rankings(df_test, col, n_long, n_short)
                if len(ranks) == 0:
                    continue
                rets    = backtest(df_test, ranks)
                if len(rets) < 12:
                    continue
                factor_results_list.append({
                    'factor':       col,
                    'total_return': float((1 + rets).prod() - 1),
                    'sharpe':       float((rets.mean() / rets.std()) * np.sqrt(12)),
                    'win_rate':     float((rets > 0).mean()),
                    'mean_monthly': float(rets.mean()),
                    'std_monthly':  float(rets.std()),
                })
            except Exception:
                pass

        factor_results = (pd.DataFrame(factor_results_list)
                          .sort_values('sharpe', ascending=False)
                          .reset_index(drop=True))
        results_store['factor_results'] = factor_results
        log_container.markdown(f"   → Tested **{len(factor_results)}** factors successfully")

        # ── STEP 9: GARCH backtest ────────────────────────────────────────
        update(10, "Running GARCH-enhanced backtest vs base strategy…")

        # Composite z-score
        top_factors = factor_results['factor'].tolist()
        for f in top_factors:
            if f in model_df.columns:
                model_df[f'{f}_zscore'] = (model_df.groupby(level='date')[f]
                                           .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)))
        zcols = [f'{f}_zscore' for f in top_factors if f'{f}_zscore' in model_df.columns]
        model_df['composite_score'] = model_df[zcols].mean(axis=1)

        comp_ranks   = create_rankings(model_df, 'composite_score', n_long, n_short)
        comp_returns = backtest(model_df, comp_ranks)
        comp_cum     = (1 + comp_returns).cumprod()

        best_factor   = factor_results.iloc[0]['factor'] if len(factor_results) else 'rsi'
        best_ranks    = create_rankings(model_df.dropna(subset=[best_factor]), best_factor, n_long, n_short)
        best_returns  = backtest(model_df, best_ranks)
        best_cum      = (1 + best_returns).cumprod()

        # Base equal-weight (first factor)
        base_returns  = comp_returns.copy()
        base_cum      = (1 + base_returns).cumprod()

        # GARCH stress
        daily_stress     = model_df.groupby(level='date')['garch_vol_ratio'].median()
        stress_thresh    = daily_stress.quantile(stress_quantile / 100)
        high_stress_days = daily_stress[daily_stress > stress_thresh].index

        garch_sized_returns = backtest_garch_sized(model_df, comp_ranks, None, stress_scale)
        garch_full_returns  = backtest_garch_sized(model_df, comp_ranks, high_stress_days, stress_scale)
        garch_sized_cum     = (1 + garch_sized_returns).cumprod()
        garch_full_cum      = (1 + garch_full_returns).cumprod()

        # SPY benchmark
        spy = yf.download('SPY', start=comp_returns.index.min(),
                          end=comp_returns.index.max(), progress=False)
        spy_returns    = spy['Close'].resample('M').last().pct_change().dropna()
        spy_cumulative = (1 + spy_returns).cumprod()

        results_store.update({
            'model_df':            model_df,
            'features_df':         features_df,
            'monthly_df':          monthly_df,
            'comp_returns':        comp_returns,
            'comp_cum':            comp_cum,
            'best_returns':        best_returns,
            'best_cum':            best_cum,
            'best_factor':         best_factor,
            'base_returns':        base_returns,
            'base_cum':            base_cum,
            'garch_sized_returns': garch_sized_returns,
            'garch_sized_cum':     garch_sized_cum,
            'garch_full_returns':  garch_full_returns,
            'garch_full_cum':      garch_full_cum,
            'spy_returns':         spy_returns,
            'spy_cumulative':      spy_cumulative,
            'daily_stress':        daily_stress,
            'stress_thresh':       stress_thresh,
            'high_stress_days':    high_stress_days,
        })

        status_text.success("✅ Pipeline complete!")
        st.session_state['results'] = results_store

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS TAB
    # ─────────────────────────────────────────────────────────────────────
    with tab_results:
        rs  = results_store
        fr  = rs['factor_results']

        # KPIs
        st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)
        kpi_cols = st.columns(4)

        if len(rs['garch_full_returns']) > 0:
            gr = rs['garch_full_returns']
            gc = rs['garch_full_cum']
            kpi_cols[0].metric("GARCH Strategy Total Return",
                               f"{float((1+gr).prod()-1):.1%}")
            kpi_cols[1].metric("GARCH Sharpe Ratio",
                               f"{float(gr.mean()/gr.std()*np.sqrt(12)):.2f}")
            kpi_cols[2].metric("GARCH Max Drawdown",
                               f"{float(drawdown(gc).min()):.1%}")
            kpi_cols[3].metric("Win Rate",
                               f"{float((gr>0).mean()):.1%}")

        # ── Factor ranking ────────────────────────────────────────────────
        st.markdown('<div class="section-title">🏆 Factor Performance Ranking</div>', unsafe_allow_html=True)
        if not fr.empty:
            display_fr = fr.copy()
            for c in ['total_return', 'win_rate', 'mean_monthly', 'std_monthly']:
                display_fr[c] = display_fr[c].map(lambda x: f"{x:.2%}")
            display_fr['sharpe'] = display_fr['sharpe'].map(lambda x: f"{x:.2f}")
            st.dataframe(display_fr, use_container_width=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fr.plot(x='factor', y='sharpe', kind='barh', ax=axes[0],
                    color='#1a73e8', legend=False)
            axes[0].set_title('Sharpe Ratio by Factor', fontweight='bold')
            axes[0].axvline(0, color='black', lw=0.8)

            fr.plot(x='factor', y='total_return', kind='barh', ax=axes[1],
                    color='#e8710a', legend=False)
            axes[1].set_title('Total Return by Factor', fontweight='bold')
            axes[1].axvline(0, color='black', lw=0.8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Cumulative returns ────────────────────────────────────────────
        st.markdown('<div class="section-title">📈 Cumulative Returns Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(rs['comp_cum'].index,        rs['comp_cum'].values,        label='Multi-Factor (composite)', lw=2,   color='#1a73e8')
        ax.plot(rs['garch_sized_cum'].index, rs['garch_sized_cum'].values, label='GARCH Inv-Vol Sizing',     lw=2,   color='#0d47a1', ls='--')
        ax.plot(rs['garch_full_cum'].index,  rs['garch_full_cum'].values,  label='GARCH + Regime Filter',    lw=2.5, color='#28a745')
        ax.plot(rs['best_cum'].index,        rs['best_cum'].values,        label=f'Best Single ({rs["best_factor"]})', lw=1.8, color='#a23b72', alpha=0.7)
        ax.plot(rs['spy_cumulative'].index,  rs['spy_cumulative'].values,  label='S&P 500',                  lw=1.8, color='gray', alpha=0.5)
        ax.set_title('Cumulative Returns: GARCH Strategies vs Benchmark', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Return (Start = $1.00)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── 2x2 Performance Grid ──────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Detailed Performance Breakdown</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Rolling Sharpe
        axes[0,0].plot(rolling_sharpe(rs['base_returns']).index,
                       rolling_sharpe(rs['base_returns']).values,
                       label='Base', color='gray', lw=1.5, alpha=0.8)
        axes[0,0].plot(rolling_sharpe(rs['garch_full_returns']).index,
                       rolling_sharpe(rs['garch_full_returns']).values,
                       label='GARCH Full', color='#28a745', lw=2)
        axes[0,0].axhline(0, color='black', ls='--', lw=0.8)
        axes[0,0].set_title('Rolling 12-Month Sharpe', fontweight='bold')
        axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)

        # Drawdown
        dd_b = drawdown(rs['base_cum'])
        dd_g = drawdown(rs['garch_full_cum'])
        axes[0,1].fill_between(dd_b.index, 0, dd_b.values, color='gray', alpha=0.3, label='Base')
        axes[0,1].fill_between(dd_g.index, 0, dd_g.values, color='#28a745', alpha=0.3, label='GARCH Full')
        axes[0,1].plot(dd_b.index, dd_b.values, color='gray', lw=1.2)
        axes[0,1].plot(dd_g.index, dd_g.values, color='#28a745', lw=1.5)
        axes[0,1].set_title('Drawdown Comparison', fontweight='bold')
        axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3)

        # Return distribution
        axes[1,0].hist(rs['comp_returns'].values,       bins=30, alpha=0.6, color='#1a73e8', label='Multi-Factor')
        axes[1,0].hist(rs['garch_full_returns'].values, bins=30, alpha=0.6, color='#28a745', label='GARCH Full')
        axes[1,0].hist(rs['spy_returns'].values,        bins=30, alpha=0.6, color='gray',    label='S&P 500')
        axes[1,0].set_title('Monthly Returns Distribution', fontweight='bold')
        axes[1,0].legend(fontsize=9); axes[1,0].grid(True, alpha=0.3)

        # Scatter base vs garch
        common = rs['base_returns'].index.intersection(rs['garch_full_returns'].index)
        axes[1,1].scatter(rs['base_returns'][common], rs['garch_full_returns'][common],
                          alpha=0.4, s=18, color='#1a73e8')
        lim = max(rs['base_returns'][common].abs().max(),
                  rs['garch_full_returns'][common].abs().max()) * 1.1
        axes[1,1].plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='y = x')
        axes[1,1].axhline(0, color='gray', lw=0.5); axes[1,1].axvline(0, color='gray', lw=0.5)
        axes[1,1].set_title('Monthly Returns: Base vs GARCH Full', fontweight='bold')
        axes[1,1].set_xlabel('Base'); axes[1,1].set_ylabel('GARCH Full')
        axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Strategy comparison table ──────────────────────────────────────
        st.markdown('<div class="section-title">📋 Strategy Comparison Table</div>', unsafe_allow_html=True)
        strat_rows = []
        for nm, rt in [('Multi-Factor',             rs['comp_returns']),
                       ('GARCH Inv-Vol Sizing',      rs['garch_sized_returns']),
                       ('GARCH + Regime Filter',     rs['garch_full_returns']),
                       (f'Best Single ({rs["best_factor"]})', rs['best_returns']),
                       ('S&P 500',                   rs['spy_returns'])]:
            strat_rows.append(perf_stats(rt, nm))
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True)

        # ── Regime filter ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔴 GARCH Regime Filter</div>', unsafe_allow_html=True)
        ds = rs['daily_stress']
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(ds.index, ds.values, color='#2E86AB', lw=1.2, label='Median GARCH Vol-Ratio')
        ax.axhline(rs['stress_thresh'], color='red', ls='--', lw=1.5,
                   label=f'Stress Threshold ({rs["stress_thresh"]:.2f})')
        ax.fill_between(ds.index, ds.values, rs['stress_thresh'],
                        where=ds.values > rs['stress_thresh'],
                        color='red', alpha=0.15, label='High-Stress Regime')
        ax.set_title('Market Stress Score (Cross-sectional Median GARCH Vol-Ratio)', fontweight='bold')
        ax.set_ylabel('Vol Ratio'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        pct_hs = len(rs['high_stress_days']) / len(ds) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Stress Threshold", f"{rs['stress_thresh']:.3f}")
        col2.metric("High-Stress Periods", f"{len(rs['high_stress_days'])}")
        col3.metric("% in High-Stress Regime", f"{pct_hs:.1f}%")

        # ── Data quality visuals ───────────────────────────────────────────
        st.markdown('<div class="section-title">🧹 Data Quality Overview</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        cov = rs['model_df'].groupby(level='date').size()
        cov.plot(ax=axes[0], color='coral')
        axes[0].set_title('Stocks per Month', fontweight='bold')
        axes[0].set_ylabel('Count'); axes[0].grid(True, alpha=0.3)

        fc = [c for c in ['rsi', 'atr', 'bb_width', 'macd_hist', 'garch_vol']
              if c in rs['model_df'].columns]
        corr = rs['model_df'][fc].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('5-Factor Correlation Matrix', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Downloads ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title">💾 Download Results</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3)

        csv_fr = fr.to_csv(index=False).encode()
        dl1.download_button("📥 Factor Results CSV", csv_fr, "factor_results.csv", "text/csv")

        csv_cr = rs['comp_returns'].to_csv().encode()
        dl2.download_button("📥 Strategy Returns CSV", csv_cr, "strategy_returns.csv", "text/csv")

        comp_tbl = pd.DataFrame(strat_rows)
        csv_comp = comp_tbl.to_csv(index=False).encode()
        dl3.download_button("📥 Comparison Table CSV", csv_comp, "strategy_comparison.csv", "text/csv")

    # ─────────────────────────────────────────────────────────────────────
    # GARCH DIAGNOSTICS TAB
    # ─────────────────────────────────────────────────────────────────────
    with tab_garch_diag:
        rs = results_store
        st.markdown('<div class="section-title">🔬 GJR-GARCH(1,1) Parameter Diagnostics</div>', unsafe_allow_html=True)
        st.info("Re-fitting GARCH on a sample of 50 tickers for parameter distribution analysis.")

        n_diag = st.slider("Number of tickers for diagnostics", 10, 100, 50, step=10)
        run_diag = st.button("Run GARCH Diagnostics")

        if run_diag:
            sample_tickers = rs['features_df'].index.get_level_values('ticker').unique()[:n_diag]
            param_records  = []
            prog = st.progress(0)
            for i, ticker in enumerate(sample_tickers):
                prog.progress((i + 1) / len(sample_tickers))
                try:
                    prices  = rs['features_df'].xs(ticker, level='ticker')['close']
                    log_ret = np.log(prices / prices.shift(1)).dropna() * 100
                    if len(log_ret) < 252:
                        continue
                    am  = arch_model(log_ret, vol='Garch', p=1, o=1, q=1, dist='Normal', rescale=False)
                    res = am.fit(disp='off', options={'maxiter': 300})
                    p   = res.params
                    param_records.append({
                        'ticker': ticker,
                        'omega':  float(p.get('omega',    np.nan)),
                        'alpha':  float(p.get('alpha[1]', np.nan)),
                        'gamma':  float(p.get('gamma[1]', np.nan)),
                        'beta':   float(p.get('beta[1]',  np.nan)),
                    })
                except Exception:
                    pass
            prog.empty()

            if param_records:
                param_df = pd.DataFrame(param_records)
                param_df['persistence'] = param_df['alpha'] + param_df['beta'] + param_df['gamma'] * 0.5

                fig, axes = plt.subplots(2, 2, figsize=(13, 8))
                for ax, col, color, title in [
                    (axes[0,0], 'alpha',       '#2E86AB', 'alpha[1] — ARCH Effect'),
                    (axes[0,1], 'gamma',       '#E76F51', 'gamma[1] — Leverage Effect (GJR)'),
                    (axes[1,0], 'beta',        '#57CC99', 'beta[1] — GARCH Persistence'),
                    (axes[1,1], 'persistence', '#9D4EDD', 'Total Persistence (α+β+γ/2)'),
                ]:
                    ax.hist(param_df[col].dropna(), bins=25, color=color, edgecolor='white')
                    ax.set_title(title, fontweight='bold'); ax.grid(True, alpha=0.3)
                    if col == 'persistence':
                        ax.axvline(1.0, color='red', ls='--', lw=1.5, label='Non-stationary')
                        ax.legend(fontsize=8)
                    if col == 'gamma':
                        ax.axvline(0, color='black', ls='--', lw=1)

                plt.suptitle(f'GJR-GARCH(1,1) Parameter Distributions (n={len(param_df)} tickers)',
                             fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("**Summary Statistics:**")
                st.dataframe(param_df[['alpha','gamma','beta','persistence']].describe().round(4),
                             use_container_width=True)

                pct_lev  = (param_df['gamma'] > 0).mean() * 100
                persist  = param_df['persistence'].mean()
                label    = "high — vol shocks are long-lasting" if persist > 0.9 else "moderate"
                col1, col2 = st.columns(2)
                col1.metric("Tickers with Positive Leverage (γ > 0)", f"{pct_lev:.0f}%")
                col2.metric("Mean Persistence (α+β+γ/2)", f"{persist:.3f} ({label})")
            else:
                st.warning("No GARCH parameters could be estimated. Check data coverage.")

elif 'results' in st.session_state:
    # Show cached results without rerunning
    with tab_results:
        st.info("Showing results from previous run. Click **🚀 Run Full Analysis** to refresh.")
