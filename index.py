"""
Advanced Stock Market Analysis — 2024 to Present
=================================================
Multi-stock technical, statistical, and risk analysis.
Stocks: TSLA, RIVN, NIO, LCID, GM, F, SPY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
import warnings
import datetime
import time

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE   = datetime.date.today().strftime("%Y-%m-%d")
FOCUS      = "TSLA"

STOCKS = {
    "TSLA": "Tesla",
    "RIVN": "Rivian",
    "NIO":  "NIO Inc.",
    "LCID": "Lucid",
    "GM":   "Gen. Motors",
    "F":    "Ford",
    "SPY":  "S&P 500",
}

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.figsize": (14, 6), "font.size": 11})

# ── 1. Data Download (batch — one request, avoids rate limits) ────────────────
tickers_list = list(STOCKS.keys())
MAX_RETRIES  = 3
RETRY_DELAY  = 8

print(f"Downloading {len(STOCKS)} stocks in one batch: {START_DATE} → {END_DATE}\n")

raw = pd.DataFrame()
for attempt in range(1, MAX_RETRIES + 1):
    try:
        raw = yf.download(tickers_list, start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=True, group_by="ticker")
        if not raw.empty:
            break
        print(f"  Attempt {attempt}: empty response, retrying in {RETRY_DELAY}s...")
    except Exception as e:
        print(f"  Attempt {attempt} failed: {e}")
    if attempt < MAX_RETRIES:
        time.sleep(RETRY_DELAY * attempt)

if raw.empty:
    raise RuntimeError("All download attempts failed. Wait a minute and retry.")

stock_data: dict[str, pd.DataFrame] = {}
for ticker, name in STOCKS.items():
    try:
        df = raw[ticker].copy().dropna(how="all")
        if not df.empty:
            stock_data[ticker] = df
            print(f"  {ticker:5s}  {name:15s}  {len(df)} trading days")
        else:
            print(f"  {ticker:5s}  no data in response")
    except KeyError:
        print(f"  {ticker:5s}  missing from batch result")

if not stock_data:
    raise RuntimeError("No data loaded — Yahoo Finance may be rate-limiting. Try again in a few minutes.")

close = pd.DataFrame({t: stock_data[t]["Close"].squeeze() for t in stock_data})
print(f"\nLoaded {len(stock_data)} stocks — {close.index[0].date()} to {close.index[-1].date()}")

# ── 2. Technical Indicators ────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]

    # Moving averages
    d["SMA_20"]  = c.rolling(20).mean()
    d["SMA_50"]  = c.rolling(50).mean()
    d["SMA_200"] = c.rolling(200).mean()
    d["EMA_12"]  = c.ewm(span=12, adjust=False).mean()
    d["EMA_26"]  = c.ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_Upper"] = mid + 2 * std
    d["BB_Mid"]   = mid
    d["BB_Lower"] = mid - 2 * std

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)

    # MACD
    d["MACD"]        = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]

    # ATR
    hl  = d["High"] - d["Low"]
    hcp = (d["High"] - c.shift()).abs()
    lcp = (d["Low"]  - c.shift()).abs()
    d["ATR"] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

    # Returns & Volatility
    d["Return"]     = c.pct_change()
    d["Log_Return"] = np.log(c / c.shift())
    d["Cum_Return"] = (1 + d["Return"]).cumprod() - 1
    d["Vol_20"]     = d["Return"].rolling(20).std() * np.sqrt(252)

    return d

tsla = add_indicators(stock_data[FOCUS])
print(f"\nIndicators computed for {FOCUS}")

# ── 3. Risk Metrics ────────────────────────────────────────────────────────────
def risk_metrics(returns: pd.Series, name: str = "") -> dict:
    r = returns.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol else 0
    cum     = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd      = (cum - rolling_max) / rolling_max
    max_dd  = dd.min()
    var_95  = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()
    return {
        "Ticker":          name,
        "Ann. Return":     f"{ann_ret:.1%}",
        "Ann. Volatility": f"{ann_vol:.1%}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{max_dd:.1%}",
        "VaR 95%":         f"{var_95:.2%}",
        "CVaR 95%":        f"{cvar_95:.2%}",
    }

daily_returns = close.pct_change()
metrics_rows  = [risk_metrics(daily_returns[t], t) for t in daily_returns.columns]
metrics_df    = pd.DataFrame(metrics_rows).set_index("Ticker")
print("\nRisk Metrics Summary:")
print(metrics_df.to_string())

# ── 4. Plots ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

# 4a. TSLA price + MAs + Bollinger Bands
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(tsla.index, tsla["Close"],   color="#1f77b4", lw=1.5, label="Close")
ax0.plot(tsla.index, tsla["SMA_20"],  color="orange",  lw=1,   label="SMA 20",  alpha=0.8)
ax0.plot(tsla.index, tsla["SMA_50"],  color="green",   lw=1,   label="SMA 50",  alpha=0.8)
ax0.plot(tsla.index, tsla["SMA_200"], color="red",     lw=1,   label="SMA 200", alpha=0.8)
ax0.fill_between(tsla.index, tsla["BB_Lower"], tsla["BB_Upper"],
                 alpha=0.1, color="purple", label="Bollinger Bands")
ax0.set_title(f"{FOCUS} Price + Moving Averages & Bollinger Bands ({START_DATE} → {END_DATE})", fontsize=13)
ax0.set_ylabel("Price (USD)")
ax0.legend(ncol=5, loc="upper left", fontsize=9)

# 4b. Volume (colored by daily direction)
ax1 = fig.add_subplot(gs[1, 0])
bar_colors = ["#2ca02c" if r >= 0 else "#d62728" for r in tsla["Return"].fillna(0)]
ax1.bar(tsla.index, tsla["Volume"], color=bar_colors, alpha=0.7, width=1)
ax1.set_title(f"{FOCUS} Volume", fontsize=12)
ax1.set_ylabel("Volume")

# 4c. RSI
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(tsla.index, tsla["RSI"], color="#9467bd", lw=1.2)
ax2.axhline(70, color="red",   ls="--", lw=1, alpha=0.7, label="Overbought (70)")
ax2.axhline(30, color="green", ls="--", lw=1, alpha=0.7, label="Oversold (30)")
ax2.fill_between(tsla.index, tsla["RSI"], 70, where=(tsla["RSI"] >= 70), alpha=0.3, color="red")
ax2.fill_between(tsla.index, tsla["RSI"], 30, where=(tsla["RSI"] <= 30), alpha=0.3, color="green")
ax2.set_title(f"{FOCUS} RSI (14)", fontsize=12)
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9)

# 4d. MACD
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(tsla.index, tsla["MACD"],        color="blue",   lw=1.2, label="MACD")
ax3.plot(tsla.index, tsla["MACD_Signal"], color="orange", lw=1.2, label="Signal")
hist_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in tsla["MACD_Hist"].fillna(0)]
ax3.bar(tsla.index, tsla["MACD_Hist"], color=hist_colors, alpha=0.5, width=1)
ax3.axhline(0, color="black", lw=0.8, ls="--")
ax3.set_title(f"{FOCUS} MACD", fontsize=12)
ax3.legend(fontsize=9)

# 4e. Normalized multi-stock performance
ax4 = fig.add_subplot(gs[2, 1])
norm = close.div(close.iloc[0]) * 100
for col in norm.columns:
    ax4.plot(norm.index, norm[col], lw=1.5, label=col)
ax4.axhline(100, color="black", ls="--", lw=0.8, alpha=0.5)
ax4.set_title("Normalized Performance (Base = 100)", fontsize=12)
ax4.set_ylabel("Indexed Price")
ax4.legend(ncol=2, fontsize=9)

# 4f. Correlation heatmap
ax5 = fig.add_subplot(gs[3, 0])
corr = daily_returns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax5, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, mask=mask, linewidths=0.5, annot_kws={"size": 9})
ax5.set_title("Return Correlation Matrix", fontsize=12)

# 4g. Returns distribution
ax6 = fig.add_subplot(gs[3, 1])
tsla_ret = tsla["Return"].dropna()
ax6.hist(tsla_ret, bins=60, edgecolor="white", color="#1f77b4", alpha=0.8, density=True)
mu, sigma = float(tsla_ret.mean()), float(tsla_ret.std())
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
ax6.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
         "r-", lw=2, label="Normal fit")
ax6.axvline(float(np.percentile(tsla_ret, 5)), color="orange", ls="--", lw=1.5, label="VaR 95%")
ax6.set_title(f"{FOCUS} Daily Returns Distribution", fontsize=12)
ax6.set_xlabel("Daily Return")
ax6.legend(fontsize=9)

fig.suptitle("Advanced Stock Market Analysis Dashboard — 2024 to Present",
             fontsize=16, fontweight="bold", y=1.01)
plt.savefig("analysis_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nDashboard saved → analysis_dashboard.png")
