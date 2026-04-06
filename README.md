# Advanced Stock Market Analysis

A comprehensive, end-to-end stock market analysis pipeline covering **Tesla and the EV sector from January 2024 to the present**. It fetches live data, computes professional-grade technical indicators, measures risk, runs a machine learning price-direction model, and backtests a trading strategy — all in a self-contained Jupyter notebook and Python script.

---

## What This Project Does

Most stock market tutorials show a single price chart and stop there. This project goes further:

1. **Pulls live, adjusted data** for 7 tickers every time you run it — no stale CSVs.
2. **Computes 20+ technical indicators** — the same ones used by professional traders.
3. **Measures real risk** using VaR, CVaR, Sharpe, Calmar, beta, and max drawdown.
4. **Compares stocks side-by-side** on a normalised performance and risk-return basis.
5. **Trains a machine learning model** (Random Forest) with walk-forward cross-validation to predict next-day return direction.
6. **Backtests a strategy** (MA crossover) and compares it against buy-and-hold with a drawdown chart.

---

## Stocks Analysed

| Ticker | Company | Why Included |
|--------|---------|-------------|
| **TSLA** | Tesla Inc. | Primary deep-dive — EV market leader |
| **RIVN** | Rivian Automotive | Direct EV truck/SUV competitor |
| **NIO** | NIO Inc. | Chinese EV market leader |
| **LCID** | Lucid Group | Luxury EV competitor |
| **GM** | General Motors | Legacy OEM transitioning to EV |
| **F** | Ford Motor | Legacy OEM with EV line (F-150 Lightning) |
| **SPY** | S&P 500 ETF | Market benchmark for beta and correlation |

---

## Section-by-Section Breakdown

### 1. Data Acquisition

Data is downloaded live using `yfinance` with `auto_adjust=True`, which folds stock splits and dividends into the price so comparisons are always clean.

**What you see at this stage:**
- A printed table showing each ticker, how many trading days were downloaded, and the latest closing price.
- A YTD return bar for each stock — instantly shows which EV names led and lagged the market since January 2024.

---

### 2. Exploratory Data Analysis

Two chart panels:

**Panel A — Log-scale price chart (all 7 stocks)**
Because TSLA trades near $200 while NIO trades near $5, a linear axis makes the small-cap EVs invisible. The log scale puts every stock on equal visual footing and shows percentage moves rather than dollar moves.

**Panel B — Normalised performance (Base = 100)**
Every stock is rebased to 100 on Jan 1 2024. A stock at 140 is up 40%; one at 60 is down 40%. This is the clearest way to compare returns across stocks at very different price levels.

**TSLA detail chart:**
A two-panel chart showing the close price with the high-low daily range shaded underneath, and a volume bar chart below it. Volume bars are coloured green on up-days and red on down-days so you can instantly see if volume confirms a move.

---

### 3. Technical Indicators

All indicators are computed from scratch using pandas — no external TA library needed.

#### Price + Moving Averages + Bollinger Bands

```
  SMA 20   (orange)  — short-term trend
  SMA 50   (green)   — medium-term trend
  SMA 200  (red)     — long-term trend / bull/bear dividing line
  EMA 12/26          — faster-reacting exponential averages (used in MACD)
  Bollinger Bands    — shaded purple band: ±2 standard deviations around SMA 20
```

The **Bollinger Band Width** subplot below the price chart acts as a volatility gauge — when the bands are narrow the stock is coiling; when they explode wide it's in a trending or volatile phase.

> **How to read it:** When price rides the upper Bollinger Band on rising volume, momentum is strong. A close back inside the bands after tagging the upper band often signals short-term exhaustion.

#### RSI (14)

The Relative Strength Index oscillates between 0 and 100.

- **Above 70** → overbought zone (shaded red)
- **Below 30** → oversold zone (shaded green)
- **50 line** → separates bullish from bearish momentum regimes

The chart fills the overbought and oversold regions so extremes jump out visually.

#### MACD (12, 26, 9)

Three components are plotted together:

- **MACD line** (blue) — difference between EMA 12 and EMA 26
- **Signal line** (orange) — 9-period EMA of MACD
- **Histogram** — coloured bars (green when MACD > signal, red when below); shrinking bars signal a trend is losing steam before a crossover happens

#### ATR, OBV, BB %B, Rolling Volatility

A 2×2 subplot grid:

| Chart | What it shows |
|-------|--------------|
| **ATR (14)** | Average True Range — raw dollar volatility per day. Useful for sizing stop-losses. |
| **OBV** | On-Balance Volume — cumulative volume flow. Rising OBV on a rising price confirms buyers are in control. |
| **Rolling Volatility (20 & 60-day)** | Annualised realised volatility. Shows when the stock is entering calm or stormy periods. |
| **Bollinger %B** | Where price sits within the bands: 1.0 = at upper band, 0.0 = at lower band, 0.5 = at midline. |

---

### 4. Returns Distribution & Rolling Volatility

#### Daily Returns Histogram

A histogram of all daily close-to-close returns overlaid with a fitted normal distribution curve. Two vertical lines mark **VaR 95%** (orange) and **VaR 99%** (red) — the worst return you'd expect to see on 5% and 1% of trading days respectively.

> **Fat tails:** TSLA's return distribution has heavier tails than a normal distribution (positive excess kurtosis). This means large moves happen more often than a normal distribution would predict.

#### Q-Q Plot

Plots empirical return quantiles against theoretical normal quantiles. Points curving away from the diagonal confirm fat tails and negative skew — standard characteristics of equity returns.

#### Cumulative Returns & Rolling Volatility

Two side-by-side charts comparing all 7 stocks:
- **Left:** Cumulative total return since Jan 2024 — shows the full journey, not just start and end.
- **Right:** 30-day rolling annualised volatility — shows which periods were calm and which were turbulent for each name.

---

### 5. Risk Metrics Table

A summary table is printed and displayed for all stocks:

| Metric | Description |
|--------|-------------|
| **Ann. Return** | Annualised average daily return × 252 |
| **Ann. Volatility** | Annualised standard deviation of daily returns |
| **Sharpe Ratio** | Return per unit of risk (higher = better) |
| **Calmar Ratio** | Annualised return ÷ max drawdown (higher = better) |
| **Max Drawdown** | Largest peak-to-trough decline in the period |
| **VaR 95%** | Worst expected daily loss 1 day in 20 |
| **CVaR 95%** | Average loss on the worst 5% of days (tail risk) |
| **VaR 99%** | Worst expected daily loss 1 day in 100 |
| **Beta (vs SPY)** | Sensitivity to S&P 500 moves (>1 = amplified moves) |

#### Drawdown Chart

Two stacked panels:
- **Top:** All cumulative returns together — lets you see which stocks recovered fastest.
- **Bottom:** Drawdown from peak for all stocks — shows how far each one fell before recovering, and which ones are still underwater.

---

### 6. Multi-Stock Comparative Analysis

#### Risk-Return Scatter Plot

Each stock is plotted as a point with **annualised volatility on the X axis** and **annualised return on the Y axis**. Points are coloured by Sharpe ratio (red = poor, green = strong).

> **What to look for:** Stocks in the upper-left are the most efficient (high return, low risk). Stocks in the lower-right are the worst (low return, high risk). The colour immediately shows which names are earning their risk.

#### Correlation Heatmap

A full return correlation matrix. Values close to **+1.0** mean two stocks move together; values near **0** mean they move independently.

> **Key insight:** High correlation between EV names means owning multiple EV stocks provides less diversification than it appears.

#### 60-Day Rolling Correlation to TSLA

A time-series chart showing how each non-TSLA stock's correlation to Tesla has evolved over time. Correlations are not static — they spike during market stress and compress in calm periods.

---

### 7. Machine Learning: Next-Day Return Direction

#### Feature Engineering

18 features are derived from price and indicator history:

| Feature Group | Features |
|--------------|---------|
| Lagged returns | 1-day through 5-day lagged returns |
| Momentum | 5, 10, 20-day price momentum |
| Indicators | RSI, MACD, MACD Signal, BB%B, BB Width |
| MA ratios | Price / SMA20, Price / SMA50 |
| Volatility | 20-day rolling annualised volatility |
| Volume | Volume ratio vs 20-day average |
| ATR | ATR as a fraction of price |

**Target:** Direction of the next day's close-to-close return (+1 up, −1 down).

#### Walk-Forward Cross-Validation

A `TimeSeriesSplit` with 5 folds is used — the model is **never trained on future data**. Each fold trains on the past and tests on the next unseen period, mimicking how a real trading system would be validated.

#### Feature Importance Chart

A horizontal bar chart shows which of the 18 features the Random Forest found most predictive. Typically momentum and RSI features rank highly, while volume features rank lower.

#### Fold Accuracy Chart

A bar chart of directional accuracy across all 5 folds with a 50% random-baseline reference line. Consistent accuracy above 50% confirms the model has learned a real pattern rather than over-fitting.

---

### 8. Backtesting — MA 20/50 Crossover Strategy

**Rules:**
- **Golden Cross** — SMA 20 crosses above SMA 50 → BUY (go long)
- **Death Cross** — SMA 20 crosses below SMA 50 → SELL (go flat)
- Long-only, fully invested when in position, cash otherwise
- No transaction costs or slippage modelled

#### Three-Panel Backtest Chart

```
Panel 1 — Price chart with SMA 20 and SMA 50
          ▲ green triangles = buy signals
          ▼ red triangles   = sell signals
          shaded green regions = periods when strategy is in position

Panel 2 — Cumulative return: Strategy (green) vs Buy & Hold (blue)
          Labels show the final total return for each

Panel 3 — Strategy drawdown
          Red shaded area shows how far the strategy fell from its peak
          at any point in time
```

The comparison makes clear when the trend-following rules help (avoiding prolonged downtrends) and when they hurt (whipsawing in choppy sideways markets).

---

### 9. Summary Dashboard

A single 6-panel figure that combines the most important charts into one image, saved to **`analysis_dashboard.png`**:

```
┌─────────────────────────────────────────────────┐
│   Normalised Performance — all 7 stocks         │  ← full-width
├───────────────────────┬─────────────────────────┤
│  Annualised Return    │  Annualised Volatility   │
├───────────────────────┴─────────────────────────┤
│   TSLA Price + SMA 20/50/200 + Bollinger Bands  │  ← full-width
├───────────────────────┬─────────────────────────┤
│  Correlation Matrix   │  Returns Distribution   │
└───────────────────────┴─────────────────────────┘
```

This single image gives a complete picture of the market environment, relative performance, risk structure, and TSLA's current technical position — everything you need at a glance.

---

## Generated Output

| File | Generated by | Contents |
|------|-------------|---------|
| `analysis_dashboard.png` | `index.py` or notebook Section 9 | 6-panel summary chart |

---

## Quick Start

```bash
# 1. Install dependencies
pip install yfinance pandas numpy matplotlib seaborn scikit-learn scipy

# 2a. Run the standalone script (produces analysis_dashboard.png)
python index.py

# 2b. Or open the full interactive notebook
jupyter notebook stock_market_analysis.ipynb
```

Data is fetched live every run — no setup or CSV maintenance needed.

---

## File Structure

```
stock_market_analysis/
├── index.py                      # Standalone script: full analysis + dashboard PNG
├── stock_market_analysis.ipynb   # Interactive notebook: all 9 sections
├── Tesla_Stock.csv               # Archive: TSLA historical data 2010–2022
├── analysis_dashboard.png        # Generated: summary dashboard (after first run)
└── README.md                     # This file
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `yfinance` | Live market data from Yahoo Finance |
| `pandas` | Data frames and time-series operations |
| `numpy` | Numerical computation |
| `matplotlib` | All charting and figure layout |
| `seaborn` | Statistical plots and heatmaps |
| `scikit-learn` | Random Forest, TimeSeriesSplit, scaling, metrics |
| `scipy` | Statistical distributions (Q-Q plot, normal fit) |
