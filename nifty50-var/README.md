# Nifty 50 Risk Metrics

A from-scratch implementation of three core market risk measures on Nifty 50 daily returns, with an interactive Streamlit dashboard.

---

## What it computes

| Metric | Method | Detail |
|---|---|---|
| **Historical VaR** | Non-parametric percentile | 99% confidence, rolling lookback window |
| **EWMA VaR** | Exponentially Weighted Moving Average | λ = 0.99 → half-life ≈ 69 days |
| **Expected Shortfall** | Mean of tail losses beyond VaR | Also known as CVaR |

---

## Why these three

Every market risk desk runs VaR. But Historical VaR alone has blind spots — it weights a day 250 days ago the same as yesterday. EWMA fixes that by letting recent volatility carry more weight. Expected Shortfall goes further by asking *how bad* losses are when they breach the threshold, not just whether they do. Together they give a fuller picture of tail risk.

---

## Run it

```bash
pip install streamlit yfinance pandas numpy plotly
streamlit run var_app.py
```

Data is fetched live from Yahoo Finance (`^NSEI`). No static files needed.

---

## Files

| File | Role |
|---|---|
| `var_calculator.py` | Core engine — data fetch, VaR/ES calculations, backtest logic |
| `var_app.py` | Streamlit dashboard — charts, metrics, exception table |
