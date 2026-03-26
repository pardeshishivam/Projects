"""
Risk Metrics Calculator — Nifty 50
====================================
Three rolling metrics, all expressed as % returns:
  1. Historical VaR  — non-parametric percentile
  2. EWMA VaR        — RiskMetrics (lambda=0.99, half-life ~69 days)
  3. Expected Shortfall (CVaR) — mean of tail losses beyond Historical VaR
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def fetch_nifty50(start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """Download Nifty 50 and return a DataFrame with column 'log_ret'."""
    raw = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No data returned. Check your internet connection or date range.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 1. HISTORICAL VaR
# ─────────────────────────────────────────────

def rolling_hist_var(
    returns: pd.Series,
    lookback: int = 252,
    confidence: float = 0.99,
) -> pd.Series:
    """
    Rolling Historical VaR (% return, negative means loss).
    At each day t, sort the prior `lookback` returns and take the alpha-th percentile.
    """
    alpha = (1 - confidence) * 100
    var = returns.rolling(lookback).quantile(alpha / 100)
    return var.rename("hist_var")


# ─────────────────────────────────────────────
# 2. EWMA VaR
# ─────────────────────────────────────────────

def rolling_ewma_var(
    returns: pd.Series,
    lam: float = 0.99,
    confidence: float = 0.99,
) -> tuple[pd.Series, pd.Series]:
    """
    EWMA (RiskMetrics) VaR — assumes conditional normality.

    Variance update: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
    VaR_t          = z_α · σ_t

    Half-life = ln(2) / ln(1/λ) ≈ 69 days for λ=0.99

    Returns (ewma_var series, ewma_vol series) both as % returns.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - confidence)          # e.g. -2.326 for 99%

    r = returns.values
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[0] = r[0] ** 2

    for i in range(1, n):
        sigma2[i] = lam * sigma2[i - 1] + (1 - lam) * r[i - 1] ** 2

    sigma = np.sqrt(sigma2)
    ewma_var = pd.Series(z * sigma, index=returns.index, name="ewma_var")  # negative
    ewma_vol = pd.Series(sigma,     index=returns.index, name="ewma_vol")
    return ewma_var, ewma_vol


# ─────────────────────────────────────────────
# 3. EXPECTED SHORTFALL
# ─────────────────────────────────────────────

def rolling_es(
    returns: pd.Series,
    lookback: int = 252,
    confidence: float = 0.99,
) -> pd.Series:
    """
    Rolling Expected Shortfall (CVaR) — mean of returns that fall beyond the VaR threshold.
    ES is always <= VaR (a larger loss estimate).
    """
    alpha = (1 - confidence)

    def _es(window):
        cutoff = np.quantile(window, alpha)
        tail = window[window <= cutoff]
        return tail.mean() if len(tail) > 0 else cutoff

    es = returns.rolling(lookback).apply(_es, raw=True)
    return es.rename("es")


# ─────────────────────────────────────────────
# COMBINED PIPELINE
# ─────────────────────────────────────────────

def compute_metrics(
    returns: pd.Series,
    lookback: int = 252,
    confidence: float = 0.99,
    lam: float = 0.99,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      actual_ret, hist_var, ewma_var, ewma_vol, es
    All expressed as decimal log-returns (multiply by 100 for %).
    """
    hist_var        = rolling_hist_var(returns, lookback, confidence)
    ewma_var, ewma_vol = rolling_ewma_var(returns, lam, confidence)
    es              = rolling_es(returns, lookback, confidence)

    df = pd.concat([returns.rename("actual_ret"), hist_var, ewma_var, ewma_vol, es], axis=1)
    df.dropna(inplace=True)
    return df


def exceptions(metrics_df: pd.DataFrame, var_col: str = "hist_var") -> pd.DataFrame:
    """Rows where actual return breached the given VaR column."""
    breached = metrics_df["actual_ret"] < metrics_df[var_col]
    return metrics_df[breached][["actual_ret"]].copy()


def halflife(lam: float) -> float:
    """Half-life in days for a given EWMA lambda."""
    return np.log(2) / np.log(1 / lam)
