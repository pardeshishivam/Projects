# streamlit_greek_attribution.py
# Pathwise incremental Greek attribution dashboard (portfolio + DTE)
# If Streamlit is unavailable, this file runs in **CLI self-test mode** so you can
# verify the core math without the UI.
#
# Run UI:      pip install streamlit pandas numpy matplotlib
#              streamlit run streamlit_greek_attribution.py
# Run CLI test: python streamlit_greek_attribution.py

from __future__ import annotations
import math
from math import erf
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Try to import streamlit. If not present, fall back to CLI mode.
try:
    import streamlit as st  # type: ignore
    import matplotlib.pyplot as plt  # Only used in Streamlit UI
    STREAMLIT_AVAILABLE = True
except Exception:  # ModuleNotFoundError or any env issue
    st = None  # type: ignore
    plt = None  # type: ignore
    STREAMLIT_AVAILABLE = False

# ==================== Black–Scholes helpers ====================

def CDF(x: float) -> float:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

def pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = 'put') -> float:
    """European option price (Black–Scholes)."""
    if sigma <= 0 or T <= 0:
        return max((K - S) if option_type == 'put' else (S - K), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * math.exp(-q * T) * CDF(d1) - K * math.exp(-r * T) * CDF(d2)
    else:
        return K * math.exp(-r * T) * CDF(-d2) - S * math.exp(-q * T) * CDF(-d1)


def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = 'put') -> Tuple[float, float, float, float, float, float]:
    """Return (delta, vega, gamma, vanna, vomma, theta). Units: vega per 1.00 vol (abs)."""
    if sigma <= 0 or T <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = pdf(d1)
    delta = (math.exp(-q * T) * CDF(d1)) if option_type == 'call' else (-math.exp(-q * T) * CDF(-d1))
    vega = S * math.exp(-q * T) * pdf_d1 * math.sqrt(T)
    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * math.sqrt(T))
    vomma = vega * d1 * d2 / sigma
    # vanna (per currency*vol): common closed-form
    vanna = vega * (1 - d1 / (sigma * math.sqrt(T))) / S
    theta = - (S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * CDF(-d2) + q * S * math.exp(-q * T) * CDF(-d1) if option_type == 'put' else 0.0
    return delta, vega, gamma, vanna, vomma, theta

# ==================== Portfolio helpers ====================

@dataclass
class Leg:
    K: float
    opt_type: str  # 'put' or 'call'
    qty: float

LegTuple = Tuple[float, str, float]


def price_portfolio(legs: List[LegTuple], S: float, r: float, q: float, sigma: float, T: float) -> float:
    total = 0.0
    for K, opt_type, qty in legs:
        total += qty * bs_price(S, K, r, q, sigma, T, opt_type)
    return total


def greeks_portfolio(legs: List[LegTuple], S: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float, float, float, float]:
    d = v = g = va = vo = 0.0
    for K, opt_type, qty in legs:
        delta, vega, gamma, vanna, vomma, _ = bs_greeks(S, K, r, q, sigma, T, opt_type)
        d += qty * delta
        v += qty * vega
        g += qty * gamma
        va += qty * vanna
        vo += qty * vomma
    return d, v, g, va, vo

# ==================== Pathwise incremental attribution ====================

def pathwise_attribution(
    legs: List[LegTuple],
    S0: float,
    r: float,
    q: float,
    sigma0: float,
    T: float,
    dx_total: float,
    dv_total: float,
    N: int,
    notional: float = 100.0,
) -> Tuple[Dict[str, float], pd.DataFrame, float, float]:
    """Portfolio-level pathwise 2nd-order Taylor accumulation.
    Returns (cumulative dict, step DataFrame, exact PL, final portfolio price per share-equivalent).
    """
    dS = dx_total / max(N, 1)
    dSig = dv_total / max(N, 1)

    S = S0
    sigma = sigma0
    V = price_portfolio(legs, S, r, q, sigma, T)

    cumulative = {k: 0.0 for k in ['delta', 'gamma', 'vega', 'vomma', 'vanna', 'total']}
    rows = []

    for i in range(N):
        delta, vega, gamma, vanna, vomma = greeks_portfolio(legs, S, r, q, sigma, T)
        dV_delta = delta * dS
        dV_gamma = 0.5 * gamma * (dS ** 2)
        dV_vega = vega * dSig
        dV_vomma = 0.5 * vomma * (dSig ** 2)
        dV_vanna = vanna * dS * dSig
        dV_local = dV_delta + dV_gamma + dV_vega + dV_vomma + dV_vanna

        V += dV_local
        S += dS
        sigma += dSig

        cumulative['delta'] += dV_delta * notional
        cumulative['gamma'] += dV_gamma * notional
        cumulative['vega'] += dV_vega * notional
        cumulative['vomma'] += dV_vomma * notional
        cumulative['vanna'] += dV_vanna * notional
        cumulative['total'] += dV_local * notional

        rows.append({
            'step': i + 1,
            'S': S,
            'sigma': sigma,
            'dV_delta': dV_delta * notional,
            'dV_gamma': dV_gamma * notional,
            'dV_vega': dV_vega * notional,
            'dV_vomma': dV_vomma * notional,
            'dV_vanna': dV_vanna * notional,
            'dV_local': dV_local * notional,
            'Delta': delta,
            'Vega': vega,
            'Gamma': gamma,
            'Vanna': vanna,
            'Vomma': vomma,
        })

    V_exact_final = price_portfolio(legs, S0 + dx_total, r, q, sigma0 + dv_total, T)
    exact_pl = (V_exact_final - price_portfolio(legs, S0, r, q, sigma0, T)) * notional

    df = pd.DataFrame(rows)
    return cumulative, df, exact_pl, V_exact_final

# ==================== STREAMLIT UI (only if available) ====================

if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Pathwise Greek Attribution", layout="wide")

    st.title("Pathwise Greek Attribution — incremental ANOVA style")
    st.markdown("Use the left panel to change option and shock parameters, and run the attribution.")

    with st.sidebar.form(key='inputs'):
        st.header('Market/Model Inputs')
        S0 = st.number_input('Spot S0', value=100.0, step=1.0, format='%f')
        r = st.number_input('Risk-free rate (r)', value=0.05, step=0.001, format='%f')
        q = st.number_input('Dividend yield (q)', value=0.0, step=0.001, format='%f')
        sigma0 = st.number_input('Initial implied vol (sigma0, abs)', value=0.20, step=0.01, format='%f')

        st.markdown('---')
        st.header('Expiry (DTE)')
        dte_days = st.number_input('Days to expiry (trading days, 252/yr)', value=63, step=1)
        T = max(int(dte_days), 0) / 252.0

        st.markdown('---')
        st.header('Portfolio Legs')
        if 'legs' not in st.session_state:
            st.session_state.legs = [
                {'type': 'put', 'K': 98.0, 'qty': 1.0, 'delete': False},
            ]

        # Editable leg rows
        for i, leg in enumerate(st.session_state.legs):
            c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 0.8])
            with c1:
                leg['type'] = st.selectbox(f'Option type #{i+1}', options=['put', 'call'], index=0 if leg['type'] == 'put' else 1, key=f'type_{i}')
            with c2:
                leg['K'] = st.number_input(f'Strike K #{i+1}', value=float(leg['K']), step=1.0, key=f'K_{i}')
            with c3:
                leg['qty'] = st.number_input(f'Qty #{i+1} (long + / short -)', value=float(leg['qty']), step=1.0, format='%f', key=f'qty_{i}')
            with c4:
                leg['delete'] = st.checkbox(f"Delete #{i+1}", key=f"del_{i}")

        notional = st.number_input('Notional scale (per leg unit → shares)', value=100.0, step=1.0)

        st.markdown('---')
        st.header('Shock / Path settings')
        spot_shock_pct = st.number_input('Spot shock (%) (negative for drop)', value=-5.0, step=0.1)
        vol_shock_abs = st.number_input('Vol shock (abs, e.g. 0.20 for +20 vol pts)', value=0.20, step=0.01, format='%f')
        N = st.slider('Number of micro-steps (N)', min_value=10, max_value=500, value=50, step=10)

        run = st.form_submit_button('Run attribution')

    # Outside the form: process add/delete and run
    # Add leg button must NOT be inside a form
    if st.sidebar.button('➕ Add leg'):
        st.session_state.legs.append({'type': 'put', 'K': round(S0 * 0.98, 2), 'qty': 1.0, 'delete': False})
        st.rerun()

    # Process deletions
    for leg in list(st.session_state.legs):
        if leg.get('delete'):
            st.session_state.legs.remove(leg)
    
    if run:
        legs: List[LegTuple] = [(float(leg['K']), leg['type'], float(leg['qty'])) for leg in st.session_state.legs]
        dx_total = (spot_shock_pct / 100.0) * S0
        dv_total = float(vol_shock_abs)

        with st.spinner('Computing pathwise attribution...'):
            cumulative, df_steps, exact_pl, V_final = pathwise_attribution(
                legs=legs,
                S0=S0,
                r=r,
                q=q,
                sigma0=sigma0,
                T=T,
                dx_total=dx_total,
                dv_total=dv_total,
                N=int(N),
                notional=float(notional),
            )

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric('Exact total P&L (per notional)', f"{exact_pl:,.4f}")
        col2.metric('Pathwise sum (per notional)', f"{cumulative['total']:,.4f}", delta=f"{(cumulative['total']-exact_pl):.4f}")
        col3.metric('Final portfolio price (per share-equivalent)', f"{V_final:,.6f}")

        contrib = pd.DataFrame({'Greek': ['Delta','Gamma','Vega','Vomma','Vanna'],
                                'Contribution': [cumulative['delta'], cumulative['gamma'], cumulative['vega'], cumulative['vomma'], cumulative['vanna']]})

        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Greeks", "Stepwise Attribution Table", "Cumulative P&L Chart", "Leg Summary"])

        with tab1:
            st.subheader('Cumulative Greek contributions (per notional)')
            st.bar_chart(contrib.set_index('Greek'))
            d0, v0, g0, va0, vo0 = greeks_portfolio(legs, S0, r, q, sigma0, T)
            st.caption(f"Start Greeks (per share): Δ={d0:.4f}, Γ={g0:.6f}, Vega={v0:.4f}, Vanna={va0:.6f}, Vomma={vo0:.4f}")

        with tab2:
            st.dataframe(df_steps[['step','S','sigma','dV_delta','dV_gamma','dV_vega','dV_vomma','dV_vanna','dV_local']])
            st.download_button('Download CSV', df_steps.to_csv(index=False), file_name='stepwise_contrib.csv')

        with tab3:
            st.subheader('How contributions build over steps')
            cum_df = df_steps[['step','dV_delta','dV_gamma','dV_vega','dV_vomma','dV_vanna']].copy().set_index('step').cumsum()
            fig, ax = plt.subplots(figsize=(10, 4))
            for col in cum_df.columns:
                ax.plot(cum_df.index, cum_df[col], label=col)
            ax.legend(); ax.set_xlabel('Micro-step'); ax.set_ylabel('Cumulative P&L (per notional)')
            st.pyplot(fig)

        with tab4:
            # per-leg risk snapshot at start
            leg_rows = []
            for (K_leg, t_leg, qty_leg) in legs:
                d, v, g, va, vo, _ = bs_greeks(S0, K_leg, r, q, sigma0, T, t_leg)
                leg_rows.append({'Type': t_leg, 'Strike': K_leg, 'Qty': qty_leg,
                                 'Delta': qty_leg*d, 'Gamma': qty_leg*g, 'Vega': qty_leg*v,
                                 'Vanna': qty_leg*va, 'Vomma': qty_leg*vo})
            st.dataframe(pd.DataFrame(leg_rows))

        st.success('Done — interactive attribution computed.')
    else:
        st.write('Fill the inputs in the left panel and click "Run attribution" to compute.')

# ==================== CLI SELF-TESTS (when Streamlit missing) ====================

if not STREAMLIT_AVAILABLE and __name__ == "__main__":
    print("[CLI MODE] Streamlit not found. Running self-tests and examples.\n")

    # ---- Test Case 1: Single-leg OTM put, small shocks ----
    legs = [(98.0, 'put', 1.0)]
    S0, r, q, sigma0, T = 100.0, 0.05, 0.0, 0.20, 63/252.0
    dx_total, dv_total, N = -0.05*S0, 0.20, 50

    cum, df, exact_pl, V_final = pathwise_attribution(legs, S0, r, q, sigma0, T, dx_total, dv_total, N)

    print("Test 1 — Exact PL:", round(exact_pl, 6))
    print("Test 1 — Pathwise sum:", round(cum['total'], 6))
    print("Difference:", cum['total'] - exact_pl)
    assert abs(cum['total'] - exact_pl) < 1e-1, "Pathwise attribution should closely match exact repricing for N=50."

    # ---- Test Case 2: Two-leg spread ----
    spread_legs = [(95.0, 'put', 1.0), (90.0, 'put', -1.0)]
    cum2, df2, exact2, Vfin2 = pathwise_attribution(spread_legs, S0, r, q, sigma0, T, dx_total, dv_total, N)
    print("Test 2 — Spread exact PL:", round(exact2, 6))
    print("Test 2 — Spread pathwise sum:", round(cum2['total'], 6))
    assert abs(cum2['total'] - exact2) < 1e-1, "Spread attribution should also be close."

    # Save CSVs for inspection
    df.to_csv('stepwise_single_leg.csv', index=False)
    df2.to_csv('stepwise_spread.csv', index=False)
    print("\nSelf-test complete. CSVs written: stepwise_single_leg.csv, stepwise_spread.csv")

# EOF
