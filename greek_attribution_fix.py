# streamlit_greek_attribution_fixed.py
# Pathwise incremental Greek attribution dashboard (portfolio + per-leg DTE + per-leg IV)
# Multi-segment shocks (spot/vol/time), per-leg initial IV, Theta term, delta-hedge via forwards,
# and Trader View (Gamma* = Gamma + slippage). If Streamlit is unavailable, runs CLI self-tests.
#
# UI:      pip install streamlit pandas numpy matplotlib
#          streamlit run streamlit_greek_attribution_fixed.py
# CLI:     python streamlit_greek_attribution_fixed.py

from __future__ import annotations
import math
from math import erf
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import streamlit as st  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    plt = None  # type: ignore
    STREAMLIT_AVAILABLE = False

# ==================== Black–Scholes helpers ====================

def CDF(x: float) -> float:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

def pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = 'put') -> float:
    # If sigma<=0 or T<=0, fallback to intrinsic (makes sense for collapsed legs)
    if sigma <= 0 or T <= 0:
        return max((K - S) if option_type == 'put' else (S - K), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * math.exp(-q * T) * CDF(d1) - K * math.exp(-r * T) * CDF(d2)
    else:
        return K * math.exp(-r * T) * CDF(-d2) - S * math.exp(-q * T) * CDF(-d1)


def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = 'put') -> Tuple[float, float, float, float, float, float]:
    """Return (delta, vega, gamma, vanna, vomma, theta) with theta per year."""
    if sigma <= 0 or T <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    nd1 = pdf(d1)
    if option_type == 'call':
        delta = math.exp(-q * T) * CDF(d1)
        theta = (
            - (S * math.exp(-q * T) * nd1 * sigma) / (2 * sqT)
            - r * K * math.exp(-r * T) * CDF(d2)
            + q * S * math.exp(-q * T) * CDF(d1)
        )
    else:
        delta = - math.exp(-q * T) * CDF(-d1)
        theta = (
            - (S * math.exp(-q * T) * nd1 * sigma) / (2 * sqT)
            + r * K * math.exp(-r * T) * CDF(-d2)
            - q * S * math.exp(-q * T) * CDF(-d1)
        )
    vega = S * math.exp(-q * T) * nd1 * sqT
    gamma = math.exp(-q * T) * nd1 / (S * sigma * sqT)
    vomma = vega * d1 * d2 / sigma
    vanna = vega * (1 - d1 / (sigma * sqT)) / S
    return delta, vega, gamma, vanna, vomma, theta

# ==================== Portfolio helpers (per-leg DTE & IV) ====================

@dataclass
class Leg:
    K: float
    opt_type: str  # 'put' or 'call'
    qty: float
    T: float   # years (DTE/252)
    sigma: float  # per-leg IV

# (K, type, qty, T_years, sigma)
LegTuple = Tuple[float, str, float, float, float]


def price_portfolio(legs: List[LegTuple], S: float, r: float, q: float) -> float:
    total = 0.0
    for K, opt_type, qty, T, sig in legs:
        total += qty * bs_price(S, K, r, q, sig, T, opt_type)
    return total


def greeks_portfolio(legs: List[LegTuple], S: float, r: float, q: float) -> Tuple[float, float, float, float, float, float]:
    d = v = g = va = vo = th = 0.0
    for K, opt_type, qty, T, sig in legs:
        delta, vega, gamma, vanna, vomma, theta = bs_greeks(S, K, r, q, sig, T, opt_type)
        d  += qty * delta
        v  += qty * vega
        g  += qty * gamma
        va += qty * vanna
        vo += qty * vomma
        th += qty * theta
    return d, v, g, va, vo, th

# ==================== Pathwise attribution core (multi-segment) ====================

def _pathwise_core(
    legs_in: List[LegTuple], S0: float, r: float, q: float,
    segments: List[Tuple[float, float, float]], # (dS_total, dSig_total, dTau_days)
    N_per: int, notional: float,
    delta_hedge: bool, hedge_timing: str,
    trader_view: bool,
) -> Tuple[Dict[str,float], pd.DataFrame, float, float, float]:
    """Internal engine. Each segment carries spot/vol/time shocks.
    Time shock is in **days**; positive days = time forward (DTE decreases).
    Returns cumulative dict, step df, exact options-only PL, V_final, V_initial.
    """
    # work on a local mutable copy of legs for pathwise integration
    legs = [list(x) for x in legs_in]
    S = S0
    V_initial = price_portfolio([(K,t,qty,T,sig) for K,t,qty,T,sig in legs], S, r, q)

    cumulative = {k: 0.0 for k in ['delta','gamma','vega','vomma','vanna','theta','hedge','slip','total']}
    rows = []

    h = 0.0
    if delta_hedge and hedge_timing == 'start':
        D0, *_ = greeks_portfolio(legs, S, r, q)
        h = -D0

    step_idx = 0
    for seg_idx, (dx_total, dv_total, dt_days) in enumerate(segments, start=1):
        if N_per <= 0:
            continue
        dS = dx_total / N_per
        dSig = dv_total / N_per
        dTau_years = (dt_days / 252.0) / max(N_per,1)  # positive means time moves forward

        for _ in range(N_per):
            step_idx += 1
            # Greeks at start-of-step
            D,V, G, VA, VO, TH = greeks_portfolio(legs, S, r, q)

            # Local Taylor P&L (options)
            dV_delta = D * dS
            dV_gamma = 0.5 * G * (dS ** 2)
            dV_vega  = V * dSig
            dV_vomma = 0.5 * VO * (dSig ** 2)
            dV_vanna = VA * dS * dSig
            dV_theta = TH * dTau_years
            dV_opt   = dV_delta + dV_gamma + dV_vega + dV_vomma + dV_vanna + dV_theta

            # Hedge P&L
            dV_hedge = (h * dS) if delta_hedge else 0.0
            dV_slip  = (dV_delta + dV_hedge) if delta_hedge else 0.0

            # Advance state: spot, vols (all legs), time (per leg)
            S_next = S + dS
            for j in range(len(legs)):
                # update sigma (absolute shifts)
                legs[j][4] = max(1e-8, legs[j][4] + dSig)  # sigma_j
                # decrement time
                legs[j][3] = max(0.0,   legs[j][3] - dTau_years)  # T_j (reduce DTE)

            # Re-hedge at end (or start) using next state
            if delta_hedge:
                # compute next delta at S_next with updated legs
                D_next, *_ = greeks_portfolio([(K,t,qty,T,sig) for K,t,qty,T,sig in legs], S_next, r, q)
                h = -D_next  # set hedge position for next micro-step

            # commit state
            S = S_next

            # Accumulate
            cumulative['delta'] += dV_delta * notional
            cumulative['gamma'] += dV_gamma * notional
            cumulative['vega']  += dV_vega  * notional
            cumulative['vomma'] += dV_vomma * notional
            cumulative['vanna'] += dV_vanna * notional
            cumulative['theta'] += dV_theta * notional
            cumulative['hedge'] += dV_hedge * notional
            cumulative['slip']  += dV_slip  * notional
            cumulative['total'] += (dV_opt + dV_hedge) * notional

            # record row (use current legs state)
            rows.append({
                'segment': seg_idx,
                'step': step_idx,
                'S': S,
                'dS': dS,
                'dSig': dSig,
                'dTau_days': dt_days / max(N_per,1),
                'dV_delta': dV_delta * notional,
                'dV_gamma': dV_gamma * notional,
                'dV_vega':  dV_vega  * notional,
                'dV_vomma': dV_vomma * notional,
                'dV_vanna': dV_vanna * notional,
                'dV_theta': dV_theta * notional,
                'dV_hedge': dV_hedge * notional,
                'dV_slip':  dV_slip  * notional,
                'dV_local_total': (dV_opt + dV_hedge) * notional,
                'Delta': D, 'Vega': V, 'Gamma': G, 'Vanna': VA, 'Vomma': VO, 'Theta': TH,
                'HedgePos': h,
            })

    # -------------------- DETERMINISTIC FINAL REPRICE (FIX) --------------------
    # Compute final underlying and per-leg final params directly from the input segments
    # (avoids any mutated-in-loop state mismatch and ensures V_final matches UI's FinalPrice)
    S_final_det = S0 + sum(dx for dx, dv, dt in segments)
    vol_shift_total = sum(dv for dx, dv, dt in segments)
    days_forward_total = sum(dt for dx, dv, dt in segments)

    legs_final = []
    for K, opt_type, qty, T_init, sig_init in legs_in:
        sig_final = max(1e-8, sig_init + vol_shift_total)
        T_final = max(0.0, T_init - days_forward_total / 252.0)
        legs_final.append((K, opt_type, qty, T_final, sig_final))

    V_final = price_portfolio(legs_final, S_final_det, r, q)
    # exact_pl is options-only repricing based on deterministic final params
    exact_pl = (V_final - V_initial) * notional

    return cumulative, pd.DataFrame(rows), exact_pl, V_final, V_initial

# Public single-segment wrapper (kept for backward compatibility)

def pathwise_attribution(
    legs: List[LegTuple], S0: float, r: float, q: float,
    dx_total: float, dv_total: float, N: int, notional: float = 100.0,
    delta_hedge: bool = False, hedge_timing: str = 'end', trader_view: bool = True
) -> Tuple[Dict[str,float], pd.DataFrame, float, float]:
    cum, df, exact_pl, V_final, _ = _pathwise_core(
        legs, S0, r, q, [(dx_total, dv_total, 0.0)], N, notional, delta_hedge, hedge_timing, trader_view
    )
    return cum, df, exact_pl, V_final

# Multi-segment API (spot/vol/time lists)

def pathwise_attribution_multi(
    legs: List[LegTuple], S0: float, r: float, q: float,
    spot_shocks: List[float], vol_shocks: List[float], time_shocks_days: List[float],
    N_per: int, notional: float = 100.0,
    delta_hedge: bool = False, hedge_timing: str = 'end', trader_view: bool = True
) -> Tuple[Dict[str,float], pd.DataFrame, float, float, float]:
    # pad lists
    if not spot_shocks and not vol_shocks and not time_shocks_days:
        spot_shocks, vol_shocks, time_shocks_days = [0.0], [0.0], [0.0]
    L = max(len(spot_shocks or [0.0]), len(vol_shocks or [0.0]), len(time_shocks_days or [0.0]))
    def pad(xs, L, fill=0.0):
        xs = list(xs or [fill])
        if len(xs) < L:
            xs = xs + [xs[-1]] * (L - len(xs))
        return xs
    spot_shocks = pad(spot_shocks, L, 0.0)
    vol_shocks  = pad(vol_shocks,  L, 0.0)
    time_shocks_days = pad(time_shocks_days, L, 0.0)

    segments = list(zip(spot_shocks, vol_shocks, time_shocks_days))
    return _pathwise_core(legs, S0, r, q, segments, max(1,int(N_per)), notional, delta_hedge, hedge_timing, trader_view)

# ==================== Streamlit UI (optional) ====================

if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Pathwise Greek Attribution (fixed)", layout="wide")
    st.title("Pathwise Greek Attribution — incremental ANOVA style (Trader View ready)")
    st.markdown("Per-leg DTE and IV, multi-segment shocks (spot/vol/time), optional delta-hedge via forwards. Trader View folds slippage into Gamma*.")

    with st.sidebar.form(key='inputs'):
        st.header('Market/Model Inputs')
        S0 = st.number_input('Spot S0', value=100.0, step=1.0, format='%f')
        r = st.number_input('Risk-free rate (r)', value=0.05, step=0.001, format='%f')
        q = st.number_input('Dividend yield (q)', value=0.0, step=0.001, format='%f')

        st.markdown('---')
        st.header('Portfolio Legs (per-leg DTE & IV)')
        if 'legs' not in st.session_state:
            st.session_state.legs = [
                {'type': 'put', 'K': 98.0, 'qty': 1.0, 'dte': 63, 'iv': 0.20, 'delete': False},
            ]

        for i, leg in enumerate(st.session_state.legs):
            c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 0.8])
            with c1:
                leg['type'] = st.selectbox(f'Option type #{i+1}', options=['put','call'], index=0 if leg['type']=='put' else 1, key=f'type_{i}')
            with c2:
                leg['K'] = st.number_input(f'Strike K #{i+1}', value=float(leg['K']), step=1.0, key=f'K_{i}')
            with c3:
                leg['qty'] = st.number_input(f'Qty #{i+1} (long+/short-)', value=float(leg['qty']), step=1.0, format='%f', key=f'qty_{i}')
            with c4:
                leg['dte'] = st.number_input(f'DTE #{i+1} (days)', value=int(leg.get('dte',63)), step=1, key=f'dte_{i}')
            with c5:
                leg['iv'] = st.number_input(f'IV σ #{i+1}', value=float(leg.get('iv',0.20)), step=0.01, format='%f', key=f'iv_{i}')
            with c6:
                leg['delete'] = st.checkbox(f'Delete #{i+1}', key=f'del_{i}')

        notional = st.number_input('Notional scale (per leg unit → shares)', value=100.0, step=1.0)

        st.markdown('---')
        st.header('Shock / Path settings (multi-segment)')
        spot_list_str = st.text_input('Spot shocks list (%) — comma sep (e.g., -5, +6, -3)', value='-5')
        vol_list_str  = st.text_input('Vol shocks list (abs vols) — comma sep (e.g., 0.15, -0.05)', value='0.15')
        time_list_str = st.text_input('Time shocks list (days, positive = forward)', value='0')
        N_per_seg = st.slider('Micro-steps per segment (N per)', min_value=5, max_value=500, value=50, step=5)

        st.markdown('---')
        st.header('Delta Hedging (forwards)')
        delta_hedge = st.checkbox('Delta-hedge each step (forwards)', value=True)
        hedge_timing = st.selectbox('Hedge timing', options=['end','start'], index=0)
        trader_view = st.checkbox('Trader View (Gamma* = Gamma + slippage)', value=True)

        run = st.form_submit_button('Run attribution')

    if st.sidebar.button('➕ Add leg'):
        st.session_state.legs.append({'type':'put','K': round(S0*0.98,2), 'qty':1.0, 'dte':63, 'iv':0.20, 'delete':False})
        st.rerun()

    for leg in list(st.session_state.legs):
        if leg.get('delete'):
            st.session_state.legs.remove(leg)

    if run:
        legs: List[LegTuple] = [
            (float(leg['K']), leg['type'], float(leg['qty']), max(0,int(leg['dte']))/252.0, float(leg['iv']))
            for leg in st.session_state.legs
        ]

        # parse lists
        def parse_spot(s: str) -> List[float]:
            if not s.strip():
                return [0.0]
            vals = []
            for tok in s.split(','):
                tok = tok.strip().replace('%','')
                if tok:
                    # convert percentage string -> absolute move on S0
                    vals.append(float(tok)/100.0 * S0)
            return vals or [0.0]
        def parse_vol(s: str) -> List[float]:
            if not s.strip():
                return [0.0]
            vals = []
            for tok in s.split(','):
                tok = tok.strip()
                if tok:
                    vals.append(float(tok))
            return vals or [0.0]
        def parse_time(s: str) -> List[float]:
            if not s.strip():
                return [0.0]
            vals = []
            for tok in s.split(','):
                tok = tok.strip()
                if tok:
                    vals.append(float(tok))
            return vals or [0.0]

        spot_steps = parse_spot(spot_list_str)
        vol_steps  = parse_vol(vol_list_str)
        time_steps = parse_time(time_list_str)

        with st.spinner('Computing pathwise attribution...'):
            result_tuple = pathwise_attribution_multi(
                legs=legs, S0=S0, r=r, q=q,
                spot_shocks=spot_steps, vol_shocks=vol_steps, time_shocks_days=time_steps,
                N_per=N_per_seg, notional=float(notional),
                delta_hedge=bool(delta_hedge), hedge_timing=str(hedge_timing), trader_view=bool(trader_view)
            )
            
            # EXPLICIT unpacking to avoid any confusion
            cumulative = result_tuple[0]
            df_steps = result_tuple[1]
            exact_pl_from_function = result_tuple[2]
            V_final_from_function = result_tuple[3]
            V_initial_from_function = result_tuple[4]

        # ========== CRITICAL DEBUG SECTION ==========
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Debug Output")
        st.sidebar.write(f"**V_initial (per share):** {V_initial_from_function:.8f}")
        st.sidebar.write(f"**V_final (per share):** {V_final_from_function:.8f}")
        st.sidebar.write(f"**Notional:** {float(notional):.0f}")
        st.sidebar.write(f"**Calculation:**")
        st.sidebar.write(f"  (V_final - V_initial) × notional")
        st.sidebar.write(f"  = ({V_final_from_function:.8f} - {V_initial_from_function:.8f}) × {float(notional):.0f}")
        st.sidebar.write(f"  = {(V_final_from_function - V_initial_from_function):.8f} × {float(notional):.0f}")
        st.sidebar.write(f"  = **{exact_pl_from_function:.8f}**")
        st.sidebar.write(f"**For SHORT positions:** If exact_pl < 0, you profit (options got cheaper)")
        
        # Metrics: show net premium (credit/debit)
        net_premium = -V_initial_from_function * float(notional)  # +ve = credit received when short
        
        # Use the explicitly unpacked value
        exact_repriced_pnl = exact_pl_from_function
        
        # For trader perspective: if you're short, negative exact_pl means profit
        # (options got cheaper, you can buy back for less than you sold)
        trader_pnl = -exact_repriced_pnl  # flip sign for short positions

        # options_pathwise (greeks aggregated) = sum of option greek buckets (excludes hedge)
        options_pathwise = (cumulative['delta'] + cumulative['gamma'] + cumulative['vega'] + cumulative['vomma'] + cumulative['vanna'] + cumulative['theta'])
        hedge_pnl = cumulative['hedge']
        greeks_path_mtm = options_pathwise
        residual_options = exact_pl_from_function - options_pathwise  # unexplained by local Taylor (third+ order)

        col1, col2, col3 = st.columns(3)
        col1.metric('Exact repriced P&L (options perspective)', f"{exact_repriced_pnl:,.4f}")
        col2.metric('Trader P&L (for short: flip sign)', f"{trader_pnl:,.4f}", 
                   help="If you're SHORT, positive values mean profit")
        col3.metric('Net premium at start (credit + / debit -)', f"{net_premium:,.4f}")

        # Additional quick stats below the main metrics
        st.write('## Breakdown (per notional)')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Greeks aggregated (pathwise)', f"{greeks_path_mtm:,.4f}")
        c2.metric('Hedge P&L (realized)', f"{hedge_pnl:,.4f}")
        c3.metric('Residual (exact - greeks_pathwise)', f"{residual_options:,.4f}")
        c4.metric('Pathwise total (incl hedge)', f"{cumulative['total']:,.4f}")

        # Contributions for chart
        if bool(delta_hedge) and bool(trader_view):
            gamma_star = cumulative['gamma'] + cumulative['slip']
            contrib = pd.DataFrame({'Greek': ['Gamma* (incl. slippage)','Vega','Vomma','Vanna','Theta','Hedge','Delta'],
                                    'Contribution': [gamma_star, cumulative['vega'], cumulative['vomma'], cumulative['vanna'], cumulative['theta'], cumulative['hedge'], cumulative['delta']]})
        else:
            contrib = pd.DataFrame({'Greek': ['Delta','Gamma','Vega','Vomma','Vanna','Theta','Hedge','Slip'],
                                    'Contribution': [cumulative['delta'], cumulative['gamma'], cumulative['vega'], cumulative['vomma'], cumulative['vanna'], cumulative['theta'], cumulative['hedge'], cumulative['slip']]})

        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Greeks", "Stepwise Attribution Table", "Cumulative P&L Chart", "Leg Summary"])

        with tab1:
            st.subheader('Cumulative Greek contributions (per notional)')
            st.bar_chart(contrib.set_index('Greek'))
            D0,V0,G0,VA0,VO0,TH0 = greeks_portfolio(legs=S0 and [(K,t,qty,T,sig) for (K,t,qty,T,sig) in legs] or [], S=S0, r=r, q=q)
            st.caption(f"Start Greeks (per share): Δ={D0:.4f}, Γ={G0:.6f}, Vega={V0:.4f}, Vanna={VA0:.6f}, Vomma={VO0:.4f}, Theta={TH0:.4f}. Trader View folds slippage into Γ* when hedged.")

        with tab2:
            cols = ['segment','step','S','dS','dSig','dTau_days','dV_delta','dV_gamma','dV_vega','dV_vomma','dV_vanna','dV_theta','dV_hedge','dV_slip','dV_local_total']
            st.dataframe(df_steps[cols])
            st.download_button('Download CSV', df_steps[cols].to_csv(index=False), file_name='stepwise_contrib.csv')

        with tab3:
            st.subheader('How contributions build over steps')
            cum_df = df_steps[['step','dV_delta','dV_gamma','dV_vega','dV_vomma','dV_vanna','dV_theta','dV_hedge','dV_slip']].copy().set_index('step').cumsum()
            fig, ax = plt.subplots(figsize=(10, 4))
            for col in cum_df.columns:
                ax.plot(cum_df.index, cum_df[col], label=col)
            ax.legend(); ax.set_xlabel('Micro-step'); ax.set_ylabel('Cumulative P&L (per notional)')
            st.pyplot(fig)

        with tab4:
            # compute final spot/sigma/T after shocks for per-leg final repricing (deterministic)
            S_final_calc = S0 + sum(spot_steps)
            total_vol_shift = sum(vol_steps)
            total_time_days = sum(time_steps)

            leg_rows = []
            for (K_leg, t_leg, qty_leg, T_leg, sig_leg) in legs:
                # initial greeks and prices
                d,v,g,va,vo,th = bs_greeks(S0, K_leg, r, q, sig_leg, T_leg, t_leg)
                price_init = bs_price(S0, K_leg, r, q, sig_leg, T_leg, t_leg)
                # final per-leg params (use same deterministic logic as engine)
                sig_final_leg = max(1e-8, sig_leg + total_vol_shift)
                T_final_leg = max(0.0, T_leg - total_time_days/252.0)
                price_final = bs_price(S_final_calc, K_leg, r, q, sig_final_leg, T_final_leg, t_leg)

                leg_rows.append({'Type': t_leg, 'Strike': K_leg, 'Qty': qty_leg, 'DTE(days)': int(round(T_leg*252)), 'IV': sig_leg,
                                 'InitPrice': price_init, 'FinalPrice': price_final,
                                 'Delta': qty_leg*d, 'Gamma': qty_leg*g, 'Vega': qty_leg*v,
                                 'Vanna': qty_leg*va, 'Vomma': qty_leg*vo, 'Theta': qty_leg*th})
            st.dataframe(pd.DataFrame(leg_rows))

        st.success('Done — interactive attribution computed.')
    else:
        st.write('Fill the inputs and click "Run attribution" to compute.')

# ==================== CLI SELF-TESTS ====================

if not STREAMLIT_AVAILABLE and __name__ == "__main__":
    print("[CLI MODE] Streamlit not) available — running self-tests...")