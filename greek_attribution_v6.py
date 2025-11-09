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
from typing import List, Tuple, Dict, Optional

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
    aftermath_vol: Optional[float] = None,
) -> Tuple[Dict[str,float], pd.DataFrame, float, float, float]:
    """Internal engine. Each segment carries spot/vol/time shocks.
    Time shock is in **days**; positive days = time forward (DTE decreases).
    Returns cumulative dict, step df, exact options-only PL, V_final, V_initial.
    
    aftermath_vol: if provided, overrides path-based vol for final reprice only.
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

    # -------------------- DETERMINISTIC FINAL REPRICE --------------------
    # Compute final underlying and per-leg final params directly from the input segments
    S_final_det = S0 + sum(dx for dx, dv, dt in segments)
    vol_shift_total = sum(dv for dx, dv, dt in segments)
    days_forward_total = sum(dt for dx, dv, dt in segments)

    # Build legs_final with updated parameters
    legs_final: List[LegTuple] = []
    for K, opt_type, qty, T_init, sig_init in legs_in:
        T_final = max(0.0, T_init - days_forward_total / 252.0)
        
        # Use aftermath_vol if provided, otherwise use path-based final vol
        if aftermath_vol is not None:
            sig_final = aftermath_vol
        else:
            sig_final = max(1e-8, sig_init + vol_shift_total)
        
        legs_final.append((K, opt_type, qty, T_final, sig_final))

    # Use the portfolio pricing function to ensure consistency
    V_final = price_portfolio(legs_final, S_final_det, r, q)
    
    # exact_pl is the change in portfolio value (options perspective)
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
    delta_hedge: bool = False, hedge_timing: str = 'end', trader_view: bool = True,
    aftermath_vol: Optional[float] = None,
) -> Tuple[Dict[str,float], pd.DataFrame, float, float, float]:
    """
    Multi-segment pathwise attribution.
    
    Args:
        spot_shocks: List of spot moves (e.g., [-5, +3] for 2 segments). DRIVES segment count.
        vol_shocks: List of vol shifts for path Greeks (e.g., [+0.15, -0.10]). Must be <= len(spot_shocks).
        time_shocks_days: Single value or list - will be SUMMED and distributed evenly across spot segments.
        aftermath_vol: Optional single value - overrides final vol for reprice only (ignores path vol).
    
    Returns:
        (cumulative, df_steps, exact_pl, V_final, V_initial)
    """
    # Handle empty inputs
    if not spot_shocks:
        spot_shocks = [0.0]
    if not vol_shocks:
        vol_shocks = []
    if not time_shocks_days:
        time_shocks_days = [0.0]
    
    # CRITICAL: Spot shocks drive the segment count
    num_segments = len(spot_shocks)
    
    # Vol shocks: must be <= spot shocks length
    vol_shocks = list(vol_shocks)
    if len(vol_shocks) > num_segments:
        raise ValueError(f"Vol shocks ({len(vol_shocks)}) cannot exceed spot shocks ({num_segments}). "
                        f"Reduce vol shocks or add more spot segments.")
    
    # Pad vol with 0 (no further changes)
    while len(vol_shocks) < num_segments:
        vol_shocks.append(0.0)
    
    # Time: sum all values and distribute evenly across segments
    time_list = list(time_shocks_days)
    total_time = sum(time_list)
    time_per_segment = total_time / num_segments
    time_shocks_distributed = [time_per_segment] * num_segments
    
    # Build segments
    segments = list(zip(spot_shocks, vol_shocks, time_shocks_distributed))
    
    return _pathwise_core(legs, S0, r, q, segments, max(1,int(N_per)), notional, 
                         delta_hedge, hedge_timing, trader_view, aftermath_vol)

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
        st.header('Shock / Path settings')
        st.caption('💡 Spot shocks drive segment count. Vol must be ≤ spot length.')
        spot_list_str = st.text_input('Spot shocks (%) — comma sep (e.g., -5, +6, -3)', value='-5, +5')
        vol_list_str  = st.text_input('Vol PATH shocks (abs) — comma sep (e.g., 0.15, -0.05)', value='0')
        time_total_str = st.text_input('Time shock (days, TOTAL distributed evenly)', value='1')
        
        st.markdown('---')
        st.subheader('Aftermath Vol (Reprice Override)')
        st.caption('⚠️ Optional: Override final IV for reprice only (ignores path vol)')
        use_aftermath = st.checkbox('Use aftermath vol override', value=False)
        aftermath_vol_input = st.number_input('Aftermath IV (single value)', value=0.25, step=0.01, format='%f', disabled=not use_aftermath)
        
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
                    vals.append(float(tok)/100.0 * S0)
            return vals
        
        def parse_vol(s: str) -> List[float]:
            if not s.strip():
                return []
            vals = []
            for tok in s.split(','):
                tok = tok.strip()
                if tok:
                    vals.append(float(tok))
            return vals
        
        def parse_time(s: str) -> List[float]:
            if not s.strip():
                return [0.0]
            vals = []
            for tok in s.split(','):
                tok = tok.strip()
                if tok:
                    vals.append(float(tok))
            return vals

        spot_steps = parse_spot(spot_list_str)
        vol_steps  = parse_vol(vol_list_str)
        time_steps = parse_time(time_total_str)
        
        # Validation
        if len(vol_steps) > len(spot_steps):
            st.error(f"❌ Error: Vol shocks ({len(vol_steps)}) cannot exceed spot shocks ({len(spot_steps)}). "
                    f"Either reduce vol shocks or add more spot segments.")
            st.stop()
        
        aftermath_vol_val = float(aftermath_vol_input) if use_aftermath else None

        with st.spinner('Computing pathwise attribution...'):
            try:
                result_tuple = pathwise_attribution_multi(
                    legs=legs, S0=S0, r=r, q=q,
                    spot_shocks=spot_steps, vol_shocks=vol_steps, time_shocks_days=time_steps,
                    N_per=N_per_seg, notional=float(notional),
                    delta_hedge=bool(delta_hedge), hedge_timing=str(hedge_timing), trader_view=bool(trader_view),
                    aftermath_vol=aftermath_vol_val
                )
                
                # EXPLICIT unpacking to avoid any confusion
                cumulative = result_tuple[0]
                df_steps = result_tuple[1]
                exact_pl_from_function = result_tuple[2]
                V_final_from_function = result_tuple[3]
                V_initial_from_function = result_tuple[4]
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                st.stop()

        # ========== DEBUG SECTION ==========
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Debug Output")
        
        # Recompute manually to verify
        V_init_manual = 0.0
        V_final_manual = 0.0
        S_final_check = S0 + sum(spot_steps)
        vol_shift_check = sum(vol_steps)
        time_shift_check = sum(time_steps)
        
        st.sidebar.write("**Manual verification:**")
        for idx, (K, opt_type, qty, T, sig) in enumerate(legs):
            p_init = bs_price(S0, K, r, q, sig, T, opt_type)
            
            # Final vol: aftermath override or path-based
            if aftermath_vol_val is not None:
                sig_f = aftermath_vol_val
            else:
                sig_f = max(1e-8, sig + vol_shift_check)
            
            T_f = max(0.0, T - time_shift_check / 252.0)
            p_final = bs_price(S_final_check, K, r, q, sig_f, T_f, opt_type)
            V_init_manual += qty * p_init
            V_final_manual += qty * p_final
            st.sidebar.write(f"Leg {idx+1}: {opt_type} K={K} qty={qty}")
            st.sidebar.write(f"  Init: {p_init:.6f} → {qty * p_init:.6f}")
            st.sidebar.write(f"  Final: {p_final:.6f} (IV={sig_f:.4f}) → {qty * p_final:.6f}")
        
        st.sidebar.write(f"**V_init (manual):** {V_init_manual:.8f}")
        st.sidebar.write(f"**V_final (manual):** {V_final_manual:.8f}")
        st.sidebar.write(f"**V_init (function):** {V_initial_from_function:.8f}")
        st.sidebar.write(f"**V_final (function):** {V_final_from_function:.8f}")
        st.sidebar.write(f"**Expected P&L:** {(V_final_manual - V_init_manual) * float(notional):.6f}")
        st.sidebar.write(f"**Actual P&L:** {exact_pl_from_function:.6f}")
        
        # Metrics
        net_premium = -V_initial_from_function * float(notional)
        exact_repriced_pnl = exact_pl_from_function
        trader_pnl = -exact_repriced_pnl

        options_pathwise = (cumulative['delta'] + cumulative['gamma'] + cumulative['vega'] + 
                           cumulative['vomma'] + cumulative['vanna'] + cumulative['theta'])
        hedge_pnl = cumulative['hedge']
        residual_options = exact_pl_from_function - options_pathwise

        col1, col2, col3 = st.columns(3)
        col1.metric('Exact repriced P&L', f"{exact_repriced_pnl:,.4f}", 
                   help="Options perspective: V_final - V_initial")
        col2.metric('Trader P&L (short flip sign)', f"{trader_pnl:,.4f}", 
                   help="For SHORT positions: positive = profit")
        col3.metric('Net premium at start', f"{net_premium:,.4f}",
                   help="Credit (+) received or Debit (-) paid")

        st.write('## Breakdown (per notional)')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Greeks aggregated', f"{options_pathwise:,.4f}")
        c2.metric('Hedge P&L', f"{hedge_pnl:,.4f}")
        c3.metric('Residual', f"{residual_options:,.4f}")
        c4.metric('Pathwise total', f"{cumulative['total']:,.4f}")

        # Contributions for chart
        if bool(delta_hedge) and bool(trader_view):
            gamma_star = cumulative['gamma'] + cumulative['slip']
            contrib = pd.DataFrame({
                'Greek': ['Gamma*','Vega','Vomma','Vanna','Theta','Hedge','Delta'],
                'Contribution': [gamma_star, cumulative['vega'], cumulative['vomma'], 
                               cumulative['vanna'], cumulative['theta'], cumulative['hedge'], cumulative['delta']]
            })
        else:
            contrib = pd.DataFrame({
                'Greek': ['Delta','Gamma','Vega','Vomma','Vanna','Theta','Hedge','Slip'],
                'Contribution': [cumulative['delta'], cumulative['gamma'], cumulative['vega'], 
                               cumulative['vomma'], cumulative['vanna'], cumulative['theta'], 
                               cumulative['hedge'], cumulative['slip']]
            })

        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Greeks", "Stepwise Table", "Cumulative Chart", "Leg Summary"])

        with tab1:
            st.subheader('Cumulative Greek contributions (per notional)')
            st.bar_chart(contrib.set_index('Greek'))
            D0,V0,G0,VA0,VO0,TH0 = greeks_portfolio(legs, S0, r, q)
            st.caption(f"Start Greeks: Δ={D0:.4f}, Γ={G0:.6f}, ν={V0:.4f}, Vanna={VA0:.6f}, Vomma={VO0:.4f}, Θ={TH0:.4f}")

        with tab2:
            cols = ['segment','step','S','dS','dSig','dTau_days','dV_delta','dV_gamma','dV_vega',
                   'dV_vomma','dV_vanna','dV_theta','dV_hedge','dV_slip','dV_local_total']
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
            # compute final spot/sigma/T after shocks for per-leg final repricing
            S_final_calc = S0 + sum(spot_steps)
            total_vol_shift = sum(vol_steps)
            total_time_days = sum(time_steps)

            leg_rows = []
            for (K_leg, t_leg, qty_leg, T_leg, sig_leg) in legs:
                # initial greeks and prices
                d,v,g,va,vo,th = bs_greeks(S0, K_leg, r, q, sig_leg, T_leg, t_leg)
                price_init = bs_price(S0, K_leg, r, q, sig_leg, T_leg, t_leg)
                
                # final per-leg params
                T_final_leg = max(0.0, T_leg - total_time_days/252.0)
                if aftermath_vol_val is not None:
                    sig_final_leg = aftermath_vol_val
                else:
                    sig_final_leg = max(1e-8, sig_leg + total_vol_shift)
                
                price_final = bs_price(S_final_calc, K_leg, r, q, sig_final_leg, T_final_leg, t_leg)

                leg_rows.append({
                    'Type': t_leg, 'Strike': K_leg, 'Qty': qty_leg, 
                    'DTE(days)': int(round(T_leg*252)), 'IV_init': sig_leg,
                    'InitPrice': price_init, 'FinalPrice': price_final, 'IV_final': sig_final_leg,
                    'Delta': qty_leg*d, 'Gamma': qty_leg*g, 'Vega': qty_leg*v,
                    'Vanna': qty_leg*va, 'Vomma': qty_leg*vo, 'Theta': qty_leg*th
                })
            st.dataframe(pd.DataFrame(leg_rows))
            
            if aftermath_vol_val is not None:
                st.info(f"ℹ️ Aftermath vol override active: Final IV = {aftermath_vol_val:.4f} (path vol ignored for reprice)")

        st.success('✅ Attribution computed successfully!')
    else:
        st.info('👈 Configure inputs in sidebar and click "Run attribution"')

# ==================== CLI SELF-TESTS ====================

if not STREAMLIT_AVAILABLE and __name__ == "__main__":
    print("[CLI MODE] Streamlit not found. Running self-tests.\n")

    S0, r, q = 100.0, 0.05, 0.0

    # Test 1: Single-leg OTM put
    legs1: List[LegTuple] = [(98.0, 'put', 1.0, 63/252.0, 0.20)]
    cum1, df1, exact1, Vfin1, Vinit1 = pathwise_attribution_multi(
        legs1, S0, r, q, spot_shocks=[-0.05*S0], vol_shocks=[0.20], time_shocks_days=[0], 
        N_per=50, delta_hedge=False
    )
    print("Test1 exact:", round(exact1,4), " pathwise:", round(cum1['total'],4))
    assert abs(cum1['total']-exact1) < 0.25

    # Test 2: Multi-segment spot with time distribution
    legs2: List[LegTuple] = [(100.0,'call', 1.0,30/252.0,0.22), (100.0,'call',-1.0,60/252.0,0.20)]
    cum2, df2, exact2, Vfin2, _ = pathwise_attribution_multi(
        legs2, S0, r, q, spot_shocks=[-0.03*S0, +0.02*S0], vol_shocks=[0.10, -0.05], 
        time_shocks_days=[2], N_per=60, delta_hedge=True
    )
    print("Test2 exact:", round(exact2,4), " pathwise:", round(cum2['total'],4))
    path_opts_2 = cum2['total'] - cum2['hedge']
    assert abs(path_opts_2-exact2) < 0.7

    # Test 3: Aftermath vol override
    legs3: List[LegTuple] = [(100.0,'put',-1.0,5/252.0,0.20)]
    cum3, df3, exact3, _, _ = pathwise_attribution_multi(
        legs3, S0, r, q, spot_shocks=[-0.05*S0, +0.05*S0], vol_shocks=[0, 0], 
        time_shocks_days=[1], N_per=50, delta_hedge=False, aftermath_vol=0.30
    )
    print("Test3 (aftermath vol) exact:", round(exact3,4), " pathwise:", round(cum3['total'],4))
    # Final reprice should use IV=0.30 regardless of path
    assert abs(cum3['total']-exact3) < 0.5

    # Test 4: Theta-only (no spot/vol movement)
    legs4: List[LegTuple] = [(100.0,'call',1.0,10/252.0,0.20)]
    cum4, df4, exact4, _, _ = pathwise_attribution_multi(
        legs4, S0, r, q, spot_shocks=[0.0], vol_shocks=[0.0], time_shocks_days=[1.0], 
        N_per=50, delta_hedge=False
    )
    print("Test4 (theta) exact:", round(exact4,4), " pathwise:", round(cum4['total'],4))
    assert abs((cum4['total']-cum4['theta'])) < 1e-3

    # Write CSVs
    df1.to_csv('stepwise_single_leg.csv', index=False)
    df2.to_csv('stepwise_multi_segment.csv', index=False)
    df3.to_csv('stepwise_aftermath_vol.csv', index=False)
    df4.to_csv('stepwise_theta.csv', index=False)
    print("\n✅ All self-tests passed! CSVs written.")

# EOF