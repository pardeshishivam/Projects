"""
Nifty 50 — Risk Metrics Dashboard (Streamlit + Plotly)
=======================================================
Run:  streamlit run var_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from var_calculator import fetch_nifty50, compute_metrics, exceptions, halflife

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 Risk Metrics",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Nifty 50 — Risk Metrics Dashboard")
st.caption("Historical VaR · EWMA VaR · Expected Shortfall")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    start_date = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end_date   = st.date_input("End date",   value=pd.Timestamp("today"))
    lookback   = st.slider("Lookback window (days)", 60, 500, 252, 10)
    confidence = st.selectbox(
        "Confidence level",
        [0.99, 0.975, 0.95],
        format_func=lambda x: f"{x*100:.1f}%",
    )
    lam = st.slider("EWMA lambda (λ)", 0.90, 0.99, 0.99, 0.01)

    st.caption(f"EWMA half-life: **{halflife(lam):.1f} days**")
    run_btn = st.button("Run Analysis", use_container_width=True)

# ─────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────
if "metrics" not in st.session_state:
    run_btn = True

if run_btn:
    with st.spinner("Fetching data…"):
        try:
            raw = fetch_nifty50(start=str(start_date), end=str(end_date))
        except Exception as e:
            st.error(str(e))
            st.stop()

    with st.spinner("Computing metrics…"):
        metrics = compute_metrics(
            returns=raw["log_ret"],
            lookback=lookback,
            confidence=confidence,
            lam=lam,
        )
    st.session_state.update(
        metrics=metrics, confidence=confidence, lam=lam, lookback=lookback
    )

if "metrics" not in st.session_state:
    st.info("Configure parameters and click **Run Analysis**.")
    st.stop()

metrics    = st.session_state["metrics"]
confidence = st.session_state["confidence"]
lam        = st.session_state["lam"]
lookback   = st.session_state["lookback"]

last = metrics.iloc[-1]

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

c1.metric(
    "Historical VaR (latest)",
    f"{last['hist_var']*100:.2f}%",
    help=f"1-day {confidence*100:.0f}% VaR — worst percentile over {lookback}-day window",
)
c2.metric(
    "EWMA VaR (latest)",
    f"{last['ewma_var']*100:.2f}%",
    help=f"λ={lam}, half-life {halflife(lam):.1f} days — conditional normal assumption",
)
c3.metric(
    "Expected Shortfall (latest)",
    f"{last['es']*100:.2f}%",
    help=f"Average loss in the worst {(1-confidence)*100:.1f}% of days (CVaR)",
)

st.divider()

# ─────────────────────────────────────────────
# Chart 1 — Rolling metrics over time
# ─────────────────────────────────────────────
st.subheader("Rolling Risk Metrics Over Time")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=metrics.index, y=metrics["actual_ret"] * 100,
    mode="lines", name="Daily Return",
    line=dict(color="steelblue", width=0.8), opacity=0.6,
))
fig.add_trace(go.Scatter(
    x=metrics.index, y=metrics["hist_var"] * 100,
    mode="lines", name=f"Historical VaR ({confidence*100:.0f}%)",
    line=dict(color="crimson", width=1.8),
))
fig.add_trace(go.Scatter(
    x=metrics.index, y=metrics["ewma_var"] * 100,
    mode="lines", name=f"EWMA VaR (λ={lam})",
    line=dict(color="darkorange", width=1.8, dash="dash"),
))
fig.add_trace(go.Scatter(
    x=metrics.index, y=metrics["es"] * 100,
    mode="lines", name="Expected Shortfall",
    line=dict(color="purple", width=1.8, dash="dot"),
))
fig.add_hline(y=0, line_color="black", line_width=0.6)

fig.update_layout(
    height=380, template="plotly_white",
    xaxis_title="Date", yaxis_title="Return (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# Chart 2 — Return distribution with all three cutoffs
# ─────────────────────────────────────────────
st.subheader("Return Distribution")

rets_pct = metrics["actual_ret"] * 100
hist_var_static = np.percentile(rets_pct, (1 - confidence) * 100)
ewma_var_static = last["ewma_var"] * 100
es_static       = last["es"] * 100

fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=rets_pct, nbinsx=80,
    marker_color="steelblue", opacity=0.7, name="Daily Returns",
))

# Draw lines as shapes (full height, yref=paper)
_lines = [
    (hist_var_static, "crimson",    "solid", f"Hist VaR<br>{hist_var_static:.2f}%",  0.92),
    (ewma_var_static, "darkorange", "dash",  f"EWMA VaR<br>{ewma_var_static:.2f}%",  0.70),
    (es_static,       "purple",     "dot",   f"ES<br>{es_static:.2f}%",               0.50),
]
for x_val, color, dash, label, y_anchor in _lines:
    fig2.add_shape(
        type="line",
        x0=x_val, x1=x_val, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=color, width=2, dash=dash),
    )
    fig2.add_annotation(
        x=x_val, y=y_anchor,
        xref="x", yref="paper",
        text=label,
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color,
        ax=-55, ay=0,
        font=dict(color=color, size=12, family="monospace"),
        bgcolor="white",
        bordercolor=color,
        borderwidth=1.5,
        borderpad=4,
        align="center",
    )

fig2.update_layout(
    height=380, template="plotly_white",
    xaxis_title="Daily Return (%)", yaxis_title="Frequency",
    showlegend=False, margin=dict(t=20, b=10),
)
st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# Exception table — date + return only
# ─────────────────────────────────────────────
st.subheader(f"Historical VaR Exceptions  ({confidence*100:.0f}% confidence)")

exc = exceptions(metrics, var_col="hist_var")

if exc.empty:
    st.success("No exceptions in this period.")
else:
    tbl = exc.copy()
    tbl.index = tbl.index.date
    tbl.index.name = "Date"
    tbl["Return (%)"] = (tbl["actual_ret"] * 100).map("{:.2f}%".format)
    tbl = tbl[["Return (%)"]].sort_index(ascending=False)
    st.dataframe(tbl, use_container_width=True, height=min(400, 35 + len(tbl) * 35))
    st.caption(f"{len(tbl)} exceptions  |  expected ~{(1-confidence)*len(metrics):.0f}")
