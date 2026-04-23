# Ticker Correlation Analysis — Session Context

## Source Document
**File:** `019db8ec-Risk_Applicant_Assessment_Final1.pdf`  
**Context:** Optiver Risk Applicant Assessment — fictional P&L scenario table for options/futures positions.

---

## Tickers Extracted from Document

| Label | Description | yfinance Ticker | Notes |
|---|---|---|---|
| A50 | FTSE China A50 Index | `000016.SS` | SSE 50 — best available proxy |
| NK225 | Nikkei 225 | `^N225` | Direct |
| Kospi | KOSPI 200 | `^KS200` | Direct |
| Kosdaq | KOSDAQ 150 | `^KQ11` | KOSDAQ Composite — proxy (150 not on yfinance) |
| DowJones | Dow Jones Industrial Average | `^DJI` | Direct |
| Nasdaq | NASDAQ 100 | `^NDX` | Direct |
| IronOre | TSI Platts Iron Ore | `TIO=F` | SGX futures — may have limited history (~5-6 yrs) |
| CrudeOil | WTI Crude Oil | `CL=F` | Front-month rolling contract |
| BrentOil | ICE Brent Crude | `BZ=F` | Front-month rolling contract |
| Gold | Comex Gold | `GC=F` | Front-month rolling contract |

---

## Analysis Plan (Fully Agreed)

### Part 1 — Normal Analysis (10-Year Period: Apr 2015 → Apr 2025)
- **1.1 Extreme Moves** — Hard % thresholds: ±5%, ±15%, ±30%, ±50%, ±75% (matching PDF scenario buckets). Count frequency of days exceeding each threshold, show top 5 up/down days per ticker.
- **1.2 Rolling Volatility** — 30-day and 90-day annualized. Plot all tickers with COVID period shaded. Summary table of mean/peak vol.
- **1.3 Correlation Matrix** — Full 10-year pairwise correlation heatmap. Print top positive and negative pairs.

### Part 2 — COVID-19 Stress Analysis (2020)
Sub-periods:
- **Pre-crash:** Jan 1 → Feb 14, 2020
- **Crash:** Feb 14 → Mar 23, 2020 (peak-to-trough)
- **Recovery:** Mar 23 → Dec 31, 2020

Analyses:
- **2.1 Drawdowns** — Per-ticker drawdown chart from Jan 1 2020 highs. Max drawdown summary table.
- **2.2 Extreme Moves** — Same ±% thresholds applied to COVID full year + crash window only. Normalized frequency (moves per 100 days) vs full period baseline.
- **2.3 Volatility Spikes** — 30d rolling vol from Oct 2019–Dec 2020, crash window shaded. Pre-crash mean vs peak crash vol comparison table (vol multiplier).
- **2.4 Correlation Regime Shift** — Side-by-side heatmaps for Pre/Crash/Recovery. Delta heatmap (crash − pre) to show correlation increases.

### Deferred (to do later)
- Statistical thresholds approach for extreme moves (2σ / 3σ based on each ticker's own distribution) — **user asked to be reminded about this**

---

## Technical Decisions
- **Output:** Jupyter notebook (`.ipynb`)
- **Library:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Returns:** Daily pct_change, pairwise dropna for mixed trading calendars
- **Volatility:** Annualized (`std * sqrt(252) * 100`)
- **Charts saved as:** PNG files alongside the notebook

---

## Known Data Caveats
1. **Iron Ore (`TIO=F`)** — SGX futures limited yfinance history, may not reach full 10 years
2. **KOSDAQ** — Using composite `^KQ11` as proxy for KOSDAQ 150 (no direct ticker)
3. **Commodities futures** — Front-month roll causes occasional artificial price jumps (fine for returns analysis)
4. **Mixed calendars** — Asian vs US trading holidays handled with pairwise dropna in correlation

---

## Current State
- Branch: `claude/ticker-correlation-analysis-SwLHj`
- Repo: `pardeshishivam/projects` at `/home/user/Projects`
- **Notebook NOT yet written** — session timed out before file was created
- Nothing committed yet

## Next Step
Write the Jupyter notebook at:
```
/home/user/Projects/ticker_correlation_analysis.ipynb
```
Then commit and push to `claude/ticker-correlation-analysis-SwLHj`.

---

## Prompt to Resume in New Session

> I have a pre-planned Jupyter notebook analysis to build. See `session_context.md` in `/home/user/Projects` for full context. Summary:
>
> - 10 tickers from an Optiver risk doc (A50, NK225, Kospi, Kosdaq, DowJones, Nasdaq, IronOre, CrudeOil, BrentOil, Gold) with yfinance tickers mapped
> - Build `/home/user/Projects/ticker_correlation_analysis.ipynb` with:
>   - Part 1 (10-year): extreme moves (hard % thresholds ±5/15/30/50/75%), rolling vol (30d+90d), correlation heatmap
>   - Part 2 (COVID 2020): drawdowns, extreme moves comparison, vol spikes, 3-regime correlation shift (pre/crash/recovery)
> - Branch: `claude/ticker-correlation-analysis-SwLHj`, commit and push when done
> - Remind user later: statistical threshold approach (2σ/3σ) for extreme moves is deferred
