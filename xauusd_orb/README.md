# XAUUSD — Order Flow + Scalping + Opening Range Breakout (30-day backtest)

Run: `pip install yfinance pandas matplotlib && python backtest.py .`

**Data:** COMEX gold futures (GC=F), 5-minute bars, 2026-10-19 → 2026-11-16 (≈21 trading days),
used as a stand-in for spot XAUUSD. Gold fell from ~4196 to ~3859 over the period.
**Order flow is a proxy:** no tick or Level-2 data is available for free, so delta is estimated from
where each bar closes within its high–low range × volume, together with cumulative session delta,
session VWAP and a volume-spike filter.

## Rules
- Sessions: London (03:00 ET) and New York / COMEX open (08:20 ET). OR = first 15 min.
- Entry: a 5m bar closes outside the OR within 2h. To confirm, the bar delta, the cumulative delta and the side of VWAP must all agree with the breakout direction, and volume must be ≥1.2× its 20-bar average.
- Exit: SL at the OR midpoint (capped at 1.5×ATR), TP 1.5R, time stop at 2h. If a bar hits both the SL and the TP, it counts as a stop-out.
- Max 1 trade/session, $0.40/oz round-trip cost.

## Results (net of costs, 1 lot = 100 oz)
| Variant | Trades | Win % | Net R | PF | Net $ (1 lot) | Max DD $/oz |
|---|---|---|---|---|---|---|
| **ORB + Order Flow** | 38 | 39.5 | −4.92 | 0.89 | −$1,795 | −63.9 |
| — London only | 20 | 35.0 | −5.93 | 0.63 | −$2,888 | −51.5 |
| — New York only | 18 | 44.4 | +1.01 | 1.13 | +$1,093 | −33.1 |
| Plain ORB (no OF filter) | 41 | 41.5 | −0.38 | 0.99 | −$137 | −52.1 |

## Sensitivity (OF filter on, both sessions)
| OR min | RR | Vol× | Trades | Win % | Net R | PF |
|---|---|---|---|---|---|---|
| 15 | 1.5 | 1.2 | 38 | 39.5 | −4.92 | 0.89 |
| 15 | 2.0 | 1.5 | 33 | 48.5 | +9.37 | 1.57 |
| 30 | 2.0 | 1.5 | 30 | 46.7 | +9.78 | 1.51 |
| 30 | 1.0 | 1.2 | 35 | 51.4 | −0.86 | 0.98 |

## Takeaways
- With the default settings, the strategy loses a little (PF 0.89). The bar-based order-flow filter does not improve on plain ORB.
- New York breakouts were profitable; London breakouts lost money.
- A stricter volume filter (1.5×) with a 2R target looked best (PF ~1.5), but that comes from about 30 trades picked in-sample. Treat it as a hypothesis and test it on more data or a forward test before trusting it.
- For true order flow (bid/ask delta, footprint, absorption), use tick data from your broker or a CME feed.

Files: `trades.csv` (every trade), `results.json`, `equity_curve.png`.
