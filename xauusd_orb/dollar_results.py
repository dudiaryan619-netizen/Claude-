"""Account simulation: $1,000 start, fixed $100 risk per trade (loss at stop = $100)."""
import pandas as pd
import highwin_search as H   # re-uses entries + sim()

START, RISK = 1000, 100
CFG = [("NY ORB + flow", 1.0, 1.0), ("NY ORB + flow", 0.75, 1.0), ("NY ORB + flow", 1.0, 2.0),
       ("S/R bounce + OF", 1.0, 2.0), ("VWAP mean reversion", 0.5, 1.5),
       ("VWAP mean reversion", 0.5, 2.0), ("Asian breakout", 2.0, 1.0)]
rows = []
for strat, k, m in CFG:
    g = H.E[H.E.strategy == strat].sort_values("entry_time")
    for per, sub in (("Last 30 days", g[g.entry_time >= H.cut]), ("Previous 30 days", g[g.entry_time < H.cut])):
        bal, low, wins, n = START, START, 0, 0
        for r in sub.itertuples():
            x = H.sim(r, k, m)
            if not x: continue
            pnl = x[0] * RISK; bal += pnl; low = min(low, bal); n += 1; wins += pnl > 0
        rows.append(dict(strategy=f"{strat} (TP {k}x / SL {m}x)", period=per, trades=n,
                         win_pct=round(100 * wins / n, 1), profit=round(bal - START),
                         end_balance=round(bal), lowest_balance=round(low)))
out = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(out.to_string(index=False)); out.to_csv("dollar_results.csv", index=False)
