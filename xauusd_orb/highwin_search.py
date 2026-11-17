"""
High-win-rate search on XAUUSD (GC=F 5m). Re-uses the ENTRY signals from earlier strategies
and re-simulates exits with a closer take-profit / wider stop (TP = k x risk, SL = m x risk).
A config 'passes' only if: last-30-day win rate >= 70% AND positive total R after costs.
Each passing config is then checked on the prior 30 days (not used for selection).
"""
import numpy as np, pandas as pd
from other_strategies import load60

COST = 0.40
df = load60()
cut = df.index.max().normalize() - pd.Timedelta(days=30)
idx = {ts: k for k, ts in enumerate(df.index)}
H, L, C, HM, SD = df.High.values, df.Low.values, df.Close.values, df.hm.values, df.sday.values

src = []
o = pd.read_csv("other_strategies_trades.csv", parse_dates=["entry_time"])
src.append(o[["strategy", "entry_time", "side", "entry", "sl"]])
s = pd.read_csv("sr_orderflow_trades.csv", parse_dates=["entry_time"])
s = s[s.orderflow]; s["strategy"] = "S/R " + s.setup + " + OF"
src.append(s[["strategy", "entry_time", "side", "entry", "sl"]])
E = pd.concat(src)
E["entry_time"] = pd.to_datetime(E.entry_time, utc=True).dt.tz_convert("America/New_York")
E = E[E.entry_time.isin(idx)]


def sim(row, k, m):
    i = idx[row.entry_time]; sd = 1 if row.side == "LONG" else -1
    r0 = abs(row.entry - row.sl)
    if r0 <= 0: return None
    sl, tp = row.entry - sd * m * r0, row.entry + sd * k * r0
    for j in range(i + 1, len(df)):
        if (L[j] <= sl) if sd == 1 else (H[j] >= sl): px = sl; break
        if (H[j] >= tp) if sd == 1 else (L[j] <= tp): px = tp; break
        if HM[j] >= 960 or SD[j] != SD[i]: px = C[j]; break
    else: return None
    return (sd * (px - row.entry) - COST) / (m * r0), sd * (px - row.entry) - COST


rows = []
for strat, g in E.groupby("strategy"):
    for k in (0.3, 0.5, 0.75, 1.0):
        for m in (1.0, 1.5, 2.0):
            res = [(r.entry_time >= cut, *sim(r, k, m)) for r in g.itertuples() if sim(r, k, m)]
            d = pd.DataFrame(res, columns=["last", "R", "usd"])
            for per, x in (("last30", d[d["last"]]), ("prior30", d[~d["last"]])):
                if len(x):
                    rows.append(dict(strategy=strat, TP_R=k, SL_R=m, period=per, trades=len(x),
                                     win=round(100 * (x.usd > 0).mean(), 1), totalR=round(x.R.sum(), 2),
                                     net1lot=round(100 * x.usd.sum())))
R = pd.DataFrame(rows)
w = R.pivot_table(index=["strategy", "TP_R", "SL_R"], columns="period",
                  values=["trades", "win", "totalR", "net1lot"]).reset_index()
w.columns = ["_".join(c).strip("_") for c in w.columns]
w.to_csv("highwin_search.csv", index=False)
pas = w[(w.win_last30 >= 70) & (w.totalR_last30 > 0) & (w.trades_last30 >= 10)]
pd.set_option("display.width", 220)
cols = ["strategy", "TP_R", "SL_R", "trades_last30", "win_last30", "totalR_last30", "net1lot_last30",
        "trades_prior30", "win_prior30", "totalR_prior30", "net1lot_prior30"]
print(f"{len(w)} configs tested, {len(pas)} pass (>=70% win, >0R, >=10 trades in last 30d)\n")
print(pas.sort_values("totalR_last30", ascending=False)[cols].to_string(index=False))
print("\nany config >=70% win in BOTH periods and profitable in both:")
both = w[(w.win_last30 >= 70) & (w.win_prior30 >= 70) & (w.totalR_last30 > 0) & (w.totalR_prior30 > 0)]
print(both[cols].to_string(index=False) if len(both) else "none")
