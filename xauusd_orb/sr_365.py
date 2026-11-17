"""S/R bounce + order flow on 365 days of spot XAUUSD (Dukascopy 5m, tick volume).
Account: $1,000 start, fixed $100 risk per trade (loss at stop = $100)."""
import sys, numpy as np, pandas as pd
from other_strategies import load_csv
from sr_orderflow import add_levels, run

START, RISK = 1000, 100
VARIANTS = {"Original (target 2x risk)": (2.0, 1.0), "High-win (target 1x, stop 2x)": (1.0, 2.0)}


def report(name, t):
    t = t.sort_values("entry_time").copy()
    t["usd"] = t.R * RISK
    bal = START + t.usd.cumsum()
    wins, losses = t[t.usd > 0], t[t.usd <= 0]
    streak = (t.usd <= 0).astype(int).groupby((t.usd > 0).cumsum()).sum().max()
    print(f"\n######## {name} ########")
    print(f"trades {len(t)} | win rate {100*len(wins)/len(t):.1f}% | avg win ${wins.usd.mean():.0f} | "
          f"avg loss ${losses.usd.mean():.0f} | realised RR 1:{wins.usd.mean()/-losses.usd.mean():.2f}")
    print(f"profit ${t.usd.sum():,.0f} | end balance ${bal.iloc[-1]:,.0f} | lowest balance ${bal.min():,.0f} | "
          f"max drawdown ${(bal - bal.cummax()).min():,.0f} | longest losing streak {streak}")
    t["month"] = t.entry_time.dt.strftime("%Y-%m")
    m = t.groupby("month").agg(trades=("usd", "size"), win_pct=("usd", lambda x: round(100 * (x > 0).mean(), 1)),
                               profit=("usd", lambda x: round(x.sum())))
    m["balance"] = START + m.profit.cumsum()
    print(m.to_string())
    print("by session:\n", t.groupby("session").agg(trades=("usd", "size"), win_pct=("usd", lambda x: round(100*(x>0).mean(),1)), profit=("usd", lambda x: round(x.sum()))).to_string())
    print("by side:\n", t.groupby("side").agg(trades=("usd", "size"), win_pct=("usd", lambda x: round(100*(x>0).mean(),1)), profit=("usd", lambda x: round(x.sum()))).to_string())
    return t


if __name__ == "__main__":
    df = add_levels(load_csv(sys.argv[1] if len(sys.argv) > 1 else "data/xauusd_5m_365d.csv"))
    print("data:", df.index.min(), "->", df.index.max(), "bars", len(df))
    out = []
    for name, (k, m) in VARIANTS.items():
        t = report(name, run(df, "bounce", True, k, m)); t["variant"] = name; out.append(t)
    pd.concat(out).to_csv("sr_365_trades.csv", index=False)
