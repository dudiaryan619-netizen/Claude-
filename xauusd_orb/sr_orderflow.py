"""
Order Flow + Support/Resistance on XAUUSD (GC=F 5m, bar-based delta proxy), 60 days,
split into last 30 days and prior 30 days (out-of-sample check).

Support / resistance levels (all known at the time - no look-ahead):
  * previous session high / low
  * Asian session high / low (18:00-02:00 ET), usable from 02:00 ET
  * confirmed swing highs / lows: 1-hour pivots (bar high/low is the extreme of +-12 bars),
    available only 12 bars after the pivot; last 10 kept
  * round numbers every $25

Setups (trading hours 03:00-12:00 ET, one trade at a time, max 3 losses/day):
  BOUNCE   : bar wicks into a level (within 0.2 ATR) and closes back on the other side
             -> trade away from the level.  SL = level -/+ 0.5 ATR (risk 0.5-1.5 ATR).
  BREAKOUT : bar closes > 0.2 ATR beyond a level it opened on the other side of
             -> trade in the breakout direction. SL = level -/+ 0.5 ATR.
  Order-flow filter: volume >= 1.5x 20-bar avg AND delta/volume >= 0.3 in trade direction
                     AND 6-bar CVD slope in trade direction.
  TP = 2R. Exit at 16:00 ET. SL+TP same bar -> SL. Cost $0.40/oz.
"""
import numpy as np, pandas as pd
from other_strategies import load60

RR, COST, TOL, PIV = 2.0, 0.40, 0.2, 12


def add_levels(df):
    df["cvd_slope"] = df.delta.groupby(df.sday).cumsum().groupby(df.sday).diff(6)
    df["imb"] = df.delta / df.Volume.replace(0, np.nan)
    asia = df[(df.hm >= 18 * 60) | (df.hm < 120)].groupby("sday").agg(AH=("High", "max"), AL=("Low", "min"))
    df["ah"] = df.sday.map(asia.AH).where((df.hm >= 120) & (df.hm < 18 * 60))
    df["al"] = df.sday.map(asia.AL).where((df.hm >= 120) & (df.hm < 18 * 60))
    w = 2 * PIV + 1
    ph = df.High.where(df.High == df.High.rolling(w, center=True).max())
    pl = df.Low.where(df.Low == df.Low.rolling(w, center=True).min())
    df["ph_conf"] = ph.shift(PIV); df["pl_conf"] = pl.shift(PIV)   # known only PIV bars later
    return df


def levels_at(df, i, swings):
    b = df.iloc[i]
    lv = [b.pdh, b.pdl, b.ah, b.al] + swings
    base = np.floor(b.Close / 25) * 25
    lv += [base - 25, base, base + 25, base + 50]
    return [x for x in lv if pd.notna(x)]


def run(df, setup, use_of):
    trades, losses, swings, i, n = [], {}, [], 250, len(df)
    while i < n - 1:
        b = df.iloc[i]
        for col in ("ph_conf", "pl_conf"):
            if pd.notna(b[col]): swings = (swings + [b[col]])[-10:]
        if not (180 <= b.hm < 720) or losses.get(b.sday, 0) >= 3 or pd.isna(b.atr):
            i += 1; continue
        tol, s, lvl = TOL * b.atr, 0, None
        for L in levels_at(df, i, swings):
            if setup == "bounce":
                if b.Low <= L + tol and b.Close > L and b.Open > L: s, lvl = 1, L
                elif b.High >= L - tol and b.Close < L and b.Open < L: s, lvl = -1, L
            else:
                if b.Open < L and b.Close > L + tol: s, lvl = 1, L
                elif b.Open > L and b.Close < L - tol: s, lvl = -1, L
            if s: break
        if s and use_of:
            ok = (b.Volume >= 1.5 * b.vol_avg and s * b.imb >= 0.3 and s * b.cvd_slope > 0)
            if not ok: s = 0
        if not s: i += 1; continue
        entry = b.Close
        risk = float(np.clip(abs(entry - (lvl - s * 0.5 * b.atr)), 0.5 * b.atr, 1.5 * b.atr))
        sl, tp = entry - s * risk, entry + s * RR * risk
        for j in range(i + 1, n):
            c = df.iloc[j]
            for col in ("ph_conf", "pl_conf"):
                if pd.notna(c[col]): swings = (swings + [c[col]])[-10:]
            if (c.Low <= sl) if s == 1 else (c.High >= sl): px, why = sl, "SL"; break
            if (c.High >= tp) if s == 1 else (c.Low <= tp): px, why = tp, "TP"; break
            if c.hm >= 960 or c.sday != b.sday: px, why = c.Close, "EOD"; break
        pnl = s * (px - entry) - COST
        if pnl < 0: losses[b.sday] = losses.get(b.sday, 0) + 1
        trades.append(dict(entry_time=df.index[i], setup=setup, orderflow=use_of,
                           side="LONG" if s == 1 else "SHORT", level=round(lvl, 2),
                           session="London" if b.hm < 500 else "NewYork",
                           entry=round(entry, 2), sl=round(sl, 2), tp=round(tp, 2), exit=round(px, 2),
                           reason=why, pnl_usd_oz=round(pnl, 2), R=round(pnl / risk, 2)))
        i = j + 1
    return pd.DataFrame(trades)


def summary(g):
    w = (g.pnl_usd_oz > 0).sum(); eq = g.R.cumsum()
    return pd.Series({"trades": len(g), "wins": w, "win%": round(100 * w / len(g), 1),
                      "totalR": round(g.R.sum(), 2), "avgR": round(g.R.mean(), 2),
                      "maxDD_R": round((eq - eq.cummax()).min(), 2),
                      "net$_1lot": round(100 * g.pnl_usd_oz.sum())})


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = add_levels(load60())
    cut = df.index.max().normalize() - pd.Timedelta(days=30)
    t = pd.concat([run(df, s, of) for s in ("bounce", "breakout") for of in (True, False)])
    t["period"] = np.where(t.entry_time >= cut, "last30", "prior30")
    t["variant"] = t.setup + np.where(t.orderflow, " + OF", " (no OF)")
    t.to_csv("sr_orderflow_trades.csv", index=False)
    print("data:", df.index.min(), "->", df.index.max())
    for p in ("last30", "prior30"):
        print(f"\n== {p} ==\n", t[t.period == p].groupby("variant").apply(summary))
    print("\n== all 60 days ==\n", t.groupby("variant").apply(summary))
    of = t[t.orderflow]
    print("\n== with OF, by session (last30 / prior30) ==\n",
          of.groupby(["variant", "session", "period"]).apply(summary)[["trades", "win%", "totalR"]])
    print("\n== with OF, by side ==\n", of.groupby(["variant", "side", "period"]).apply(summary)[["trades", "win%", "totalR"]])
