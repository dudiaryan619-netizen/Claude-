"""
XAUUSD Order-Flow Scalping backtest (no ORB) - last 30 days, GC=F 5m bars.

Order flow (bar proxy): delta = volume * close-location-in-range; cumulative delta (CVD)
and VWAP reset each session day at 18:00 ET.

Rules (active hours 03:00-12:00 ET, one position at a time):
  LONG  when: close > VWAP  (buyers in control)
              CVD rising over last 6 bars (net buying)
              volume spike >= VOL_MULT x 20-bar avg
              strong buy bar: delta/volume >= IMB (close in top of bar)
  SHORT mirrored.
  Entry at bar close. SL = signal bar low (long) / high (short), risk clamped to
  [0.5, 1.5] x ATR. TP = RR x risk. Time stop MAX_HOLD bars (30 min).
  SL+TP in same bar -> SL. Cost $0.40/oz round trip.
"""
import sys, json
import numpy as np, pandas as pd
from backtest import load, add_features

RR, MAX_HOLD, VOL_MULT, IMB, COST = 1.5, 6, 1.5, 0.4, 0.40


def prepare(df):
    day = (df.index - pd.Timedelta(hours=18)).date        # futures day starts 18:00 ET
    tp = (df.High + df.Low + df.Close) / 3
    df["vwap"] = (tp * df.Volume).groupby(day).cumsum() / df.Volume.groupby(day).cumsum()
    df["cvd"] = df.delta.groupby(day).cumsum()
    df["cvd_slope"] = df.cvd - df.cvd.groupby(day).shift(6)
    df["imb"] = df.delta / df.Volume.replace(0, np.nan)
    return df


def backtest(df, rr=RR, vol_mult=VOL_MULT, imb=IMB):
    trades, i, n = [], 0, len(df)
    hrs = df.index.hour + df.index.minute / 60
    while i < n - 1:
        b = df.iloc[i]
        if not (3 <= hrs[i] < 12) or pd.isna(b.cvd_slope) or pd.isna(b.atr):
            i += 1; continue
        side = 0
        if b.Volume >= vol_mult * b.vol_avg:
            if b.Close > b.vwap and b.cvd_slope > 0 and b.imb >= imb: side = 1
            elif b.Close < b.vwap and b.cvd_slope < 0 and b.imb <= -imb: side = -1
        if side == 0:
            i += 1; continue
        entry = b.Close
        risk = entry - b.Low if side == 1 else b.High - entry
        risk = float(np.clip(risk, 0.5 * b.atr, 1.5 * b.atr))
        sl, tp = entry - side * risk, entry + side * rr * risk
        j_end = min(i + MAX_HOLD, n - 1)
        exit_px, reason, j = None, "time", i
        for j in range(i + 1, j_end + 1):
            c = df.iloc[j]
            if (c.Low <= sl) if side == 1 else (c.High >= sl): exit_px, reason = sl, "SL"; break
            if (c.High >= tp) if side == 1 else (c.Low <= tp): exit_px, reason = tp, "TP"; break
        if exit_px is None:
            exit_px = df.Close.iloc[j]
        pnl = side * (exit_px - entry) - COST
        trades.append(dict(entry_time=df.index[i], exit_time=df.index[j],
                           session="London" if hrs[i] < 8 else "NewYork",
                           side="LONG" if side == 1 else "SHORT", entry=round(entry, 2),
                           sl=round(sl, 2), tp=round(tp, 2), exit=round(exit_px, 2),
                           reason=reason, pnl_usd_oz=round(pnl, 2), R=round(pnl / risk, 2)))
        i = j + 1
    return pd.DataFrame(trades)


def summary(g):
    w = (g.pnl_usd_oz > 0).sum()
    return pd.Series({"trades": len(g), "wins": w, "losses": len(g) - w,
                      "win%": round(100 * w / len(g), 1), "totalR": round(g.R.sum(), 2),
                      "avgR": round(g.R.mean(), 2),
                      "avgWinR": round(g[g.R > 0].R.mean(), 2),
                      "avgLossR": round(g[g.R <= 0].R.mean(), 2),
                      "net$_1lot": round(100 * g.pnl_usd_oz.sum())})


if __name__ == "__main__":
    df = prepare(add_features(load()))
    t = backtest(df)
    t.to_csv("scalp_trades.csv", index=False)
    pd.set_option("display.width", 200)
    print("== Order-flow scalp, default (RR 1.5, vol 1.5x, imbalance 0.4) ==")
    print(summary(t).to_string())
    print(t.groupby("session").apply(summary)); print(t.groupby("side").apply(summary))
    print(t.reason.value_counts().to_dict())
    eq = t.R.cumsum(); print("max DD (R):", round((eq - eq.cummax()).min(), 2))
    print("\n== Sensitivity ==")
    rows = []
    for rr in (1.0, 1.5, 2.0):
        for vm in (1.5, 2.0):
            for im in (0.4, 0.6):
                s = summary(backtest(df, rr, vm, im))
                rows.append(dict(RR=rr, vol=vm, imb=im, trades=s.trades, win=s["win%"], totalR=s.totalR))
    print(pd.DataFrame(rows).to_string(index=False))
