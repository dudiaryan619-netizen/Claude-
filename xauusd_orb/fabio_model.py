"""
Approximation of Fabio Valentini's publicly described order-flow model, applied to XAUUSD
(GC=F 5m bars, bar-based delta proxy -- NOT real footprint/big-trade data).

Session: New York only, 08:30-12:00 ET.  Daily rules: max 3 losses -> stop for the day.
1. Bias      : long bias if EMA50 > EMA200 (15m-ish structure) AND close > session VWAP; short mirror.
2. Location  :
     Trend model (imbalance): price pulls back into VWAP..VWAP+1sd (long) and holds.
     Mean-reversion model (balance): price at/beyond VWAP -/+ 2sd (fade back toward value),
                                     taken in either direction regardless of trend.
3. Aggression: confirmation bar with volume >= 1.5x avg and delta/volume >= 0.4 in trade direction
               ("big aggressive orders" proxy).
Risk: SL beyond confirmation bar (risk clamped 0.5-1.5 ATR), TP 2R, stop to breakeven after +1R.
Cost $0.40/oz. SL+TP in same bar -> SL.
"""
import numpy as np, pandas as pd
from other_strategies import load60

VOL, IMB, RR, COST = 1.5, 0.4, 2.0, 0.40


def signal(b):
    if b.Volume < VOL * b.vol_avg or b.vsd <= 0: return 0, None
    imb = b.delta / b.Volume if b.Volume else 0
    up, dn = b.ema50 > b.ema200, b.ema50 < b.ema200
    if up and b.Close > b.vwap and b.Low <= b.vwap + b.vsd and imb >= IMB: return 1, "trend"
    if dn and b.Close < b.vwap and b.High >= b.vwap - b.vsd and imb <= -IMB: return -1, "trend"
    if b.Low <= b.vwap - 2 * b.vsd and imb >= IMB: return 1, "meanrev"
    if b.High >= b.vwap + 2 * b.vsd and imb <= -IMB: return -1, "meanrev"
    return 0, None


def run(df):
    trades, losses, i, n = [], {}, 250, len(df)
    while i < n - 1:
        b = df.iloc[i]
        if not (510 <= b.hm < 720) or losses.get(b.sday, 0) >= 3:
            i += 1; continue
        s, model = signal(b)
        if not s: i += 1; continue
        entry = b.Close
        risk = float(np.clip(entry - b.Low if s == 1 else b.High - entry, 0.5 * b.atr, 1.5 * b.atr))
        sl, tp, be = entry - s * risk, entry + s * RR * risk, False
        for j in range(i + 1, n):
            c = df.iloc[j]
            if (c.Low <= sl) if s == 1 else (c.High >= sl): px, why = sl, "BE" if be else "SL"; break
            if (c.High >= tp) if s == 1 else (c.Low <= tp): px, why = tp, "TP"; break
            if not be and ((c.High >= entry + risk) if s == 1 else (c.Low <= entry - risk)):
                sl, be = entry, True
            if c.hm >= 16 * 60 or c.sday != b.sday: px, why = c.Close, "EOD"; break
        pnl = s * (px - entry) - COST
        if pnl < 0: losses[b.sday] = losses.get(b.sday, 0) + 1
        trades.append(dict(entry_time=df.index[i], model=model, side="LONG" if s == 1 else "SHORT",
                           entry=round(entry, 2), exit=round(px, 2), reason=why,
                           pnl_usd_oz=round(pnl, 2), R=round(pnl / risk, 2)))
        i = j + 1
    return pd.DataFrame(trades)


def summary(g):
    w = (g.pnl_usd_oz > 0).sum(); eq = g.R.cumsum()
    return pd.Series({"trades": len(g), "wins": w, "win%": round(100 * w / len(g), 1),
                      "totalR": round(g.R.sum(), 2), "avgR": round(g.R.mean(), 2),
                      "avgWinR": round(g[g.R > 0].R.mean(), 2), "avgLossR": round(g[g.R <= 0].R.mean(), 2),
                      "maxDD_R": round((eq - eq.cummax()).min(), 2), "net$_1lot": round(100 * g.pnl_usd_oz.sum())})


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load60(); t = run(df)
    cut = df.index.max().normalize() - pd.Timedelta(days=30)
    t["period"] = np.where(t.entry_time >= cut, "last30", "prior30")
    t.to_csv("fabio_model_trades.csv", index=False)
    print("exits:", t.reason.value_counts().to_dict())
    print(t.groupby("period").apply(summary)); print(summary(t).rename("all60").to_frame().T)
    print(t.groupby(["period", "model"]).apply(summary)[["trades", "win%", "totalR", "avgR"]])
