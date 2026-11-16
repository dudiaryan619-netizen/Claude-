"""
XAUUSD pure Order-Flow strategies - last 30 days, GC=F 5m bars (bar-based delta proxy).
No ORB, no VWAP, no scalp time-stop. Protective stop only; positions flat at 16:55 ET.

 A) CVD Trend      : fast EMA(8) of CVD crosses slow EMA(21) -> enter in that direction.
                     Exit on the opposite cross (order flow flips). SL 2 x ATR.
 B) Delta Divergence: price closes at a 12-bar high but CVD is below its 12-bar high
                     (buying not confirming) -> SHORT; mirror -> LONG.
                     SL 1.5 x ATR, TP 2R.
 C) Absorption     : volume >= 1.5x avg, bar range <= 0.8 x ATR, 12-bar low made in last 3 bars,
                     negative delta (sellers absorbed) -> LONG; mirror -> SHORT.
                     SL 1.5 x ATR, TP 2R.
CVD is continuous (not reset). SL+TP same bar -> SL. Cost $0.40/oz round trip.
"""
import numpy as np, pandas as pd
from backtest import load, add_features

COST, LB = 0.40, 12


def prepare(df):
    df["cvd"] = df.delta.cumsum()
    df["f"] = df.cvd.ewm(span=8).mean(); df["s"] = df.cvd.ewm(span=21).mean()
    up = (df.f > df.s).astype(int) - (df.f <= df.s).astype(int)
    df["cross"] = up.where(up != up.shift(), 0)
    ph, ch = df.Close.rolling(LB).max().shift(), df.cvd.rolling(LB).max().shift()
    pl, cl = df.Close.rolling(LB).min().shift(), df.cvd.rolling(LB).min().shift()
    df["div"] = np.where((df.Close > ph) & (df.cvd < ch), -1,
                np.where((df.Close < pl) & (df.cvd > cl), 1, 0))
    big = (df.Volume >= 1.5 * df.vol_avg) & ((df.High - df.Low) <= 0.8 * df.atr)
    lo3, hi3 = df.Low.rolling(3).min(), df.High.rolling(3).max()
    df["abs"] = np.where(big & (lo3 <= df.Low.rolling(LB).min()) & (df.delta < 0), 1,
                np.where(big & (hi3 >= df.High.rolling(LB).max()) & (df.delta > 0), -1, 0))
    return df


def run(df, sig, sl_atr, rr=None, exit_on_flip=False):
    trades, i, n = [], 0, len(df)
    hm = df.index.hour * 60 + df.index.minute
    flat = (hm >= 16 * 60 + 55) & (hm < 18 * 60)
    while i < n - 1:
        s = df[sig].iloc[i]
        if s == 0 or flat[i] or pd.isna(df.atr.iloc[i]):
            i += 1; continue
        entry, risk = df.Close.iloc[i], sl_atr * df.atr.iloc[i]
        sl = entry - s * risk; tp = entry + s * rr * risk if rr else None
        exit_px, reason = None, None
        for j in range(i + 1, n):
            c = df.iloc[j]
            if (c.Low <= sl) if s == 1 else (c.High >= sl): exit_px, reason = sl, "SL"; break
            if tp and ((c.High >= tp) if s == 1 else (c.Low <= tp)): exit_px, reason = tp, "TP"; break
            if exit_on_flip and df[sig].iloc[j] == -s: exit_px, reason = c.Close, "flip"; break
            if flat[j] or (df.index[j] - df.index[j - 1]) > pd.Timedelta(minutes=30):
                exit_px, reason = df.Close.iloc[j - 1] if not flat[j] else c.Close, "EOD"; break
        if exit_px is None: break
        pnl = s * (exit_px - entry) - COST
        trades.append(dict(entry_time=df.index[i], exit_time=df.index[j], side="LONG" if s == 1 else "SHORT",
                           entry=round(entry, 2), exit=round(exit_px, 2), reason=reason,
                           pnl_usd_oz=round(pnl, 2), R=round(pnl / risk, 2),
                           session="Asia" if df.index[i].hour >= 18 or df.index[i].hour < 3
                           else "London" if df.index[i].hour < 8 else "NewYork"))
        i = j if reason == "flip" else j + 1   # flip = stop and reverse
    return pd.DataFrame(trades)


def summary(g):
    w = (g.pnl_usd_oz > 0).sum(); eq = g.R.cumsum()
    return pd.Series({"trades": len(g), "wins": w, "win%": round(100 * w / len(g), 1),
                      "totalR": round(g.R.sum(), 2), "avgR": round(g.R.mean(), 2),
                      "avgWinR": round(g[g.R > 0].R.mean(), 2), "avgLossR": round(g[g.R <= 0].R.mean(), 2),
                      "maxDD_R": round((eq - eq.cummax()).min(), 2),
                      "net$_1lot": round(100 * g.pnl_usd_oz.sum())})


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = prepare(add_features(load()))
    strategies = {"A_CVD_Trend": run(df, "cross", 2.0, exit_on_flip=True),
                  "B_Delta_Divergence": run(df, "div", 1.5, rr=2),
                  "C_Absorption": run(df, "abs", 1.5, rr=2)}
    print(pd.DataFrame({k: summary(v) for k, v in strategies.items()}).T)
    for k, t in strategies.items():
        t.to_csv(f"of_{k}.csv", index=False)
        print(f"\n--- {k}  exits={t.reason.value_counts().to_dict()}")
        print(t.groupby("session").apply(summary)[["trades", "win%", "totalR", "avgR"]])
        print(t.groupby("side").apply(summary)[["trades", "win%", "totalR", "avgR"]])
