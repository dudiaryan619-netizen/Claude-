"""
Alternative XAUUSD intraday strategies on GC=F 5m bars.
Tested on the last 30 days AND the 30 days before (out-of-sample check).
Common: entry at bar close, SL+TP same bar -> SL, $0.40/oz cost, flat by 16:55 ET, 1 trade at a time.

 1 Asian-range breakout : range 18:00-02:00 ET; first 5m close outside it 03:00-08:00 -> trade.
                          SL = range mid (capped 1.5 ATR), TP 2R. 1 trade/day.
 2 EMA trend pullback   : trend up if EMA50>EMA200 and close>EMA200; price dips to touch EMA20
                          and closes back above with a bullish bar -> LONG (mirror short).
                          SL = 6-bar swing low, risk in [0.5,2] ATR, TP 2R. 03:00-15:00 ET.
 3 VWAP mean reversion  : close >= 2 sd from session VWAP and RSI(14) > 75 -> SHORT (mirror).
                          TP = VWAP, SL 1 ATR beyond entry. 03:00-15:00 ET.
 4 Prev-day H/L breakout: 5m close above previous session high -> LONG (mirror). SL 1.5 ATR,
                          TP 2R. 1 trade per side per day.
 5 NY-only ORB + flow   : 08:20 ET 15-min OR, close outside + order-flow confirm (vol 1.5x),
                          SL OR mid, TP 2R (best variant from the earlier test).
"""
import numpy as np, pandas as pd, yfinance as yf

COST = 0.40


def load60():
    df = yf.download("GC=F", period="60d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_convert("America/New_York")
    c = df.Close
    tr = pd.concat([df.High - df.Low, (df.High - c.shift()).abs(), (df.Low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["sday"] = (df.index + pd.Timedelta(hours=6)).date          # session day starts 18:00 ET
    df["hm"] = df.index.hour * 60 + df.index.minute
    rng = (df.High - df.Low).replace(0, np.nan)
    df["delta"] = (df.Volume * ((c - df.Low) - (df.High - c)) / rng).fillna(0)
    df["vol_avg"] = df.Volume.rolling(20, min_periods=5).mean().shift()
    for n in (20, 50, 200): df[f"ema{n}"] = c.ewm(span=n).mean()
    tp = (df.High + df.Low + c) / 3
    g = df.groupby("sday")
    v = df.Volume.groupby(df.sday).cumsum().replace(0, np.nan)
    df["vwap"] = (tp * df.Volume).groupby(df.sday).cumsum() / v
    var = (tp ** 2 * df.Volume).groupby(df.sday).cumsum() / v - df.vwap ** 2
    df["vsd"] = np.sqrt(var.clip(lower=0))
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)
    df["cvd"] = df.delta.groupby(df.sday).cumsum()
    daily = g.agg(H=("High", "max"), L=("Low", "min"))
    prev = daily.shift()
    df["pdh"] = df.sday.map(prev.H); df["pdl"] = df.sday.map(prev.L)
    return df


def simulate(df, i, side, sl, tp, name):
    entry = df.Close.iloc[i]; risk = abs(entry - sl)
    if risk <= 0: return None, i + 1
    for j in range(i + 1, len(df)):
        c = df.iloc[j]
        if (c.Low <= sl) if side == 1 else (c.High >= sl): px, why = sl, "SL"; break
        if (c.High >= tp) if side == 1 else (c.Low <= tp): px, why = tp, "TP"; break
        if c.hm >= 16 * 60 + 55 and c.hm < 18 * 60 or c.sday != df.sday.iloc[i]:
            px, why = c.Open if c.sday != df.sday.iloc[i] else c.Close, "EOD"; break
    else:
        return None, len(df)
    pnl = side * (px - entry) - COST
    return dict(strategy=name, entry_time=df.index[i], side="LONG" if side == 1 else "SHORT",
                entry=round(entry, 2), sl=round(sl, 2), tp=round(tp, 2), exit=round(px, 2),
                reason=why, pnl_usd_oz=round(pnl, 2), R=round(pnl / risk, 2)), j + 1


def asian_breakout(df):
    out = []
    for day, d in df.groupby("sday"):
        asia = d[(d.hm >= 18 * 60) | (d.hm < 2 * 60)]
        if len(asia) < 50: continue
        hi, lo = asia.High.max(), asia.Low.min(); mid = (hi + lo) / 2
        win = d[(d.hm >= 3 * 60) & (d.hm < 8 * 60)]
        for ts, b in win.iterrows():
            s = 1 if b.Close > hi else -1 if b.Close < lo else 0
            if not s: continue
            i = df.index.get_loc(ts); risk = min(abs(b.Close - mid), 1.5 * b.atr)
            t, _ = simulate(df, i, s, b.Close - s * risk, b.Close + s * 2 * risk, "Asian breakout")
            if t: out.append(t)
            break
    return out


def loop(df, name, signal):
    out, i = [], 250
    while i < len(df) - 1:
        r = signal(df, i)
        if r is None: i += 1; continue
        t, i = simulate(df, i, *r, name)
        if t: out.append(t)
    return out


def ema_pullback(df, i):
    b = df.iloc[i]; p = df.iloc[i - 1]
    if not (3 * 60 <= b.hm < 15 * 60): return None
    if b.ema50 > b.ema200 and b.Close > b.ema200 and p.Low <= p.ema20 and b.Close > b.ema20 and b.Close > b.Open:
        sl = df.Low.iloc[i - 5:i + 1].min(); s = 1
    elif b.ema50 < b.ema200 and b.Close < b.ema200 and p.High >= p.ema20 and b.Close < b.ema20 and b.Close < b.Open:
        sl = df.High.iloc[i - 5:i + 1].max(); s = -1
    else: return None
    risk = float(np.clip(abs(b.Close - sl), 0.5 * b.atr, 2 * b.atr))
    return s, b.Close - s * risk, b.Close + s * 2 * risk


def vwap_revert(df, i):
    b = df.iloc[i]
    if not (3 * 60 <= b.hm < 15 * 60) or b.vsd <= 0: return None
    if b.Close >= b.vwap + 2 * b.vsd and b.rsi > 75: return -1, b.Close + b.atr, b.vwap
    if b.Close <= b.vwap - 2 * b.vsd and b.rsi < 25: return 1, b.Close - b.atr, b.vwap
    return None


def pd_breakout_factory():
    taken = set()
    def sig(df, i):
        b = df.iloc[i]
        if pd.isna(b.pdh) or (b.hm >= 16 * 60 and b.hm < 18 * 60): return None
        for s, lvl in ((1, b.pdh), (-1, b.pdl)):
            if (b.sday, s) in taken: continue
            if (s == 1 and b.Close > lvl) or (s == -1 and b.Close < lvl):
                taken.add((b.sday, s)); r = 1.5 * b.atr
                return s, b.Close - s * r, b.Close + s * 2 * r
        return None
    return sig


def ny_orb(df):
    out = []
    for day, d in df.groupby("sday"):
        orb = d[(d.hm >= 500) & (d.hm < 515)]
        if len(orb) < 3: continue
        hi, lo = orb.High.max(), orb.Low.min(); mid = (hi + lo) / 2
        cvd0 = d[d.hm >= 500].delta.cumsum()
        for ts, b in d[(d.hm >= 515) & (d.hm < 635)].iterrows():
            s = 1 if b.Close > hi else -1 if b.Close < lo else 0
            if not s: continue
            if not (np.sign(b.delta) == s and np.sign(cvd0[ts]) == s and s * (b.Close - b.vwap) > 0
                    and b.Volume >= 1.5 * b.vol_avg): continue
            risk = min(abs(b.Close - mid), 1.5 * b.atr)
            t, _ = simulate(df, df.index.get_loc(ts), s, b.Close - s * risk, b.Close + s * 2 * risk, "NY ORB + flow")
            if t: out.append(t)
            break
    return out


def summary(g):
    w = (g.pnl_usd_oz > 0).sum(); eq = g.R.cumsum()
    return pd.Series({"trades": len(g), "win%": round(100 * w / len(g), 1), "totalR": round(g.R.sum(), 2),
                      "avgR": round(g.R.mean(), 2), "PF": round(g[g.pnl_usd_oz > 0].pnl_usd_oz.sum() / -g[g.pnl_usd_oz <= 0].pnl_usd_oz.sum(), 2),
                      "maxDD_R": round((eq - eq.cummax()).min(), 2), "net$_1lot": round(100 * g.pnl_usd_oz.sum())})


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load60()
    t = pd.DataFrame(asian_breakout(df) + loop(df, "EMA trend pullback", ema_pullback)
                     + loop(df, "VWAP mean reversion", vwap_revert)
                     + loop(df, "Prev-day H/L breakout", pd_breakout_factory()) + ny_orb(df))
    cut = df.index.max().normalize() - pd.Timedelta(days=30)
    t["period"] = np.where(t.entry_time >= cut, "last30", "prior30")
    t.to_csv("other_strategies_trades.csv", index=False)
    print("data:", df.index.min(), "->", df.index.max(), " split at", cut)
    for p in ("last30", "prior30"):
        print(f"\n===== {p} =====")
        print(t[t.period == p].groupby("strategy").apply(summary).sort_values("totalR", ascending=False))
    print("\n===== full 60 days =====")
    print(t.groupby("strategy").apply(summary).sort_values("totalR", ascending=False))
