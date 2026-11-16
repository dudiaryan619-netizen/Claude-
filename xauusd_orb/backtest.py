"""
XAUUSD Order-Flow + Scalping + Opening Range Breakout (ORB) backtest.

Data   : COMEX gold futures (GC=F) 5-minute bars from Yahoo Finance, used as a
         proxy for spot XAUUSD (intraday moves are near-identical; absolute price
         differs by the futures basis). Timestamps in America/New_York.
Order flow (bar-based proxy -- no tick / L2 data is available for free):
   delta      = volume * ((close-low) - (high-close)) / (high-low)   (buy vs sell pressure)
   cum delta  = running sum of delta since session open
   VWAP       = session VWAP
   vol spike  = bar volume / 20-bar average volume

Rules (per session: London 03:00 ET, New York 08:20 ET COMEX open)
   1. Opening range = high/low of the first OR_MIN minutes of the session.
   2. Within ENTRY_WINDOW minutes after the OR, a 5m bar CLOSES beyond the range.
   3. Order-flow confirmation (long; mirrored for short):
        bar delta > 0, cum session delta > 0, close > VWAP, volume >= VOL_MULT x avg.
   4. Scalp exit: SL = OR midpoint (capped to MAX_RISK_ATR x ATR), TP = RR x risk,
      time stop after MAX_HOLD bars. If SL and TP hit in the same bar -> SL (worst case).
   5. Max 1 trade per session. Costs: COST $/oz round trip (spread+commission).
"""
import sys, json
import numpy as np, pandas as pd, yfinance as yf

DAYS = 30
OR_MIN = 15
ENTRY_WINDOW = 120
RR = 1.5
MAX_HOLD = 24            # 24 x 5m = 2h
MAX_RISK_ATR = 1.5
VOL_MULT = 1.2
COST = 0.40              # $/oz round trip
SESSIONS = {"London": "03:00", "NewYork": "08:20"}


def load(days=DAYS):
    df = yf.download("GC=F", period="60d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_convert("America/New_York")
    start = df.index.max().normalize() - pd.Timedelta(days=days)
    return df[df.index >= start].copy()


def add_features(df):
    rng = (df.High - df.Low).replace(0, np.nan)
    df["delta"] = (df.Volume * ((df.Close - df.Low) - (df.High - df.Close)) / rng).fillna(0)
    df["vol_avg"] = df.Volume.rolling(20, min_periods=5).mean().shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(),
                    (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def backtest(df, use_orderflow=True):
    trades = []
    for day in sorted(set(df.index.date)):
        if pd.Timestamp(day).weekday() >= 5:
            continue
        for sname, t0 in SESSIONS.items():
            s0 = pd.Timestamp(f"{day} {t0}").tz_localize("America/New_York")
            or_end = s0 + pd.Timedelta(minutes=OR_MIN)
            win_end = or_end + pd.Timedelta(minutes=ENTRY_WINDOW)
            orb = df[(df.index >= s0) & (df.index < or_end)]
            if len(orb) < OR_MIN // 5:
                continue
            hi, lo = orb.High.max(), orb.Low.min()
            mid = (hi + lo) / 2
            sess = df[df.index >= s0].iloc[: 400].copy()
            pv = (sess.High + sess.Low + sess.Close) / 3 * sess.Volume
            sess["vwap"] = pv.cumsum() / sess.Volume.cumsum().replace(0, np.nan)
            sess["cumdelta"] = sess.delta.cumsum()
            cand = sess[(sess.index >= or_end) & (sess.index < win_end)]
            for ts, b in cand.iterrows():
                side = 1 if b.Close > hi else -1 if b.Close < lo else 0
                if side == 0:
                    continue
                if use_orderflow:
                    ok = (np.sign(b.delta) == side and np.sign(b.cumdelta) == side
                          and side * (b.Close - b.vwap) > 0
                          and b.Volume >= VOL_MULT * b.vol_avg)
                    if not ok:
                        continue
                entry = b.Close
                risk = abs(entry - mid)
                risk = min(risk, MAX_RISK_ATR * b.atr) if b.atr > 0 else risk
                if risk <= 0:
                    continue
                sl = entry - side * risk
                tp = entry + side * RR * risk
                after = sess[sess.index > ts].iloc[:MAX_HOLD]
                exit_px, exit_ts, reason = None, None, "time"
                for ts2, c in after.iterrows():
                    hit_sl = (c.Low <= sl) if side == 1 else (c.High >= sl)
                    hit_tp = (c.High >= tp) if side == 1 else (c.Low <= tp)
                    if hit_sl:
                        exit_px, exit_ts, reason = sl, ts2, "SL"; break
                    if hit_tp:
                        exit_px, exit_ts, reason = tp, ts2, "TP"; break
                if exit_px is None:
                    if after.empty:
                        break
                    exit_px, exit_ts = after.Close.iloc[-1], after.index[-1]
                pnl = side * (exit_px - entry) - COST
                trades.append(dict(session=sname, entry_time=ts, exit_time=exit_ts,
                                   side="LONG" if side == 1 else "SHORT",
                                   or_high=round(hi, 2), or_low=round(lo, 2),
                                   entry=round(entry, 2), sl=round(sl, 2), tp=round(tp, 2),
                                   exit=round(exit_px, 2), reason=reason,
                                   pnl_usd_oz=round(pnl, 2), R=round(pnl / risk, 2)))
                break  # one trade per session
    return pd.DataFrame(trades)


def stats(t):
    if t.empty:
        return {"trades": 0}
    eq = t.pnl_usd_oz.cumsum()
    wins, losses = t[t.pnl_usd_oz > 0], t[t.pnl_usd_oz <= 0]
    return {
        "trades": len(t),
        "win_rate_%": round(100 * len(wins) / len(t), 1),
        "net_usd_per_oz": round(t.pnl_usd_oz.sum(), 2),
        "net_usd_1lot_100oz": round(100 * t.pnl_usd_oz.sum(), 0),
        "total_R": round(t.R.sum(), 2),
        "avg_R": round(t.R.mean(), 3),
        "profit_factor": round(wins.pnl_usd_oz.sum() / -losses.pnl_usd_oz.sum(), 2) if len(losses) and losses.pnl_usd_oz.sum() < 0 else float("inf"),
        "max_drawdown_usd_oz": round((eq - eq.cummax()).min(), 2),
        "avg_win": round(wins.pnl_usd_oz.mean(), 2) if len(wins) else 0,
        "avg_loss": round(losses.pnl_usd_oz.mean(), 2) if len(losses) else 0,
        "exits": t.reason.value_counts().to_dict(),
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    df = add_features(load())
    print(f"Data: GC=F 5m  {df.index.min()} -> {df.index.max()}  bars={len(df)}  "
          f"price {df.Close.iloc[0]:.1f} -> {df.Close.iloc[-1]:.1f}")
    res = {}
    for name, of in [("ORB + OrderFlow (strategy)", True), ("Plain ORB (no order-flow filter)", False)]:
        t = backtest(df, of)
        res[name] = {"overall": stats(t),
                     "by_session": {s: stats(g) for s, g in t.groupby("session")} if len(t) else {}}
        if of:
            t.to_csv(f"{out}/trades.csv", index=False)
            main = t
        else:
            base = t
    print(json.dumps(res, indent=2, default=str))
    json.dump(res, open(f"{out}/results.json", "w"), indent=2, default=str)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    for t, lab in [(main, "ORB + Order Flow"), (base, "Plain ORB")]:
        if len(t):
            ax.plot(t.exit_time, 100 * t.pnl_usd_oz.cumsum(), marker="o", ms=3, label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("XAUUSD (GC=F 5m) ORB backtest - last 30 days, 1 lot = 100 oz, net of costs")
    ax.set_ylabel("Cumulative P&L (USD)"); ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f"{out}/equity_curve.png")
