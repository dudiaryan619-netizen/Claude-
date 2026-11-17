"""Download 1-minute XAUUSD BID candles from Dukascopy's free datafeed and save 5-minute bars.
Usage: python fetch_dukascopy.py <days> <out.csv>"""
import sys, lzma, struct, datetime as dt, concurrent.futures as cf, urllib.request
import pandas as pd

def day(d):
    url = f"https://datafeed.dukascopy.com/datafeed/XAUUSD/{d.year}/{d.month-1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for _ in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=30).read()
            break
        except Exception:
            continue
    else:
        return []
    if not raw: return []
    data = lzma.decompress(raw); base = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    rows = []
    for k in range(len(data) // 24):
        t, o, c, l, h, v = struct.unpack(">5if", data[k*24:(k+1)*24])
        if v > 0:
            rows.append((base + dt.timedelta(seconds=t), o/1000, h/1000, l/1000, c/1000, v))
    return rows

if __name__ == "__main__":
    days, out = int(sys.argv[1]), sys.argv[2]
    end = dt.date.today(); dates = [end - dt.timedelta(days=i) for i in range(days, 0, -1)]
    with cf.ThreadPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(day, dates) for r in rs]
    m1 = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close", "Volume"]).set_index("time").sort_index()
    m5 = m1.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    m5.to_csv(out); print(f"1m bars {len(m1)}  5m bars {len(m5)}  {m5.index.min()} -> {m5.index.max()}")
