# file: fe_behavior.py  -> 生成 outputs/behavior_features.csv（按 event_id 对齐）
import numpy as np, pandas as pd
from pathlib import Path

DATA_DIR = Path(r"D:\chan\chan.py\ml_dataset\20250902_194053")  # 有 train/valid/test.parquet
KLINE_DIR = Path(r"D:\chan\chan.py\data\stooq\1d_15y")     # 放各品种K线：AAPL_1d.parquet 之类
OUT = Path(r"D:\chan\chan.py\ml_dataset\outputs"); OUT.mkdir(exist_ok=True, parents=True)

def load_kline(symbol, timeframe):
    # 你可以按自己的命名规则改这行
    path = KLINE_DIR / f"{symbol}_1d_15y.parquet"
    df = pd.read_parquet(path).copy()

    # 如果日期在索引上，拉出来；否则在列里找时间列
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "timestamp"})

    # 建一个小写映射，方便匹配不同大小写/写法
    lower2orig = {c.lower(): c for c in df.columns}

    # 依次尝试常见的时间列名
    time_candidates = ["timestamp", "datetime", "date", "time", "dt"]
    time_col = None
    for cand in time_candidates:
        if cand in lower2orig:
            time_col = lower2orig[cand]
            break

    if time_col is None:
        # 到这里说明既不在索引也没有常见列名；抛出可读性强的错误
        raise KeyError(
            f"No timestamp-like column found in {path}. "
            f"Got columns: {list(df.columns)}. "
            f"Expected one of {time_candidates} or a DatetimeIndex."
        )

    # 统一成 'timestamp'
    df["timestamp"] = pd.to_datetime(df[time_col])

    # 统一价格/成交量列名为小写：open, high, low, close, volume
    wanted = ["open", "high", "low", "close", "volume"]
    rename_map = {}
    for w in wanted:
        for cand in (w, w.capitalize(), w.upper()):
            if cand in df.columns:
                rename_map[cand] = w
                break
    df = df.rename(columns=rename_map)

    # 可选：确保这些列存在（给出更早、更清晰的错误）
    missing_ohlcv = [c for c in wanted if c not in df.columns]
    if missing_ohlcv:
        raise KeyError(
            f"Missing OHLCV columns in {path}: {missing_ohlcv}. "
            f"Got columns: {list(df.columns)}"
        )

    # 排序 + 重置索引
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def atr(df, n=20):
    cprev = df["close"].shift(1)
    tr = np.maximum(df["high"]-df["low"], np.maximum(abs(df["high"]-cprev), abs(df["low"]-cprev)))
    return tr.rolling(n, min_periods=1).mean()

def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()

def max_drawdown(close, lookback=100):
    roll_max = close.rolling(lookback, min_periods=1).max()
    dd = close/roll_max - 1.0
    return dd.rolling(lookback, min_periods=1).min()

def build_for_split(df_split):
    # 预加载/缓存每个 (symbol,timeframe) 的K线
    cache = {}
    rows = []
    for i, row in df_split.iterrows():
        sid = row["event_id"]
        sym = row["symbol"]
        tf  = row["timeframe"] if "timeframe" in row else "1d"
        t   = pd.to_datetime(row["t_entry_ts"])  # 或 t_entry_ts

        key = (sym, tf)
        if key not in cache:
            k = load_kline(sym, tf)
            k["ret1"] = k["close"].pct_change()
            k["atr14"] = atr(k, 14)
            k["atr20"] = atr(k, 20)
            k["ema20"] = ema(k["close"], 20)
            k["ema60"] = ema(k["close"], 60)
            k["vol_z20"] = (k["volume"] - k["volume"].rolling(20,1).mean()) / (k["volume"].rolling(20,1).std(ddof=0)+1e-9)
            cache[key] = k

        k = cache[key]
        k2 = k[k["timestamp"] <= t]
        if len(k2) < 60:  # 太短无法算特征
            continue

        # === 行为/结构特征（只看 t_entry 之前） ===
        def tail(series, n): return series.tail(n)
        atr_now = k2["atr20"].iloc[-1]
        ema20_now = k2["ema20"].iloc[-1]
        ema60_now = k2["ema60"].iloc[-1]

        # 动量/趋势
        mom_20 = k2["close"].pct_change(20).iloc[-1]
        mom_60 = k2["close"].pct_change(60).iloc[-1]
        # 近20根回归斜率（标准化到 /ATR）
        y = tail(k2["close"], 20).reset_index(drop=True)
        x = np.arange(len(y)); x = (x - x.mean())
        slope = ( ((x*(y - y.mean())).sum()) / ( (x**2).sum() + 1e-9) ) / (atr_now + 1e-9)

        # 波动/回撤
        rv20 = tail(k2["ret1"], 20).std(ddof=0) * np.sqrt(20)
        dd_100 = max_drawdown(k2["close"], 100).iloc[-1]
        atr_trend = (atr_now / (tail(k2["atr20"], 60).mean() + 1e-9))  # ATR 上/下行

        # 价量异常/背离
        volz20 = k2["vol_z20"].iloc[-1]
        # 近20根：收益与成交量变化的相关系数（背离时为负）
        vchg = tail(k2["volume"].pct_change(), 20)
        r20  = tail(k2["ret1"], 20)
        corr_pr_vol = r20.corr(vchg)

        # 与均线的偏离（归一到ATR）
        dev_ema20 = (k2["close"].iloc[-1] - ema20_now) / (atr_now + 1e-9)
        dev_ema60 = (k2["close"].iloc[-1] - ema60_now) / (atr_now + 1e-9)

        # K线形态粗粒度（近20根平均上下影线占比）
        hl = tail(k2["high"],20) - tail(k2["low"],20)
        up_wick = tail(k2["high"],20) - np.maximum(tail(k2["open"],20), tail(k2["close"],20))
        lo_wick = np.minimum(tail(k2["open"],20), tail(k2["close"],20)) - tail(k2["low"],20)
        up_wick_ratio = (up_wick/(hl+1e-9)).mean()
        lo_wick_ratio = (lo_wick/(hl+1e-9)).mean()

        rows.append({
            "event_id": sid,
            # 动量/趋势
            "mom20": mom_20, "mom60": mom_60, "slope20_over_atr": slope,
            # 波动/回撤
            "rv20": rv20, "atr_trend": atr_trend, "dd_100": dd_100,
            # 价量
            "volz20": volz20, "corr_price_vol20": corr_pr_vol,
            # 结构/位置
            "dev_ema20_atr": dev_ema20, "dev_ema60_atr": dev_ema60,
            # 形态
            "up_wick_ratio20": up_wick_ratio, "lo_wick_ratio20": lo_wick_ratio,
        })
    return pd.DataFrame(rows)

def main():
    # 把 train/valid/test 拼一起算特征，再按 event_id 回写
    dfs = []
    for name in ["train","valid","test"]:
        df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
        dfs.append(df[["event_id","symbol","timeframe","t_entry_ts"]].copy())
    base = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates("event_id")
    feats = build_for_split(base)
    feats.to_csv(OUT/"behavior_features.csv", index=False)
    print("✅ saved:", (OUT/"behavior_features.csv").resolve())

if __name__ == "__main__":
    main()
