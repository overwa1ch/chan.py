# -*- coding: utf-8 -*-
"""
zs_features.py
产出“贴近中枢”的结构/上下文/执行相关特征，按 event_id 对齐：
  - 输入：打过标签的事件 CSV（建议用你生成的 aapl_events_labeled.csv 模板）
          至少包含：event_id, symbol, timeframe(可无), t_entry_ts(或 t_entry), entry_price(可无)
  - 辅助：本地 K 线目录（与 behavior_features.py 的读取方式一致）
  - 输出：OUT/zs_features.csv  （仅包含 event_id 与若干特征列）
用法示例：
  python zs_features.py \
    --events_csv "/path/to/aapl_events_labeled.csv" \
    --kline_dir  "/path/to/stooq/1d_15y" \
    --out_dir    "/path/to/ml_dataset/outputs" \
    --kltype 1d --higher_kltype 60m --atr_n 20 --pre_L 60
"""

import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

# ====== 如需直接调用 chan.py，请保证仓库根目录在 PYTHONPATH ======
# import sys; sys.path.append("/path/to/chan.py/repo")
try:
    from Chan import CChan
    from ChanConfig import CChanConfig
    from Common.CEnum import KL_TYPE, DATA_SRC, AUTYPE
    CHAN_AVAILABLE = True
except Exception:
    # 如果没有装好 chan.py，也可以退化为“近似中枢”版本（不报错，输出空值，便于流程不中断）
    CHAN_AVAILABLE = False

# -----------------------
# 工具函数（与 behavior_features.py 风格保持一致）
# -----------------------
def atr(df, n=20):
    cprev = df["close"].shift(1)
    tr = np.maximum(df["high"]-df["low"],
                    np.maximum((df["high"]-cprev).abs(), (df["low"]-cprev).abs()))
    return tr.rolling(n, min_periods=1).mean()

def load_kline_one(symbol, kline_dir: Path, timeframe: str):
    """Load a single-symbol kline parquet robustly and normalize columns & timestamps."""
    from pathlib import Path
    import glob, re
    kline_dir = Path(kline_dir)

    # Normalize timeframe suffix
    tf = "1d_15y" if timeframe.lower() == "1d" else timeframe.lower()

    # Build symbol variants
    sym = str(symbol)
    variants = {
        sym,
        sym.upper(), sym.lower(),
        sym.replace("-", "_"), sym.replace("_", "-"),
        sym.upper().replace("-", "_"), sym.upper().replace("_", "-"),
        sym.lower().replace("-", "_"), sym.lower().replace("_", "-"),
    }

    # 1) Exact candidates
    for s in variants:
        p = kline_dir / f"{s}_{tf}.parquet"
        if p.exists():
            df = pd.read_parquet(p).copy()
            break
    else:
        # 2) Prefix glob (e.g., BRK -> BRK-A/BRK-B)
        g = glob.glob(str(kline_dir / f"{sym.upper()}*_{tf}.parquet"))
        if g:
            p = Path(sorted(g)[0])
            df = pd.read_parquet(p).copy()
        else:
            raise FileNotFoundError(str(kline_dir / f"{sym}_{tf}.parquet"))

    # ---- Normalize column names ----
    def norm_name(s):  # letters only, lowercase
        return re.sub(r'[^a-z]', '', str(s).lower())

    colmap = {c: norm_name(c) for c in df.columns}

    # Timestamp column detection (case-insensitive, flexible)
    ts_cands = {'timestamp','time','datetime','date','dt','t'}
    ts_col = None
    for orig, nm in colmap.items():
        if nm in ts_cands:
            ts_col = orig
            break
    if ts_col is None and hasattr(df.index, 'dtype'):
        # try index if datetime-like
        try:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index().rename(columns={df.columns[0]: 'timestamp'})
                ts_col = 'timestamp'
        except Exception:
            pass
    if ts_col is None:
        # give a helpful error
        raise KeyError("timestamp")

    # Rename timestamp to 'timestamp'
    if ts_col != 'timestamp':
        df = df.rename(columns={ts_col: 'timestamp'})

    # ---- Normalize OHLC columns ----
    # Build reverse lookup of normalized name -> original
    rev = {}
    for orig, nm in colmap.items():
        rev.setdefault(nm, orig)

    def pick(orig_names):
        # from a list of normalized candidates, return first present original name
        for nm in orig_names:
            if nm in rev:
                return rev[nm]
        return None

    open_col  = pick(['open','o'])
    high_col  = pick(['high','h'])
    low_col   = pick(['low','l'])
    close_col = pick(['close','adjclose','adjustedclose','closeadj','c'])

    rename_dict = {}
    if open_col and open_col != 'open':   rename_dict[open_col]  = 'open'
    if high_col and high_col != 'high':   rename_dict[high_col]  = 'high'
    if low_col  and low_col  != 'low':    rename_dict[low_col]   = 'low'
    if close_col and close_col != 'close':rename_dict[close_col] = 'close'
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Basic presence check
    for c in ['open','high','low','close']:
        if c not in df.columns:
            raise KeyError(f"missing required column: {c}")

    # Normalize timestamp to tz-naive
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    df["timestamp"] = ts
    if df["timestamp"].isna().any():
        raise ValueError("invalid timestamp values after normalization")

    return df

def to_kltype(s: str):
    s = s.lower()
    if s in ("1d","d","day","daily"): return KL_TYPE.K_DAY
    if s in ("60m","1h","60min"):     return KL_TYPE.K_60M
    if s in ("30m","30min"):          return KL_TYPE.K_30M
    if s in ("15m","15min"):          return KL_TYPE.K_15M
    if s in ("5m","5min"):            return KL_TYPE.K_5M
    raise ValueError(f"Unsupported timeframe: {s}")

# -----------------------
# 从 CChan / ZS 对象拿中枢
# -----------------------
def build_chan(symbol, start_date, end_date, lv_list, conf=None):
    if conf is None: conf = CChanConfig()
    ch = CChan(code=symbol, begin_time=start_date, end_time=end_date,
           data_src="custom:myParquetAPI.MyParquetAPI",
           lv_list=lv_list, autype=AUTYPE.QFQ, config=conf)
    return ch

def iter_lv_kl_lists(chan_obj):
    # 典型版本中，CChan.kl_datas[KL_TYPE] 是 KLine_List
    for lv, kl in getattr(chan_obj, "kl_datas", {}).items():
        yield lv, kl

def get_zs_list_from_kl(kl_obj):
    for attr in ["zs_list","zs","zs_lst"]:
        if hasattr(kl_obj, attr):
            return getattr(kl_obj, attr)
    return None

def pick_active_zs(zs_list, t_event):
    """t_event 时刻最近的（起点不晚于 t_event）的中枢"""
    if zs_list is None: return None
    cand = None
    for zs in zs_list:
        tb, te = _zs_begin_end_time(zs)
        if tb is None or te is None: continue
        if tb <= t_event and (cand is None or te >= _zs_begin_end_time(cand)[1]):
            cand = zs
    return cand

def _zs_begin_end_time(zs):
    # begin / end 可能是 CKLine_Unit；有的版本需从 begin_bi / end_bi -> klu 获取
    def get_time_from_klu(klu):
        for a in ["time","timestamp","dt","datetime"]:
            if hasattr(klu, a): return pd.to_datetime(getattr(klu, a))
        return None
    tb = te = None
    if hasattr(zs,"begin"): tb = get_time_from_klu(zs.begin)
    if hasattr(zs,"end"):   te = get_time_from_klu(zs.end)
    if tb is None and hasattr(zs,"begin_bi") and hasattr(zs.begin_bi,"get_begin_klu"):
        tb = get_time_from_klu(zs.begin_bi.get_begin_klu())
    if te is None and hasattr(zs,"end_bi") and hasattr(zs.end_bi,"get_end_klu"):
        te = get_time_from_klu(zs.end_bi.get_end_klu())
    return tb, te

def zs_bounds(zs):
    """统一取 low/high/mid、bi 列表与峰值包络"""
    def g(obj, names):
        for n in names:
            if hasattr(obj, n): return getattr(obj, n)
        return None
    low  = g(zs, ["low","__low"])
    high = g(zs, ["high","__high"])
    mid  = g(zs, ["mid","__mid"])
    bi_lst = g(zs, ["bi_lst","__bi_lst"]) or []
    peak_low  = g(zs, ["peak_low","__peak_low"])
    peak_high = g(zs, ["peak_high","__peak_high"])
    return low, high, mid, bi_lst, peak_low, peak_high

# -----------------------
# 特征计算
# -----------------------
def calc_structure_features(zs, kline_df, t_event, atr_n=20):
    low, high, mid, bi_lst, peak_low, peak_high = zs_bounds(zs)
    if any(v is None for v in [low, high, mid]): 
        return {}

    height_pct = (high - low) / (abs(mid) + 1e-9)            # 中枢相对高度
    width_bi   = len(bi_lst)                                  # 中枢覆盖笔数

    win = kline_df[kline_df["timestamp"] <= t_event].tail(20).copy()
    win["m"] = (win["high"] + win["low"]) / 2.0               # 用 (H+L)/2 近似中轴
    x = np.arange(len(win)); x = x - x.mean()
    slope_mid = float(((x * (win["m"] - win["m"].mean())).sum()) / ((x**2).sum() + 1e-9))

    k2 = kline_df[kline_df["timestamp"] <= t_event].copy()
    k2["atr"] = atr(k2, n=atr_n)
    atr_now = float(k2["atr"].iloc[-1]) if len(k2) else np.nan
    entry_price = float(k2["close"].iloc[-1]) if len(k2) else np.nan

    entry_vs_mid_atr = (entry_price - mid) / (atr_now + 1e-9) if not math.isnan(atr_now) else np.nan
    dist_to_upper_atr = (high - entry_price) / (atr_now + 1e-9) if not math.isnan(atr_now) else np.nan
    dist_to_lower_atr = (entry_price - low) / (atr_now + 1e-9) if not math.isnan(atr_now) else np.nan

    peak_span = (peak_high - peak_low) if (peak_high is not None and peak_low is not None) else np.nan
    overlap_strength = ((high - low) / (abs(peak_span) + 1e-9)
                        if (peak_span is not None and not math.isnan(peak_span)) else np.nan)

    return {
        "zs_height_pct": float(height_pct),
        "zs_width_bi":   int(width_bi),
        "zs_slope_mid":  float(slope_mid),
        "entry_vs_mid_atr":   float(entry_vs_mid_atr),
        "dist_to_upper_atr":  float(dist_to_upper_atr),
        "dist_to_lower_atr":  float(dist_to_lower_atr),
        "zs_overlap_strength": float(overlap_strength),
    }

def calc_context_features(kline_df, t_event, pre_L=60):
    k2 = kline_df[kline_df["timestamp"] <= t_event].copy()
    if len(k2) < max(20, pre_L): return {}
    k2["ret1"] = k2["close"].pct_change()
    k2["atr20"] = atr(k2, 20)

    mom_L = k2["close"].pct_change(pre_L).iloc[-1]                         # 进入动量
    dd_L  = (k2["close"]/k2["close"].rolling(pre_L,1).max() - 1.0).rolling(pre_L,1).min().iloc[-1]
    atr_trend = (k2["atr20"].iloc[-1] / (k2["atr20"].tail(pre_L).mean() + 1e-9))
    return {"pre_mom": float(mom_L), "pre_dd": float(dd_L), "atr_trend_ctx": float(atr_trend)}

def higher_tf_zs_overlap(chan_obj, higher_lv, t_event):
    try:
        for lv, kl in iter_lv_kl_lists(chan_obj):
            if lv != higher_lv: continue
            zs_list = get_zs_list_from_kl(kl)
            zs = pick_active_zs(zs_list, t_event)
            return 1 if zs is not None else 0
    except Exception:
        return 0
    return 0

# -----------------------
# 主流程
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_csv", type=str, default=r"D:\chan\chan.py\events_outputs\labels\all_events_labeled.csv", help="打好标签的事件文件（含 event_id,symbol,t_entry_ts 或 t_entry）")
    ap.add_argument("--kline_dir",  type=str, default=r"D:\chan\chan.py\data\stooq\1d_15y",help="K线 parquet 目录（与 behavior_features.py 约定一致）")
    ap.add_argument("--out_dir",    type=str, default=r"D:\chan\chan.py\ml_dataset\zs_features_outputs",help="输出目录")
    ap.add_argument("--kltype",         type=str, default="1d")
    ap.add_argument("--higher_kltype",  type=str, default="60m")
    ap.add_argument("--atr_n",          type=int, default=20)
    ap.add_argument("--pre_L",          type=int, default=60)
    args = ap.parse_args()

    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    KLINE_DIR = Path(args.kline_dir)

    ev = pd.read_csv(args.events_csv)
    # 统一时间列：优先 t_entry_ts / 其次 t_entry / 再其次 t_signal
    tcol = None
    for c in ["t_entry_ts","t_entry","t_signal"]:
        if c in ev.columns: tcol = c; break
    if tcol is None: raise KeyError("需要 t_entry_ts / t_entry / t_signal 之一")
    # 统一解析为 UTC -> 再去掉时区，得到 naive 时间戳
    ev["t_entry_ts"] = pd.to_datetime(ev[tcol], utc=True, errors="coerce").dt.tz_convert(None)
    if "timeframe" not in ev.columns: ev["timeframe"] = args.kltype
    if "entry_price" not in ev.columns: ev["entry_price"] = np.nan

    # 预加载 K 线
    k_cache = {}
    for sym, tf in ev[["symbol","timeframe"]].drop_duplicates().itertuples(index=False):
        try:
            k_cache[(sym, tf)] = load_kline_one(sym, KLINE_DIR, tf)
        except Exception as e:
            print(f"[warn] load_kline {sym}-{tf} failed: {e}")

    rows = []
    for r in ev.itertuples(index=False):
        sid = getattr(r, "event_id")
        sym = getattr(r, "symbol")
        tf  = getattr(r, "timeframe")
        t   = pd.to_datetime(getattr(r, "t_entry_ts"))
        kdf = k_cache.get((sym, tf))
        if kdf is None:
            rows.append({"event_id": sid}); continue

        k_pre = kdf[kdf["timestamp"] <= t]
        if len(k_pre) < max(60, args.pre_L):
            rows.append({"event_id": sid}); continue

        # 默认输出字典：即使拿不到中枢，也占位 event_id 方便左连接
        feat = {"event_id": sid}

        if CHAN_AVAILABLE:
            # 用 chan.py 在 [t-2*L, t] 的窗口内构建中枢
            start_date = (k_pre["timestamp"].iloc[-min(len(k_pre), 2*args.pre_L)]).date()
            end_date = (t + pd.Timedelta(days=1)).date()
            # —— 检查更高周期 parquet 是否存在；不存在就只跑单级 —— 

        # 1) 预判 higher tf parquet 是否存在
        def parquet_path(symbol, tf):
            return (Path(args.kline_dir) / (f"{symbol}_1d_15y.parquet" if tf.lower()=="1d"
                                            else f"{symbol}_{tf.lower()}.parquet"))

        need_higher  = args.higher_kltype.lower() != "1d"
        has_higher   = (not need_higher) or parquet_path(sym, args.higher_kltype).exists()

        # Build level list and force order: big -> small
        cands = {to_kltype(tf)}
        if has_higher:
            cands.add(to_kltype(args.higher_kltype))
        _order = {KL_TYPE.K_DAY: 5, KL_TYPE.K_60M: 4, KL_TYPE.K_30M: 3, KL_TYPE.K_15M: 2, KL_TYPE.K_5M: 1}
        lv_list = sorted(cands, key=lambda k: _order.get(k, 0), reverse=True)

        try:
            conf = CChanConfig()
            chan_obj = build_chan(sym, start_date, end_date, lv_list=lv_list, conf=conf)
            # 当前级别活动中枢
            zs_active = None
            for lv, kl in iter_lv_kl_lists(chan_obj):
                if lv == to_kltype(tf):
                    zs_active = pick_active_zs(get_zs_list_from_kl(kl), t)
                    break
            if zs_active is not None:
                feat.update(calc_structure_features(zs_active, k_pre, t_event=t, atr_n=args.atr_n))

            # 更高周期中枢存在性标记（仅当我们真的加载了 higher 才评估）
            feat["higher_has_zs"] = 0
            if has_higher:
                feat["higher_has_zs"] = higher_tf_zs_overlap(
                    chan_obj, higher_lv=to_kltype(args.higher_kltype), t_event=t
                )
        except Exception as e:
            print(f"[warn] build/eval chan failed {sym} {t}: {e}")
            # 明确兜底，避免下游出现缺列/空值不一致
            feat["higher_has_zs"] = 0


        # 上下文特征：无论是否拿到中枢都可以计算
        feat.update(calc_context_features(k_pre, t_event=t, pre_L=args.pre_L))
        rows.append(feat)

    out = (pd.DataFrame(rows)
             .sort_values("event_id")
             .drop_duplicates("event_id", keep="last"))
    out.to_csv(OUT / "zs_features.csv", index=False)
    print("✅ saved:", (OUT / 'zs_features.csv').resolve())

if __name__ == "__main__":
    main()
