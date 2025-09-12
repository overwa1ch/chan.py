# -*- coding: utf-8 -*-
from pathlib import Path
import json, time
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

BASE = Path(r"D:\chan\chan.py")
DIR_KLINES   = BASE / r"D:\chan\chan.py\data\stooq\1d_15y"           # AAPL_1d.parquet
DIR_LABELED  = BASE / r"D:\chan\chan.py\events_outputs\labels"   # *_events_labeled.csv

run_tag = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = BASE / r"ml_dataset" / run_tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECK_TENTRY_IN_KLINE = True   # 开着更安全，出 WARNING 方便排查

# ---------- 1) 建立 K 线符号映射（忽略大小写） ----------
def build_kline_symbol_map():
    """
    扫描 K 线目录，收集形如 <SYMBOL>_1d.parquet 里的 SYMBOL。
    返回: {lower_symbol: ProperCaseSymbol}
    """
    m = {}
    for p in DIR_KLINES.glob("*_1d_15y.parquet"):
        stem = p.stem  # e.g., "AAPL_1d"
        sym = stem[:-7] if stem.lower().endswith("_1d_15y") else stem
        #这是一个字符串切片操作。[:-3] 的意思是“获取从字符串开头到倒数第4个字符（不含）的所有字符”。
        m[sym.lower()] = sym
    return m

SYM_MAP = build_kline_symbol_map()
if not SYM_MAP:
    raise FileNotFoundError(f"在 {DIR_KLINES} 没找到 *_1d_15y.parquet，无法建立符号映射。")

# ---------- 2) 从 labeled 文件名里拿到“符号候选” ----------
def symbol_candidate_from_labeled(fp: Path) -> str:
    """
    aapl_events_labeled.csv -> 'aapl'
    AAPL_1d_events_labeled.csv -> 'AAPL' （会把尾部 _1d 切掉）
    """
    s = fp.stem  # 例如 "aapl_events_labeled"
    s_low = s.lower()
    if s_low.endswith("_events_labeled"):
        s = s[: -len("_events_labeled")]
    if s.lower().endswith("_1d"):  # 兼容有人带了 _1d
        s = s[:-3]
    return s.strip()

def resolve_symbol(cand: str) -> str | None:
    """
    cand 在 K 线映射里则返回规范写法，否则 None
    """
    return SYM_MAP.get(cand.lower())#当传入的 cand 参数（转换为小写后）在 SYM_MAP 字典中不存在时，.get() 方法会默认返回 None。

# ---------- 3) 读 K 线为 UTC 索引 ----------
def load_klines(symbol: str, timeframe: str = "1d_15y") -> pd.DataFrame:
    f = DIR_KLINES / f"{symbol}_{timeframe}.parquet"
    if not f.exists():
        return None  # 交给上游判断是否跳过
    df = pd.read_parquet(f)

    if is_datetime64_any_dtype(df.index):
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return df.sort_index()

    for cand in ["timestamp", "datetime", "date", "time"]:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], utc=True, errors="coerce")
            if ts.isna().any():
                raise ValueError(f"{f} 的 {cand} 列有无法解析的时间戳。")
            df = df.set_index(ts).drop(columns=[cand])
            return df.sort_index()

    raise ValueError(f"{f} 缺少可用的时间列，且索引也不是时间类型。")

# ---------- 4) 读 labeled 并标准化 ----------
def read_and_standardize_labeled(fp: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_csv(fp)

    # t_entry_ts：统一为 UTC
    if "t_entry_ts" in df.columns:
        ts = pd.to_datetime(df["t_entry_ts"], utc=True, errors="coerce")
    elif "t_entry" in df.columns:
        ts = pd.to_datetime(df["t_entry"], utc=True, errors="coerce")
    else:
        raise KeyError(f"{fp.name} 缺少 t_entry 或 t_entry_ts 列。")
    if ts.isna().any():
        raise ValueError(f"{fp.name} 中有 {int(ts.isna().sum())} 条时间戳无法解析。")
    df["t_entry_ts"] = ts

    # 必需列
    for c in ["entry_price", "direction", "y", "ret"]:
        if c not in df.columns:
            raise KeyError(f"{fp.name} 缺少必需列: {c}")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce").astype("Int64")
    df["y"]         = pd.to_numeric(df["y"], errors="coerce").astype("Int64")
    df["ret"]       = pd.to_numeric(df["ret"], errors="coerce")

    # event_id
    if "event_id" not in df.columns or df["event_id"].isna().any():
        df["event_id"] = [f"{symbol}_1d_{t.isoformat()}" for t in df["t_entry_ts"]]

    df["symbol"]    = symbol
    df["timeframe"] = timeframe

    keep = ["event_id","symbol","timeframe","t_entry_ts","entry_price","direction","y","ret"]
    for extra in ["exit_reason","bars_held","exit_loc","k","m","H","fee_pct","tie"]:
        if extra in df.columns: keep.append(extra)
    out = df[keep].copy().sort_values("t_entry_ts").reset_index(drop=True)

    # 可选检查：t_entry_ts 是否在该标的 K 线里
    if CHECK_TENTRY_IN_KLINE:
        kl = load_klines(symbol, "1d_15y")
        if kl is None:
            print(f"⚠️  {symbol}: 缺少 K 线文件，跳过索引检查。")
        else:
            missing = (~out["t_entry_ts"].isin(kl.index)).sum()
            if missing > 0:
                print(f"⚠️  {fp.name}: 有 {int(missing)} 条 t_entry_ts 不在 {symbol}_1d 的K线索引中（粒度/时区不一致或事件落在非交易日）。")

    return out

# ---------- 5) 主流程 ----------
def main():
    files = sorted(DIR_LABELED.glob("*_events_labeled.csv"))
    if not files:
        raise FileNotFoundError(f"在 {DIR_LABELED} 未找到 *_events_labeled.csv")

    pieces, skipped = [], []

    print(f"📦 K线可用标的数: {len(SYM_MAP)}，示例: {list(SYM_MAP.values())[:5]}")

    for fp in files:
        cand = symbol_candidate_from_labeled(fp)
        symbol = resolve_symbol(cand)
        if symbol is None:
            skipped.append((fp.name, cand, "no_matching_kline_file"))
            print(f"⏭️  SKIP {fp.name}: 解析到符号候选 '{cand}'，但在 {DIR_KLINES} 找不到对应 *_1d_15y.parquet。")
            continue

        # 再次确认 K线是否存在（防御）
        kl_file = DIR_KLINES / f"{symbol}_1d_15y.parquet"
        if not kl_file.exists():
            skipped.append((fp.name, symbol, "kline_parquet_missing"))
            print(f"⏭️  SKIP {fp.name}: '{symbol}_1d_15y.parquet' 不存在。")
            continue

        
        df_one = read_and_standardize_labeled(fp, symbol, "1d")
        pieces.append(df_one)

    if not pieces:
        raise RuntimeError("没有可用的 labeled 文件被处理。请检查命名或目录内容。")

    dataset = pd.concat(pieces, ignore_index=True)
    dataset = dataset[dataset["t_entry_ts"].notna()]
    dataset = dataset[dataset["y"].isin([0,1])]
    dataset = dataset.drop_duplicates(subset=["event_id"]).sort_values("t_entry_ts").reset_index(drop=True)

    # 70/15/15 时间切分
    #训练集(70%)：用于模型训练，学习历史 patterns
    #验证集(15%)：用于调参和模型选择，模拟"近期过去"
    #测试集(15%)：用于最终评估，模拟"未知未来"
    q70 = dataset["t_entry_ts"].quantile(0.70)
    q85 = dataset["t_entry_ts"].quantile(0.85)
    train = dataset[dataset["t_entry_ts"] <= q70].copy()
    valid = dataset[(dataset["t_entry_ts"] > q70) & (dataset["t_entry_ts"] <= q85)].copy()
    test  = dataset[dataset["t_entry_ts"] > q85].copy()

    # 导出
    dataset.to_parquet(OUT_DIR / "dataset_all.parquet", index=False)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    valid.to_parquet(OUT_DIR / "valid.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)

    manifest = {
        "run_tag": run_tag,
        "paths": {
            "all":   str(OUT_DIR / "dataset_all.parquet"),
            "train": str(OUT_DIR / "train.parquet"),
            "valid": str(OUT_DIR / "valid.parquet"),
            "test":  str(OUT_DIR / "test.parquet"),
        },
        "counts": {
            "all": len(dataset),
            "train": len(train),
            "valid": len(valid),
            "test":  len(test),
        },
        "positives_ratio": {
            "all": float(dataset["y"].mean()) if len(dataset) else None,
            "train": float(train["y"].mean()) if len(train) else None,
            "valid": float(valid["y"].mean()) if len(valid) else None,
            "test":  float(test["y"].mean())  if len(test)  else None,
        },
        "by_symbol_counts": dataset.groupby("symbol")["event_id"].count().sort_values(ascending=False).to_dict(),
        "columns": dataset.dtypes.astype(str).to_dict(),
        "skipped_files": skipped,
        "note": "事件时间统一为 UTC；仅合并能在K线目录中找到的标的。",
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n📁 输出目录：", OUT_DIR)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
