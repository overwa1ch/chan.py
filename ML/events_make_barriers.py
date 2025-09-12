#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量版：用 ATR 把“波动”量化（给边界定尺度），为多标的 events 生成 TP/SL。

亮点：
- 兼容单文件 & 目录批量两种用法
- 自动识别 K 线列名（Open/High/Low/Close）与时间列（timestamp/datetime/...）
- events 的 t_entry 支持“时间戳”或“K线行号”；entry_price 缺失会自动补
- 默认按文件名自动配对（AAPL_1d.parquet ↔ AAPL_events.csv）
- 可选自定义正则 --regex 来抽取“标的代码”（命名不统一也能搞定）
- 并行处理 --jobs N
- 为每个标的输出各自 CSV，且可一键合并为一个总表 --merge_all all.csv

用法示例：
1) 处理单个标的（完全兼容旧脚本）
   python events_make_barriers_batch.py \
       --ohlcv data/AAPL_1d.parquet \
       --events events/AAPL_events.csv \
       --out out/AAPL_events_with_barriers.csv \
       --entry_when next_open --k 2.0 --m 1.0 --atr_window 20

2) 一次性处理 20 个标的（目录批量）
   python events_make_barriers_batch.py \
       --ohlcv data/kline_dir \
       --events data/events_dir \
       --out_dir out/batch \
       --entry_when next_open --k 2.0 --m 1.0 --atr_window 20 \
       --merge_all out/batch/all_events_with_barriers.csv \
       --jobs 4

3) 命名不统一？用正则指定“代码”抽取规则（提取命名中的 (?P<sym>...)）
   python events_make_barriers_batch.py \
       --ohlcv data/kline_dir --events data/events_dir --out_dir out/batch \
       --regex "(?P<sym>[A-Z]{1,5}(?:\.[A-Z]{2})?)"  # 例如 AAPL、TSLA、HK.00700

备注：
- ATR 前 n 根为 NaN 属正常；这些时刻的事件会被丢弃（无足够历史波动）
- direction: 1=做多，-1=做空；若缺失默认按 1 处理
- entry_when: next_open=以下一根开盘价入场；this_close=以当前根收盘价入场
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================== 工具：列名清洗与匹配 ======================

def _clean(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")

def _pick_col(cols: Iterable[str], options: Iterable[str]) -> Optional[str]:
    options = set(options)
    for c in cols:
        if _clean(c) in options:
            return c
    return None

# ====================== 1) 读取 K 线（自动识别时间列/索引 & OHLC 变体） ======================

def load_ohlcv_flexible(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"未找到K线文件: {p}")
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    ts_col = _pick_col(df.columns, {"timestamp","time","datetime","date"})
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    else:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            ts = pd.to_datetime(df.index, utc=True)
        else:
            idx_col = _pick_col(df.columns, {"index"})
            if idx_col is not None:
                ts = pd.to_datetime(df[idx_col], utc=True, errors="coerce")
            else:
                raise ValueError("无法在 K 线里找到时间列（timestamp/time/datetime/date）或时间索引。")

    if ts.isna().all():
        raise ValueError("K 线时间列解析失败（全是 NaN）")

    cols = list(df.columns)
    c_open  = _pick_col(cols, {"open","o"})
    c_high  = _pick_col(cols, {"high","h"})
    c_low   = _pick_col(cols, {"low","l"})
    c_close = _pick_col(cols, {"close","c"})
    if c_close is None:
        c_close = _pick_col(cols, {"adjclose","adjustedclose","adjcloseprice"})

    need_map: Dict[str, str] = {}
    if not c_open or not c_high or not c_low or not c_close:
        missing = [k for k,v in [("open",c_open),("high",c_high),("low",c_low),("close",c_close)] if not v]
        raise ValueError(f"K线缺少必要列（接受大小写/变体）：{missing}")

    need_map["open"]  = c_open
    need_map["high"]  = c_high
    need_map["low"]   = c_low
    need_map["close"] = c_close

    out = pd.DataFrame({
        "timestamp": ts,
        "open":  pd.to_numeric(df[need_map["open"]],  errors="coerce"),
        "high":  pd.to_numeric(df[need_map["high"]],  errors="coerce"),
        "low":   pd.to_numeric(df[need_map["low"]],   errors="coerce"),
        "close": pd.to_numeric(df[need_map["close"]], errors="coerce"),
    })
    out = out.dropna(subset=["timestamp","open","high","low","close"])\
             .sort_values("timestamp")\
             .drop_duplicates("timestamp")\
             .set_index("timestamp")
    if len(out) == 0:
        raise ValueError("K 线清洗后为空，请检查数据内容。")
    return out

# ====================== 2) 计算 ATR ======================

def compute_atr(df: pd.DataFrame, n: int = 20, method: str = "rma") -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    if method == "sma":
        return tr.rolling(window=n, min_periods=n).mean()
    else:  # rma（RMA=EMA的特殊参数）
        return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

# ====================== 3) 读取 events（自动分辨 t_entry 是“时间戳”还是“行号”） ======================

def load_events_auto(path: str, df: pd.DataFrame, entry_when: str) -> pd.DataFrame:
    ev = pd.read_csv(path)

    if "t_entry" not in ev.columns:
        for alt in ["entry_ts","entry_time","entry_datetime","entry_loc","entry_idx","entry_i"]:
            if alt in ev.columns:
                ev = ev.rename(columns={alt:"t_entry"})
                break
    if "t_entry" not in ev.columns:
        raise ValueError("events 中找不到 t_entry（或别名 entry_ts/entry_time/entry_loc 等）")

    if "direction" not in ev.columns:
        ev["direction"] = 1
    ev["direction"] = ev["direction"].astype(int)

    if "entry_price" in ev.columns:
        ev["entry_price"] = pd.to_numeric(ev["entry_price"], errors="coerce")
    else:
        ev["entry_price"] = np.nan

    is_numeric = pd.api.types.is_integer_dtype(ev["t_entry"]) or pd.api.types.is_float_dtype(ev["t_entry"])
    if is_numeric:
        ev["t_entry"] = ev["t_entry"].astype(int)
        ev["mode"] = "positional"
    else:
        ev["t_entry"] = pd.to_datetime(ev["t_entry"], utc=True, errors="coerce")
        if ev["t_entry"].isna().any():
            raise ValueError("events 里有无法解析的时间值（t_entry）")
        ev["mode"] = "timestamp"

    if ev["entry_price"].isna().any():
        if ev["mode"].iloc[0] == "positional":
            price_series = df["open"] if entry_when == "next_open" else df["close"]
            ev.loc[ev["entry_price"].isna(), "entry_price"] = ev.loc[ev["entry_price"].isna(), "t_entry"].map(
                lambda i: float(price_series.iloc[int(i)]) if 0 <= int(i) < len(df) else np.nan
            )
        else:
            price_series = df["open"] if entry_when == "next_open" else df["close"]
            ev = ev.set_index("t_entry").join(price_series.rename("_price_snap"), how="left")
            ev["entry_price"] = ev["entry_price"].fillna(ev["_price_snap"])
            ev = ev.drop(columns=["_price_snap"]).reset_index()

    ev = ev.dropna(subset=["entry_price"]).reset_index(drop=True)
    return ev

# ====================== 4) 生成 TP/SL ======================

def add_barriers_auto(df: pd.DataFrame, ev: pd.DataFrame, atr: pd.Series,
                      k: float, m: float, entry_when: str) -> pd.DataFrame:
    atr_for_entry = atr.shift(1) if entry_when == "next_open" else atr

    if ev["mode"].iloc[0] == "positional":
        ev["atr_entry"] = ev["t_entry"].map(lambda i: float(atr_for_entry.iloc[int(i)]) if 0 <= int(i) < len(df) else np.nan)
    else:
        ev = ev.set_index("t_entry").join(atr_for_entry.rename("atr_entry"), how="left").reset_index()

    ev = ev.dropna(subset=["atr_entry"]).reset_index(drop=True)

    is_long  = ev["direction"] == 1
    is_short = ev["direction"] == -1

    ev.loc[is_long,  "tp"] = ev.loc[is_long,  "entry_price"] + k * ev.loc[is_long,  "atr_entry"]
    ev.loc[is_long,  "sl"] = ev.loc[is_long,  "entry_price"] - m * ev.loc[is_long,  "atr_entry"]

    ev.loc[is_short, "tp"] = ev.loc[is_short, "entry_price"] - k * ev.loc[is_short, "atr_entry"]
    ev.loc[is_short, "sl"] = ev.loc[is_short, "entry_price"] + m * ev.loc[is_short, "atr_entry"]

    ev["check_ok"] = True
    ev.loc[is_long,  "check_ok"] &= (ev.loc[is_long,  "tp"] > ev.loc[is_long,  "entry_price"]) & (ev.loc[is_long,  "sl"] < ev.loc[is_long,  "entry_price"])
    ev.loc[is_short, "check_ok"] &= (ev.loc[is_short, "tp"] < ev.loc[is_short, "entry_price"]) & (ev.loc[is_short, "sl"] > ev.loc[is_short, "entry_price"])

    return ev

# ====================== 5) 文件配对（自动 or 正则） ======================

def _list_ohlcv(ohlcv_input: str) -> List[Path]:
    p = Path(ohlcv_input)
    if p.is_file():
        return [p]
    if p.is_dir():
        files: List[Path] = []
        files.extend(sorted(p.glob("*.parquet")))
        files.extend(sorted(p.glob("*.csv")))
        return files
    raise FileNotFoundError(f"未找到路径: {p}")


def _list_events(events_input: str) -> List[Path]:
    p = Path(events_input)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    raise FileNotFoundError(f"未找到路径: {p}")


def _extract_sym_from_name(path: Path, regex: Optional[str]) -> str:
    name = path.name
    if regex:
        m = re.search(regex, name)
        if m:
            if "sym" in m.groupdict():
                return str(m.group("sym")).lower()
            if m.groups():
                return str(m.group(1)).lower()
    # fallback：取文件名里最长的字母数字片段作为“代码”
    stem = path.stem#获取文件的主干名（即去掉路径和扩展名后的部分）。例如：Path("/data/AAPL_20230101.csv").stem 得到 "AAPL_20230101"。
    tokens = re.findall(r"[A-Za-z0-9\.]+", stem)
    blacklist = {"events","event","ohlcv","kline","data","day","1d","d","15y","y","parquet","csv","bars","quotes",}
    tokens = [t for t in tokens if t.lower() not in blacklist]
    if not tokens:
        return stem.lower()
    tokens.sort(key=lambda t: (-len(t), t))
    return tokens[0].lower()


def pair_files(ohlcv_list: List[Path], events_list: List[Path], regex: Optional[str]) -> List[Tuple[str, Path, Path]]:
    o_map: Dict[str, Path] = {}
    for f in ohlcv_list:
        key = _extract_sym_from_name(f, regex)
        if key in o_map:#这是一个冲突处理机制。如果当前提取出的代码 key 已经存在于 o_map 字典中了，说明我们遇到了同一个股票有多个OHLCV文件的情况。
            # 尽量更偏好 parquet（更快更稳）
            prefer = f if f.suffix.lower()==".parquet" else o_map[key]
            o_map[key] = prefer
        else:
            o_map[key] = f

    e_map: Dict[str, Path] = {}
    for f in events_list:
        key = _extract_sym_from_name(f, regex)
        if key in e_map:
            # 同一个代码出现多个 events：保留“最近修改”的那个
            prefer = f if f.stat().st_mtime >= e_map[key].stat().st_mtime else e_map[key]
            #f.stat().st_mtime： 这里用到了 .stat() 方法，它可以获取文件的元信息（或称 metadata）。
            # .st_mtime 是元信息中的一项，代表文件最后的修改时间（modification time），它是一个时间戳数字。
            e_map[key] = prefer
        else:
            e_map[key] = f

    keys = sorted(set(o_map) & set(e_map))
    pairs = [(k, o_map[k], e_map[k]) for k in keys]

    # 发出提示：哪些 unmatched
    miss_ohlcv = sorted(set(e_map) - set(o_map))
    miss_events = sorted(set(o_map) - set(e_map))
    if miss_ohlcv:
        print(f"⚠️ 这些 events 没找到对应的 K 线（按代码抽取逻辑）：{miss_ohlcv}")
    if miss_events:
        print(f"⚠️ 这些 K 线没找到对应的 events（按代码抽取逻辑）：{miss_events}")

    return pairs

# ====================== 6) 单标的执行 & 批量驱动 ======================

def run_single(ohlcv_path: Path, events_path: Path, out_csv: Path,
               atr_window: int, atr_method: str, entry_when: str, k: float, m: float,
               symbol: Optional[str]=None) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """返回 (symbol, df_out or None, err_msg or None)"""
    sym = symbol or _extract_sym_from_name(ohlcv_path, None)
    try:
        print(f"📦 读取 K 线: {ohlcv_path}")
        df = load_ohlcv_flexible(str(ohlcv_path))
        print(f"   条数: {len(df)}  时间范围(UTC): {df.index.min()} ~ {df.index.max()}")

        print(f"📦 读取 events: {events_path}")
        ev = load_events_auto(str(events_path), df, entry_when)
        print(f"   事件条数(原始且已补价): {len(ev)}  模式: {ev['mode'].iloc[0]}")

        print(f"🧮 计算 ATR(n={atr_window}, method={atr_method}) ...")
        atr = compute_atr(df, n=atr_window, method=atr_method)
        print(f"   ATR可用比例: {atr.notna().mean():.2%}（前 {atr_window} 根 NaN 正常）")

        print(f"🎯 生成 TP/SL（入场={entry_when}，k={k}，m={m}） ...")
        out = add_barriers_auto(df, ev, atr, k=k, m=m, entry_when=entry_when)

        # 若已有 symbol 列则覆盖；没有则插入到首列
        if "symbol" in out.columns:
            out["symbol"] = sym
        else:
            out.insert(0, "symbol", sym)

        out["src_ohlcv"] = ohlcv_path.name
        out["src_events"] = events_path.name

        # 统一把 symbol 放到最前（如果是覆盖的情况）
        cols_order = ["symbol"] + [c for c in out.columns if c != "symbol"]
        out = out[cols_order]

        # 写盘：确保目录存在（无论 out_csv 是 str 还是 Path）
        out_path = out_csv if isinstance(out_csv, Path) else Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已导出: {out_path}  条数: {len(out)}")
        return sym, out, None
    except Exception as e:
        return sym, None, f"{type(e).__name__}: {e}"


def run_batch(ohlcv_input: str, events_input: str, out_dir: str,
              atr_window: int, atr_method: str, entry_when: str, k: float, m: float,
              regex: Optional[str], jobs: int, merge_all: Optional[str]) -> None:
    ohlcv_list = _list_ohlcv(ohlcv_input)
    events_list = _list_events(events_input)

    # 若两边都是单文件，走单标的；否则批量
    if len(ohlcv_list) == 1 and len(events_list) == 1:
        out_csv = Path(out_dir)
        if out_csv.is_dir():
            sym = _extract_sym_from_name(ohlcv_list[0], regex)
            out_csv = out_csv / f"{sym}_events_with_barriers.csv"
        run_single(ohlcv_list[0], events_list[0], out_csv, atr_window, atr_method, entry_when, k, m)
        return

    # 批量：配对
    pairs = pair_files(ohlcv_list, events_list, regex)
    if not pairs:
        raise SystemExit("未配对到任何 (ohlcv, events)。请检查命名或使用 --regex。")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    results: List[pd.DataFrame] = []

    def _task(pair: Tuple[str, Path, Path]):
        sym, o_path, e_path = pair
        out_csv = out_dir_p / f"{sym}_events_with_barriers.csv"
        return run_single(o_path, e_path, out_csv, atr_window, atr_method, entry_when, k, m, symbol=sym)

    if jobs and jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_task, p): p[0] for p in pairs}
            for fut in as_completed(futs):
                sym = futs[fut]
                s, df_out, err = fut.result()
                if err:
                    print(f"❌ {sym}: {err}")
                elif df_out is not None:
                    results.append(df_out)
    else:
        for p in pairs:
            sym, df_out, err = _task(p)
            if err:
                print(f"❌ {sym}: {err}")
            elif df_out is not None:
                results.append(df_out)

    if merge_all and results:
        all_df = pd.concat(results, ignore_index=True)
        out_all = Path(merge_all)
        out_all.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(out_all, index=False, encoding="utf-8-sig")
        print(f"📚 汇总已导出: {out_all}  条数: {len(all_df)}")

# ====================== 7) CLI ======================

def main():
    ap = argparse.ArgumentParser(description="批量用 ATR 为 events 生成 TP/SL（自动配对/可正则/可并行）")
    ap.add_argument("--ohlcv", required=True, help="K线文件或目录（.parquet/.csv）")
    ap.add_argument("--events", required=True, help="events.csv 文件或目录（.csv）")

    # 单文件模式下：--out 是输出文件；目录批量下：--out_dir 是输出目录
    ap.add_argument("--out", help="单文件模式下输出 CSV 路径；若是目录则自动命名")
    ap.add_argument("--out_dir", help="批量模式下输出目录")

    ap.add_argument("--atr_window", type=int, default=20)
    ap.add_argument("--atr_method", choices=["rma","sma"], default="rma")
    ap.add_argument("--entry_when", choices=["next_open","this_close"], default="next_open")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--m", type=float, default=1.0)

    ap.add_argument("--regex", help="文件名中抽取标的代码用的正则；可包含命名分组 (?P<sym>...)")
    ap.add_argument("--jobs", type=int, default=0, help="并行线程数（>1 启用并行）")
    ap.add_argument("--merge_all", help="合并全部标的输出为一个 CSV 的路径（可选）")

    args = ap.parse_args()

    # 判定运行模式
    ohlcv_is_file = Path(args.ohlcv).is_file()
    events_is_file = Path(args.events).is_file()

    if ohlcv_is_file and events_is_file:
        if not args.out and not args.out_dir:
            raise SystemExit("单文件模式需要 --out（或给 --out_dir 一个目录，我会自动命名）。")
        out_target = args.out or args.out_dir
        run_batch(args.ohlcv, args.events, out_target, args.atr_window, args.atr_method,
                  args.entry_when, args.k, args.m, args.regex, jobs=0, merge_all=None)
    else:
        out_dir = args.out_dir or args.out
        if not out_dir:
            raise SystemExit("目录批量模式需要提供 --out_dir（或用 --out 指向一个目录）。")
        run_batch(args.ohlcv, args.events, out_dir, args.atr_window, args.atr_method,
                  args.entry_when, args.k, args.m, args.regex, args.jobs, args.merge_all)


if __name__ == "__main__":
    main()
#python .\events_make_barriers.py `--ohlcv "D:\chan\chan.py\data\stooq\1d_15y" `--events "D:\chan\chan.py\events_outputs" `--out_dir "D:\chan\chan.py\events_outputs\barriers" `--entry_when next_open --k 2.0 --m 1.0 --atr_window 20 `--merge_all "D:\chan\chan.py\events_outputs\barriers\all_events_with_barriers.csv" `--jobs 4
