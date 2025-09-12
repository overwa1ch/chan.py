#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# events_label_with_right_barrier_batch.py —— 优化版
# 变更点（对比原版 + 融合 step4 特性）：
# 1) 入场定位更健壮：若给的是时间戳，先做时区对齐；精确匹配失败时可取“最近一根”（可设容差 --nearest_tol）。
# 2) 入场价策略可选：--entry_price_policy 支持 'same_open'（同根开盘）与 'next_open'（后一根开盘，默认）。
#    若 events 内已有 entry_price 列，则优先生效；否则按策略回退；若缺 open 列则回退到 close。
# 3) 输出新增 t_entry（入场时间戳），并沿用 exit_time、ret、exit_reason 等；保留 bars_held，同时新增 holding_bars（同义）。
# 4) 其余逻辑（TP/SL 优先级、费用、右边界按 H_bars/H_time）保持兼容；CLI 增加上述新参数。
#
# 用法示例：
"""
python events_label_with_barrier_optimized.py `
--ohlcv "D:\chan\chan.py\data\stooq\1d_15y" `
--ewb "D:\chan\chan.py\events_outputs\barriers" `
--out_dir "D:/chan/chan.py/events_outputs/labels" `
--H_bars 10 `
--fee 0.0005 `
--tie sl_first `
--entry_price_policy next_open `
--nearest_tol 5min `
--jobs 4 `
--merge_all "D:/chan/chan.py/events_outputs/labels/all_events_labeled.csv" 
  """


from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
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

    # 时间列或索引 -> DatetimeIndex（尽量 UTC-aware，若已有 tz 则保留）
    ts_col = _pick_col(df.columns, {"timestamp","time","datetime","date"})
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")  # 统一转为 UTC-aware
    else:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            # 尝试保留原索引时区（若有的话）；若无 tz，则仍为 naive
            ts = pd.to_datetime(df.index, utc=False, errors="coerce")
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
    missing = [name for name, v in [("open",c_open),("high",c_high),("low",c_low),("close",c_close)] if not v]
    if missing:
        raise ValueError(f"K线缺少必要列（接受大小写/变体）：{missing}")

    out = pd.DataFrame({
        "timestamp": ts,
        "open":  pd.to_numeric(df[c_open],  errors="coerce"),
        "high":  pd.to_numeric(df[c_high],  errors="coerce"),
        "low":   pd.to_numeric(df[c_low],   errors="coerce"),
        "close": pd.to_numeric(df[c_close], errors="coerce"),
    })
    out = out.dropna(subset=["timestamp","open","high","low","close"]) \
             .sort_values("timestamp") \
             .drop_duplicates("timestamp") \
             .set_index("timestamp")

    if len(out) == 0:
        raise ValueError("K 线清洗后为空，请检查数据内容。")
    return out

# ====================== 右边界与标签逻辑 ======================

def _to_timedelta(x) -> Optional[pd.Timedelta]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, pd.Timedelta):
        return x
    import datetime as _dt
    if isinstance(x, _dt.timedelta):
        return pd.to_timedelta(x)
    if isinstance(x, np.timedelta64):
        return pd.to_timedelta(x)
    if isinstance(x, (str, int, float)):
        # 纯数字视为“天”；字符串如 '90min','2H','3D' 直接交给 to_timedelta
        return pd.to_timedelta(x, unit="D") if isinstance(x, (int, float)) else pd.to_timedelta(x)
    raise TypeError(f"H_time 类型不支持: {type(x)}")

def right_barrier_loc(df: pd.DataFrame, entry_loc: int, H_bars: Optional[int]=None, H_time: Optional[Union[str,int,float,pd.Timedelta]]=None) -> int:
    n = len(df)
    if not (0 <= int(entry_loc) < n):
        raise IndexError(f"entry_loc 越界: {entry_loc}, 数据行数: {n}")
    if (H_bars is None) == (H_time is None):
        raise ValueError("H_BARS 与 H_TIME 必须二选一（一个有值、另一个为 None）。")
    if H_bars is not None:
        Hb = int(H_bars)
        if Hb < 0:
            raise ValueError("H_BARS 不能为负数")
        return min(int(entry_loc) + Hb, n - 1)

    Ht = _to_timedelta(H_time)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception as e:
            raise ValueError("使用 H_TIME 时需要 DatetimeIndex（或可被 to_datetime 转换）。") from e
    if not idx.is_monotonic_increasing:
        raise ValueError("索引需要按时间升序。请先 sort_index().")
    entry_ts = idx[int(entry_loc)]
    deadline = entry_ts + Ht
    loc = int(np.searchsorted(idx.values, deadline, side="right") - 1)
    loc = max(loc, int(entry_loc))
    loc = min(loc, n - 1)
    return loc

def label_one_given(df: pd.DataFrame, entry_loc: int, entry_price: float, tp: float, sl: float, direction: int,
                    H_bars=None, H_time=None, fee_pct: float=0.0005, tie: str="sl_first"):
    end_loc = right_barrier_loc(df, entry_loc, H_bars=H_bars, H_time=H_time)
    cost = 2 * float(fee_pct)
    for loc in range(int(entry_loc) + 1, int(end_loc) + 1):
        hi = float(df.iloc[loc]['high']); lo = float(df.iloc[loc]['low'])
        if direction == 1:
            hit_tp = hi >= tp; hit_sl = lo <= sl
        else:
            hit_tp = lo <= tp; hit_sl = hi >= sl
        if hit_tp and hit_sl:
            if tie == "sl_first":
                exit_price = sl; gross = direction * (exit_price / entry_price - 1.0)
                return 0, gross - cost, loc, "SL&TP_samebar->SL", loc - int(entry_loc)
            else:
                exit_price = tp; gross = direction * (exit_price / entry_price - 1.0)
                return 1, gross - cost, loc, "SL&TP_samebar->TP", loc - int(entry_loc)
        if hit_tp:
            exit_price = tp; gross = direction * (exit_price / entry_price - 1.0)
            return 1, gross - cost, loc, "TP", loc - int(entry_loc)
        if hit_sl:
            exit_price = sl; gross = direction * (exit_price / entry_price - 1.0)
            return 0, gross - cost, loc, "SL", loc - int(entry_loc)
    exit_loc = int(end_loc)
    exit_price = float(df.iloc[exit_loc]['close'])
    gross = direction * (exit_price / entry_price - 1.0)
    y = 1 if (direction==1 and exit_price >= tp) or (direction==-1 and exit_price <= tp) else 0
    return y, gross - cost, exit_loc, "TIMEOUT", exit_loc - int(entry_loc)

# ====================== 解析 entry_loc / entry_price ======================

def _pick_price_from_row_or_policy(df: pd.DataFrame, row: pd.Series, loc: int, entry_price_policy: str) -> float:
    # 显式 entry_price 优先
    if "entry_price" in row and pd.notna(row["entry_price"]):
        return float(row["entry_price"])

    cols = set(c.lower() for c in df.columns)
    col_open  = "open"  if "open"  in cols else next((c for c in df.columns if c.lower()=="open"),  None)
    col_close = "close" if "close" in cols else next((c for c in df.columns if c.lower()=="close"), None)
    if not col_open and not col_close:
        raise KeyError("K线数据缺少 open/close 列，无法推导入场价")

    if entry_price_policy == "next_open":
        if loc + 1 < len(df) and col_open:
            return float(df.iloc[loc+1][col_open])
        # 后一根不存在或缺 open → 回退
        fallback_col = col_open or col_close
        return float(df.iloc[loc][fallback_col])
    else:  # same_open
        if col_open:
            return float(df.iloc[loc][col_open])
        return float(df.iloc[loc][col_close])

def _align_ts_to_index(entry_dt: pd.Timestamp, idx: pd.DatetimeIndex, nearest_tol: Optional[pd.Timedelta]) -> Optional[int]:
    """把 entry_dt 映射到 idx 的 iloc。优先精确匹配；不行则最近（可设置容差）。"""
    # 与索引对齐时区
    if getattr(idx, "tz", None) is None:
        entry_dt = entry_dt.tz_localize(None)  # 索引是 naive
    else:
        entry_dt = entry_dt.tz_convert(idx.tz) # 索引是 tz-aware

    # 精确匹配
    try:
        return int(idx.get_loc(entry_dt))
    except Exception:
        pass

    # 最近匹配
    try:
        pos = int(idx.get_indexer([entry_dt], method="nearest")[0])
    except Exception:
        return None
    if pos == -1:
        return None
    if nearest_tol is not None:
        dt_near = idx[pos]
        try:
            if abs(dt_near - entry_dt) > nearest_tol:
                return None
        except Exception:
            # 假如不同 tz/类型比较失败，就直接接受最近
            pass
    return pos

def resolve_entry_loc(df: pd.DataFrame, row: pd.Series,
                      entry_price_policy: str = "next_open",
                      nearest_tol: Optional[pd.Timedelta] = None) -> Tuple[int, float]:
    """返回 (iloc, entry_price)；支持整数 iloc 或时间戳入场."""
    # 1) 直接整数行号
    for key in ["t_entry_idx", "entry_idx", "entry_loc", "i_entry"]:
        if key in row and pd.notna(row[key]):
            loc = int(row[key])
            if not (0 <= loc < len(df)):
                raise ValueError(f"{key} 越界: {loc}")
            price = _pick_price_from_row_or_policy(df, row, loc, entry_price_policy)
            return loc, price

    # 2) 时间戳
    for key in ["t_entry","entry_time","signal_time"]:
        if key in row and pd.notna(row[key]):
            ts = pd.to_datetime(row[key], utc=True, errors="coerce")  # 先统一成 UTC-aware
            if pd.isna(ts):
                raise ValueError(f"无法解析入场时间: {row[key]}")
            idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index, errors="coerce", utc=False)
            loc = _align_ts_to_index(ts, idx, nearest_tol)
            if loc is None:
                raise KeyError(f"事件时间 {ts} 无法在 K 线索引中找到匹配（含最近逻辑）。")
            price = _pick_price_from_row_or_policy(df, row, int(loc), entry_price_policy)
            return int(loc), price

    raise ValueError("事件缺少 t_entry_idx 或 t_entry/entry_time/signal_time")

# ====================== 配对工具 ======================

def _extract_sym_from_name(path: Path, regex: Optional[str]) -> str:
    name = path.name
    if regex:
        m = re.search(regex, name)
        if m:
            if "sym" in m.groupdict():
                return str(m.group("sym")).lower()
            if m.groups():
                return str(m.group(1)).lower()
    stem = path.stem
    tokens = re.findall(r"[A-Za-z0-9\.]+", stem)
    blacklist = {"events","event","ohlcv","kline","data","day","1d","d","15y","y","parquet","csv","bars","quotes","with","barriers"}
    tokens = [t for t in tokens if t.lower() not in blacklist]
    if not tokens:
        return stem.lower()
    tokens.sort(key=lambda t: (-len(t), t))
    return tokens[0].lower()

def list_ohlcv(ohlcv_input: str) -> List[Path]:
    p = Path(ohlcv_input)
    if p.is_file():
        return [p]
    if p.is_dir():
        files: List[Path] = []
        files.extend(sorted(p.glob("*.parquet")))
        files.extend(sorted(p.glob("*.csv")))
        return files
    raise FileNotFoundError(f"未找到路径: {p}")

def list_ewb(ewb_input: str) -> Tuple[bool, List[Path]]:
    """返回 (is_single_csv, 文件列表)。单 CSV 模式下列表只含该文件。"""
    p = Path(ewb_input)
    if p.is_file():
        return True, [p]
    if p.is_dir():
        return False, sorted(p.glob("*_events_with_barriers.csv"))
    raise FileNotFoundError(f"未找到路径: {p}")

# ====================== 主流程：单标的与批量 ======================

def run_one_symbol(sym: str, df: pd.DataFrame, ev: pd.DataFrame, out_csv: Path,
                   H_bars: Optional[int], H_time: Optional[str], fee_pct: float, tie: str,
                   entry_price_policy: str, nearest_tol: Optional[pd.Timedelta]) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    try:
        # 统一 direction / 必要列
        if "direction" not in ev.columns:
            ev["direction"] = 1
        ev["direction"] = ev["direction"].astype(int)

        out_rows = []
        for _, row in ev.iterrows():
            try:
                loc, price = resolve_entry_loc(df, row, entry_price_policy=entry_price_policy, nearest_tol=nearest_tol)
            except Exception:
                # 入场定位失败：跳过该条
                continue
            direction = int(row.get("direction", 1))
            if pd.notna(row.get("tp")) and pd.notna(row.get("sl")):
                tp = float(row["tp"]); sl = float(row["sl"])
            else:
                # 保险：若缺失 tp/sl，跳过（也可在此添加 ATR×倍数的自动生成）
                continue
            y, r, exit_loc, reason, bars = label_one_given(
                df, int(loc), float(price), tp, sl, direction,
                H_bars=H_bars, H_time=H_time, fee_pct=fee_pct, tie=tie
            )
            rec = dict(row)
            rec.update({
                "symbol": sym,
                "t_entry": df.index[int(loc)] if isinstance(df.index, pd.DatetimeIndex) else None,
                "t_entry_idx": int(loc),
                "entry_price": float(price),
                "y": int(y),
                "ret": float(r),
                "exit_idx": int(exit_loc),
                "exit_time": df.index[int(exit_loc)] if isinstance(df.index, pd.DatetimeIndex) else None,
                "exit_reason": reason,
                "bars_held": int(bars),
                "holding_bars": int(bars),
            })
            out_rows.append(rec)

        out = pd.DataFrame(out_rows)
        out_path = out_csv if isinstance(out_csv, Path) else Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已导出: {out_path}  条数: {len(out)}")
        return sym, out, None
    except Exception as e:
        return sym, None, f"{type(e).__name__}: {e}"

def run_batch(ohlcv_input: str, ewb_input: str, out_dir: str,
              H_bars: Optional[int], H_time: Optional[str], fee_pct: float, tie: str,
              regex: Optional[str], jobs: int, merge_all: Optional[str],
              entry_price_policy: str, nearest_tol: Optional[pd.Timedelta]) -> None:
    ohlcv_list = list_ohlcv(ohlcv_input)
    is_single_csv, ewb_list = list_ewb(ewb_input)

    # 建立 symbol -> K线文件 的映射
    sym2k: Dict[str, Path] = {}
    for f in ohlcv_list:
        k = _extract_sym_from_name(f, regex)
        sym2k[k] = f

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    results: List[pd.DataFrame] = []

    if is_single_csv:
        # 读取总表，按 symbol 分组
        big = pd.read_csv(ewb_list[0])
        if 'symbol' not in big.columns:
            raise ValueError("总表缺少 symbol 列")
        # 时间列若存在，转成 UTC，避免 get_loc 失败
        for col in ["t_entry","entry_time","signal_time"]:
            if col in big.columns:
                big[col] = pd.to_datetime(big[col], utc=True, errors="coerce")
        groups = big.groupby(big['symbol'].astype(str))

        def _task(sym: str, sub: pd.DataFrame):
            key = str(sym).lower()
            if key not in sym2k:
                print(f"⚠️ 没找到 {sym} 对应的 K 线文件，已跳过。")
                return sym, None, None
            df = load_ohlcv_flexible(str(sym2k[key]))
            out_csv = out_dir_p / f"{key}_events_labeled.csv"
            return run_one_symbol(key, df, sub, out_csv, H_bars, H_time, fee_pct, tie, entry_price_policy, nearest_tol)

        if jobs and jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_task, s, g.copy()): str(s) for s, g in groups}
                for fut in as_completed(futs):
                    sym = futs[fut]
                    s, df_out, err = fut.result()
                    if err:
                        print(f"❌ {sym}: {err}")
                    elif df_out is not None:
                        results.append(df_out)
        else:
            for s, g in groups:
                sym, df_out, err = _task(s, g.copy())
                if err:
                    print(f"❌ {sym}: {err}")
                elif df_out is not None:
                    results.append(df_out)
    else:
        # 逐文件处理
        def _task_file(ewb_fp: Path):
            # 根据 events 文件名推断 symbol
            sym = _extract_sym_from_name(ewb_fp, regex)
            key = str(sym).lower()
            if key not in sym2k:
                print(f"⚠️ 没找到 {sym} 对应的 K 线文件，已跳过。")
                return sym, None, None
            df = load_ohlcv_flexible(str(sym2k[key]))
            ev = pd.read_csv(ewb_fp)
            # 防止时间列类型不一
            for col in ["t_entry","entry_time","signal_time"]:
                if col in ev.columns:
                    ev[col] = pd.to_datetime(ev[col], utc=True, errors="coerce")
            out_csv = out_dir_p / f"{key}_events_labeled.csv"
            return run_one_symbol(key, df, ev, out_csv, H_bars, H_time, fee_pct, tie, entry_price_policy, nearest_tol)

        if jobs and jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_task_file, fp): fp for fp in ewb_list}
                for fut in as_completed(futs):
                    fp = futs[fut]
                    s, df_out, err = fut.result()
                    if err:
                        print(f"❌ {fp.name}: {err}")
                    elif df_out is not None:
                        results.append(df_out)
        else:
            for fp in ewb_list:
                sym, df_out, err = _task_file(fp)
                if err:
                    print(f"❌ {fp.name}: {err}")
                elif df_out is not None:
                    results.append(df_out)

    # 合并输出
    if merge_all:
        merged = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        out_path = Path(merge_all)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"🧩 已合并导出: {out_path}  总条数: {len(merged)}")

# ====================== CLI ======================

def main():
    ap = argparse.ArgumentParser(description="批量对带 barriers 的事件进行 TP/SL/H 打标（支持近邻时间匹配与入场价策略）。")
    ap.add_argument("--ohlcv", required=True, help="K 线文件或目录（.parquet/.csv）")
    ap.add_argument("--ewb",   required=True, help="events_with_barriers 文件或目录（支持单CSV或目录）")
    ap.add_argument("--out_dir", required=True, help="输出目录")

    # H 参数：二选一
    ap.add_argument("--H_bars", type=int, default=None, help="右边界向前看的 bar 数。与 --H_time 二选一")
    ap.add_argument("--H_time", type=str, default=None, help="右边界向前看的时间跨度，如 '90min','2D' 等。与 --H_bars 二选一")

    ap.add_argument("--fee", type=float, default=0.0005, help="单边费率（含滑点），默认 0.0005")
    ap.add_argument("--tie", choices=["sl_first","tp_first"], default="sl_first", help="同根同时命中 TP/SL 时的优先级")

    ap.add_argument("--regex", type=str, default=None, help="从文件名抽取 symbol 的正则；若包含命名组 (?P<sym>...) 则优先使用")

    ap.add_argument("--jobs", type=int, default=1, help="并发线程数")
    ap.add_argument("--merge_all", type=str, default=None, help="把所有标的结果合并输出到该 CSV 路径")

    # 新增：入场价策略 & 最近匹配容差
    ap.add_argument("--entry_price_policy", choices=["same_open","next_open"], default="next_open",
                    help="当事件未给 entry_price 时的回退策略：same_open=同根开盘；next_open=后一根开盘（默认）。若缺 open 列则回退到 close。")
    ap.add_argument("--nearest_tol", type=str, default=None,
                    help="把时间戳映射到最近 bar 时允许的最大时间差（如 '5min'、'2H'、'3D'）。缺省不限制。")

    args = ap.parse_args()

    if (args.H_bars is None) == (args.H_time is None):
        raise SystemExit("需在 --H_bars 与 --H_time 中二选一。")

    nearest_tol = pd.to_timedelta(args.nearest_tol) if args.nearest_tol else None

    run_batch(args.ohlcv, args.ewb, args.out_dir,
              H_bars=args.H_bars, H_time=args.H_time, fee_pct=args.fee, tie=args.tie,
              regex=args.regex, jobs=args.jobs, merge_all=args.merge_all,
              entry_price_policy=args.entry_price_policy, nearest_tol=nearest_tol)

if __name__ == "__main__":
    main()
