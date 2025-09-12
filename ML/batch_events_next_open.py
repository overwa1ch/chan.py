# -*- coding: utf-8 -*-
"""
合并版：build_events_next_open.py + batch_events_next_open2.py
——不新增功能，只把两份脚本合在一个文件里，保留两种用法：
1) 单文件模式（原 build_events_next_open.py 的 CLI）：
   python events_next_open_merged.py --ohlcv <path.parquet> --signals <signals.csv> --out <events.csv>
2) 批量模式（原 batch_events_next_open2.py 的行为）：
   直接运行不带参数，会读取常量中的目录与路径批量生成 events，并产出汇总 CSV。

注：保留了原有函数/常量/打印语句与校验逻辑，尽量不改动语义。
"""
from __future__ import annotations

import os, glob, sys
import pandas as pd
import numpy as np
from pathlib import Path

# 复用你提供的工具：规范信号、对齐到 df.index、体检
# （保持原始导入路径不变）
from signals_build import build_signals_df, validate_signals

# ========================
# 单文件模式（原 build_events_next_open.py 的函数）
# ========================

def load_ohlcv_parquet(path_parquet: str) -> pd.DataFrame:
    """
    读取 parquet 的 OHLCV，并确保：
      1) 索引为 DatetimeIndex（升序）
      2) 索引为 UTC（若原本无时区，统一本地化为 UTC）
      3) 列名至少包含 open/high/low/close
    """
    df = pd.read_parquet(path_parquet)
    # 如果 parquet 中是普通列而非索引，优先尝试 'timestamp' 列
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            raise ValueError("parquet 中未发现 DatetimeIndex 或 'timestamp' 列。")

    # 确保时区：无 tz 则本地化为 UTC；有 tz 则转为 UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # 统一列名到小写
    df.columns = [str(c).lower() for c in df.columns]

    need = {"open","high","low","close"}
    if not need.issubset(df.columns):
        raise ValueError(f"OHLCV 缺少必要列：{need - set(df.columns)}")

    df = df.sort_index()
    return df


def load_signals_csv(path_csv: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    读取 csv 的信号，列至少包含：
      - t_signal：时间戳
      - signal_type：'3buy' 或 '3sell'（大小写/别名会在 build_signals_df 里规范）
    然后用 build_signals_df 将其：
      * 统一为 UTC
      * 对齐到 df.index 的某一根 K 线（nearest）
      * 去除重复
      * 仅保留 df 的时间范围内的信号
    """
    raw = pd.read_csv(path_csv)
    if "t_signal" not in raw.columns or "signal_type" not in raw.columns:
        raise ValueError("signals.csv 需要包含列：t_signal, signal_type")

    # 交给现成的工具做标准化（含对齐到 df.index）
    signals = build_signals_df(raw, df, align_method="nearest")
    # 体检（保持原有行为）
    rpt = validate_signals(df, signals)
    print("signals 验收：", rpt)
    if not rpt.get("PASS", False):
        print("⚠️ 警告：signals 体检未通过，请先修正后再继续。")

    return signals


def build_events_next_open(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    把每条信号映射为“下一根开盘入场”的事件表：
      event_id, t_signal, t_entry, entry_price, direction
    方向：3buy -> +1；3sell -> -1
    """
    events = []
    for i, row in signals.iterrows():
        t_sig = pd.Timestamp(row["t_signal"])  # 已是 UTC
        stype = str(row["signal_type"]).lower()

        # 找到信号所在的 K 线位置（signals 已经对齐过，这里一般能 get_loc 成功）
        try:
            pos_sig = df.index.get_loc(t_sig)
        except KeyError:
            # 兜底：nearest
            pos_sig = df.index.get_indexer([t_sig], method="nearest")[0]
            t_sig = df.index[pos_sig]

        pos_entry = pos_sig + 1
        if pos_entry >= len(df):
            # 没有下一根，无法入场 → 跳过
            continue

        t_entry = df.index[pos_entry]
        entry_price = float(df.iloc[pos_entry]["open"])

        if stype == "3buy":
            direction = 1
        elif stype == "3sell":
            direction = -1
        else:
            # 未知类型，跳过
            continue

        # event_id：用入场时间 + 一个序号，避免潜在重复
        events.append({
            "event_id": f"EVT_{t_entry.strftime('%Y%m%d_%H%M%S')}_{i}",
            "t_signal": t_sig,
            "t_entry":  t_entry,
            "entry_price": entry_price,
            "direction": direction
        })

    events = pd.DataFrame(events).sort_values("t_entry").reset_index(drop=True)
    return events


def validate_events(df: pd.DataFrame, events: pd.DataFrame, rtol=1e-10, atol=1e-12) -> dict:
    """
    对 events 做验收：
      * 列齐全
      * t_entry 晚于 t_signal
      * entry_price ≈ df.loc[t_entry,'open'] （浮点容差）
      * direction 只在 {+1, -1}
      * event_id 唯一
      * 时间在 df 范围内、无越界
    """
    rpt = {"PASS": True, "errors": [], "warnings": [], "stats": {}}
    need_cols = {"event_id","t_signal","t_entry","entry_price","direction"}
    if not need_cols.issubset(events.columns):
        rpt["PASS"] = False
        rpt["errors"].append(f"events 缺少列: {need_cols - set(events.columns)}")
        return rpt

    if events.empty:
        rpt["warnings"].append("events 为空。")
        rpt["stats"]["rows"] = 0
        return rpt

    # 基础空值检查
    for c in need_cols:
        if events[c].isna().any():
            rpt["PASS"] = False
            rpt["errors"].append(f"{c} 存在空值")

    # 时间与范围
    if ((events["t_signal"] < df.index[0]).any() or
        (events["t_signal"] > df.index[-1]).any() or
        (events["t_entry"]  < df.index[0]).any() or
        (events["t_entry"]  > df.index[-1]).any()):
        rpt["PASS"] = False
        rpt["errors"].append("存在超出 df 时间范围的 t_signal 或 t_entry")

    # t_entry 晚于 t_signal（严格下一根）
    if not (events["t_entry"] > events["t_signal"]).all():
        rpt["PASS"] = False
        rpt["errors"].append("发现 t_entry ≤ t_signal 的行（不是“下一根”）")

    # entry_price 与 df 的 open 对齐（浮点容差）
    opens = df["open"].reindex(events["t_entry"]).astype(float).values
    ok = np.isclose(events["entry_price"].astype(float).values, opens, rtol=rtol, atol=atol)
    if not ok.all():
        bad = int((~ok).sum())
        rpt["PASS"] = False
        rpt["errors"].append(f"{bad} 条事件的 entry_price 与 df.open 不一致")

    # direction 合法值
    if not set(events["direction"].unique()).issubset({-1, 1}):
        rpt["PASS"] = False
        rpt["errors"].append("direction 存在非 {-1, +1} 的值")

    # event_id 唯一
    if events["event_id"].duplicated().any():
        dupn = int(events["event_id"].duplicated().sum())
        rpt["PASS"] = False
        rpt["errors"].append(f"event_id 存在 {dupn} 个重复")

    # 统计信息
    rpt["stats"]["rows"] = int(len(events))
    rpt["stats"]["start"] = str(events["t_entry"].min())
    rpt["stats"]["end"]   = str(events["t_entry"].max())
    rpt["stats"]["longs"] = int((events["direction"] == 1).sum())
    rpt["stats"]["shorts"]= int((events["direction"] == -1).sum())
    return rpt


def main_single(path_parquet: str, path_signals_csv: str, out_csv: str = "events.csv"):
    print(f"读取 OHLCV：{path_parquet}")
    df = load_ohlcv_parquet(path_parquet)

    print(f"读取 signals：{path_signals_csv}")
    signals = load_signals_csv(path_signals_csv, df)

    print("映射到“下一根开盘入场”…")
    events = build_events_next_open(df, signals)

    # 验收
    rpt = validate_events(df, events)
    print("events 验收：", rpt)

    # 保存
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{out_csv}  （共 {len(events)} 条）")

    # 小结预览
    print("\n预览前5行：")
    print(events.head())


# ========================
# 批量模式（原 batch_events_next_open2.py）
# ========================

# 路径配置（保持原常量名与默认值）
KLINES_DIR   = r"D:\chan\chan.py\data\stooq\1d_15y"
OUTPUTS_DIR  = r"D:\chan\chan.py\events_outputs"
SUMMARY_CSV  = r"D:\chan\chan.py\events_outputs\events_batch_summary.csv"

# signals 的优先查找顺序（保持可格式化 {sym} 的写法）
SIGNALS_PATTERNS = [
    r"D:\chan\chan.py\outputs_signals\{sym}_1d_15y_signals.csv",
]

def find_signals_path(sym: str) -> str | None:
    for pat in SIGNALS_PATTERNS:
        p = pat.format(sym=sym)
        if os.path.exists(p):
            return p
    return None


def main_batch():
    files = glob.glob(os.path.join(KLINES_DIR, "*_1d_15y.parquet"))
    assert files, f"在 {KLINES_DIR} 下没有找到 *_1d_15y.parquet"

    rows = []
    ok, fail, skipped = 0, 0, 0

    for i, f in enumerate(sorted(files), 1):
        sym = os.path.basename(f).split("_")[0].upper()
        sig_path = find_signals_path(sym)
        if sig_path is None:
            print(f"[{i}/{len(files)}] {sym}: 未找到 signals，跳过")
            skipped += 1
            continue

        Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
        out_csv = os.path.join(OUTPUTS_DIR, f"{sym}_events.csv")

        try:
            # 1) 读取
            df = load_ohlcv_parquet(f)
            signals = load_signals_csv(sig_path, df)

            # 2) 生成 events（下一根开盘入场）
            events = build_events_next_open(df, signals)

            # —— 确保全局唯一：给 event_id 加上 symbol 前缀，并写入 t_entry_idx/symbol 两列 ——
            events["event_id"] = events["event_id"].apply(lambda s: f"{sym}_" + str(s))
            idx_map = {ts: ix for ix, ts in enumerate(df.index)}
            events["t_entry_idx"] = events["t_entry"].map(idx_map)
            events["symbol"] = sym

            # 3) 验收
            rpt = validate_events(df, events)
            if not rpt.get("PASS", False):
                fail += 1
                print(f"[{i}/{len(files)}] {sym}: 验收未通过 → {rpt}")
            else:
                ok += 1
                events.to_csv(out_csv, index=False)
                print(f"[{i}/{len(files)}] {sym}: OK  events={len(events)}  → {out_csv}")

            rows.append({
                "symbol": sym,
                "events": len(events),
                "pass": bool(rpt.get("PASS", False)),
                "errors": "|".join(rpt.get("errors", [])),
                "warnings": "|".join(rpt.get("warnings", [])),
                "start": rpt.get("stats", {}).get("start", ""),
                "end": rpt.get("stats", {}).get("end", ""),
                "longs": rpt.get("stats", {}).get("longs", 0),
                "shorts": rpt.get("stats", {}).get("shorts", 0),
            })

        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] {sym}: 失败 → {e}")
            rows.append({
                "symbol": sym, "events": 0, "pass": False,
                "errors": str(e), "warnings": ""
            })

    # 汇总
    df_sum = pd.DataFrame(rows).sort_values(["pass","events"], ascending=[False, False])
    Path(os.path.dirname(SUMMARY_CSV)).mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(SUMMARY_CSV, index=False)

    print("\n=== 批量完成 ===")
    print(df_sum.to_string(index=False))
    print(f"\n汇总已保存：{SUMMARY_CSV}")
    print(f"OK: {ok}  FAIL: {fail}  SKIP(no signals): {skipped}")


# ========================
# 入口：兼容两种老用法
# ========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            "合并版：单文件模式（--ohlcv/--signals）或不带参数运行批量模式。\n"
            "保持原始行为，不新增功能。"
        ),
        add_help=True,
    )
    ap.add_argument("--ohlcv", help="parquet 格式的 OHLCV 文件路径")
    ap.add_argument("--signals", help="csv 格式的信号路径（含 t_signal, signal_type 列）")
    ap.add_argument("--out", default="outputs/events.csv", help="输出 events.csv 路径（单文件模式）")
    args = ap.parse_args()

    # 如果给了 --ohlcv 与 --signals → 走单文件模式；否则走批量模式
    if args.ohlcv and args.signals:
        main_single(args.ohlcv, args.signals, args.out)
    elif any([args.ohlcv, args.signals]):
        ap.error("需要同时提供 --ohlcv 与 --signals；否则不带参数将运行批量模式。")
    else:
        # 保持原 batch 的“无参数即运行”的习惯
        main_batch()
