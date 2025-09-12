# estimate_event_rate.py
# 目的：估算每个标的、每个“结果类别”(win/lose/neutral) 的年发生率 r，
# 并基于 r 的中位数推算：在给定的 Valid/Test 年限内，为达到研究级总量（≥300）
# 大约需要多少个标的；同时反推“每类每标的≥100”需要的年限。
#
# 已适配你的样例列：
# - 日期：t_signal / t_entry / exit_time（优先 t_signal，可切换）
# - 结果：label / outcome / result / y / status / exit_reason（再不行才用 direction 兜底）
# - y=1→win；y=0 或 -1→lose；exit_reason: TP→win, SL→lose, BE→neutral

from pathlib import Path
import sys, math, re
import pandas as pd
import numpy as np

# ======= 可配置区 =======

# 你的 events CSV 所在目录与文件模式
EVENTS_DIR = r"D:\chan\chan.py\events_outputs\labels"
FILE_GLOB  = "*_events_labeled.csv"  # 例如 AAPL_events.csv；也可改成 *_labeled.csv

# 日期优先级：signal=用 t_signal，entry=用 t_entry（若不存在则自动回退候选）
DATE_PREFERENCE = "signal"         # 可选 "entry"
# 若 t_signal 与 t_entry 同时存在，是否二选一：'earliest' / 'latest' / None
DATE_RESOLVE    = None

# 1) 把候选列表稍微扩充一下（放在可配置区）
DATE_COL_CANDIDATES  = ["t_signal", "t_entry", "exit_time", "signal_time", "entry_time",
                        "timestamp", "event_time", "datetime", "date", "time"]
LABEL_COL_CANDIDATES = ["label", "outcome", "result", "y", "status", "exit_reason", "direction"]

# 你的计划切窗长度（只用于估算“总量是否达标”，方便给建议）
TRAIN_YEARS = 10
VALID_YEARS = 3
TEST_YEARS  = 3

# 研究级目标
PER_CLASS_PER_SYMBOL_TARGET = 100   # 每类每标的 ≥100（用于计算 T_needed_years）
VALID_TOTAL_TARGET = 300            # Valid 总样本 ≥300
TEST_TOTAL_TARGET  = 300            # Test  总样本 ≥300
SELECTED_N_TARGET  = 150            # 近似用总样本量校验

# =======================


def pick_first_existing_ci(cols, candidates):
    """大小写不敏感：在 candidates 里按顺序找，返回真实列名"""
    lower2real = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower2real:
            return lower2real[cand.lower()]
    return None

def find_label_col(cols):
    """
    智能探测“标签列”：
    1) 先在优先列表里做大小写不敏感精确匹配
    2) 再做“结果类”模糊匹配（win/lose/tp/sl/…）
    3) 实在没有，再做“方向类”模糊匹配（direction/side/dir/…）
    返回：(列名, 语义) —— 语义: 'result' 或 'direction' 或 None
    """
    # 1) 精确候选（大小写不敏感）
    exact = ["label", "outcome", "result", "y", "status", "exit_reason", "direction"]
    col = pick_first_existing_ci(cols, exact)
    if col:
        return col, ("direction" if col.lower() == "direction" else "result")

    # 2) 结果类模糊关键词
    result_kw = [
        "label", "outcome", "result", "status", "y",
        "win", "lose", "loss", "tp", "sl", "be", "breakeven",
        "pnl", "profit", "ret", "return", "stop", "hit", "exit"
    ]
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in result_kw):
            return c, "result"

    # 3) 方向类模糊关键词
    dir_kw = ["direction", "dir", "side", "long_short", "signal_dir", "signal_side", "pos", "position"]
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in dir_kw):
            return c, "direction"

    return None, None



def drop_tz(series_dt):
    """安全去时区：有时区 -> tz_convert(None)，无时区 -> tz_localize(None)"""
    try:
        if getattr(series_dt.dt, "tz", None) is not None:
            return series_dt.dt.tz_convert(None)
        else:
            return series_dt.dt.tz_localize(None)
    except Exception:
        try:
            return series_dt.dt.tz_convert(None)
        except Exception:
            return series_dt.dt.tz_localize(None)


def normalize_label(value, column_name=None):
    if pd.isna(value):
        return "other"
    s = str(value).strip().lower()
    col = (column_name or "").lower()

    # —— 方向语义 —— 
    if col in {"direction", "side", "dir"}:
        if s in {"1", "+1", "1.0", "long", "buy", "多"}:
            return "long"
        if s in {"-1", "-1.0", "short", "sell", "空"}:
            return "short"
        return "other"

    # —— 结果语义（胜负）——
    if col == "y":
        if s in {"1", "+1", "1.0"}:  return "win"
        if s in {"0", "0.0", "-1", "-1.0"}: return "lose"
    if col == "exit_reason":
        if s in {"tp", "takeprofit", "hit_tp"}: return "win"
        if s in {"sl", "stoploss", "hit_sl"}:   return "lose"
        if s in {"be", "breakeven"}:            return "neutral"

    if s in {"win", "tp", "takeprofit", "profit", "success", "hit_tp", "up"}:
        return "win"
    if s in {"lose", "loss", "sl", "stoploss", "hit_sl", "down", "fail"}:
        return "lose"
    if s in {"neutral", "0", "0.0", "be", "breakeven", "flat", "draw", "noop", "no_trade"}:
        return "neutral"

    return "other"



def infer_symbol_from_filename(path: Path):
    """从文件名推断代码：优先匹配 'XXXX_events.csv' 前缀"""
    m = re.match(r"^([A-Za-z0-9\-\._]+)_events\.csv$", path.name)
    if m:
        return m.group(1).upper()
    return path.stem.upper()


def read_one_csv(path: Path):
    df = pd.read_csv(path)

    # —— 日期列选择：优先 t_signal/t_entry，否则在候选里大小写不敏感找 ——
    has_sig = any(c.lower() == "t_signal" for c in df.columns)
    has_ent = any(c.lower() == "t_entry"  for c in df.columns)

    if DATE_PREFERENCE == "signal" and has_sig:
        chosen_col = [c for c in df.columns if c.lower()=="t_signal"][0]
    elif DATE_PREFERENCE == "entry" and has_ent:
        chosen_col = [c for c in df.columns if c.lower()=="t_entry"][0]
    else:
        chosen_col = pick_first_existing_ci(df.columns, DATE_COL_CANDIDATES)

    if not chosen_col:
        raise ValueError(f"{path.name} 找不到日期列，候选={DATE_COL_CANDIDATES}")

    df["dt"] = pd.to_datetime(df[chosen_col], errors="coerce", utc=False)
    df = df.dropna(subset=["dt"])
    df["dt"] = drop_tz(df["dt"])

    # —— 标签列：用 find_label_col 智能探测 ——
    label_col, label_sem = find_label_col(df.columns)
    if not label_col:
        df["label"] = "other"
    else:
        df["label"] = df[label_col].apply(lambda v: normalize_label(v, label_col if label_sem=="result" else "direction"))

    # —— 代码列：有就用，没有就文件名推断 ——
    sym_col = None
    for c in df.columns:
        if c.lower() == "symbol":
            sym_col = c; break
    if sym_col:
        df["symbol"] = df[sym_col].astype(str).str.upper()
    else:
        df["symbol"] = infer_symbol_from_filename(path)

    return df[["symbol", "dt", "label"]]


def load_all_events(events_dir: Path, pattern: str):
    """批量读取并拼接所有 events CSV"""
    frames = []
    files = sorted(events_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {events_dir} 下没有找到 {pattern}")
    for p in files:
        try:
            frames.append(read_one_csv(p))
        except Exception as e:
            print(f"[WARN] 跳过 {p.name}: {e}")
    if not frames:
        raise RuntimeError("没有成功读入任何文件（日期/标签列可能不匹配）")
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("dt")
    return all_df


def compute_yearly_rates(df: pd.DataFrame):
    """
    计算每个 (symbol, label) 的年发生率 r：
      r = total_events / 覆盖年数
      覆盖年数 = (最晚年 - 最早年 + 1)  —— 偏保守
    """
    df["year"] = df["dt"].dt.year
    counts = df.groupby(["symbol", "label"]).size().rename("total").reset_index()
    yrange = df.groupby("symbol")["year"].agg(["min", "max"]).rename(columns={"min":"ymin","max":"ymax"}).reset_index()
    merged = counts.merge(yrange, on="symbol", how="left")
    merged["years"] = (merged["ymax"] - merged["ymin"] + 1).clip(lower=1)
    merged["r_per_year"] = merged["total"] / merged["years"]

    # 达到“每类每标的≥100”所需年限
    target = PER_CLASS_PER_SYMBOL_TARGET
    merged["T_needed_years"] = merged["r_per_year"].apply(lambda r: math.inf if r <= 0 else math.ceil(target / r))
    return merged.sort_values(["symbol", "label"])


def planning_suggestion(rates_df: pd.DataFrame):
    """
    主分支：若识别到 win/lose/neutral，则用它们的“中位 r”来估算需要的标的数；
    否则兜底：用“所有标签合计”的 r 来估。
    """
    sub = rates_df[rates_df["label"].isin(["win","lose","neutral"])]
    if sub.empty:
        print("\n[提示] 未识别出 win/lose/neutral；将以所有标签合计的 r 来估算。")
        tmp = rates_df.groupby("symbol")["r_per_year"].sum().median()
        r_valid = tmp * VALID_YEARS
        r_test  = tmp * TEST_YEARS
        n_valid = math.inf if r_valid <= 0 else math.ceil(VALID_TOTAL_TARGET / r_valid)
        n_test  = math.inf if r_test  <= 0 else math.ceil(TEST_TOTAL_TARGET  / r_test )
        print(f"[估算-兜底] 每标的在 Valid≈{r_valid:.1f}、Test≈{r_test:.1f} 条；"
              f"为满足总量，需标的数 ≥ {max(n_valid, n_test)}。")
        return

    med = sub.groupby("label")["r_per_year"].median().to_dict()

    def need_symbols(total_target, years):
        # 保守：取 win/lose 的较小中位 r（因为你希望每类都够）
        r_small = min(med.get("win", 0), med.get("lose", 0))
        per_symbol = r_small * years
        return math.inf if per_symbol <= 0 else math.ceil(total_target / per_symbol)

    n_valid = need_symbols(VALID_TOTAL_TARGET, VALID_YEARS)
    n_test  = need_symbols(TEST_TOTAL_TARGET,  TEST_YEARS)

    print("\n===== 规划建议（主分支：基于 win/lose 的中位 r）=====")
    print(f"r_win(中位) ≈ {med.get('win',0):.2f} /年")
    print(f"r_lose(中位)≈ {med.get('lose',0):.2f} /年")
    print(f"窗口设置：Valid={VALID_YEARS} 年，Test={TEST_YEARS} 年。")
    if math.isfinite(n_valid) and math.isfinite(n_test):
        print(f"- 达到总量目标（Valid≥{VALID_TOTAL_TARGET}、Test≥{TEST_TOTAL_TARGET}），"
              f"大约需要标的数 ≥ {max(n_valid, n_test)}（按每类口径保守估计）。")
    else:
        print("- 当前 r 太低，单靠加标的难以达标；建议延长年限或提高触发频率（分钟级/多起点采样）。")

    need_years_win  = math.inf if med.get("win",0)  <= 0 else math.ceil(PER_CLASS_PER_SYMBOL_TARGET / med["win"])
    need_years_lose = math.inf if med.get("lose",0) <= 0 else math.ceil(PER_CLASS_PER_SYMBOL_TARGET / med["lose"])
    print(f"- 若想“每类每标的≥{PER_CLASS_PER_SYMBOL_TARGET}”，以中位 r 估："
          f"win≈{need_years_win} 年、lose≈{need_years_lose} 年。")


def probe_labels(events_dir: Path, pattern: str, max_files=10, max_rows=50000):
    """探针：打印每个样本文件里被选中的“标签列”与标准化后的频次"""
    files = sorted(events_dir.glob(pattern))[:max_files]
    if not files:
        print(f"[Probe] 没有文件匹配 {pattern}"); return

    picked = []   # (filename, picked_col, semantic)
    norm_vals = []

    for p in files:
        df = pd.read_csv(p, nrows=max_rows)
        col, sem = find_label_col(df.columns)
        picked.append((p.name, col, sem))
        if col:
            ser = df[col].apply(lambda v: normalize_label(v, col if sem=="result" else "direction"))
            norm_vals.extend(ser.astype(str).tolist())

    print("\n[Probe] 每个文件被选中的标签列：")
    for name, col, sem in picked:
        print(f"  {name:40s} -> {col or 'None'} ({sem or '—'})")

    if norm_vals:
        print("\n[Probe] 标准化后标签频次（Top 20）：")
        print(pd.Series([s.lower().strip() for s in norm_vals]).value_counts().head(20))
    else:
        print("\n[Probe] 仍未识别到任何标签列。下面是常见解决方案：")
        print("  1) 确认事件表里确实存在“结果列”（如 y、result、exit_reason、win_or_loss 等）；")
        print("  2) 若列名很异类，把它加入 LABEL_COL_CANDIDATES 或在 find_label_col 的关键词里补上；")
        print("  3) 临时回退用方向列（direction/side/dir 等），确保能先跑通 r 的计算。")

def main():
    # 支持：python estimate_event_rate.py probe   —— 先探针再主流程
    if len(sys.argv) > 1 and sys.argv[1].lower() == "probe":
        probe_labels(Path(EVENTS_DIR), FILE_GLOB)
        return

    events_dir = Path(EVENTS_DIR)
    all_df = load_all_events(events_dir, FILE_GLOB)

    rates = compute_yearly_rates(all_df)

    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "event_rate_summary.csv"
    rates.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 已生成报告：{out_path.resolve()}")

    planning_suggestion(rates)


if __name__ == "__main__":
    main()
