# -*- coding: utf-8 -*-
"""
QC Step 6 — multi symbol (v3)
- 修复：labels 小写、OHLCV 大写导致的匹配失败
- 新增：符号归一 + 别名映射（BRK -> BRK_B 等）
- 保留：v2 的时间戳 t_entry 适配、Wilson 区间、Laplace 平滑、自动回放 & 绘图
"""
from pathlib import Path
from datetime import datetime
import math, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ========== 目录配置 ==========
BASE = Path(r"D:\chan\chan.py")
DIR_OHLCV   = BASE / r"data\stooq\1d_15y"
DIR_LABELS  = BASE / r"events_outputs\labels"
DIR_BARR    = BASE / r"events_outputs\barriers"
OUTDIR      = BASE / r"qc_reports" / f"{datetime.now():%Y%m%d_%H%M%S}_step6"
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "spotcheck").mkdir(parents=True, exist_ok=True)

# ========== 参数 ==========
N_MIN_PER_SYMBOL = 20
SPOTCHECK_TARGET = 60
WINDOW_LEFT = 40
WINDOW_RIGHT = 15
TIE_RULE = "tp_first"  # 或 "tp_first"
START_FROM_ENTRY = False   # 若你的打标从“入场当根”开始，改为 True

# ===== 符号归一 & 别名 =====
def norm_sym(s: str) -> str:
    """大小写统一、去除非字母数字"""
    s = "".join(ch for ch in str(s).upper() if ch.isalnum())
    return s

# 你可以按需要增删映射；左边写“labels 里出现的样子”，右边写“行情目录里的规范名字”
ALIAS = {
    "BRK": "BRK_B",   # 若你行情目录用 BRK_A 就改成 BRK_A
    # "BF": "BF_B",
    # 其它有点/横线/下划线的，都可以在这里补
}

def detect_direction_from_row(row, entry_price, tp, sl):
    """
    优先用 labeled 的 direction；否则用 tp/sl 相对 entry_price 来推断。
    返回 +1（做多）或 -1（做空）；无法判断时默认 +1。
    """
    if "direction" in row and pd.notna(row["direction"]):
        try:
            d = int(row["direction"])
            if d in (-1, 1): 
                return d
        except Exception:
            pass
    # 用 tp/sl 推断：多 -> tp>entry & sl<entry；空 -> tp<entry & sl>entry
    try:
        if (tp > entry_price) and (sl < entry_price):
            return +1
        if (tp < entry_price) and (sl > entry_price):
            return -1
    except Exception:
        pass
    return +1


def wilson_ci(p_hat, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    denom = 1 + z**2/n
    center = (p_hat + z*z/(2*n)) / denom
    margin = (z * math.sqrt((p_hat*(1-p_hat) + z*z/(4*n)) / n)) / denom
    return (center - margin, center + margin)

def to_utc_ts(x): return pd.to_datetime(x, utc=True, errors="coerce")

def find_ts_column(df: pd.DataFrame):
    cands = [c for c in df.columns if any(k in str(c).lower() for k in ["ts","time","date","stamp"])]
    for c in cands:
        if to_utc_ts(df[c]).notna().any(): return c
    for c in df.columns:
        if to_utc_ts(df[c]).notna().any(): return c
    return None

# —— 建立“可用行情符号索引”：归一键 -> 规范 symbol（文件名中的）
def build_symbol_index():
    idx = {}
    for p in DIR_OHLCV.glob("*_1d_15y.parquet"):
        stem = p.stem  # e.g., "AAPL_1d_15y"
        if not stem.endswith("_1d_15y"): continue
        sym = stem[:-len("_1d_15y")]     # "AAPL"
        idx[norm_sym(sym)] = sym
    return idx

# —— 更稳健地从 labels 文件名解析 symbol（不强制大小写）
def parse_symbol_from_labels_filename(fp: Path) -> str:
    base = fp.stem  # e.g., "aapl_events_labeled", "AAPL_1d_15y_events_labeled"
    if base.endswith("_events_labeled"):
        base = base[:-len("_events_labeled")]
    for suf in ["_1d_15y", "_1d", "_D", "_daily"]:
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return base  # 可能是大小写混合，后续统一归一

# ========== 数据加载 ==========
def load_all_labeled():
    sym_index = build_symbol_index()  # norm -> canonical
    rows, skipped = [], []

    for fp in sorted(DIR_LABELS.glob("*_events_labeled.csv")):
        raw_sym = parse_symbol_from_labels_filename(fp)  # 原样 e.g., "aapl"
        key = norm_sym(raw_sym)                          # 归一键 e.g., "AAPL"
        # alias 尝试
        if key not in sym_index:
            # 例如 raw_sym="brk" => alias "BRK_B"
            alias_target = ALIAS.get(raw_sym.upper())
            if alias_target and norm_sym(alias_target) in sym_index:
                key = norm_sym(alias_target)
        if key not in sym_index:
            skipped.append((fp.name, raw_sym))
            continue

        canonical = sym_index[key]  # e.g., "AAPL"
        df = pd.read_csv(fp)
        df["symbol"] = canonical
        df["__src_labels__"] = str(fp)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No usable labeled CSV in {DIR_LABELS}. Skipped: {skipped}")

    all_lab = pd.concat(rows, ignore_index=True)
    if skipped:
        with open(OUTDIR / "SKIPPED_LABEL_FILES.txt", "w", encoding="utf-8") as f:
            for name, sym in skipped:
                f.write(f"skip {name} (parsed='{sym}') — no matching OHLCV parquet\n")
    print(f"[load_all_labeled] loaded symbols: {sorted(all_lab['symbol'].unique().tolist())[:10]} ... "
          f"({all_lab['symbol'].nunique()} syms). Skipped {len(skipped)} files.")
    return all_lab

def load_barriers_for_symbol(sym: str) -> pd.DataFrame | None:
    fp = DIR_BARR / f"{sym}_events_with_barriers.csv"
    if not fp.exists(): return None
    df = pd.read_csv(fp)
    if "t_entry" in df.columns:
        df["barr_ts"] = to_utc_ts(df["t_entry"])
    elif "entry_time" in df.columns:
        df["barr_ts"] = to_utc_ts(df["entry_time"])
    else:
        df["barr_ts"] = pd.NaT
    return df

def load_ohlcv_for_symbol(sym: str) -> pd.DataFrame:
    fp = DIR_OHLCV / f"{sym}_1d_15y.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"OHLCV parquet not found: {fp}")
    df = pd.read_parquet(fp)
    df.columns = [str(c).lower() for c in df.columns]
    need = {"open","high","low","close"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"OHLCV missing {need - set(df.columns)} in {fp}")
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df["ts"] = to_utc_ts(df.index)
    else:
        ts_col = find_ts_column(df)
        df["ts"] = to_utc_ts(df[ts_col]) if ts_col else pd.Series(pd.date_range("2000-01-01", periods=len(df), freq="D", tz="UTC"))
    df = df.reset_index(drop=True)
    df.insert(0, "iloc", np.arange(len(df), dtype=int))
    return df[["iloc","ts","open","high","low","close"] + [c for c in df.columns if c not in ["iloc","ts","open","high","low","close"]]]

# ========== 自检 1：标签占比 ==========
def check_class_balance(all_lab: pd.DataFrame) -> pd.DataFrame:
    df = all_lab.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    n_all, pos_all = len(df), int((df["y"]==1).sum())
    p_all = pos_all / n_all if n_all else np.nan
    lb_all, ub_all = wilson_ci(p_all, n_all)
    total = pd.DataFrame([{
        "symbol": "__ALL__", "n": n_all, "pos": pos_all,
        "pos_rate": p_all, "pos_rate_wilson_lb": lb_all, "pos_rate_wilson_ub": ub_all,
        "pos_rate_laplace": (pos_all + 1)/(n_all + 2) if n_all else np.nan
    }])
    agg = (df.groupby("symbol")
             .agg(n=("y","size"), pos=("y","sum"))
             .reset_index())
    agg["pos_rate"] = agg["pos"]/agg["n"]
    agg["pos_rate_laplace"] = (agg["pos"] + 1)/(agg["n"] + 2)
    ci = agg.apply(lambda r: pd.Series(wilson_ci(r["pos_rate"], r["n"]),
                     index=["pos_rate_wilson_lb","pos_rate_wilson_ub"]), axis=1)
    out = pd.concat([total, pd.concat([agg, ci], axis=1)], ignore_index=True)
    out.to_csv(OUTDIR / "qc_class_balance.csv", index=False)
    return out

# ========== 自检 2：持仓根数 ==========
def check_bars_held(all_lab: pd.DataFrame) -> pd.DataFrame:
    need_cols = {"bars_held","exit_reason"}
    if not need_cols.issubset(all_lab.columns):
        raise ValueError(f"labeled CSV missing {need_cols - set(all_lab.columns)}")
    df = all_lab.dropna(subset=["bars_held"]).copy()
    df["bars_held"] = df["bars_held"].astype(int)
    df["timeout"] = (df["exit_reason"].astype(str)=="TIMEOUT").astype(int)
    df["firstbar"] = (df["bars_held"]==1).astype(int)
    total = pd.DataFrame([{
        "symbol":"__ALL__", "n":len(df),
        "timeout_rate": df["timeout"].mean() if len(df) else np.nan,
        "firstbar_rate": df["firstbar"].mean() if len(df) else np.nan,
        "median_bars": df["bars_held"].median() if len(df) else np.nan,
        "p90_bars": df["bars_held"].quantile(0.90) if len(df) else np.nan
    }])
    bysym = (df.groupby("symbol")
               .agg(n=("bars_held","size"),
                    timeout_rate=("exit_reason", lambda s: (s.astype(str)=="TIMEOUT").mean()),
                    firstbar_rate=("bars_held",  lambda s: (s.astype(int)==1).mean()),
                    median_bars=("bars_held","median"),
                    p90_bars=("bars_held",  lambda s: s.quantile(0.90)))
               .reset_index())
    out = pd.concat([total, bysym], ignore_index=True)
    out.to_csv(OUTDIR / "qc_bars_held.csv", index=False)
    if len(df):
        plt.figure(figsize=(8,4.5))
        df["bars_held"].plot(kind="hist", bins=40)
        plt.title("Bars Held Histogram (ALL)")
        plt.xlabel("bars_held")
        plt.tight_layout()
        plt.savefig(OUTDIR / "bars_held_hist_all.png", dpi=150)
        plt.close()
    return out

# ========== 自检 3：抽检 & 自动回放 ==========
def find_first_hit(ohlcv: pd.DataFrame, entry_iloc: int, tp: float, sl: float, H: int,
                   direction: int = +1, tie: str = "sl_first", start_from_entry: bool = False):
    """
    direction=+1 做多：TP 在上、SL 在下；命中条件：high>=TP、low<=SL
    direction=-1 做空：TP 在下、SL 在上；命中条件：low<=TP、high>=SL
    start_from_entry: True 则从 entry_iloc 开始检查；False 则从 entry_iloc+1 开始
    """
    start = entry_iloc if start_from_entry else entry_iloc + 1
    end_iloc = min(entry_iloc + H, int(ohlcv["iloc"].max()))

    for loc in range(start, end_iloc + 1):
        row = ohlcv.iloc[loc]
        if direction == +1:
            hit_tp = row["high"] >= tp
            hit_sl = row["low"]  <= sl
        else:  # 做空
            hit_tp = row["low"]  <= tp
            hit_sl = row["high"] >= sl

        if hit_tp and hit_sl:
            return (loc, "SL&TP_samebar->SL") if tie=="sl_first" else (loc, "SL&TP_samebar->TP")
        if hit_tp:
            return loc, "TP"
        if hit_sl:
            return loc, "SL"
    return end_iloc, "TIMEOUT"


def resolve_entry_iloc_from_labeled_row(r: pd.Series, ohlcv: pd.DataFrame) -> int:
    val = r["t_entry"]
    try:
        return int(val)
    except Exception:
        pass
    t = to_utc_ts(val)
    if pd.isna(t):
        for alt in ["t_entry_ts","entry_time","t_signal","signal_time"]:
            if alt in r.index:
                t = to_utc_ts(r[alt]); 
                if not pd.isna(t): break
    if pd.isna(t): raise ValueError(f"Cannot parse t_entry as time: {val}")
    ts = to_utc_ts(ohlcv["ts"])
    mask = (ts == t)
    if mask.any(): return int(ohlcv.loc[mask, "iloc"].iloc[0])
    mask2 = (ts.dt.date == t.date())
    if mask2.any(): return int(ohlcv.loc[mask2, "iloc"].iloc[0])
    pos = ts.searchsorted(t); pos = int(np.clip(pos, 0, len(ts)-1))
    return int(ohlcv.iloc[pos]["iloc"])

def spotcheck(all_lab: pd.DataFrame):
    need = {"y","t_entry","exit_reason"}
    for c in need:
        if c not in all_lab.columns:
            raise ValueError(f"labeled CSV missing column: {c}")
    df = all_lab.dropna(subset=list(need)).copy()
    df["y"] = df["y"].astype(int)

    rows = []
    cache_ohlcv, cache_barr = {}, {}

    # 分层抽样
    pos = df[df["y"]==1]; neg = df[df["y"]==0]
    n_pos = min(SPOTCHECK_TARGET//2, len(pos))
    n_neg = min(SPOTCHECK_TARGET - n_pos, len(neg))
    picked = pd.concat([
        pos.sample(n=n_pos, random_state=42) if n_pos>0 else pos.head(0),
        neg.sample(n=n_neg, random_state=42) if n_neg>0 else neg.head(0)
    ], ignore_index=True)
    if picked.empty:
        return pd.DataFrame(columns=["symbol","event_id","match","reason_expected","reason_replay"])

    skipped_syms = set()
    for _, r in picked.iterrows():
        sym = r["symbol"]

        # 加载行情（不存在就跳过并记录）
        try:
            if sym not in cache_ohlcv:
                cache_ohlcv[sym] = load_ohlcv_for_symbol(sym)
            ohlcv = cache_ohlcv[sym]
        except FileNotFoundError:
            skipped_syms.add(sym); continue

        entry_iloc = resolve_entry_iloc_from_labeled_row(r, ohlcv)
        entry_price = float(r["entry_price"]) if "entry_price" in r and pd.notna(r["entry_price"]) \
                       else float(ohlcv.iloc[entry_iloc]["open"])
        exit_reason = str(r["exit_reason"])
        event_id = r.get("event_id", None)
        y = int(r["y"])

        # 载 barriers（event_id 优先，其次时间戳）
        if sym not in cache_barr:
            cache_barr[sym] = load_barriers_for_symbol(sym)
        barr = cache_barr[sym]

        tp = np.nan; sl = np.nan; H = int(r.get("H", r.get("max_holding", 60)))
        br = None
        if barr is not None:
            if "event_id" in barr.columns and pd.notna(event_id):
                hit = barr[barr["event_id"]==event_id]
                if not hit.empty: br = hit.iloc[0]
            if br is None:
                r_ts = to_utc_ts(r["t_entry"])
                if "barr_ts" in barr.columns and pd.notna(r_ts):
                    hit = barr[barr["barr_ts"]==r_ts]
                    if hit.empty:
                        hit = barr[to_utc_ts(barr["barr_ts"]).dt.date == r_ts.date()]
                    if not hit.empty: br = hit.iloc[0]
        if br is not None:
            tp = float(br.get("tp", np.nan)); sl = float(br.get("sl", np.nan))
            H  = int(br.get("H", br.get("max_holding", H)))

        if (np.isnan(tp) or np.isnan(sl)):
            atr_e = r.get("atr_entry", np.nan); k = r.get("k", np.nan); m = r.get("m", np.nan)
            if not (pd.isna(atr_e) or pd.isna(k) or pd.isna(m)):
                tp = entry_price + float(k) * float(atr_e)
                sl = entry_price - float(m) * float(atr_e)
            else:
                continue

        direction = detect_direction_from_row(r, entry_price, float(tp), float(sl))
        replay_loc, replay_reason = find_first_hit(
            ohlcv, entry_iloc, float(tp), float(sl), H,
            direction=direction, tie=TIE_RULE, start_from_entry=START_FROM_ENTRY)

        match = (replay_reason == exit_reason) or (
            replay_reason.startswith("SL&TP_samebar") and exit_reason.startswith("SL&TP_samebar"))

        rows.append({
            "symbol": sym,
            "event_id": event_id if pd.notna(event_id) else f"t{r['t_entry']}",
            "y": y, "t_entry": r["t_entry"], "entry_iloc": entry_iloc,
            "reason_expected": exit_reason, "reason_replay": replay_reason, "match": bool(match),
        })

        # 画图
        left = max(0, entry_iloc - WINDOW_LEFT)
        right = min(int(ohlcv["iloc"].max()), replay_loc + WINDOW_RIGHT)
        seg = ohlcv.iloc[left:right+1]
        plt.figure(figsize=(9.6,4.8))
        plt.plot(seg["iloc"], seg["close"], label="close")
        plt.plot(seg["iloc"], seg["high"], alpha=0.5, label="high")
        plt.plot(seg["iloc"], seg["low"],  alpha=0.5, label="low")
        plt.axhline(entry_price, linestyle="--", label="entry")
        plt.axhline(tp, linestyle="--", label="TP")
        plt.axhline(sl, linestyle="--", label="SL")
        plt.axvline(entry_iloc, color="k", linestyle=":")
        plt.axvline(replay_loc, color="gray", linestyle=":")
        plt.title(f"{sym} | y={y} | exp={exit_reason} | replay={replay_reason}")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUTDIR / "spotcheck" / f"{sym}_{str(event_id) if pd.notna(event_id) else 'noid'}.png", dpi=140)
        plt.close()

    rep = pd.DataFrame(rows)
    if len(rep): rep.to_csv(OUTDIR / "qc_spotcheck_compare.csv", index=False)
    if skipped_syms:
        with open(OUTDIR / "SKIPPED_SYMBOLS.txt", "w", encoding="utf-8") as f:
            for s in sorted(skipped_syms):
                f.write(f"spotcheck skip: OHLCV parquet missing for '{s}'\n")
    return rep

# ========== 主流程 ==========
def main():
    print("Loading labeled...")
    all_lab = load_all_labeled()

    print("== Check 1/3: Class Balance ==")
    balance = check_class_balance(all_lab)

    print("== Check 2/3: Bars Held ==")
    bars_stats = check_bars_held(all_lab)

    print("== Check 3/3: Spot Check ==")
    report = spotcheck(all_lab)

    lines = []
    row_all = balance[balance["symbol"]=="__ALL__"].iloc[0]
    lines.append(f"[ClassBalance] ALL n={int(row_all['n'])}, pos_rate={row_all['pos_rate']:.3f}, "
                 f"wilson95%=[{row_all['pos_rate_wilson_lb']:.3f},{row_all['pos_rate_wilson_ub']:.3f}], "
                 f"laplace={row_all['pos_rate_laplace']:.3f}")

    sub = balance[(balance["symbol"]!="__ALL__") & (balance["n"]>=N_MIN_PER_SYMBOL)]
    bad = sub[(sub["pos_rate"]<0.10) | (sub["pos_rate"]>0.90)]
    if len(bad):
        lines.append("  WARN extreme pos_rate (n>=N_MIN): " + ", ".join(
            f"{r.symbol}:n={int(r.n)},p={r.pos_rate:.2f}" for _,r in bad.iterrows()))

    bar_all = bars_stats[bars_stats["symbol"]=="__ALL__"].iloc[0]
    lines.append(f"[BarsHeld] ALL timeout={bar_all['timeout_rate']:.3f}, firstbar={bar_all['firstbar_rate']:.3f}, "
                 f"median={bar_all['median_bars']}, p90={bar_all['p90_bars']}")

    if len(report):
        lines.append(f"[SpotCheck] auto-replay match_rate={report['match'].mean():.3f} over {len(report)} samples")
    else:
        lines.append("[SpotCheck] no samples (check barriers/t_entry/keys).")

    with open(OUTDIR / "QC_README.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nArtifacts saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
