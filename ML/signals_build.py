# signals_build.py — Slim D-only + Batch Parquet
import pandas as pd
from pathlib import Path

# ========= 工具与校验 =========

def to_datetime_utc(ts):
    """把任意字符串/时间戳转为 UTC 时间戳（DatetimeIndex/Timestamp）"""
    t = pd.to_datetime(ts, utc=True)
    if isinstance(t, pd.DatetimeIndex):
        return t.tz_convert("UTC")
    return t

def normalize_signal_type(x: str) -> str:
    """把各种大小写/别名规范成 '3buy'/'3sell'"""
    s = str(x).strip().lower()
    mapping = {
        "3buy": "3buy", "buy3": "3buy", "third_buy": "3buy",
        "3sell": "3sell", "sell3": "3sell", "third_sell": "3sell",
        "buy": "3buy", "sell": "3sell"
    }
    return mapping.get(s, s)

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 df.index 为升序 UTC 时间戳"""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    df.sort_index(inplace=True)
    return df

def snap_to_index(df: pd.DataFrame, t: pd.Timestamp, method="nearest") -> pd.Timestamp:
    """把任意时间戳对齐到 df.index 上的一根K线。"""
    if t in df.index:
        return t
    pos = df.index.get_indexer([t], method=method)[0]
    return df.index[pos]

def build_signals_df(raw, df, align_method="nearest"):
    """把“原始信号”统一打磨成标准两列表：[t_signal, signal_type]"""
    if isinstance(raw, (list, tuple)):
        sig = pd.DataFrame(raw, columns=["t_signal", "signal_type"])
    else:
        sig = raw.copy()
    sig["signal_type"] = sig["signal_type"].map(normalize_signal_type)
    sig["t_signal"] = sig["t_signal"].map(lambda x: to_datetime_utc(x))
    sig = sig[sig["signal_type"].isin(["3buy", "3sell"])]
    sig = sig[(sig["t_signal"] >= df.index[0]) & (sig["t_signal"] <= df.index[-1])]
    sig["t_signal"] = sig["t_signal"].map(lambda t: snap_to_index(df, t, method=align_method))
    sig = sig.drop_duplicates(subset=["t_signal"]).sort_values("t_signal").reset_index(drop=True)
    return sig[["t_signal", "signal_type"]]

# ========= D：缠论「第三类买卖点」近似检测（强化版） =========

def _find_fractals(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """滚动窗口找顶/底分型；返回 ['ts','ptype','price']，ptype ∈ {'H','L'}"""
    high = df["high"].astype(float); low  = df["low"].astype(float)
    win = left + right + 1
    ph = high.rolling(window=win, center=True).apply(lambda w: 1.0 if w[left] == w.max() else 0.0, raw=False)
    pl = low .rolling(window=win, center=True).apply(lambda w: 1.0 if w[left] == w.min() else 0.0, raw=False)

    pivots = []
    for ts, is_h, is_l in zip(high.index, ph.fillna(0).astype(int), pl.fillna(0).astype(int)):
        if is_h: pivots.append((ts, 'H', float(high.loc[ts])))
        if is_l: pivots.append((ts, 'L', float(low.loc[ts])))

    piv = pd.DataFrame(pivots, columns=["ts", "ptype", "price"]).sort_values("ts").reset_index(drop=True)

    # 清除连续同类分型中“没更极端”的点，保持 HLHL 交替
    cleaned, last = [], None
    for row in piv.itertuples(index=False):
        if last is None: last = row; continue
        if row.ptype == last.ptype:
            if row.ptype == 'H':
                if row.price > last.price: last = row
            else:
                if row.price < last.price: last = row
        else:
            cleaned.append(last); last = row
    if last is not None: cleaned.append(last)
    return pd.DataFrame(cleaned, columns=["ts","ptype","price"])

def _build_strokes(piv: pd.DataFrame, min_change_pct: float = 0.002) -> pd.DataFrame:
    """由分型串成“笔”，返回 ['t0','p0','t1','p1','dir','hi','lo']"""
    strokes, last = [], None
    for row in piv.itertuples(index=False):
        if last is None: last = row; continue
        if row.ptype == last.ptype:
            if row.ptype == 'H' and row.price > last.price: last = row
            elif row.ptype == 'L' and row.price < last.price: last = row
            continue
        p0, p1 = last.price, row.price
        chg = abs(p1 - p0) / max(1e-12, p0)
        if chg < min_change_pct:
            if (row.ptype == 'H' and row.price > last.price) or (row.ptype == 'L' and row.price < last.price):
                last = row
            continue
        direction = 'up' if p1 > p0 else 'down'
        hi, lo = (max(p0, p1), min(p0, p1))
        strokes.append({"t0": last.ts, "p0": p0, "t1": row.ts, "p1": p1, "dir": direction, "hi": hi, "lo": lo})
        last = row
    return pd.DataFrame(strokes)

def _build_centers(strokes: pd.DataFrame, min_strokes: int = 3) -> pd.DataFrame:
    """三笔重叠求交集形成中枢，能延展则右移；返回 ['i_start','i_end','z_low','z_high','t_start','t_end']"""
    cz, i, n = [], 0, len(strokes)
    while i <= n - min_strokes:
        lo = max(strokes.loc[i:i+min_strokes-1, "lo"])
        hi = min(strokes.loc[i:i+min_strokes-1, "hi"])
        if lo < hi:
            j = i + min_strokes
            while j < n:
                lo2 = max(lo, strokes.loc[j, "lo"])
                hi2 = min(hi, strokes.loc[j, "hi"])
                if lo2 < hi2: lo, hi, j = lo2, hi2, j+1
                else: break
            cz.append({"i_start": i, "i_end": j-1, "z_low": lo, "z_high": hi,
                       "t_start": strokes.loc[i, "t0"], "t_end": strokes.loc[j-1, "t1"]})
            i = j
        else:
            i += 1
    return pd.DataFrame(cz)

# —— 改良版三买/三卖工具 —— #
def _ix_range(df, t0, t1):
    i0 = df.index.get_loc(pd.Timestamp(t0)); i1 = df.index.get_loc(pd.Timestamp(t1))
    if isinstance(i0, slice): i0 = i0.stop - 1
    if isinstance(i1, slice): i1 = i1.stop - 1
    return int(i0), int(i1)

def _first_bar_break(df, start_i, end_i, *, up, level):
    highs, lows = df["high"].astype(float), df["low"].astype(float)
    for k in range(start_i, end_i + 1):
        if up and highs.iloc[k] > level: return k
        if (not up) and lows.iloc[k] < level: return k
    return None

def _find_opposite_pull_stroke(leave_i, strokes, df, *, up, z_low, z_high, min_bars):
    """从离开点后寻找一笔完整反向笔，且整笔不回到中枢带。返回 (pb_end_i, anchor_px) 或 (None, None)。"""
    t_leave = df.index[leave_i]
    for st in strokes.itertuples(index=False):
        if pd.Timestamp(st.t0) <= t_leave: continue
        if up and st.dir != "down": continue
        if (not up) and st.dir != "up": continue
        i0, i1 = _ix_range(df, st.t0, st.t1)
        highs = df["high"].astype(float).iloc[i0:i1+1]
        lows  = df["low"].astype(float).iloc[i0:i1+1]
        if up:
            if len(lows) >= min_bars and float(lows.min()) >= float(z_high):
                return i1, float(df["high"].iloc[leave_i])   # anchor=离开当根的 swing high
        else:
            if len(highs) >= min_bars and float(highs.max()) <= float(z_low):
                return i1, float(df["low"].iloc[leave_i])    # anchor=离开当根的 swing low
    return None, None

def _dynamic_lookahead(df, strokes, center_row, *, alpha=1.0, beta=1.0, hard_min=30, hard_max=300):
    i_c0, i_c1 = _ix_range(df, center_row.t_start, center_row.t_end)
    span_center = max(1, i_c1 - i_c0 + 1)
    st = strokes.iloc[int(center_row.i_end)]
    i_s0, i_s1 = _ix_range(df, st.t0, st.t1)
    span_last_stroke = max(1, i_s1 - i_s0 + 1)
    L = int(alpha * span_center + beta * span_last_stroke)
    return max(hard_min, min(hard_max, L))

def _detect_third_points(df: pd.DataFrame,
                         strokes: pd.DataFrame,
                         centers: pd.DataFrame,
                         *,
                         pullback_min_bars: int = 1,
                         lookahead_bars: int | None = None,
                         lookahead_alpha: float = 1.0,
                         lookahead_beta: float = 1.0,
                         confirm_force: bool = False,
                         min_force_ratio: float = 0.7,
                         higher_centers: pd.DataFrame | None = None) -> list:
    """三买/三卖：高/低价破位 → 完整反向笔回/反抽（不回中枢带）→ 再破高/低；可选力度 & 高一级过滤。"""
    res = []
    highs, lows = df["high"].astype(float), df["low"].astype(float)
    for cz in centers.itertuples(index=False):
        LA = lookahead_bars or _dynamic_lookahead(df, strokes, cz, alpha=lookahead_alpha, beta=lookahead_beta)
        start_i = df.index.get_loc(pd.Timestamp(cz.t_end)); start_i = start_i.stop - 1 if isinstance(start_i, slice) else start_i
        i0, i1 = int(start_i) + 1, min(len(df) - 1, int(start_i) + 1 + LA)

        # 三买（向上离开）
        leave_up = _first_bar_break(df, i0, i1, up=True, level=float(cz.z_high))
        if leave_up is not None:
            pb_end, anchor = _find_opposite_pull_stroke(leave_up, strokes, df, up=True,
                                                        z_low=float(cz.z_low), z_high=float(cz.z_high),
                                                        min_bars=pullback_min_bars)
            if pb_end is not None:
                trigger = _first_bar_break(df, pb_end + 1, i1, up=True, level=anchor)
                if trigger is not None:
                    ok = True
                    if confirm_force:
                        st_ix = strokes[(strokes["t0"] <= df.index[trigger]) & (strokes["t1"] >= df.index[trigger])].index
                        if len(st_ix) > 0:
                            j = int(st_ix[0]); last_up = strokes.iloc[j]; prev = strokes.iloc[max(0, j-1)]
                            def _force(st):
                                s0, s1 = _ix_range(df, st.t0, st.t1)
                                return abs(float(st.p1) - float(st.p0)) / max(1, s1 - s0 + 1)
                            ok = (_force(last_up) >= min_force_ratio * _force(prev))
                    if ok and higher_centers is not None and len(higher_centers) > 0:
                        hc = higher_centers[higher_centers["t_start"] <= df.index[trigger]]
                        if len(hc) > 0:
                            hc = hc.iloc[-1]; ok = float(lows.iloc[trigger]) >= float(hc["z_high"])
                    if ok: res.append((df.index[trigger], "3buy"))

        # 三卖（向下离开）
        leave_dn = _first_bar_break(df, i0, i1, up=False, level=float(cz.z_low))
        if leave_dn is not None:
            pb_end, anchor = _find_opposite_pull_stroke(leave_dn, strokes, df, up=False,
                                                        z_low=float(cz.z_low), z_high=float(cz.z_high),
                                                        min_bars=pullback_min_bars)
            if pb_end is not None:
                trigger = _first_bar_break(df, pb_end + 1, i1, up=False, level=anchor)
                if trigger is not None:
                    ok = True
                    if confirm_force:
                        st_ix = strokes[(strokes["t0"] <= df.index[trigger]) & (strokes["t1"] >= df.index[trigger])].index
                        if len(st_ix) > 0:
                            j = int(st_ix[0]); last_dn = strokes.iloc[j]; prev = strokes.iloc[max(0, j-1)]
                            def _force(st):
                                s0, s1 = _ix_range(df, st.t0, st.t1)
                                return abs(float(st.p1) - float(st.p0)) / max(1, s1 - s0 + 1)
                            ok = (_force(last_dn) >= min_force_ratio * _force(prev))
                    if ok and higher_centers is not None and len(higher_centers) > 0:
                        hc = higher_centers[higher_centers["t_start"] <= df.index[trigger]]
                        if len(hc) > 0:
                            hc = hc.iloc[-1]; ok = float(highs.iloc[trigger]) <= float(hc["z_low"])
                    if ok: res.append((df.index[trigger], "3sell"))
    return res

def make_signals_chan_third(
        df: pd.DataFrame,
        fractal_left: int = 2,
        fractal_right: int = 2,
        stroke_min_change_pct: float = 0.002,
        center_min_strokes: int = 3,
        pullback_min_bars: int = 2,
        lookahead_bars: int | None = None,
        lookahead_alpha: float = 1.0,
        lookahead_beta: float = 1.0,
        confirm_force: bool = False,
        min_force_ratio: float = 0.7,
        higher_df: pd.DataFrame | None = None,
        higher_center_min_strokes: int = 5) -> pd.DataFrame:
    """改良：高/低价破位；回抽/反抽用一整笔；自适应 lookahead；可选力度+高一级中枢过滤。"""
    piv = _find_fractals(df, left=fractal_left, right=fractal_right)
    if len(piv) < 4: return pd.DataFrame(columns=["t_signal", "signal_type"])
    strokes = _build_strokes(piv, min_change_pct=stroke_min_change_pct)
    if len(strokes) < 3: return pd.DataFrame(columns=["t_signal", "signal_type"])
    centers = _build_centers(strokes, min_strokes=center_min_strokes)
    if len(centers) == 0: return pd.DataFrame(columns=["t_signal", "signal_type"])

    higher_centers = None
    if higher_df is not None:
        hpiv = _find_fractals(higher_df, left=fractal_left, right=fractal_right)
        hstk = _build_strokes(hpiv, min_change_pct=stroke_min_change_pct)
        higher_centers = _build_centers(hstk, min_strokes=higher_center_min_strokes)
    elif center_min_strokes < higher_center_min_strokes:
        # 没有更大周期，就用“更严格 min_strokes”模拟更高一级中枢作过滤
        higher_centers = _build_centers(strokes, min_strokes=higher_center_min_strokes)

    sig_list = _detect_third_points(
        df, strokes, centers,
        pullback_min_bars=pullback_min_bars,
        lookahead_bars=lookahead_bars,
        lookahead_alpha=lookahead_alpha,
        lookahead_beta=lookahead_beta,
        confirm_force=confirm_force,
        min_force_ratio=min_force_ratio,
        higher_centers=higher_centers
    )
    return build_signals_df(sig_list, df, align_method="nearest")

# ========= 体检报告 =========

def validate_signals(df: pd.DataFrame, signals: pd.DataFrame) -> dict:
    rpt = {"PASS": True, "errors": [], "warnings": [], "stats": {}}
    need_cols = {"t_signal", "signal_type"}
    if not need_cols.issubset(signals.columns):
        rpt["PASS"] = False; rpt["errors"].append(f"signals 缺少列: {need_cols - set(signals.columns)}"); return rpt
    if signals["t_signal"].isna().any(): rpt["PASS"] = False; rpt["errors"].append("t_signal 存在空值")
    if signals["signal_type"].isna().any(): rpt["PASS"] = False; rpt["errors"].append("signal_type 存在空值")
    if len(signals):
        if (signals["t_signal"] < df.index[0]).any() or (signals["t_signal"] > df.index[-1]).any():
            rpt["PASS"] = False; rpt["errors"].append("存在超出 df 时间范围的 t_signal")
        not_in = ~signals["t_signal"].isin(df.index)
        if not_in.any():
            rpt["PASS"] = False; rpt["errors"].append(f"{int(not_in.sum())} 条信号未对齐到 df.index")
    dup = signals["t_signal"].duplicated().sum()
    if dup > 0: rpt["warnings"].append(f"发现 {dup} 条重复 t_signal（已建议去重）")
    counts = signals["signal_type"].value_counts().to_dict()
    rpt["stats"]["rows"] = int(len(signals))
    rpt["stats"]["types"] = counts
    rpt["stats"]["start"] = str(signals["t_signal"].min()) if len(signals) else None
    rpt["stats"]["end"]   = str(signals["t_signal"].max()) if len(signals) else None
    return rpt

# ========= 主入口：单文件 or 批处理文件夹 =========

def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    else:
        raise ValueError("仅支持 .parquet 或 .csv")
    return ensure_utc_index(df)

def process_one_df(df: pd.DataFrame,
                   *,
                   fractal_left=2, fractal_right=2,
                   stroke_min_change_pct=0.002,
                   center_min_strokes=3,
                   pullback_min_bars=2,
                   lookahead_bars=None,
                   lookahead_alpha=1.0, lookahead_beta=1.0,
                   confirm_force=False, min_force_ratio=0.7,
                   higher_df=None, higher_center_min_strokes=7) -> pd.DataFrame:
    return make_signals_chan_third(
        df,
        fractal_left=fractal_left, fractal_right=fractal_right,
        stroke_min_change_pct=stroke_min_change_pct,
        center_min_strokes=center_min_strokes,
        pullback_min_bars=pullback_min_bars,
        lookahead_bars=lookahead_bars,
        lookahead_alpha=lookahead_alpha, lookahead_beta=lookahead_beta,
        confirm_force=confirm_force, min_force_ratio=min_force_ratio,
        higher_df=higher_df, higher_center_min_strokes=higher_center_min_strokes
    )

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="D-only 三买/三卖信号生成（支持单文件或批处理文件夹）")
    parser.add_argument("--path", type=str, required=True,
                        help="单个 .parquet/.csv 文件，或包含 .parquet 的文件夹路径")
    parser.add_argument("--recursive", action="store_true", help="文件夹模式下递归查找所有 .parquet")
    #递归子目录：不仅处理你指定文件夹里的 .parquet，还会顺着目录树往下，把所有子文件夹里的 .parquet 一并找出来处理。
    parser.add_argument("--outdir", type=str, default="outputs_signals", help="输出目录（默认 outputs）")

    # 常用‘旋钮’
    parser.add_argument("--stroke-min-change-pct", type=float, default=0.001)
    parser.add_argument("--fractal-left", type=int, default=2)
    parser.add_argument("--fractal-right", type=int, default=2)
    parser.add_argument("--center-min-strokes", type=int, default=3)
    parser.add_argument("--pullback-min-bars", type=int, default=2)
    parser.add_argument("--lookahead-bars", type=int, default=None)
    parser.add_argument("--lookahead-alpha", type=float, default=1.1)
    parser.add_argument("--lookahead-beta", type=float, default=0.9)
    parser.add_argument("--confirm-force", action="store_true")
    parser.add_argument("--min-force-ratio", type=float, default=0.8)

    args = parser.parse_args()
    in_path = Path(args.path)
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    # 单文件
    if in_path.is_file():
        df = _read_one(in_path)
        signals = process_one_df(
            df,
            fractal_left=args.fractal_left, fractal_right=args.fractal_right,
            stroke_min_change_pct=args.stroke_min_change_pct,
            center_min_strokes=args.center_min_strokes,
            pullback_min_bars=args.pullback_min_bars,
            lookahead_bars=args.lookahead_bars,
            lookahead_alpha=args.lookahead_alpha, lookahead_beta=args.lookahead_beta,
            confirm_force=args.confirm_force, min_force_ratio=args.min_force_ratio,
        )
        rpt = validate_signals(df, signals)
        print(f"[{in_path.name}] 验证：", rpt)
        out_file = outdir / f"{in_path.stem}_signals.csv"
        signals.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"已保存 -> {out_file}")

    # 文件夹（批处理所有 .parquet）
    elif in_path.is_dir():
        pattern = "**/*.parquet" if args.recursive else "*.parquet"
        files = sorted(in_path.glob(pattern))
        if not files:
            print("该文件夹内未发现 .parquet 文件", file=sys.stderr); sys.exit(1)
        for f in files:
            try:
                df = _read_one(f)
                signals = process_one_df(
                    df,
                    fractal_left=args.fractal_left, fractal_right=args.fractal_right,
                    stroke_min_change_pct=args.stroke_min_change_pct,
                    center_min_strokes=args.center_min_strokes,
                    pullback_min_bars=args.pullback_min_bars,
                    lookahead_bars=args.lookahead_bars,
                    lookahead_alpha=args.lookahead_alpha, lookahead_beta=args.lookahead_beta,
                    confirm_force=args.confirm_force, min_force_ratio=args.min_force_ratio,
                )
                rpt = validate_signals(df, signals)
                print(f"[{f.name}] 验证：", rpt)
                out_file = outdir / f"{f.stem}_signals.csv"
                signals.to_csv(out_file, index=False, encoding="utf-8-sig")
                print(f"已保存 -> {out_file}")
            except Exception as e:
                print(f"[{f.name}] 处理失败：{e}", file=sys.stderr)
        print("批处理完成。")
    else:
        raise ValueError("path 既不是文件也不是目录。")
    #python signals_build.py --path D:\chan\chan.py\data\stooq\1d_15y
