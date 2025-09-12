import numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
import argparse

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score

# ===================== CLI 参数 =====================
parser = argparse.ArgumentParser(
    description="Train→Calibrate→Validate(阈值/Top‑K)→Test 一体化评估脚本")
parser.add_argument("--data_dir", default=r"D:\\chan\\chan.py\\ml_dataset\\20250902_194053",
                    help="包含 train/valid/test parquet 的目录")
parser.add_argument("--behavior_csv", default=r"D:\\chan\\chan.py\\ml_dataset\\outputs\\behavior_features.csv",
                    help="行为/结构特征 CSV（需含 event_id）")
parser.add_argument("--out_dir", default=r"D:\\chan\\chan.py\\ml_dataset\\outputs",
                    help="输出目录")
parser.add_argument("--fee", type=float, default=0.0006,
                    help="双边费用/滑点假设（例如 0.0006 = 0.06%）")
parser.add_argument("--thr_min", type=float, default=0.20, help="阈值扫描最小值")
parser.add_argument("--thr_max", type=float, default=0.80, help="阈值扫描最大值")
parser.add_argument("--thr_step", type=float, default=0.05, help="阈值步长")
parser.add_argument("--min_n", type=int, default=40, help="阈值下最少成交数")
parser.add_argument("--lam", type=float, default=0.5, help="左尾风险惩罚系数 λ")
parser.add_argument("--topk_max", type=int, default=5, help="在 VALID 上试的最大 K 值（1..K）")
parser.add_argument("--mode", default="both", choices=["both","threshold","topk"],
                    help="只跑固定阈值、只跑Top‑K，或两者都跑")
parser.add_argument("--random_state", type=int, default=42, help="随机种子")
# 当无法从数据中解析出日期时的兜底分组策略
parser.add_argument("--fallback_grouping", default="chunk", choices=["chunk"],
                    help="当日期全为NaT时的兜底分组策略；目前提供 chunk（按样本块伪造日历日）")
parser.add_argument("--chunk_size", type=int, default=64,
                    help="fallback_grouping=chunk 时，每多少条样本视作同一天")
parser.add_argument("--synthetic_start", default="2000-01-01",
                    help="fallback_grouping=chunk 时，合成日期的起始日")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
BEHAV_FEATS = Path(args.behavior_csv)
OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
FEE = float(args.fee)

# ===================== 工具函数 =====================
def normalize_event_id(df):
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)
    return df

def load_split(name, behav_df):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet").copy()
    normalize_event_id(df)
    Xraw = df.merge(behav_df, on="event_id", how="left")

    # 缺失率报告（针对一组核心行为特征）
    probe_cols = [c for c in ["mom20","rv20","atr_trend"] if c in Xraw.columns]
    miss_rate = Xraw[probe_cols].isna().mean().mean() if probe_cols else 0.0
    print(f"[merge check] behavior feature missing rate = {miss_rate:.3f}")

    # 时间衍生
    for col in ["t_entry","t_signal"]:
        if col in Xraw.columns:
            t = pd.to_datetime(Xraw[col], errors="coerce")
            Xraw[f"{col}_dow"] = t.dt.dayofweek
            Xraw[f"{col}_min"] = t.dt.hour*60 + t.dt.minute

    # 相对化 + 方向交互
    eps = 1e-9
    for c in ["tp","sl","atr_entry","entry_price"]:
        if c in Xraw.columns: Xraw[c] = Xraw[c].astype(float)
    if {"tp","sl","atr_entry"} <= set(Xraw.columns):
        Xraw["tp_over_atr"] = Xraw["tp"] / (Xraw["atr_entry"] + eps)
        Xraw["sl_over_atr"] = Xraw["sl"] / (Xraw["atr_entry"] + eps)
        Xraw["rr_ratio"]    = np.abs(Xraw["tp_over_atr"]) / (np.abs(Xraw["sl_over_atr"]) + eps)
    if "direction" in Xraw.columns:
        if Xraw["direction"].dtype == "object":
            Xraw["direction"] = Xraw["direction"].map({
                "short":-1, "Short":-1, "sell":-1,
                "long":1,  "Long":1,  "buy":1
            }).fillna(0)
        Xraw["dir_x_tp_rel"] = Xraw.get("direction",0) * Xraw.get("tp_over_atr",0)
        Xraw["dir_x_rr"]     = Xraw.get("direction",0) * Xraw.get("rr_ratio",0)

    behavior_cols = [
        "mom20","mom60","slope20_over_atr","rv20","atr_trend","dd_100",
        "volz20","corr_price_vol20","dev_ema20_atr","dev_ema60_atr",
        "up_wick_ratio20","lo_wick_ratio20"
    ]
    base_cols = [
        "direction","entry_price","atr_entry",
        "tp_over_atr","sl_over_atr","rr_ratio","dir_x_tp_rel","dir_x_rr",
        "mode","check_ok","t_entry_dow","t_entry_min","t_signal_dow","t_signal_min"
    ]

    use_cols = [c for c in base_cols+behavior_cols if c in Xraw.columns]
    X = Xraw[use_cols].copy()

    y = Xraw["y"].astype(int)
    r = Xraw["ret"].astype(float)

    # —— 稳健日期提取：t_entry → t_signal → 索引；自动去时区；可统计 NaT 占比 —— 
    def _extract_date(df):
        for col in ["t_entry", "t_signal", "dt", "timestamp"]:
            if col in df.columns:
                t = pd.to_datetime(df[col], errors="coerce")
                try:
                    t = t.dt.tz_convert(None)
                except Exception:
                    try:
                        t = t.dt.tz_localize(None)
                    except Exception:
                        pass
                if t.notna().any():
                    return t.dt.date
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                return df.index.tz_convert(None).date
            except Exception:
                return df.index.tz_localize(None).date
        return pd.Series([pd.NaT] * len(df))

    date = _extract_date(Xraw)
    _nat_ratio = float(pd.isna(pd.to_datetime(date, errors="coerce")).mean())

    # 若完全拿不到日期，按样本块合成“伪日期”
    if _nat_ratio == 1.0 and args.fallback_grouping == "chunk":
        n = len(Xraw)
        if n > 0:
            day_ids = (np.arange(n) // max(1, args.chunk_size)).astype(int)
            synth = pd.to_datetime(args.synthetic_start) + pd.to_timedelta(day_ids, unit="D")
            date = pd.Series(synth.date)
            _nat_ratio = 0.0
    print(f"[date check] {name}: NaT ratio = {_nat_ratio:.3f}")

    cat_cols = [c for c in ["mode","check_ok"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    idx = Xraw.index.copy()

    return X, y, r, date, cat_cols, num_cols, idx

    cat_cols = [c for c in ["mode","check_ok"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 额外输出原始索引用于净值曲线排序
    idx = Xraw.index.copy()

    return X, y, r, date, cat_cols, num_cols, idx


def cvar(x, q=0.05):
    if len(x)==0: return 0.0
    cut = np.quantile(x, q)
    return float(np.mean(x[x<=cut]))


def choose_threshold(p, y, r, min_n=40, lam=0.5, thr_grid=None, fee=0.0):
    """
    在 VALID 上扫描阈值，基于 (均值收益 - λ*|CVaR5%|) 选择最优；收益与CVaR均扣费。
    返回 (stat_df, best_row)
    """
    if thr_grid is None:
        thr_grid = [round(x,2) for x in np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)]

    p = np.asarray(p); y = np.asarray(y); r = np.asarray(r)
    rows = []
    for thr in thr_grid:
        pick = p >= thr
        n = int(pick.sum())
        if n == 0:
            rows.append({"thr":thr,"n":0,"mean_ret":0.0,"winrate":0.0,"cvar05":0.0,
                         "score":-1e9,"valid":False})
            continue
        r_net = r[pick] - fee
        mean_ret = float(r_net.mean())
        winrate  = float((y[pick]==1).mean())
        c05      = cvar(r_net, 0.05)
        score    = mean_ret - lam*abs(c05)
        rows.append({"thr":thr,"n":n,"mean_ret":mean_ret,"winrate":winrate,
                     "cvar05":c05,"score":score,"valid": n>=min_n})

    stat = pd.DataFrame(rows)

    cand = stat[stat["valid"]]
    if len(cand) > 0:
        best = cand.sort_values(["score","mean_ret","n"], ascending=[False,False,False]).iloc[0]
    else:
        # 兜底：n 最大 → mean_ret 最大
        best = stat.sort_values(["n","mean_ret"], ascending=[False,False]).iloc[0]
    # 为了可读性，输出排好序的表
    stat_sorted = stat.sort_values(["valid","score","mean_ret","n"], ascending=[False,False,False,False])
    return stat_sorted, best


def eval_topk(date, p, y, r, K, fee=0.0):
    df = pd.DataFrame({"date": date, "p": p, "y": y, "r": r})
    out = []
    for dt, g in df.groupby("date"):
        out.append(g.sort_values("p", ascending=False).head(K))
    sel = pd.concat(out) if len(out) else pd.DataFrame(columns=df.columns)
    if len(sel)==0:
        return {"K":K, "n":0, "mean_ret":0.0, "winrate":0.0}
    r_net = sel["r"].values - fee
    return {
        "K":K,
        "n": int(len(sel)),
        "mean_ret": float(r_net.mean()),
        "winrate": float((sel["y"]==1).mean())
    }

# ===================== 主流程 =====================
def main():
    # 读行为特征
    behav = pd.read_csv(BEHAV_FEATS)
    normalize_event_id(behav)

    # 读数据集
    Xtr, ytr, rtr, dtr, cat_tr, num_tr, idx_tr = load_split("train", behav)
    Xva, yva, rva, dva, cat_va, num_va, idx_va = load_split("valid", behav)
    Xte, yte, rte, dte, cat_te, num_te, idx_te = load_split("test",  behav)

    # 列对齐
    cat_cols = sorted(set(cat_tr)|set(cat_va)|set(cat_te))
    num_cols = sorted(set(num_tr)|set(num_va)|set(num_te))

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    base = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=500, class_weight="balanced", random_state=args.random_state))
    ])

    # —— 训练 ——
    base.fit(Xtr, ytr)

    # —— 概率校准（train 内交叉，valid 留给选择阈值/Top‑K）——
    # 注：为避免 sklearn 未来弃用 prefit 写法，使用 cv=5 的交叉校准；
    #     这样不会污染 valid/test 的独立性。
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(Xtr, ytr)

    # —— VALID 上评估（用于挑阈值/挑 K）——
    p_va = cal.predict_proba(Xva)[:,1]
    ap = average_precision_score(yva, p_va)
    print(f"VALID PR-AUC: {ap:.4f}")

    # 阈值扫描
    thr_grid = [round(x,2) for x in np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)]
    stat, best = choose_threshold(p_va, yva, rva, min_n=args.min_n, lam=args.lam, thr_grid=thr_grid, fee=FEE)
    stat.to_csv(OUT_DIR/"valid_threshold_scan_with_behavior.csv", index=False)
    print("Chosen THR on VALID:", dict(best))
    thr = float(best["thr"]) if len(stat)>0 else 0.5

    # —— TEST：固定阈值 ——
    p_te = cal.predict_proba(Xte)[:,1]
    pick = p_te >= thr
    n = int(pick.sum())
    r_net_te = (rte[pick] - FEE) if n else np.array([])
    mean_ret = float(r_net_te.mean()) if n else 0.0
    winrate  = float((yte[pick]==1).mean()) if n else 0.0
    print(f"TEST(threshold) picked {n}/{len(p_te)} | mean_ret={mean_ret:.6f} | winrate={winrate:.3f}")

    # 净值曲线（基于原索引排序；重叠持仓仅作粗略参考）
    if n:
        pd.Series(r_net_te, index=pd.Index(idx_te[pick], name="idx"))\
          .sort_index().cumsum().to_csv(OUT_DIR/"test_equity_curve_with_behavior.csv")

    # —— VALID 上 Top‑K 选择 ——
    if args.mode in ("both","topk"):
        val_topk_rows = [eval_topk(dva, p_va, yva, rva, K, fee=FEE) for K in range(1, args.topk_max+1)]
        val_topk = pd.DataFrame(val_topk_rows)
        val_topk.to_csv(OUT_DIR/"valid_topk_scan.csv", index=False)
        # 选 K：先 mean_ret，再 winrate，再 n
        bestK = int(val_topk.sort_values(["mean_ret","winrate","n"], ascending=[False,False,False]).iloc[0]["K"]) if len(val_topk)>0 else 1
        print("Chosen K on VALID:", bestK)

        # —— TEST：Top‑K/日 ——
        df_te = pd.DataFrame({"date": dte, "p": p_te, "ret": rte, "y": yte, "idx": idx_te})
        # 去掉 NaT 日期
        df_te = df_te[df_te["date"].notna()].copy()

        # 若仍无可分组的日期，自动按“年-周”兜底
        use_week = False
        if df_te.empty:
            # 使用与训练/验证相同的“伪日期”策略：按样本块合成日期
            n = len(p_te)
            if n > 0 and args.fallback_grouping == "chunk":
                day_ids = (np.arange(n) // max(1, args.chunk_size)).astype(int)
                synth = pd.to_datetime(args.synthetic_start) + pd.to_timedelta(day_ids, unit="D")
                df_te = pd.DataFrame({
                    "date": synth.date,
                    "p": p_te,
                    "ret": rte,
                    "y": yte,
                    "idx": idx_te
                })
            # 若仍想按周聚合，可在伪日期基础上派生 week
            if not df_te.empty:
                df_te.index = pd.to_datetime(df_te["date"], errors="coerce")
                df_te["week"] = df_te.index.to_period("W-SUN")
                use_week = True

        picks = []
        if not df_te.empty:
            if use_week:
                for wk, g in df_te.groupby("week"):
                    picks.append(g.sort_values("p", ascending=False).head(bestK))
            else:
                for dt, g in df_te.groupby("date"):
                    picks.append(g.sort_values("p", ascending=False).head(bestK))

        if len(picks):
            topk = pd.concat(picks)
            topk_ret_net = topk["ret"].values - FEE
            topk_mean = float(topk_ret_net.mean())
            topk_win  = float((topk["y"]==1).mean())
            label = "week" if use_week else "date"
            print(f"[Top-K/{label}] groups={topk[label].nunique()}, trades={len(topk)}, "
                  f"mean_ret={topk_mean:.6f}, winrate={topk_win:.3f}")
            # 仍用时间顺序（idx）生成净值
            topk.sort_values("idx")["ret"].sub(FEE).cumsum().to_csv(OUT_DIR/"test_equity_curve_topk.csv")
        else:
            print("[Top-K] 兜底后仍无可交易样本（检查 t_entry/t_signal 是否为空）")

    # 元信息落盘
    meta = {
        "fee": FEE,
        "thr_grid": thr_grid,
        "chosen_thr": thr,
        "min_n": args.min_n,
        "lam": args.lam,
        "mode": args.mode,
        "topk_max": args.topk_max,
        "valid_ap": float(ap)
    }
    pd.Series(meta).to_csv(OUT_DIR/"run_meta.csv")

    print("Saved:",
          (OUT_DIR/"valid_threshold_scan_with_behavior.csv").resolve(),
          (OUT_DIR/"run_meta.csv").resolve(),
          (OUT_DIR/"test_equity_curve_with_behavior.csv").resolve() if (OUT_DIR/"test_equity_curve_with_behavior.csv").exists() else "(no threshold curve)",
          (OUT_DIR/"valid_topk_scan.csv").resolve() if (OUT_DIR/"valid_topk_scan.csv").exists() else "(no topk scan)",
          (OUT_DIR/"test_equity_curve_topk.csv").resolve() if (OUT_DIR/"test_equity_curve_topk.csv").exists() else "(no topk curve)")

if __name__ == "__main__":
    main()
