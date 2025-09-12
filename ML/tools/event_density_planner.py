# -*- coding: utf-8 -*-
# 事件密度评估 + 扩样规划
# 路径按你的工程约定，如需修改仅改 PATH_EVENTS / TARGET_N 即可

import os, math
import pandas as pd
from datetime import datetime

# ===== 路径配置 =====
PATH_EVENTS = r"D:\chan\chan.py\outputs\events_batch_summary.csv"
OUT_SUMMARY = r"D:\chan\chan.py\density_outputs\event_density_summary.csv"
OUT_BY_YEAR = r"D:\chan\chan.py\density_outputs\event_density_by_year.csv"

# 目标样本量
TARGET_N = 300

# 你打算考虑的扩样组合（可自行加/减）
YEARS_CHOICES   = [1, 2, 3, 5, 8, 10]          # 计划再覆盖多少年历史
SYMBOLS_CHOICES = [1, 3, 5, 10, 20]            # 并行标的数量
TF_MULTIPLIERS  = {
    "1D": 1.0,   # 日线作为基准
    "4H": 2.5,   # 经验估计：4H 事件密度 ≈ 日线的 ~2–3 倍（先用 2.5 做保守中位）
    "1H": 6.0    # 经验估计：1H 事件密度 ≈ 日线的 ~5–7 倍（先用 6 做中值）
}
# 上面倍数只是“起步估计”。一旦你真的切到 4H/1H，记得再跑一次本脚本，用实际 events 重新校准。

# ===== 读入事件，并解析入场时间 =====
assert os.path.exists(PATH_EVENTS), f"找不到 {PATH_EVENTS}"
ev = pd.read_csv(PATH_EVENTS)

# 兼容不同列名：优先 t_entry / entry_time / signal_time
time_col = None
for c in ["t_entry", "entry_time", "signal_time", "time", "timestamp"]:
    if c in ev.columns:
        time_col = c
        break
if time_col is None:
    raise ValueError("events.csv 里找不到时间列（尝试了 t_entry/entry_time/signal_time/time/timestamp）")

ev[time_col] = pd.to_datetime(ev[time_col])
ev = ev.sort_values(time_col).reset_index(drop=True)

# 统计覆盖范围
n_total = len(ev)
if n_total == 0:
    raise ValueError("events.csv 是空的，先跑信号→事件流水线再来评估密度。")

t0 = ev[time_col].iloc[0]
t1 = ev[time_col].iloc[-1]
years_covered = max(1.0, (t1 - t0).days / 365.25)  # 用实际跨度近似“覆盖年数”
# 每年事件数（按日历年分组）
ev["year"] = ev[time_col].dt.year
by_year = ev.groupby("year").size().rename("events").reset_index()
avg_per_year = by_year["events"].mean()  # 只对包含事件的年份求平均

# ===== 生成扩样计划表 =====
rows = []
for yrs in YEARS_CHOICES:
    for syms in SYMBOLS_CHOICES:
        for tf, mult in TF_MULTIPLIERS.items():
            # 估算：期望事件数 ≈ 平均每年事件数 × yrs × syms × mult
            est_n = avg_per_year * yrs * syms * mult
            rows.append({
                "years": yrs,
                "symbols": syms,
                "tf": tf,
                "tf_multiplier": mult,
                "est_events": round(est_n, 1),
                "meets_target(>=300)": est_n >= TARGET_N
            })
plan = pd.DataFrame(rows).sort_values(
    ["meets_target(>=300)", "est_events"], ascending=[False, False]
).reset_index(drop=True)

# ===== 汇总结果 =====
summary = pd.DataFrame([{
    "events_total": n_total,
    "first_event": t0.strftime("%Y-%m-%d"),
    "last_event":  t1.strftime("%Y-%m-%d"),
    "years_covered_approx": round(years_covered, 2),
    "avg_events_per_year(on_years_with_events)": round(avg_per_year, 2)
}])

# ===== 保存并打印 =====
os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
summary.to_csv(OUT_SUMMARY, index=False)
by_year.to_csv(OUT_BY_YEAR, index=False)

print("=== 事件密度摘要 ===")
print(summary.to_string(index=False))
print("\n=== 每年事件数 ===")
print(by_year.to_string(index=False))
print("\n=== 扩样计划（按是否满足目标 & 预估事件数排序）===")
print(plan.head(20).to_string(index=False))

print(f"\n已保存：\n- {OUT_SUMMARY}\n- {OUT_BY_YEAR}")
