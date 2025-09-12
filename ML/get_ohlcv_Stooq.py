# -*- coding: utf-8 -*-
"""
Stooq 批量抓取（路线 A：日线不变，多标的 + 拉年限）
- 免 API Key，适合大陆网络
- 新增: --symbols_file / --normalize_us / --workers
- 输出：每标的 Parquet + 汇总验收报告 CSV
当前的代码没有任何错误重试机制，一旦网络波动导致 requests.get() 行出现 SSLEOFError、ConnectionError 或 Timeout，函数就会立刻失败。
"""

import os, argparse, time, re
import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import io, requests, random

# ===== 工具函数 =====

def _resolve_range(period: str | None, start: str | None, end: str | None):
    tznow = pd.Timestamp.utcnow()
    tznow = tznow.tz_convert("UTC") if tznow.tzinfo is not None else tznow.tz_localize("UTC")
    if period:
        period = period.strip().lower()
        n = int(''.join(ch for ch in period if ch.isdigit()))
        unit = ''.join(ch for ch in period if ch.isalpha())
        if unit in ("d","day","days"):      start_dt = tznow - pd.Timedelta(days=n)
        elif unit in ("wk","w","week","weeks"): start_dt = tznow - pd.Timedelta(weeks=n)
        elif unit in ("mo","mon","month","months"): start_dt = tznow - pd.DateOffset(months=n)
        elif unit in ("y","yr","year","years"):    start_dt = tznow - pd.DateOffset(years=n)
        else: raise ValueError(f"不支持的 period 单位: {period}")
        end_dt = tznow
        return start_dt, end_dt
    start_dt = pd.to_datetime(start, utc=True) if start else None
    end_dt   = pd.to_datetime(end,   utc=True) if end   else None
    return start_dt, end_dt

def normalize_symbol(sym: str) -> str:
    """把常见写法统一为 pandas-datareader/stooq 更稳的形式。"""
    s = sym.strip().upper()
    if not s: return s
    # 去掉尾部国家后缀 .US （Stooq 源一般直接用代码即可）
    if s.endswith(".US"):
        s = s[:-3]
    # BRK.B / BRK/B -> BRK-B
    s = s.replace("/", "-")
    if re.match(r"^[A-Z]+[.][A-Z]$", s):
        s = s.replace(".", "-")
    return s

def load_from_stooq(symbol: str, interval: str = "1d", period: str | None = None,
                    start: str | None = None, end: str | None = None) -> pd.DataFrame:
    start_dt, end_dt = _resolve_range(period, start, end)
    df = DataReader(symbol, "stooq", start=start_dt, end=end_dt)
    if df is None or df.empty:
        raise ValueError(f"Stooq 返回空数据: {symbol}")
    df = df.rename(columns=str.lower)
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[keep]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.tz_convert("UTC") if df.index.tz is not None else df.tz_localize("UTC")
    df = df.sort_index()
    if interval.lower() in ("1wk","1w"):
        df = resample_ohlcv(df, "1W")
    elif interval.lower() in ("1mo","1m","1ms"):
        df = resample_ohlcv(df, "1M")
    return df

def _fetch_via_http_csv(symbol, period=None, start=None, end=None, interval="1d"):
    sdt, edt = _resolve_range(period, start, end)
    def stooq_sym(s):
        s = s.strip().lower()
        return s if "." in s else f"{s}.us"  # 默认当作美股
    base = f"https://stooq.com/q/d/l/?s={stooq_sym(symbol)}&i=d"
    if sdt is not None and edt is not None:
        d1 = pd.Timestamp(sdt).tz_convert(None).strftime("%Y%m%d")
        d2 = pd.Timestamp(edt).tz_convert(None).strftime("%Y%m%d")
        base += f"&d1={d1}&d2={d2}"
    r = requests.get(base, timeout=20, headers={"User-Agent":"Mozilla/5.0"})

    r.raise_for_status()
    csv = r.text.strip()
    if not csv or csv.startswith("<"):
        raise RuntimeError(f"HTTP CSV 返回异常: {base}")
    #解析和处理 CSV 数据
    df = pd.read_csv(io.StringIO(csv))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")[["open","high","low","close","volume"]].dropna()

    if interval.lower() in ("1wk","1w"):
        df = resample_ohlcv(df, "1W")
    elif interval.lower() in ("1mo","1m","1ms"):
        df = resample_ohlcv(df, "1M")
    return df.sort_index()

def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if set(["open","high","low","close"]).issubset(df.columns):
        bad = ((df["low"] > df[["open","close"]].min(axis=1)) |
               (df["high"] < df[["open","close"]].max(axis=1)) |
               (df["low"] > df["high"]))
        df = df[~bad]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.tz_convert("UTC") if df.index.tz is not None else df.tz_localize("UTC")
    df = df.sort_index()
    return df

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df.resample(rule, label="right", closed="right").agg(agg).dropna()

def validate_ohlcv(df: pd.DataFrame, expected_rule: str | None = None,
                   tolerate_gap_ratio: float = 0.02) -> dict:
    rpt = {"PASS": True, "errors": [], "warnings": [], "stats": {}}
    if not isinstance(df.index, pd.DatetimeIndex):
        rpt["PASS"]=False; rpt["errors"].append("索引不是 DatetimeIndex"); return rpt
    if df.index.tz is None: rpt["PASS"]=False; rpt["errors"].append("索引必须带时区(UTC)")
    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns: rpt["PASS"]=False; rpt["errors"].append(f"缺少列:{c}")
        elif not np.issubdtype(df[c].dtype, np.number): rpt["PASS"]=False; rpt["errors"].append(f"{c} 不是数值")
    if df[need].isna().any().any(): rpt["PASS"]=False; rpt["errors"].append("OHLCV 有 NaN")
    if set(["open","high","low","close"]).issubset(df.columns):
        bad = ((df["low"] > df[["open","close"]].min(axis=1)) |
               (df["high"] < df[["open","close"]].max(axis=1)) |
               (df["low"] > df["high"]))
        if bad.any(): rpt["PASS"]=False; rpt["errors"].append(f"价格逻辑异常 {int(bad.sum())} 条")
    if (df["volume"] < 0).any(): rpt["PASS"]=False; rpt["errors"].append("存在负成交量")
    if not df.index.is_monotonic_increasing: rpt["PASS"]=False; rpt["errors"].append("时间非递增")
    if df.index.has_duplicates: rpt["PASS"]=False; rpt["errors"].append(f"重复时间 {int(df.index.duplicated().sum())} 条")

    if len(df) >= 3:
        diffs = (df.index[1:] - df.index[:-1]).asi8 // 10**9
        uniq, cnts = np.unique(diffs, return_counts=True)
        main_step_sec = int(uniq[np.argmax(cnts)])
        gap_ratio_raw = 1.0 - cnts.max()/cnts.sum()
        rpt["stats"]["main_step_seconds"] = main_step_sec
        rpt["stats"]["gap_ratio_raw"] = float(gap_ratio_raw)

        exp_seconds = None; exp_is_bd = False
        if expected_rule:
            m = expected_rule.upper()
            if   m.endswith("T"): exp_seconds = int(m[:-1])*60
            elif m.endswith("H"): exp_seconds = int(m[:-1])*3600
            elif m.endswith("D"): exp_seconds = int(m[:-1])*86400
            elif m.endswith("W"): exp_seconds = int(m[:-1])*7*86400
            elif m.endswith("B"): exp_is_bd = True
            elif m.endswith("M"): exp_seconds = None

        gap_ratio_relaxed = None
        if exp_is_bd or (exp_seconds == 86400) or (main_step_sec == 86400):
            allowed = {86400, 2*86400, 3*86400, 4*86400}
            ok = np.isin(diffs, list(allowed))
            gap_ratio_relaxed = float(1.0 - ok.sum()/ok.size)
            rpt["stats"]["gap_ratio_relaxed_daily"] = gap_ratio_relaxed

        if expected_rule and (exp_seconds is not None):
            if abs(main_step_sec - exp_seconds) > 1:
                rpt["warnings"].append(f"主频 {main_step_sec}s ≠ 期望 {exp_seconds}s")

        eff_gap = gap_ratio_relaxed if gap_ratio_relaxed is not None else gap_ratio_raw
        rpt["stats"]["gap_ratio"] = float(eff_gap)
        if eff_gap > tolerate_gap_ratio:
            rpt["warnings"].append(f"间隔不规则比例较高: {eff_gap:.2%}")

    rpt["stats"]["rows"] = int(len(df))
    rpt["stats"]["start"] = str(df.index[0]); rpt["stats"]["end"] = str(df.index[-1])
    return rpt

# ===== 批量主流程 =====

DEFAULT_SYMBOLS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","COST","JPM",
    "V","MA","JNJ","UNH","XOM","CVX","HD","PG","KO","PEP"
]

def fetch_one(
    symbol,
    interval,
    period,
    start,
    end,
    expected,
    outdir,
    *,
    max_retries=6,
    retry_backoff=0.8,
    http_fallback=False
):
    """
    单个标的抓取 + 校验 + 保存
    这里不再依赖外层 args，全部通过显式参数传入
    """
    last_err = None
    for k in range(int(max_retries)):
        try:
            df = load_from_stooq(symbol, interval=interval, period=period, start=start, end=end)
            df = standardize_ohlcv(df)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"{symbol}_{interval}_{period}.parquet")
            df.to_parquet(outfile)
            rpt = validate_ohlcv(df, expected_rule=expected)
            return outfile, df, rpt
        except Exception as e:
            last_err = e
            # 指数退避 + 少量抖动
            sleep_s = float(retry_backoff) * (2 ** k) + random.uniform(0, 0.3)
            print(f"[retry {k+1}/{max_retries}] {symbol} -> {e} ; sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)

    # 多次失败后，可选 HTTP CSV 回退
    if http_fallback:
        try:
            df = _fetch_via_http_csv(symbol, period=period, start=start, end=end, interval=interval)
            df = standardize_ohlcv(df)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"{symbol}_{interval}_{period}.parquet")
            df.to_parquet(outfile)
            rpt = validate_ohlcv(df, expected_rule=expected)
            print(f"[fallback:HTTP] {symbol}: saved {len(df)} → {outfile}")
            return outfile, df, rpt
        except Exception as e2:
            last_err = e2

    # 兜底把最后一次错误抛出去
    raise last_err



def read_symbols(args) -> list[str]:
    """优先用 --symbols_file，其次 --symbols；统一清洗、去重、保序。"""
    syms: list[str] = []
    # 1) 文件
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or raw.startswith("//"):
                    continue
                # 支持逗号分隔的一行多码
                parts = [p for p in re.split(r"[\s,;]+", raw) if p]
                syms.extend(parts)
    # 2) 命令行
    if args.symbols:
        syms.extend([s for s in args.symbols.split(",") if s.strip()])

    # 清洗/规范化
    cleaned = []
    seen = set()
    for s in syms:
        t = s.strip().upper()
        if args.normalize_us:
            t = normalize_symbol(t)
        if t and t not in seen:
            seen.add(t); cleaned.append(t)
    if not cleaned:
        cleaned = DEFAULT_SYMBOLS.copy()
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="逗号分隔的标的列表；默认 20 个大票")
    ap.add_argument("--symbols_file", default=None,
                    help="从文本文件读取标的（每行一个；支持空行/注释/逗号分隔）")
    ap.add_argument("--normalize_us", action="store_true",
                    help="把 AAPL.US→AAPL、BRK.B/BRK/B→BRK-B 等统一写法（推荐开启）")
    ap.add_argument("--interval", default="1d", help="1d / 1wk / 1mo（周/月线由 resample 生成）")
    ap.add_argument("--period", default="15y", help="也可配合 --start/--end")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--expected", default="1B", help="期望频率校验，日线建议 1B（交易日）或 1D")
    ap.add_argument("--outdir", default=r"data\stooq\1d_15y")
    ap.add_argument("--report_csv", default=r"reports\stooq_eod_report.csv")
    ap.add_argument("--sleep", type=float, default=0.8, help="串行模式下的请求间隔秒数")
    ap.add_argument("--workers", type=int, default=1, help="并发线程数；>1 启用并发")
    ap.add_argument("--max_retries", type=int, default=6, help="失败重试次数")
    ap.add_argument("--retry_backoff", type=float, default=0.8, help="退避起始秒")
    ap.add_argument("--http_fallback", action="store_true", help="HTTPS 多次失败后尝试 HTTP CSV 接口")

    args = ap.parse_args()

    syms = read_symbols(args)
    rows = []
    ok, fail = 0, 0

    def _job(sym):
        try:
            outfile, df, rpt = fetch_one(
                sym,
                interval=args.interval,
                period=args.period,
                start=args.start,
                end=args.end,
                expected=args.expected,
                outdir=args.outdir,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                http_fallback=args.http_fallback,
            )
            return ("ok", sym, outfile, df, rpt, None)
        except Exception as e:
            return ("fail", sym, "", None, None, e)


    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, sym): sym for sym in syms}
            for i, fut in enumerate(as_completed(futs), 1):
                sym = futs[fut]
                status, sym, outfile, df, rpt, err = fut.result()
                if status == "ok":
                    ok += 1
                    print(f"[{i}/{len(syms)}] {sym}: saved {len(df)} → {outfile}")
                    rows.append({
                        "symbol": sym, "outfile": outfile, "PASS": True,
                        "errors": "", "warnings": "",
                        **({f"stat_{k}": v for k, v in rpt["stats"].items()} if rpt else {})
                    })
                else:
                    fail += 1
                    print(f"[{i}/{len(syms)}] {sym}: FAILED - {err}")
                    rows.append({
                        "symbol": sym, "outfile": "", "PASS": False,
                        "errors": str(err), "warnings": "", "stat_rows": 0
                    })
    else:
        for i, sym in enumerate(syms, 1):
            status, sym, outfile, df, rpt, err = ("", sym, "", None, None, None)
            try:
                outfile, df, rpt = fetch_one(
                    sym,
                    interval=args.interval,
                    period=args.period,
                    start=args.start,
                    end=args.end,
                    expected=args.expected,
                    outdir=args.outdir,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                    http_fallback=args.http_fallback,
                )
                status = "ok"
            except Exception as e:
                status, err = "fail", e
                
            if status == "ok":
                ok += 1
                print(f"[{i}/{len(syms)}] {sym}: saved {len(df)} → {outfile}")
                rows.append({
                    "symbol": sym, "outfile": outfile, "PASS": True,
                    "errors": "", "warnings": "",
                    **({f"stat_{k}": v for k, v in rpt["stats"].items()} if rpt else {})
                })
            else:
                fail += 1
                print(f"[{i}/{len(syms)}] {sym}: FAILED - {err}")
                rows.append({
                    "symbol": sym, "outfile": "", "PASS": False,
                    "errors": str(err), "warnings": "", "stat_rows": 0
                })
            time.sleep(args.sleep)

    # 汇总报告
    os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(args.report_csv, index=False)
    print(f"\n=== 批量完成 ===  成功: {ok}  失败: {fail}")
    print(f"报告: {args.report_csv}")
    if not rep.empty:
        passed = (rep["PASS"] == True).sum()
        print(f"通过验收: {passed}/{len(rep)}")
        if "stat_gap_ratio" in rep.columns:
            print("gap_ratio 95 分位：", rep["stat_gap_ratio"].quantile(0.95))

if __name__ == "__main__":
    main()
"""
python D:\chan\chan.py\get_ohlcv_Stooq.py `
  --symbols_file "D:\chan\chan.py\symbols_usa_2025.txt" 
"""