#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_parquet.py
用来查看 parquet 文件的列名、类型、以及 pandas 索引信息（如果有）。
默认只读 parquet 元数据（很快），可选 --sample N 预览前 N 行（需要安装 pandas）。
"""

import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_path", help="Parquet 文件路径")
    ap.add_argument("--sample", type=int, default=0, help="预览前 N 行（需要 pandas）；默认 0 表示不读取数据")
    args = ap.parse_args()

    p = Path(args.parquet_path)
    if not p.exists():
        print(f"[ERR] file not found: {p}", file=sys.stderr)
        sys.exit(1)

    # --- 使用 pyarrow 读取 schema 和 pandas 元数据 ---
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(p))

        print("=== Parquet Schema (arrow) ===")
        sch = pf.schema_arrow
        for i, f in enumerate(sch):
            print(f"[{i}] {f.name}: {f.type}")

        # 解析 pandas 元数据（其中可能包含 index_columns 等）
        md = pf.metadata
        pandas_meta = None
        if md is not None and md.metadata and b"pandas" in md.metadata:
            try:
                pandas_meta = json.loads(md.metadata[b"pandas"].decode("utf-8"))
            except Exception:
                pandas_meta = None

        print("\n=== pandas metadata ===")
        if pandas_meta:
            # 可能包含 'index_columns', 'columns'（含 name / field_name / pandas_type / numpy_type）
            idx_cols = pandas_meta.get("index_columns", [])
            print("index_columns:", idx_cols)
            # 列映射信息更详细（可选打印）
            # print("columns:", pandas_meta.get("columns", []))
        else:
            print("(no pandas metadata)")

        # 标注“疑似时间列”
        print("\n=== candidate time-like columns ===")
        candidates = {"timestamp","datetime","date","time","dt"}
        arrow_names = {f.name for f in sch}
        found = sorted(candidates & arrow_names)
        if found:
            print("found:", found)
        else:
            print("found: (none in columns)  # 注意：时间可能存放在 pandas 索引中")

    except ModuleNotFoundError:
        print("[WARN] pyarrow 未安装，无法读取 schema/metadata；请先 `pip install pyarrow`。", file=sys.stderr)

    # --- 可选：预览数据（需要 pandas） ---
    if args.sample > 0:
        try:
            import pandas as pd
            print(f"\n=== pandas preview (head {args.sample}) ===")
            df = pd.read_parquet(str(p))  # 简单读取；如果很大可考虑只读部分列
            # 展示列与 dtypes
            print("columns:", list(df.columns))
            print("\ndtypes:")
            print(df.dtypes)

            # 检测 DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                print("\nindex: DatetimeIndex (from pandas index)")
            else:
                print("\nindex:", type(df.index).__name__)

            print("\nhead:")
            print(df.head(args.sample))
        except ModuleNotFoundError:
            print("[WARN] pandas 未安装，跳过预览；如需预览请 `pip install pandas`。", file=sys.stderr)

if __name__ == "__main__":
    main()
