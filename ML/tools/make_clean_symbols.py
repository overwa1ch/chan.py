# -*- coding: utf-8 -*-
"""
交互式清单整理：
- 从抓数报告中生成两份清单：
  1) symbols_usa_2025_clean.txt  —— 干净（行数 >= 阈值，默认3600）
  2) symbols_usa_2025_short.txt  —— 历史较短但可用（0 < 行数 < 阈值）
- 打印两份清单，并询问是否保留“短史清单”：Y/N
  * Y: 不做任何修改
  * N: 打开 D:\chan\chan.py\symbols_usa_2025.txt，删除短史清单中的标的（自动 .bak 备份）
"""

import os, re, sys, shutil
import pandas as pd
from datetime import datetime

# ---- 可按需改的默认路径 ----
DEFAULT_REPORT     = r"D:\chan\chan.py\reports\stooq_eod_report.csvN"
DEFAULT_LISTS_DIR  = r"D:\chan\chan.py\lists"
DEFAULT_SYMBOLS_TXT= r"D:\chan\chan.py\symbols_usa_2025.txt"
ROW_THRESHOLD      = 3600  # 约等于 >=14.3 年

def normalize_symbol(sym: str) -> str:
    """统一写法，便于匹配与去重：AAPL.US->AAPL、BRK.B/BRK/B->BRK-B。"""
    s = sym.strip().upper()
    if not s: return s
    if s.endswith(".US"): s = s[:-3]
    s = s.replace("/", "-")
    if re.match(r"^[A-Z]+[.][A-Z]$", s):  # 如 BRK.B
        s = s.replace(".", "-")
    return s

def read_rows_column(df: pd.DataFrame) -> pd.Series:
    # 优先 stat_rows，其次 rows
    cand = [c for c in df.columns if c.lower().endswith("stat_rows")] + \
           [c for c in df.columns if c.lower()=="rows"]
    if not cand:
        # 尝试从 start/end 推断（不够严谨，仅兜底）
        return pd.Series([0]*len(df))
    return pd.to_numeric(df[cand[0]], errors="coerce").fillna(0).astype(int)

def format_block(lst, per_line=10):
    out = []
    for i in range(0, len(lst), per_line):
        out.append("  " + "  ".join(lst[i:i+per_line]))
    return "\n".join(out) if out else "  （空）"

def unique_keep_order(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def read_symbols_file(path: str) -> list[str]:
    """解析 symbols_usa_2025.txt，支持注释(#,//)、空行、逗号/空格/分号分隔。"""
    if not os.path.exists(path): return []
    tokens=[]
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = [p for p in re.split(r"[\s,;]+", line) if p]
            tokens.extend(parts)
    # 规范化并去重（保持原有顺序）
    tokens = [normalize_symbol(t) for t in tokens]
    return unique_keep_order([t for t in tokens if t])

def write_symbols_file(path: str, symbols: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")

def main(report_path=DEFAULT_REPORT, lists_dir=DEFAULT_LISTS_DIR,
         symbols_txt=DEFAULT_SYMBOLS_TXT, row_threshold=ROW_THRESHOLD):
    # 读报告
    if not os.path.exists(report_path):
        print(f"[错误] 找不到报告文件：{report_path}")
        sys.exit(1)
    os.makedirs(lists_dir, exist_ok=True)
    df = pd.read_csv(report_path)

    # 标准化 PASS 与 rows
    pass_mask = df["PASS"].astype(str).str.lower().isin(["true","1","y","yes"])
    rows = read_rows_column(df)
    df["rows_std"] = rows
    pass_df = df[pass_mask & (rows > 0)].copy()

    clean_df = pass_df[pass_df["rows_std"] >= row_threshold]
    short_df = pass_df[(pass_df["rows_std"] > 0) & (pass_df["rows_std"] < row_threshold)]

    clean_syms = [normalize_symbol(s) for s in clean_df["symbol"].astype(str).tolist()]
    short_syms = [normalize_symbol(s) for s in short_df["symbol"].astype(str).tolist()]
    clean_syms = unique_keep_order(clean_syms)
    short_syms = unique_keep_order(short_syms)

    # 写两份清单
    clean_path = os.path.join(lists_dir, "symbols_usa_2025_clean.txt")
    short_path = os.path.join(lists_dir, "symbols_usa_2025_short.txt")
    write_symbols_file(clean_path, clean_syms)
    write_symbols_file(short_path, short_syms)

    # 打印结果
    print("\n=== 干净清单（满足行数阈值，建议用于正式抓数/事件生成）===")
    print(f"共 {len(clean_syms)} 个，文件：{clean_path}")
    print(format_block(clean_syms))
    print("\n=== 历史较短但可用（0 < 行数 < 阈值）===")
    print(f"共 {len(short_syms)} 个，文件：{short_path}")
    print(format_block(short_syms))

    # 交互：是否保留“短史清单”中的标的于 symbols_usa_2025.txt
    if len(short_syms) == 0:
        print("\n没有短史标的需要处理，任务结束。")
        return

    choice = input("\n是否保留历史较短但可用的清单？(Y/N)：").strip().lower()
    if choice in ("y","yes","是","shi"):
        print("保留短史标的，不修改 symbols_usa_2025.txt。")
        return

    if choice in ("n","no","否","fou"):
        # 读取原始 symbols 文件，删除短史标的
        if not os.path.exists(symbols_txt):
            print(f"[提示] 找不到 {symbols_txt}，跳过修改。")
            return
        original = read_symbols_file(symbols_txt)
        if not original:
            print(f"[提示] {symbols_txt} 内没有可解析的标的，跳过修改。")
            return

        short_set = set(short_syms)
        kept = [s for s in original if normalize_symbol(s) not in short_set]

        # 备份原文件
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = symbols_txt + f".{ts}.bak"
        shutil.copy2(symbols_txt, bak)

        # 写回（保持一行一个代码，规范化写法）
        write_symbols_file(symbols_txt, kept)

        removed = [s for s in original if s not in kept]
        print("\n已根据你的选择更新 symbols 文件：")
        print("  原始数量：", len(original))
        print("  删除数量：", len(removed))
        print("  保留数量：", len(kept))
        print("  备份文件：", bak)
        if removed:
            print("  已删除：")
            print(format_block(removed))
        return

    print("输入未识别（接受 Y/N）。未做任何修改。")

if __name__ == "__main__":
    # 允许通过命令行改路径/阈值（可选）
    # 使用示例：
    #   python tools\make_clean_symbols.py
    #   或
    #   python tools\make_clean_symbols.py "D:\...\report.csv" "D:\...\lists" "D:\...\symbols_usa_2025.txt" 3600
    args = sys.argv[1:]
    if len(args) >= 1: DEFAULT_REPORT    = args[0]
    if len(args) >= 2: DEFAULT_LISTS_DIR = args[1]
    if len(args) >= 3: DEFAULT_SYMBOLS_TXT = args[2]
    if len(args) >= 4: ROW_THRESHOLD     = int(args[3])
    main(DEFAULT_REPORT, DEFAULT_LISTS_DIR, DEFAULT_SYMBOLS_TXT, ROW_THRESHOLD)
