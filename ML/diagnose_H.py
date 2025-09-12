import pandas as pd
from pathlib import Path

LABELS_DIR = Path(r"D:\chan\chan.py\events_outputs\labels")

def load_all():
    frames=[]
    for fp in LABELS_DIR.glob("*_events_labeled.csv"):
        df = pd.read_csv(fp)
        df["symbol"]=fp.stem.split("_")[0].upper()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def summarize(df):
    out={}
    df_tp = df[df["exit_reason"].str.contains("TP", na=False)]
    out["N_total"]=len(df)
    out["TP_ratio"]=round(len(df_tp)/len(df),3) if len(df) else None
    out["TIMEOUT_ratio"]=round((df["exit_reason"]=="TIMEOUT").mean(),3)
    if len(df_tp):
        out["bars_held_p50"]=int(df_tp["bars_held"].quantile(0.5))
        out["bars_held_p80"]=int(df_tp["bars_held"].quantile(0.8))
        out["bars_held_p90"]=int(df_tp["bars_held"].quantile(0.9))
    return out

if __name__=="__main__":
    all_df = load_all()
    print("== 全市场汇总 ==")
    print(summarize(all_df))
    print("\n== 分标的 ==")
    for sym, sub in all_df.groupby("symbol"):
        print(sym, summarize(sub))
