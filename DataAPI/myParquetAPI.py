# DataAPI/myParquetAPI.py
# 读取本地 parquet，接口对齐 CSV_API
import os
import glob
import pandas as pd
from pathlib import Path

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit
from .CommonStockAPI import CCommonStockApi

CHAN_DATA_HOME = r"D:\chan\chan.py\data\stooq\1d_15y"

class MyParquetAPI(CCommonStockApi):
    """
    与 CSV_API 一样：
      - 继承 CCommonStockApi
      - 构造签名: (code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None)
      - 通过 get_kl_data() 逐条 yield CKLine_Unit(...)
      - 时间过滤在产出前完成（与 CSV_API 逻辑一致）
      - 文件路径规则：{DataAPI}/../{code}_{k_type}.parquet ；其中 k_type 取 self.k_type.name[2:].lower()
    """
    
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        # 与 CSV_API 一样，定义列（最小可用集合）
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            # 如需成交量等，可按需加入 DATA_FIELD.FIELD_VOLUME 等字段
        ]
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        super(MyParquetAPI, self).__init__(code, k_type, begin_date, end_date, autype)

    def _filepath(self):
        # Build robust file path with symbol variants and prefix glob
        data_home = self.data_home or CHAN_DATA_HOME
        base = Path(data_home).expanduser().resolve()
        symbol_in = str(self.code)
        # timeframe suffix mapping
        if self.k_type == KL_TYPE.K_DAY:
            tail = "1d_15y"
        else:
            # Fall back to lower-cased enum name after 'K_'
            nm = getattr(self.k_type, 'name', '').lower()
            tail = nm[2:] if nm.startswith('k_') else nm
        # Build variants
        variants = set([
            symbol_in,
            symbol_in.upper(), symbol_in.lower(),
            symbol_in.replace('-', '_'), symbol_in.replace('_', '-'),
            symbol_in.upper().replace('-', '_'), symbol_in.upper().replace('_', '-'),
            symbol_in.lower().replace('-', '_'), symbol_in.lower().replace('_', '-')
        ])
        for s in variants:
            p = base / f"{s}_{tail}.parquet"
            if p.exists():
                return str(p)
        # Try prefix glob (e.g., BRK -> BRK-A/BRK-B)
        g = list(base.glob(f"{symbol_in.upper()}*_{tail}.parquet"))
        if g:
            g = sorted(g)
            return str(g[0])
        # Fallback to default path (for error message clarity)
        return str(base / f"{symbol_in}_{tail}.parquet")

    def _to_ctime(self, ts) -> CTime:
    ts = pd.Timestamp(ts)
    if getattr(ts, 'tzinfo', None) is not None:
        # normalize to naive for CTime
        ts = ts.tz_convert(None)
    return CTime(int(ts.year), int(ts.month), int(ts.day), int(ts.hour), int(ts.minute))

    def get_kl_data(self):
        fp = self._filepath()
        if not os.path.exists(fp):
            raise CChanException(f"{fp} not exists", ErrCode.SRC_DATA_NOT_FOUND)
        df = pd.read_parquet(fp)

        # ---- Normalize columns (case-insensitive) ----
        import re as _re
        def _norm(s):
            return _re.sub(r'[^a-z]', '', str(s).lower())
        colmap = {c: _norm(c) for c in df.columns}
        rev = {}
        for orig, nm in colmap.items():
            rev.setdefault(nm, orig)

        # timestamp
        ts_cands = ['timestamp','time','datetime','date','dt','t']
        ts_col = None
        for nm in ts_cands:
            if nm in rev:
                ts_col = rev[nm]
                break
        if ts_col is None and pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index().rename(columns={df.columns[0]: 'timestamp'})
            ts_col = 'timestamp'
        if ts_col is None:
            raise CChanException("no timestamp column", ErrCode.SRC_DATA_FORMAT_ERROR)
        if ts_col != 'timestamp':
            df = df.rename(columns={ts_col: 'timestamp'})

        # OHLC columns
        def _pick(cands):
            for nm in cands:
                if nm in rev:
                    return rev[nm]
            return None
        open_col  = _pick(['open','o'])
        high_col  = _pick(['high','h'])
        low_col   = _pick(['low','l'])
        close_col = _pick(['close','adjclose','adjustedclose','closeadj','c'])

        ren = {}
        if open_col and open_col != 'open':   ren[open_col]  = 'open'
        if high_col and high_col != 'high':   ren[high_col]  = 'high'
        if low_col  and low_col  != 'low':    ren[low_col]   = 'low'
        if close_col and close_col != 'close':ren[close_col] = 'close'
        if ren:
            df = df.rename(columns=ren)
        for c in ['open','high','low','close']:
            if c not in df.columns:
                raise CChanException(f"missing column {c}", ErrCode.SRC_DATA_FORMAT_ERROR)

        # Normalize timestamp to tz-naive pandas datetime
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        df['timestamp'] = ts
        if df['timestamp'].isna().any():
            raise CChanException("invalid timestamps", ErrCode.SRC_DATA_FORMAT_ERROR)

        # Parse begin/end
        def _to_ts(s, end=False):
            if s is None:
                return None
            if isinstance(s, str) and len(s) == 10:
                s = s + (" 23:59" if end else " 00:00")
            return pd.to_datetime(s)
        bdt = _to_ts(self.begin_date, end=False)
        edt = _to_ts(self.end_date,   end=True)
        df = df.sort_values('timestamp')
        if bdt is not None:
            df = df[df['timestamp'] >= bdt]
        if edt is not None:
            df = df[df['timestamp'] <= edt]
        if df.empty:
            raise CChanException(f"no data in range [{self.begin_date}, {self.end_date}]", ErrCode.SRC_DATA_NOT_FOUND)

        # Yield
        for r in df.itertuples(index=False):
            item = {
                DATA_FIELD.FIELD_TIME:  self._to_ctime(getattr(r, 'timestamp')),
                DATA_FIELD.FIELD_OPEN:  float(getattr(r, 'open')),
                DATA_FIELD.FIELD_HIGH:  float(getattr(r, 'high')),
                DATA_FIELD.FIELD_LOW:   float(getattr(r, 'low')),
                DATA_FIELD.FIELD_CLOSE: float(getattr(r, 'close')),
            }
            yield CKLine_Unit(item)

    def _to_ts(s, end=False):
            if s is None:
                return None
            if isinstance(s, str) and len(s) == 10:
                s = s + (" 23:59" if end else " 00:00")
            return pd.to_datetime(s)
        bdt = _to_ts(self.begin_date, end=False)
        edt = _to_ts(self.end_date,   end=True)
        df = df.sort_values('timestamp')
        if bdt is not None:
            df = df[df['timestamp'] >= bdt]
        if edt is not None:
            df = df[df['timestamp'] <= edt]
        if df.empty:
            raise CChanException(f"no data in range [{self.begin_date}, {self.end_date}]", ErrCode.SRC_DATA_NOT_FOUND)
        # Validate price columns
        for c in ['open','high','low','close']:
            if c not in df.columns:
                raise CChanException(f"missing column {c}", ErrCode.SRC_DATA_FORMAT_ERROR)
        # Yield
        for r in df.itertuples(index=False):
            item = {
                DATA_FIELD.FIELD_TIME:  self._to_ctime(getattr(r, 'timestamp')),
                DATA_FIELD.FIELD_OPEN:  float(getattr(r, 'open')),
                DATA_FIELD.FIELD_HIGH:  float(getattr(r, 'high')),
                DATA_FIELD.FIELD_LOW:   float(getattr(r, 'low')),
                DATA_FIELD.FIELD_CLOSE: float(getattr(r, 'close')),
            }
            yield CKLine_Unit(item)

    def SetBasciInfo(self):
        pass

    @classmethod
    def load_iterator(cls, code, k_type, begin_date=None, end_date=None, autype=None,**kwargs):
            # 返回可迭代的 CKLine_Unit 序列
            return cls(code, k_type, begin_date, end_date, autype).get_kl_data()

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

