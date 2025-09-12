# 第1-6行: 导入Python标准库
import copy          # 用于深拷贝对象
import datetime      # 处理日期时间
import pickle        # 用于序列化和反序列化Python对象
import sys           # 系统相关功能，这里用于调整递归限制
from collections import defaultdict  # 创建默认字典，避免KeyError
from typing import Dict, Iterable, List, Optional, Union  # 类型注解

# 第8-16行: 导入项目内部模块
from BuySellPoint.BS_Point import CBS_Point        # 买卖点相关类
from ChanConfig import CChanConfig                 # 缠论配置类
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE # 枚举类型：复权类型、数据源、K线类型
from Common.ChanException import CChanException, ErrCode # 异常处理类
from Common.CTime import CTime                     # 时间处理类
from Common.func_util import check_kltype_order, kltype_lte_day # 工具函数
from DataAPI.CommonStockAPI import CCommonStockApi # 股票数据API基类
from KLine.KLine_List import CKLine_List           # K线列表类
from KLine.KLine_Unit import CKLine_Unit           # K线单元类


class CChan:  # 缠论分析的核心类
    def __init__(
        self,
        code,                                              # 股票代码，比如"000001"
        begin_time=None,                                   # 开始时间，可以为空
        end_time=None,                                     # 结束时间，可以为空
        data_src: Union[DATA_SRC, str] = DATA_SRC.BAO_STOCK, # 数据源，默认用包钢股份数据
        lv_list=None,                                      # 级别列表，比如[日线, 60分钟线]
        config=None,                                       # 配置对象
        autype: AUTYPE = AUTYPE.QFQ,                      # 复权类型，默认前复权
    ):
        # 第30-32行: 如果没有指定级别列表，就用默认的[日线, 60分钟线]
        # 这里的 = DATA_SRC.BAO_STOCK 和 = AUTYPE.QFQ 是默认值，不是固定赋值。它们的作用是：
        # 1.如果调用时不传入这些参数，就使用默认值
        # 2.如果调用时传入了参数，就使用传入的值
        if lv_list is None:
            lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M]  # 默认分析日线和60分钟线
        check_kltype_order(lv_list)  # 检查级别顺序是否正确（从高到低）
        
        # 第33-38行: 保存基本参数
        self.code = code                                   # 保存股票代码
        # 如果 begin_time 是 datetime.date 类型的对象，就把它转换成字符串；如果不是，就保持原样。
        # 三元运算符（条件表达式）：条件为真时返回值1，否则返回值2
        # isinstance(对象, 类型)是Python的内置函数，用来检查对象的类型,它还支持继承关系检查
        self.begin_time = str(begin_time) if isinstance(begin_time, datetime.date) else begin_time
        self.end_time = str(end_time) if isinstance(end_time, datetime.date) else end_time
        self.autype = autype                              # 复权类型
        self.data_src = data_src                          # 数据源
        self.lv_list: List[KL_TYPE] = lv_list            # 要分析的K线级别列表
        
        # 第40-42行: 配置对象处理
        if config is None:
            config = CChanConfig()                        # 如果没有配置，创建默认配置
        self.conf = config                                # 保存配置对象
        
        # 第44-47行: 初始化统计和缓存变量
        self.kl_misalign_cnt = 0                         # K线不对齐计数器
        self.kl_inconsistent_detail = defaultdict(list)  # K线时间不一致的详细记录
        self.g_kl_iter = defaultdict(list)               # 各级别K线数据的迭代器列表
        
        # 第49行: 执行初始化操作
        self.do_init()
        
        # 第51-53行: 如果不是分步触发模式，立即加载所有数据
        if not config.trigger_step:
            for _ in self.load():  # 加载数据，返回值被忽略
                ...  # 这里的...表示什么都不做，只是为了消费迭代器

    def __deepcopy__(self, memo):  # 自定义深拷贝方法，用于复制整个缠论分析对象
        cls = self.__class__                    # 获取当前类的类型
        obj: CChan = cls.__new__(cls)          # 创建一个新的实例，但不调用__init__
        memo[id(self)] = obj                   # 在memo中记录，防止循环引用
        
        # 第59-68行: 复制基本属性
        obj.code = self.code                   # 复制股票代码
        obj.begin_time = self.begin_time       # 复制开始时间
        obj.end_time = self.end_time          # 复制结束时间
        obj.autype = self.autype              # 复制复权类型
        obj.data_src = self.data_src          # 复制数据源
        obj.lv_list = copy.deepcopy(self.lv_list, memo)  # 深拷贝级别列表
        obj.conf = copy.deepcopy(self.conf, memo)        # 深拷贝配置对象
        obj.kl_misalign_cnt = self.kl_misalign_cnt       # 复制K线不对齐计数
        obj.kl_inconsistent_detail = copy.deepcopy(self.kl_inconsistent_detail, memo)  # 深拷贝不一致详情
        obj.g_kl_iter = copy.deepcopy(self.g_kl_iter, memo)  # 深拷贝迭代器列表
        
        # 第69-72行: 复制缓存属性（如果存在的话）
        if hasattr(self, 'klu_cache'):         # 如果有K线缓存
            obj.klu_cache = copy.deepcopy(self.klu_cache, memo)
        if hasattr(self, 'klu_last_t'):       # 如果有最后K线时间记录
            obj.klu_last_t = copy.deepcopy(self.klu_last_t, memo)
        
        # 第73-82行: 复制K线数据并重建引用关系
        obj.kl_datas = {}                      # 初始化K线数据字典
        for kl_type, ckline in self.kl_datas.items():  # 遍历每个级别的K线数据
            obj.kl_datas[kl_type] = copy.deepcopy(ckline, memo)  # 深拷贝K线列表
        
        # 重建K线之间的父子引用关系
        for kl_type, ckline in self.kl_datas.items():
            for klc in ckline:                 # 遍历K线组合
                for klu in klc.lst:            # 遍历每个K线单元
                    assert id(klu) in memo     # 确保K线单元已被拷贝
                    if klu.sup_kl:             # 如果有上级K线引用
                        memo[id(klu)].sup_kl = memo[id(klu.sup_kl)]  # 重建上级引用
                    # 重建下级K线列表引用
                    memo[id(klu)].sub_kl_list = [memo[id(sub_kl)] for sub_kl in klu.sub_kl_list]
        return obj                             # 返回拷贝完成的对象

    def do_init(self):  # 初始化K线数据容器
        self.kl_datas: Dict[KL_TYPE, CKLine_List] = {}  # 创建空的K线数据字典
        for idx in range(len(self.lv_list)):            # 遍历每个要分析的级别
            # 为每个级别创建一个K线列表对象
            self.kl_datas[self.lv_list[idx]] = CKLine_List(self.lv_list[idx], conf=self.conf)

    def load_stock_data(self, stockapi_instance: CCommonStockApi, lv) -> Iterable[CKLine_Unit]:
        # 从股票API实例加载指定级别的K线数据
        for KLU_IDX, klu in enumerate(stockapi_instance.get_kl_data()):  # 遍历获取的K线数据
            klu.set_idx(KLU_IDX)    # 设置K线的索引号
            klu.kl_type = lv        # 设置K线的级别类型
            yield klu               # 生成器返回K线单元

    def get_load_stock_iter(self, stockapi_cls, lv):  # 获取指定级别的股票数据迭代器
        # 创建股票API实例，传入股票代码、K线类型、时间范围、复权类型等参数
        stockapi_instance = stockapi_cls(
            code=self.code, 
            k_type=lv, 
            begin_date=self.begin_time, 
            end_date=self.end_time, 
            autype=self.autype
        )
        return self.load_stock_data(stockapi_instance, lv)  # 返回数据加载迭代器

    def add_lv_iter(self, lv_idx, iter):  # 添加级别迭代器到全局迭代器列表
        if isinstance(lv_idx, int):        # 如果传入的是索引号
            self.g_kl_iter[self.lv_list[lv_idx]].append(iter)  # 根据索引获取级别类型
        else:                              # 如果传入的是级别类型
            self.g_kl_iter[lv_idx].append(iter)  # 直接使用级别类型

    def get_next_lv_klu(self, lv_idx):    # 获取指定级别的下一个K线单元
        if isinstance(lv_idx, int):        # 如果传入的是索引，转换为级别类型
            lv_idx = self.lv_list[lv_idx]
        if len(self.g_kl_iter[lv_idx]) == 0:  # 如果没有可用的迭代器
            raise StopIteration            # 抛出停止迭代异常
        try:
            return self.g_kl_iter[lv_idx][0].__next__()  # 尝试获取下一个K线
        #__next__() 方法会记住当前的位置，每次调用都返回下一个元素，而不是固定返回某个位置的元素。
        except StopIteration:              # 如果当前迭代器已耗尽
            self.g_kl_iter[lv_idx] = self.g_kl_iter[lv_idx][1:]  # 移除第一个迭代器
            if len(self.g_kl_iter[lv_idx]) != 0:  # 如果还有其他迭代器
                return self.get_next_lv_klu(lv_idx)  # 递归调用，使用下一个迭代器
            else:
                raise                      # 所有迭代器都耗尽，重新抛出异常

    def step_load(self):  # 分步加载模式，用于回放或逐步分析
        assert self.conf.trigger_step      # 确保配置为分步触发模式
        self.do_init()                     # 清空数据，防止重复运行时数据残留
        yielded = False                    # 标记是否已经返回过结果
        for idx, snapshot in enumerate(self.load(self.conf.trigger_step)):  # 逐步加载数据
            if idx < self.conf.skip_step:  # 如果还没到指定的跳过步数
                continue                   # 跳过当前步骤
            yield snapshot                 # 返回当前分析快照
            yielded = True                 # 标记已返回结果
        if not yielded:                    # 如果没有返回任何结果
            yield self                     # 返回自身

    def trigger_load(self, inp):  # 触发式加载，用于传入外部K线数据
        # inp格式: {KL_TYPE: [CKLine_Unit, ...]}
        if not hasattr(self, 'klu_cache'):  # 如果没有K线缓存
            # 为每个级别创建一个空的缓存位置
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]
        if not hasattr(self, 'klu_last_t'):  # 如果没有最后时间记录
            # 为每个级别初始化一个很早的时间
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]
        
        for lv_idx, lv in enumerate(self.lv_list):  # 遍历每个级别
            if lv not in inp:              # 如果输入数据中没有这个级别
                if lv_idx == 0:            # 如果是最高级别（最重要的级别）
                    raise CChanException(f"最高级别{lv}没有传入数据", ErrCode.NO_DATA)
                continue                   # 跳过次要级别
            for klu in inp[lv]:            # 为每个K线单元设置级别类型
                klu.kl_type = lv
            assert isinstance(inp[lv], list)  # 确保传入的是列表
            self.add_lv_iter(lv, iter(inp[lv]))  # 添加到迭代器列表
        
        # 开始加载和分析数据
        for _ in self.load_iterator(lv_idx=0, parent_klu=None, step=False):
            ...                            # 消费迭代器，执行分析逻辑
        
        if not self.conf.trigger_step:     # 如果不是回放模式
            for lv in self.lv_list:        # 为每个级别计算线段和中枢
                self.kl_datas[lv].cal_seg_and_zs()

    def init_lv_klu_iter(self, stockapi_cls):  # 初始化各级别的K线迭代器
        # 这个方法用于跳过一些获取数据失败的级别
        lv_klu_iter = []                   # 存储有效的迭代器
        valid_lv_list = []                 # 存储有效的级别列表
        for lv in self.lv_list:            # 遍历每个级别
            try:
                # 尝试获取该级别的数据迭代器
                lv_klu_iter.append(self.get_load_stock_iter(stockapi_cls, lv))
                valid_lv_list.append(lv)   # 记录成功的级别
            except CChanException as e:    # 如果获取数据失败
                # 如果是数据源找不到错误，且配置允许自动跳过
                if e.errcode == ErrCode.SRC_DATA_NOT_FOUND and self.conf.auto_skip_illegal_sub_lv:
                    if self.conf.print_warning:  # 如果配置要打印警告
                        print(f"[WARNING-{self.code}]{lv}级别获取数据失败，跳过")
                    del self.kl_datas[lv]  # 删除该级别的数据容器
                    continue               # 继续下一个级别
                raise e                    # 其他错误重新抛出
        self.lv_list = valid_lv_list      # 更新为有效的级别列表
        return lv_klu_iter                # 返回有效的迭代器列表

    def GetStockAPI(self):  # 根据配置选择合适的股票数据API
        # --- 先处理内置数据源（保持原有行为） ---
        _dict = {}
        if self.data_src == DATA_SRC.BAO_STOCK:
            from DataAPI.BaoStockAPI import CBaoStock
            _dict[DATA_SRC.BAO_STOCK] = CBaoStock
        elif self.data_src == DATA_SRC.CCXT:
            from DataAPI.ccxt import CCXT
            _dict[DATA_SRC.CCXT] = CCXT
        elif self.data_src == DATA_SRC.CSV:
            from DataAPI.csvAPI import CSV_API
            _dict[DATA_SRC.CSV] = CSV_API

        if self.data_src in _dict:
            return _dict[self.data_src]

        # --- 自定义数据源：custom:<module_path>.<attr> ---
        assert isinstance(self.data_src, str)
        if "custom:" not in self.data_src:
            raise CChanException("load src type error", ErrCode.SRC_DATA_TYPE_ERR)

        import importlib, inspect

        provider_path = self.data_src.split(":", 1)[1].strip()
        if not provider_path:
            raise CChanException("empty custom provider", ErrCode.SRC_DATA_TYPE_ERR)

        parts = provider_path.split(".")
        if len(parts) < 2:
            raise CChanException(
                f"custom provider must be 'module.attr', got '{provider_path}'",
                ErrCode.SRC_DATA_TYPE_ERR
            )

        attr_name  = parts[-1]
        module_path = ".".join(parts[:-1])

        # 先按原样导入；若失败且未以 DataAPI. 开头，则尝试加上 DataAPI. 前缀做一次回退
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError:
            if not module_path.startswith("DataAPI."):
                mod = importlib.import_module("DataAPI." + module_path)
            else:
                raise
        provider = getattr(mod, attr_name)

        # ---- 统一包装：自动补齐 code/k_type/时间/复权，并吞掉 lv_idx 等多余参数 ----
        def _resolve_k_type(kw):
            # 从 kwargs 中识别 k 线级别；优先显式字段，其次从 lv_idx -> self.lv_list
            for key in ("k_type", "kl_type", "level", "lv", "tf", "timeframe"):
                if kw.get(key) is not None:
                    return kw[key]
            if "lv_idx" in kw and hasattr(self, "lv_list"):
                try:
                    return self.lv_list[kw["lv_idx"]]
                except Exception:
                    pass
            return None

        def _wrap_call(callable_like, *args, **kw):
            # 统一从 self/kwargs 补齐必要形参
            code       = kw.pop("code", None) or getattr(self, "code", None)
            k_type     = _resolve_k_type(kw)
            begin_date = kw.pop("begin_date", None) or getattr(self, "begin_time", None)
            end_date   = kw.pop("end_date",   None) or getattr(self, "end_time",   None)
            autype     = kw.pop("autype",     None) or getattr(self, "autype",     None)

            # 若调用方把 code/k_type 作为位置参数给了，也做兜底兼容
            if (code is None or k_type is None) and args:
                if code is None and len(args) > 0: code = args[0]
                if k_type is None and len(args) > 1: k_type = args[1]

            if code is None or k_type is None:
                raise CChanException("custom provider needs code & k_type", ErrCode.SRC_DATA_TYPE_ERR)

            # 用关键字方式调用，剩余未知参数（如 lv_idx）留在 kw 里由 provider 自行忽略/接收
            return callable_like(code=code, k_type=k_type,
                                begin_date=begin_date, end_date=end_date,
                                autype=autype, **kw)

        # ---- 函数/可调用对象型提供者 ----
        if inspect.isfunction(provider) or (hasattr(provider, "__call__") and not inspect.isclass(provider)):
            def _li(*args, **kw):
                return _wrap_call(provider, *args, **kw)
            self.load_iterator = _li
            self.do_init  = getattr(provider, "do_init",  lambda *a, **k: None)
            self.do_close = getattr(provider, "do_close", lambda *a, **k: None)
            return provider

        # ---- 类型提供者：优先 classmethod load_iterator；否则实例.get_kl_data 兜底 ----
        if inspect.isclass(provider):
            if hasattr(provider, "load_iterator"):
                def _li(*args, **kw):
                    # 包一层以吞掉 lv_idx 等，并补齐必需参数
                    return _wrap_call(lambda **pp: provider.load_iterator(**pp), *args, **kw)
                self.load_iterator = _li
            elif hasattr(provider, "get_kl_data"):
                def _li(*args, **kw):
                    def _call(**pp):
                        inst = provider(pp["code"], pp["k_type"], pp["begin_date"], pp["end_date"], pp["autype"])
                        return inst.get_kl_data()
                    return _wrap_call(_call, *args, **kw)
                self.load_iterator = _li
            else:
                raise CChanException(
                    "custom class must provide 'load_iterator' or 'get_kl_data'",
                    ErrCode.SRC_DATA_TYPE_ERR
                )

            self.do_init  = getattr(provider, "do_init",  lambda *a, **k: None)
            self.do_close = getattr(provider, "do_close", lambda *a, **k: None)
            return provider

        # 其它不可用类型
        raise CChanException(
            f"unsupported custom provider type: {type(provider)}",
            ErrCode.SRC_DATA_TYPE_ERR
        )



    def load(self, step=False):  # 主要的数据加载方法
        stockapi_cls = self.GetStockAPI()      # 获取对应的股票API类
        try:
            stockapi_cls.do_init()             # 初始化API（比如连接数据库）
            # 初始化各级别的K线迭代器并添加到全局迭代器列表
            for lv_idx, klu_iter in enumerate(self.init_lv_klu_iter(stockapi_cls)):
                self.add_lv_iter(lv_idx, klu_iter)
            
            # 初始化缓存和时间记录
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]
            
            # 开始递归加载和分析，从最高级别（索引0）开始
            yield from self.load_iterator(lv_idx=0, parent_klu=None, step=step)
            
            if not step:                       # 如果不是分步模式
                for lv in self.lv_list:        # 全部加载完成后，计算线段和中枢
                    self.kl_datas[lv].cal_seg_and_zs()
        except Exception:
            raise                              # 重新抛出任何异常
        finally:
            stockapi_cls.do_close()            # 无论如何都要关闭API连接
        
        if len(self[0]) == 0:                  # 如果最高级别没有任何数据
            raise CChanException("最高级别没有获得任何数据", ErrCode.NO_DATA)

    def set_klu_parent_relation(self, parent_klu, kline_unit, cur_lv, lv_idx):
        # 设置K线的父子关系
        # 如果启用数据检查且都是日级别以下，检查时间一致性
        if self.conf.kl_data_check and kltype_lte_day(cur_lv) and kltype_lte_day(self.lv_list[lv_idx-1]):
            self.check_kl_consitent(parent_klu, kline_unit)
        parent_klu.add_children(kline_unit)    # 父K线添加子K线
        kline_unit.set_parent(parent_klu)      # 子K线设置父K线引用

    def add_new_kl(self, cur_lv: KL_TYPE, kline_unit):  # 添加新的K线到对应级别
        try:
            self.kl_datas[cur_lv].add_single_klu(kline_unit)  # 调用K线列表的添加方法
        except Exception:
            if self.conf.print_err_time:       # 如果配置要打印错误时间
                print(f"[ERROR-{self.code}]在计算{kline_unit.time}K线时发生错误!")
            raise                              # 重新抛出异常

    def try_set_klu_idx(self, lv_idx: int, kline_unit: CKLine_Unit):  # 尝试设置K线索引
        if kline_unit.idx >= 0:               # 如果已经有索引了
            return                             # 直接返回
        if len(self[lv_idx]) == 0:            # 如果是该级别的第一根K线
            kline_unit.set_idx(0)             # 索引设为0
        else:
            # 否则设为前一根K线的索引+1
            kline_unit.set_idx(self[lv_idx][-1][-1].idx + 1)

## 核心迭代加载逻辑 (第234-268行)

def load_iterator(self, lv_idx, parent_klu, step):  # 核心的递归加载迭代器
    # 注释：K线时间含义
    # - 天级别以下：描述结束时间（如60分钟线，每天第一根是10:30的）
    # - 天级别以上：是当天日期
    
    cur_lv = self.lv_list[lv_idx]          # 当前处理的级别
    # 获取前一根K线，用于设置链表关系
    pre_klu = self[lv_idx][-1][-1] if len(self[lv_idx]) > 0 and len(self[lv_idx][-1]) > 0 else None
    
    while True:                            # 无限循环处理K线
        if self.klu_cache[lv_idx]:         # 如果缓存中有K线
            kline_unit = self.klu_cache[lv_idx]  # 使用缓存的K线
            assert kline_unit is not None
            self.klu_cache[lv_idx] = None  # 清空缓存
        else:                              # 如果缓存中没有K线
            try:
                kline_unit = self.get_next_lv_klu(lv_idx)  # 获取下一根K线
                self.try_set_klu_idx(lv_idx, kline_unit)   # 设置K线索引
                # 检查K线时间是否递增（保证时间序列的单调性）
                if not kline_unit.time > self.klu_last_t[lv_idx]:
                    raise CChanException(f"kline time err, cur={kline_unit.time}, last={self.klu_last_t[lv_idx]}, or refer to quick_guide.md, try set auto=False in the CTime returned by your data source class", ErrCode.KL_NOT_MONOTONOUS)
                self.klu_last_t[lv_idx] = kline_unit.time  # 更新最后时间记录
            except StopIteration:          # 如果没有更多K线了
                break                      # 退出循环
        
        # 如果有父K线且当前K线时间超过了父K线时间
        if parent_klu and kline_unit.time > parent_klu.time:
            self.klu_cache[lv_idx] = kline_unit  # 将当前K线放入缓存
            break                          # 退出当前级别的处理
        
        kline_unit.set_pre_klu(pre_klu)    # 设置前一根K线的引用
        pre_klu = kline_unit               # 更新前一根K线为当前K线
        self.add_new_kl(cur_lv, kline_unit) # 添加K线到对应级别
        
        if parent_klu:                     # 如果有父K线
            self.set_klu_parent_relation(parent_klu, kline_unit, cur_lv, lv_idx)
        
        if lv_idx != len(self.lv_list)-1:  # 如果不是最后一个级别
            # 递归处理下一个级别，当前K线作为父K线
            for _ in self.load_iterator(lv_idx+1, kline_unit, step):
                ...                        # 消费迭代器
            self.check_kl_align(kline_unit, lv_idx)  # 检查K线对齐情况
        
        if lv_idx == 0 and step:           # 如果是最高级别且是分步模式
            yield self                     # 返回当前分析状态的快照

## 数据校验和访问方法 (第269-307行)

def check_kl_consitent(self, parent_klu, sub_klu):  # 检查父子K线时间一致性
    # 检查父子K线是否在同一天（年月日都要相同）
    if parent_klu.time.year != sub_klu.time.year or \
       parent_klu.time.month != sub_klu.time.month or \
       parent_klu.time.day != sub_klu.time.day:
        # 记录不一致的详情
        self.kl_inconsistent_detail[str(parent_klu.time)].append(sub_klu.time)
        if self.conf.print_warning:       # 如果配置要打印警告
            print(f"[WARNING-{self.code}]父级别时间是{parent_klu.time}，次级别时间却是{sub_klu.time}")
        # 如果不一致的条数超过限制，抛出异常
        if len(self.kl_inconsistent_detail) >= self.conf.max_kl_inconsistent_cnt:
            raise CChanException(f"父&子级别K线时间不一致条数超过{self.conf.max_kl_inconsistent_cnt}！！", ErrCode.KL_TIME_INCONSISTENT)

def check_kl_align(self, kline_unit, lv_idx):  # 检查K线对齐情况
    # 如果启用数据检查且当前K线没有找到对应的次级别K线
    if self.conf.kl_data_check and len(kline_unit.sub_kl_list) == 0:
        self.kl_misalign_cnt += 1          # 增加不对齐计数
        if self.conf.print_warning:       # 如果配置要打印警告
            print(f"[WARNING-{self.code}]当前{kline_unit.time}没在次级别{self.lv_list[lv_idx+1]}找到K线！！")
        # 如果不对齐的条数超过限制，抛出异常
        if self.kl_misalign_cnt >= self.conf.max_kl_misalgin_cnt:
            raise CChanException(f"在次级别找不到K线条数超过{self.conf.max_kl_misalgin_cnt}！！", ErrCode.KL_DATA_NOT_ALIGN)

def __getitem__(self, n) -> CKLine_List:  # 支持索引访问K线数据
    if isinstance(n, KL_TYPE):            # 如果传入的是级别类型
        return self.kl_datas[n]           # 直接返回对应级别的K线列表
    elif isinstance(n, int):              # 如果传入的是索引
        return self.kl_datas[self.lv_list[n]]  # 根据索引获取级别，再返回K线列表
    else:
        raise CChanException("unspoourt query type", ErrCode.COMMON_ERROR)

def get_bsp(self, idx=None) -> List[CBS_Point]:  # 获取买卖点（已废弃）
    print('[deprecated] use get_latest_bsp instead')  # 提示使用新方法
    if idx is not None:                   # 如果指定了级别索引
        return self[idx].bs_point_lst.getSortedBspList()  # 返回该级别的排序买卖点列表
    assert len(self.lv_list) == 1         # 确保只有一个级别
    return self[0].bs_point_lst.getSortedBspList()     # 返回第一个级别的买卖点

def get_latest_bsp(self, idx=None, number=1) -> List[CBS_Point]:  # 获取最新的买卖点
    # number=0则取全部买卖点，从最新到最旧排序
    if idx is not None:                   # 如果指定了级别索引
        return self[idx].bs_point_lst.get_latest_bsp(number)
    assert len(self.lv_list) == 1         # 确保只有一个级别
    return self[0].bs_point_lst.get_latest_bsp(number)


## 序列化和反序列化方法 (第309-372行)


def chan_dump_pickle(self, file_path):  # 将缠论分析对象序列化保存到文件
    _pre_limit = sys.getrecursionlimit()   # 保存当前递归限制
    sys.setrecursionlimit(0x100000)       # 设置更大的递归限制（1048576）
    
    # 断开所有链表引用，避免序列化时的循环引用问题
    for kl_list in self.kl_datas.values():     # 遍历每个级别的K线列表
        for klc in kl_list.lst:                 # 遍历每个K线组合
            for klu in klc.lst:                 # 遍历每个K线单元
                klu.pre = None                  # 断开前向引用
                klu.next = None                 # 断开后向引用
            klc.set_pre(None)                   # 断开K线组合的前向引用
            klc.set_next(None)                  # 断开K线组合的后向引用
        
        for bi in kl_list.bi_list:              # 遍历每个笔
            bi.pre = None                       # 断开笔的前向引用
            bi.next = None                      # 断开笔的后向引用
        
        for seg in kl_list.seg_list:            # 遍历每个线段
            seg.pre = None                      # 断开线段的前向引用
            seg.next = None                     # 断开线段的后向引用
        
        for segseg in kl_list.segseg_list:      # 遍历每个段段
            segseg.pre = None                   # 断开段段的前向引用
            segseg.next = None                  # 断开段段的后向引用
    
    with open(file_path, "wb") as f:           # 以二进制写模式打开文件
        pickle.dump(self, f)                   # 序列化并保存对象
    
    sys.setrecursionlimit(_pre_limit)          # 恢复原来的递归限制

@staticmethod
def chan_load_pickle(file_path) -> 'CChan':    # 从文件反序列化缠论分析对象
    with open(file_path, "rb") as f:           # 以二进制读模式打开文件
        chan = pickle.load(f)                  # 反序列化对象
    
    # 重建所有的链表引用关系
    last_klu = None                            # 记录最后一个K线单元
    last_klc = None                            # 记录最后一个K线组合
    last_bi = None                             # 记录最后一个笔
    last_seg = None                            # 记录最后一个线段
    last_segseg = None                         # 记录最后一个段段
    
    for kl_list in chan.kl_datas.values():    # 遍历每个级别的K线列表
        for klc in kl_list.lst:                # 遍历每个K线组合
            for klu in klc.lst:                # 遍历每个K线单元
                klu.pre = last_klu             # 设置前向引用
                if last_klu:                   # 如果有前一个K线单元
                    last_klu.next = klu        # 设置前一个的后向引用
                last_klu = klu                 # 更新最后一个K线单元
            
            klc.set_pre(last_klc)              # 设置K线组合的前向引用
            if last_klc:                       # 如果有前一个K线组合
                last_klc.set_next(klc)         # 设置前一个的后向引用
            last_klc = klc                     # 更新最后一个K线组合
        
        for bi in kl_list.bi_list:             # 遍历每个笔
            bi.pre = last_bi                   # 设置笔的前向引用
            if last_bi:                        # 如果有前一个笔
                last_bi.next = bi              # 设置前一个的后向引用
            last_bi = bi                       # 更新最后一个笔
        
        for seg in kl_list.seg_list:           # 遍历每个线段
            seg.pre = last_seg                 # 设置线段的前向引用
            if last_seg:                       # 如果有前一个线段
                last_seg.next = seg            # 设置前一个的后向引用
            last_seg = seg                     # 更新最后一个线段
        
        for segseg in kl_list.segseg_list:     # 遍历每个段段
            segseg.pre = last_segseg           # 设置段段的前向引用
            if last_segseg:                    # 如果有前一个段段
                last_segseg.next = segseg      # 设置前一个的后向引用
            last_segseg = segseg               # 更新最后一个段段
    
    return chan                                # 返回重建完成的对象
