import datetime
import numpy as np

# 假设这些类已在你的项目中定义好，并可直接导入
from factor_func_240 import Uranus45XGBModel

# -----------------------------------------------------------------------------
# 模拟市场数据和策略
# -----------------------------------------------------------------------------
class MockMarketData15m:
    def __init__(self, start):
        self.datetime_list = [start + datetime.timedelta(minutes=15*i) for i in range(200)]
        self.close = np.linspace(100, 140, 40)
        self.open_interest = np.random.randint(100, 200, 40)

class MockMarketData1d:
    def __init__(self):
        self.idx_open_interest = np.random.randint(500, 1000, 40)

class MockStrategy:
    def __init__(self):
        self.target_list = ['contract1', 'contract2']
        self.target_symbol_name_dict = {
            'contract1': 'ticker1',
            'contract2': 'ticker2'
        }
        # 15 分钟数据的当前索引
        self.target_am15_pos_dict = {c: 0 for c in self.target_list}
        self.target_am15_dict = {
            c: MockMarketData15m(datetime.datetime(2023,1,1,9,0))
            for c in self.target_list
        }
        # 标记哪些合约有“新数据”，WavesXGBModel 会检查这个字段
        self.new_data_15m_dict = {c: True for c in self.target_list}
        
        # 日线持仓量索引和数据
        self.target_am1d_pos_dict = {c: 0 for c in self.target_list}
        self.target_am1d_dict = {c: MockMarketData1d() for c in self.target_list}
        
        # 策略全局字典，用于缓存模型实例等
        self.global_dict = {}

# -----------------------------------------------------------------------------
# 因子函数（与策略框架对接）
# -----------------------------------------------------------------------------
import pickle
def factor_func240(strategy):
    # 模型首次初始化
    if 'factor_func240' not in strategy.global_dict:
        universe = list(strategy.target_symbol_name_dict.values())
        ticker2contract = {v: k for k, v in strategy.target_symbol_name_dict.items()}
       
        mss_file = '/Users/tiantianwu/Desktop/HYTP/project4/mss_mock_for_test.bin'
        bst_file = '/Users/tiantianwu/Desktop/HYTP/project4/bst_20240701.json'
        with open(mss_file, "rb") as f:
            mss = pickle.load(f)
        model = Uranus45XGBModel(strategy, universe, mss, ticker2contract, bst_file)
        model._xgboostLearner._XgboostLearner__lastTrainDate = None
        model._trainMinSize = 0
        # 初始化后

        strategy.global_dict['factor_func240'] = {
            "model": model,
            "prev_date": None
        }

    model = strategy.global_dict['factor_func240']["model"]

    # 获取最新的 15 分钟时间点
    times = []
    for c in strategy.target_list:
        pos = strategy.target_am15_pos_dict[c]
        md = strategy.target_am15_dict[c]
        if pos < len(md.datetime_list):
            times.append(md.datetime_list[pos])
    if not times:
        return {}
    current_intvtime = max(times)

    # 日切换
    current_date = current_intvtime.date()
    datestr = current_date.strftime("%Y%m%d")
    prev_date = strategy.global_dict['factor_func240']["prev_date"]
    if prev_date != current_date:
        model.initForNewDay(datestr)
        strategy.global_dict['factor_func240']["prev_date"] = current_date

    # 区间更新
    model.initForNewInterval(current_intvtime)

    # 生成信号
    factor_dict = {}
    for contract in strategy.target_list:
        ticker = strategy.target_symbol_name_dict[contract]
        raw = model.getTickerForecast(ticker)
        factor_dict[contract] = np.clip(raw, -1.0, 1.0)

    return factor_dict

# -----------------------------------------------------------------------------
# 主函数，用于本地测试
# -----------------------------------------------------------------------------
def main():
    strategy = MockStrategy()

    # 模拟逐根 15 分钟 K 线
    max_steps = len(next(iter(strategy.target_am15_dict.values())).datetime_list)
    for step in range(max_steps):
        # 步进每个合约的 15 分钟指针（不超出上限）
        for contract in strategy.target_list:
            strategy.target_am15_pos_dict[contract] = min(
                strategy.target_am15_pos_dict[contract] + 1,
                max_steps - 1
            )

        # 调用因子函数并打印输出
        signals = factor_func240(strategy)
        sample_contract = strategy.target_list[0]
        sample_md = strategy.target_am15_dict[sample_contract]
        sample_pos = strategy.target_am15_pos_dict[sample_contract]
        current_time = sample_md.datetime_list[sample_pos]
        print(f"{current_time} -> signals: {signals}")

if __name__ == "__main__":
    main()