import pickle
import numpy as np

mock_mss = {
    "xgboostmodelmss": {
        "xgboostlearner.features": [
            np.array([0.2] * 19),
            np.array([-0.3] * 19),
            np.array([0.5] * 19),
        ],
        "xgboostlearner.targets": [1.2, -0.3, 1.0],  # 目标值
        "xgboostlearner.labels": ["ticker1_20230101", "ticker2_20230101", "ticker3_20230101"],
        "xgboostlearner.lastTrainDate": "20221231",
    },
    "ticker2wavesmss": {},
    "ticker2emamss": {
        "ticker1": [[1.0]*5, [2.0]*5, [11.0]*5],  # 模拟下跌趋势
        "ticker2": [[1.0]*5, [2.0]*5, [11.0]*5],  
        "ticker3": [[1.0]*5, [2.0]*5, [11.0]*5],  
    },
    "currentprediction": np.array([1.5, -1.8, 2.5]),  # 预测值
    "onmarketdata": np.array([110, 110, 110], dtype=np.int64),  # 市场数据值
    "ticker2feature": {
        "ticker1": np.array([0.2] * 19),
        "ticker2": np.array([-0.3] * 19),
        "ticker3": np.array([0.5] * 19),
    },
    "ticker2vol20s": {
        "ticker1": np.array([1.0] * 40),
        "ticker2": np.array([1.0] * 40),
        "ticker3": np.array([1.0] * 40),
    },
    "ticker2totalopis": {
        "ticker1": np.array([100] * 130),
        "ticker2": np.array([100] * 130),
        "ticker3": np.array([100] * 130),
    },
    "cumlagret": np.zeros(3),
    "datauniverse": ["ticker1", "ticker2", "ticker3"],
    "msslastdate": "20230101",
    "ticker2intv2forecast": {},
}

with open("mss_mock_for_test.bin", "wb") as f:
    pickle.dump(mock_mss, f)

print("✅ 完整 mock mss 已儲存為 mss_mock_for_test.bin")