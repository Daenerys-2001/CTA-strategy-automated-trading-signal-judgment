import pickle
import numpy as np

mock_mss = {
    "xgboostmodelmss": {
        "xgboostlearner.features": [
            np.array([0.2] * 19),
            np.array([-0.3] * 19),
            np.array([0.5] * 19),
        ],
        "xgboostlearner.targets": [1.2, -0.3, 1.0],
        "xgboostlearner.labels": ["ticker1_20230101", "ticker2_20230101", "ticker3_20230101"],
        "xgboostlearner.lastTrainDate": "20221231",
    },
    "ticker2wavesmss": {},
    # 上升趋势：短期 EMA > 中期 EMA > 长期 EMA
    "ticker2emamss": {
        # 举例：短期=4, 中期=2, 长期=1
        "ticker1": [[4.0]*5, [2.0]*5, [1.0]*5],
        "ticker2": [[3.5]*5, [1.5]*5, [0.5]*5],
        "ticker3": [[5.0]*5, [3.0]*5, [1.0]*5],
    },
    # 设为负值，使 forecast = -prediction * lastState 得到正数
    "currentprediction": np.array([-1.2, -0.8, -2.0]),
    "onmarketdata": np.array([110, 110, 110], dtype=np.int64),
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

print("✅ 上升趋势的 mock mss 已儲存為 mss_mock_for_test.bin")
