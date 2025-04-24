import numpy as np

class Waves:
    def __init__(self, h, span, ticker):
        self.h = h
        self.span = span
        self.ticker = ticker
        self._waves = [0.1 * i for i in range(10)]
        self._sample_index = 0
        self._states = [1.0]
    
    def update(self, cumret, high, low):
        self._waves.append(cumret)
        if len(self._waves) > 20:
            self._waves = self._waves[-20:]
        
        # 新增逻辑：根据累计收益符号更新状态值
        state = 1.0 if cumret >= 0 else -1.0
        self._states.append(state)
        if len(self._states) > 10:
            self._states = self._states[-10:]

    def checkNewExtremum(self):
        return True

    def getWaves(self):
        return self._waves[-10:]

    def getSampleRet(self):
        return 0.01  # 固定 sample return

    def setSampleIndex(self):
        self._sample_index = 0

    def getLastWaveSR(self, n):
        return [0.5] * n

    def getADXRatio(self, short, long):
        return 1.1

    def getATRRatio(self, short, long):
        return 0.9

    def getADX(self, period):
        return 15.0

    def getLastStates(self, n):
        return self._states[-n:]

    def getmodelspecificstate(self):
        return {"waves": self._waves, "index": self._sample_index}

    def setmodelspecificstate(self, mss):
        self._waves = mss.get("waves", [])
        self._sample_index = mss.get("index", 0)
