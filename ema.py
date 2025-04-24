import math
class EMA:
    """Exponential moving average class."""

    LOG2 = math.log(2.0)

    def __init__(self, h):
        """Initialize an EMA with a half life of h days.
        If h is a list, then all the results will be lists too."""
        if type(h) == int or type(h) == float:
            self.halflife = [ h ]
            self.len = 1
        elif type(h) == list:
            self.halflife = h
            self.len = len(h)
        else:
            raise TypeError
        self.__sum = [0.0] * self.len
        self.__weight = [0.0] * self.len
        self.__sumSq = [0.0] * self.len

    def forceDecay(self, rate) :
        for i in range(self.len):
            self.__sum[i] *= rate
            self.__sumSq[i] *= rate
            self.__weight[i] *= rate


    def add(self, value, daysSinceLast):
        """Add an observation to the EMA,
        and decay according to days since last observation."""
        if math.isnan(value) or math.isnan(daysSinceLast):
            return
        if daysSinceLast < 0:
            daysSinceLast = 0
        for i in range(self.len):
            if self.halflife[i] > 0:
                decay = math.exp(- daysSinceLast / self.halflife[i] * self.LOG2)
            else:
                decay = 1
            self.__sum[i] = self.__sum[i] * decay + value
            self.__sumSq[i] = self.__sumSq[i] * decay + value * value
            self.__weight[i] = self.__weight[i] * decay + 1

    def getMean(self):
        "Return the mean of the EMA."
        result = [ 0.0 if w == 0 else s / w for s, w in zip(self.__sum, self.__weight) ]
        return result[0] if self.len == 1 else result



    def getSquaredMean(self):
        result = [0.0] * self.len
        for i in range(self.len):
            if self.__weight[i] == 0:
                result[i] = 0.0
            else:
                result[i] = self.__sumSq[i] / self.__weight[i]
        if self.len == 1:
            return result[0]
        else:
            return result


    def getVar(self):
        """Return the variance of the EMA.
        Variance is defined as \sum_i a^i (x_{n-i} - \bar x)^2 / \sum_i a^i."""
        result = [ 0.0 if w == 0 else max(sq / w - (s / w) ** 2, 0.0) for w, s, sq in zip(self.__weight, self.__sum, self.__sumSq) ]
        return result[0] if self.len == 1 else result


    def getImpulseMean(self):
        "Return the mean of the impulse EMA."
        result = [0.0] * self.len
        for i in range(self.len):
            result[i] = self.__sum[i]
        if self.len == 1:
            return result[0]
        else:
            return result

    def getImpulseVar(self):
        """Return the variance of the impulse EMA.
        Variance is defined as \sum_i a^i (x_{n-i} - \bar x)^2 / \sum_i a^i."""
        result = [0.0] * self.len
        for i in range(self.len):
            var = self.__sumSq[i] - (self.__sum[i]) ** 2
            if var >= 0:
                result[i] = var
            else:
                result[i] = 0.0
        if self.len == 1:
            return result[0]
        else:
            return result

    def getParameters(self):
        "Return parameters [sum, weight and sum-squared]"
        return [self.__sum, self.__weight, self.__sumSq]

    def setParameters(self, initSum, initWeight, initSumSq):
        "Set parameters sum, weight and sum-squared"
        if self.len != len(initSum) or self.len != len(initWeight) or \
           self.len != len(initSumSq):
            raise Exception('Incorrect init parameters')

        self.__sum = initSum
        self.__weight = initWeight
        self.__sumSq = initSumSq

    def reset(self):
        "Clear observation history, but decay rate remains the same."
        self.__sum = [0.0] * self.len
        self.__weight = [0.0] * self.len
        self.__sumSq = [0.0] * self.len