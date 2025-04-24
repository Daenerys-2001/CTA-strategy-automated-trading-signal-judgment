import numpy
class Model:
    """
        Base Class of model
            Derived class can get model state data by calling members of modelState
            universe: a list of future series
            __lastForecast: private attribute which is the forecast of last interval
    """

    def __init__(self, strategy, universe, mss = None):
        self._strategy = strategy
        if mss is None:
            self._dataUniverse = universe
    
        
    def initForNewInterval(self, intv):
        self._currentIntv = intv.strftime("%Y%m%d%H%M")
        

    def initForNewDay(self, curdate):
        pass
        

    def getAllForecasts(self):       
        pass

    def getTickerForecast(self, ticker):
        pass

    def getModelSpecificState(self):        
        pass

    def onEndOfSim(self):
        pass



    def target_am1d_pos_dict(self, strategy):
        """
        This function is used to get the target am1d pos dict
        :param strategy: the strategy object
        :return: the target am1d pos dict
        """
        return strategy.getTargetAm1dPosDict()

        
    
    