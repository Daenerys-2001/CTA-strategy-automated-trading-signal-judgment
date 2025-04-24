from scipy.stats.mstats import winsorize
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy

import xgboost
import datetime

class csvwriter:
    def __init__(self, columns, path):
        self.__columns = columns
        self.__path = path 
        self.__lines = []
        self.__lines.append(",".join(columns))
        
    def writeline(self, line):
        self.__lines.append(line)
    
    def dump(self):
        f = open(self.__path, "w")
        for line in self.__lines:
            f.writelines(line + "\n")
        f.close()
        

def dumpdatatocsv(featureArray, targets, labels, featurenames , begindate, enddate, path):
    writer  = csvwriter(["labels"] + featurenames + ["target"], path)
    featureArray = numpy.array(featureArray)
    for l in range(len(labels)):
        label = labels[l]
        ticker, intv = label.split("_")
        if intv[:8] < begindate: 
            continue 
        elif intv[:8] > enddate:
            break
        feature = featureArray[l, :]
        feature_str = [str(f) for f in feature]
        line = ",".join([label] + feature_str + [str(targets[l])])
        writer.writeline(line)
    writer.dump()
    
class MLEngine:
    
    def __init__(self, trainMinSize, winsorizePecentile, demeanFactor, normResponse = False, featureNames = None):
        self._trainMinSize = trainMinSize
        self._winsorizePecentile = winsorizePecentile
        self._demeanFactor = demeanFactor
      
        self._regressor = None 
        self._trainSizeInterval  = None 
        self._features = []
        self._labels = []
        self._featureArray = None 
        self._targets = []
        self._responseArray = None
        self.__lastTrainIndex = 0
        self._lastTrainedMonth = None
        self._normalizeFactors = None
        self._normResponse = normResponse 
        self._featureNames = featureNames 
        self._predictions = []
    
   
            
    def update(self, feature, target, prediction = numpy.nan, label = None):        
        self._features.append(feature)
        self._targets.append(target)
        self._labels.append(label)
        self._predictions.append(prediction)

    def prepareArray(self, curMonth):
        print(">>> prepareArray:", len(self._targets), self.__lastTrainDate)

        train = False
        if len(self._targets) > self._trainMinSize:
            if self._regressor is None:
                train = True 
            
            if  curMonth > self._lastTrainedMonth:
                train = True            
       
        if not train:
            return False
        
        trainIndex = len(self._targets)
        featureArray = numpy.array(self._features[:trainIndex])
        targetArray = numpy.array(self._targets[:trainIndex])#feature winsorize
        for i in range(featureArray.shape[1]):
            featureArray[:, i] = winsorize(featureArray[:, i], 
                                                limits = (self._winsorizePecentile, self._winsorizePecentile))
        self._normalizeFactors = 1.0 / numpy.std(featureArray, axis = 0)
        
        featureArray = featureArray * numpy.tile(self._normalizeFactors, (len(self._targets), 1))

        responseArray = winsorize(targetArray, limits = (self._winsorizePecentile, self._winsorizePecentile))
        
        responseArray = (responseArray - self._demeanFactor * numpy.mean(responseArray)) / numpy.nanstd(responseArray) 
        #if curMonth > "202001":
        #    knnvisual.plotdistribution(responseArray)   
        self._featureArray = featureArray
        self._responseArray = responseArray
        self._lastTrainedMonth = curMonth
        self.__lastTrainIndex = trainIndex
        return True


class KNNLearner(MLEngine):
    def __init__(self, trainMinSize, winsorizePecentile, demeanFactor, neighbors, weights, normResponse = True, featureNames = None):
        MLEngine.__init__(self, trainMinSize, winsorizePecentile, demeanFactor, normResponse, featureNames)
        self.__neighbors = neighbors
        self._weights = weights
        self._regressor = None
        
    
    def train(self, plot = False):
        if self.__neighbors < 1:
            neighbors = int(self.__neighbors * len(self._responseArray))
        else:
            neighbors = self.__neighbors
        self._regressor = KNeighborsRegressor(n_neighbors = neighbors, weights = self._weights)
        self._regressor.fit(self._featureArray, self._responseArray)
        """
        if plot:
            print(self._featureNames)
            knnvisual.heat(self._featureArray[:, [0, 2]], self._responseArray, self._featureNames[0], self._featureNames[2])
        """
        
    def predict(self, feature): 
        if self._regressor is None:
            return numpy.nan 
        
        return self._regressor.predict([feature * self._normalizeFactors])[0]

class KMeansLearner(MLEngine):
    def __init__(self, trainMinSize, winsorizePecentile, demeanFactor, n_clusters, name):
        MLEngine.__init__(self, trainMinSize, winsorizePecentile, demeanFactor, False)
        self._nClusters = n_clusters
        self.__name = name
    
    def prepareArray(self, curMonth):
        train = False
    
        if len(self._targets) > self._trainMinSize:
            #if  (len(self._targets) - self._lastTrainIndex) > self._trainSizeInterval:
            #    train = True        
            if self._regressor is None:
                train = True 
            if  curMonth > self._lastTrainedMonth:
                train = True
       
        if not train:
            return False
        
        trainIndex = len(self._targets)
        featureArray = numpy.array(self._features[:trainIndex])
        targetArray = numpy.array(self._targets[:trainIndex])#feature winsorize
        for i in range(featureArray.shape[1]):
            featureArray[:, i] = winsorize(featureArray[:, i], 
                                                limits = (self._winsorizePecentile, self._winsorizePecentile))
        #self._normalizeFactors = 1.0 
        #featureArray = featureArray * numpy.tile(self._normalizeFactors, (len(self._targets), 1))

        responseArray = winsorize(targetArray, limits = (self._winsorizePecentile, self._winsorizePecentile))
        #responseArray = responseArray - self._demeanFactor * numpy.mean(responseArray)            
        self._featureArray = featureArray
        #self._responseArray = numpy.array(self._targets[:trainIndex])
        self._responseArray = targetArray
        #if curMonth > "201501":
        #    knnvisual.plotdistribution(self._responseArray)
        self._lastTrainedMonth = curMonth
      
        return True
    
   
    def train(self):
       
        self._regressor = KMeans(self._nClusters, init = 'k-means++', n_init = 20, random_state = 42)
        self._regressor.fit(self._featureArray)
        #self._regressor.transform(self._featureArray)
    
    def predict(self, feature):
        if self._regressor is None:
            return numpy.nan 

        return self._regressor.predict([feature])[0]

class ZigKMeansLearner(KMeansLearner):
    def __init__(self, trainMinSize, winsorizePecentile, demeanFactor, n_clusters, name, height, pos = False):
        KMeansLearner.__init__(self, trainMinSize, winsorizePecentile, demeanFactor, n_clusters, name)
        self.__height = height
        self.__pos = pos

    def train(self, plot = False):
        KMeansLearner.train(self)
        labels = self._regressor.labels_
        targetMean = numpy.zeros(self._nClusters)
        counts = numpy.zeros(self._nClusters)
        for label in range(self._nClusters):
            targetMean[label] = self._responseArray[labels == label].mean()
            counts[label] = numpy.sum(labels == label)
            
        counts = counts /  len(self._responseArray)   
        
        centers = self._regressor.cluster_centers_
   
        if self.__pos:
            sortedIndex = numpy.argsort(targetMean)
        else:
            sortedIndex = numpy.argsort(targetMean)[::-1]
        
       
        self._regressor.cluster_centers_ = centers[sortedIndex, :]
      
        # if self.__pos:
        #     print("pos center", self._regressor.cluster_centers_, targetMean[sortedIndex], counts[sortedIndex])
        # else:
        #     print("neg center", self._regressor.cluster_centers_, targetMean[sortedIndex], counts[sortedIndex])

        return targetMean
    
    def getmodelspecificstate(self):
        mss = {"kmeanslearner.features": self._features,
               "kmeanslearner.targets": self._targets,
               "kmeanslearner.labels": self._labels,
               "kmeanslearner.lastTrainedMonth": self._lastTrainedMonth,
               "kmeanslearner.cluster_centers": self._regressor.cluster_centers_}
        return mss

    def setmodelspecificstate(self, mss, jsonpath):
        self._features = mss.get("xgboostlearner.features", [])
        self._targets = mss["kmeanslearner.targets"]
        self._labels = mss["kmeanslearner.labels"]
        self._lastTrainedMonth = mss["kmeanslearner.lastTrainedMonth"]
        self._regressor = KMeans(self._nClusters, init = mss["kmeanslearner.cluster_centers"] , n_init = 20, random_state = 42)
        self._regressor.cluster_centers_ = mss["kmeanslearner.cluster_centers"]
        self._regressor._n_threads = 10
    


def r2_score(preds, dtrain):
    labels = dtrain.get_label()
    ss_res = numpy.sum((labels - preds) ** 2)
    ss_tot = numpy.sum((labels - numpy.mean(labels)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return 'r2', r2



class XgboostLearner(MLEngine):
    def __init__(self, trainMinSize, winsorizePecentile, demeanFactor, featureNames, training_interval_days = 30, random_state = 42):
        MLEngine.__init__(self, trainMinSize, winsorizePecentile, demeanFactor , True, featureNames)
        self.__param = [('tree_method', 'auto'), ('max_depth', 3), ('objective', 'reg:squarederror'), ("booster", "gbtree"),
        ("min_child_weight", 1),  ("eta", 0.1), ("gamma", 0), ("nthread", 10), ("seed", 0)]
        self.__lastTrainDate = None
        self.__dates = []
        self.__mins = []
        self.__label2predict = {}
        #self.__csvwriter = util.csvwriter(["label"] + featureNames + ["predict", "target"], "D:/projects/runbacktest/ap.csv")
        self.__training_interval_days  = training_interval_days
        self.__random_state = random_state 
        
    def prepareArray(self, curDate):
        train = False
        print(len(self._targets), self.__lastTrainDate) 
        if len(self._targets) > self._trainMinSize:
            if self._regressor is None and self.__lastTrainDate is None:
                train = True
            else:
                curdt = datetime.datetime.strptime(curDate, "%Y%m%d")
                lasttraindt = datetime.datetime.strptime(self.__lastTrainDate, "%Y%m%d")
               
                if (curdt - lasttraindt).days >= self.__training_interval_days:
                    train = True
          
        else:
            self.__lastTrainDate = curDate
        if not train:
            return False
       
      
        featureArray = numpy.array(self._features)
        targetArray = numpy.array(self._targets)#feature winsorize
     
        responseArray = winsorize(targetArray, limits = (self._winsorizePecentile, self._winsorizePecentile))
    
        responseArray = responseArray
     
        rangeMask = numpy.zeros(len(self._targets), dtype = numpy.bool_)
      
       
        curYear = int(curDate[:4])
        trainStartYear = curYear - 9
        trainStartDate = str(trainStartYear) + "0101"
        
        labeldates = []
        for l, label in enumerate(self._labels):
            ticker, intv = label.split("_")
            labeldates.append(intv[:8]) 
        labeldates = numpy.array(labeldates)
       
        rangeMask =  (labeldates > trainStartDate)
        self.__trainx, self.__testx, self.__trainy, self.__testy = train_test_split(featureArray[rangeMask, :], responseArray[rangeMask], test_size = 0.05, random_state = self.__random_state)
        # if curDate > "20240601":
        #     dumpdatatocsv(featureArray, responseArray, self._labels,  self._featureNames, "20240501", curDate, "traindata_" + curDate + ".csv")
      
        self.__lastTrainDate = curDate
      
        return True

    def getmodelspecificstate(self):
        self._regressor.save_model("bst.json")
        mss = {"xgboostlearner.features": self._features,
               "xgboostlearner.targetrs": self._targets,
               "xgboostlearner.labels": self._labels,
               "xgboostlearner.lastTrainDate": self.__lastTrainDate
               }
        return mss 

    def setmodelspecificstate(self, mss, pathtojsonfile):
        self._features = mss["xgboostlearner.features"]
        self._targets = mss["xgboostlearner.targets"]
        self._labels = mss["xgboostlearner.labels"]
        self.__lastTrainDate = mss["xgboostlearner.lastTrainDate"]
        self._regressor = xgboost.Booster(self.__param)
        self._regressor.load_model(pathtojsonfile)
        
    def train(self):
     
        dtrain = xgboost.DMatrix(self.__trainx, label = self.__trainy, missing = numpy.nan, feature_names = self._featureNames)
        dtest = xgboost.DMatrix(self.__testx, label = self.__testy, missing = numpy.nan, feature_names = self._featureNames)
        
        evallist = [(dtrain, "train"), (dtest, "test")]
        numRounds = 300
        
        
        early_stopping_rounds = 20
        self._regressor = xgboost.train(self.__param, dtrain, numRounds, evallist, \
                                        early_stopping_rounds = early_stopping_rounds, \
                                        feval = r2_score, maximize = True, verbose_eval = False)
        
            
    def predict(self, feature, label = None):
        if self._regressor is None:
            return numpy.nan
       
        featureArray = numpy.array([feature])
        dfeature = xgboost.DMatrix(featureArray, missing = numpy.nan, feature_names = self._featureNames)
    
        
        predict = self._regressor.predict(dfeature)[0]
        self.__label2predict[label] = predict
 
        predict = predict
        return predict
