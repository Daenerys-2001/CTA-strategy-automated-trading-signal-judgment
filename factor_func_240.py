
import numpy
import waves
import model 
import mlengine
import ema
import os 
import pickle
import numpy as np
class WavesXGBModel(model.Model):
    def __init__(self, strategy, universe, mss, ticker2contract):
        model.Model.__init__(self, strategy, universe, mss)
        self._mss = mss 
        self._ticker2waves = None 
        self._xgboostLearner = None 
        self.__HALFLIVES = [10, 20, 30, 60, 120]
        self._xgboostlearnermss = None
        self._ticker2wavesmss = None
        self.__bondset = set(["t", "tf", "ts"])
        
        if self._mss is not None:
            self._ticker2wavesmss = self._mss["ticker2wavesmss"] 
            self._xgboostlearnermss = self._mss["xgboostmodelmss"]
            ticker2emamss = self._mss["ticker2emamss"]
            self._dataUniverse = self._mss["datauniverse"]
            self.__msslastdate = self._mss["msslastdate"]
            
            self.__ticker2varema = {ticker: ema.EMA(self.__HALFLIVES) for ticker in self._dataUniverse}    
            for ticker in self.__ticker2varema:
                ticker_emaparameters = ticker2emamss[ticker]
                self.__ticker2varema[ticker].setParameters(*ticker_emaparameters)
                
            self._ticker2feature = self._mss["ticker2feature"]
        
            self.__ticker2intv2forecast = self._mss["ticker2intv2forecast"]
            self._currentprediction = self._mss["currentprediction"]
           
            self.__cumlagret = self._mss["cumlagret"]
            self._ticker2vol20s = self._mss["ticker2vol20s"]
            self._ticker2totalopis  = self._mss["ticker2totalopis"]
            self._onmarketdates = self._mss["onmarketdata"]
        else:
            self.__msslastdate = "20100101"
            self._dataUniverse = universe 
            self.__ticker2varema = {ticker: ema.EMA(self.__HALFLIVES) for ticker in self._dataUniverse} 
            self._ticker2feature = {}
            self._xgboostLearner = None 
            self._ticker2waves = None
            self._currentprediction = numpy.empty(len(self._dataUniverse))
            self._currentprediction[:] = numpy.nan
            self.__cumlagret = numpy.zeros(len(self._dataUniverse))
            self._ticker2vol20s = {ticker: numpy.array([]) for ticker in self._dataUniverse}
            self._ticker2totalopis = {ticker: numpy.array([]) for ticker in self._dataUniverse}
            self.__ticker2intv2forecast = {ticker: {} for ticker in self._dataUniverse}
            self._onmarketdates = numpy.zeros(len(self._dataUniverse), dtype = numpy.int64)
            self._onmarketdates[:] = 100 
        
        self._trainMinSize = 3000
        self._winsorizePercentile = None
        self._demeanFactor = None
     
        self._h = 0
        self._span = 0

        self._ticker2index = {ticker: t for t, ticker in enumerate(self._dataUniverse)}
     
        self._normResponse = False
        self._featureNames = None 
        self.__lastclose = numpy.empty(len(self._dataUniverse), dtype = numpy.float64)
        self.__lastclose[:] = numpy.nan
        
        self._vol10 = numpy.zeros(len(self._dataUniverse), dtype = numpy.float64)
        self._vol20 = numpy.zeros(len(self._dataUniverse), dtype = numpy.float64)
        self._vol30 = numpy.zeros(len(self._dataUniverse), dtype = numpy.float64)
        self._vol120 = numpy.zeros(len(self._dataUniverse), dtype = numpy.float64)
        self._updateintvduration = 30 
        self.__lastdateclose = numpy.empty(len(self._dataUniverse), dtype = numpy.float64)
        self.__lastdateclose[:] = numpy.nan 
     
           
        self.__ticker2contract = ticker2contract
        self.__contract2ticker = {ticker2contract[ticker]: ticker for ticker in ticker2contract}
        
    def getmodelspecificstate(self):
        wavesmss = {ticker: self._ticker2waves[ticker].getmodelspecificstate() for ticker in self._ticker2waves}
        emamss = {ticker: self.__ticker2varema[ticker].getParameters() for ticker in self.__ticker2varema}
        mlenginemss = {} 
        mlenginemss["xgboostmodelmss"] = self._xgboostLearner.getmodelspecificstate()
        mss = {"ticker2wavesmss": wavesmss,
               "ticker2emamss": emamss,
               "currentprediction": self._currentprediction,
               "onmarketdata": self._onmarketdates,
               "ticker2feature": self._ticker2feature,
               "ticker2vol20s": self._ticker2vol20s,
               "ticker2totalopis": self._ticker2totalopis,
               "cumlagret": self.__cumlagret,
               "datauniverse": self._dataUniverse,
               "msslastdate": self._currentIntv[:8],
               "ticker2intv2forecast": self.__ticker2intv2forecast
               }

        mss.update(mlenginemss)
        return mss
    
    
    def initForNewDay(self, curdate):
        model.Model.initForNewDay(self, curdate)
      

        train = self._xgboostLearner.prepareArray(curdate)
        if train:
            self._xgboostLearner.train()
                            
        dailylogret = numpy.log(self.__lastclose / self.__lastdateclose)
        dailylogret = numpy.round(dailylogret, 8)
        self.__lastdateclose = self.__lastclose.copy()
        
        if curdate > self.__msslastdate:
            for t, ticker in enumerate(self._dataUniverse):
                tickerlogret = dailylogret[t]
                if numpy.isnan(tickerlogret):
                    continue
                self.__ticker2varema[ticker].add(tickerlogret, 1)
                
            for t, ticker in enumerate(self._dataUniverse):
                vars = self.__ticker2varema[ticker].getVar()
                self._vol10[t] = numpy.sqrt(260 * vars[0])
                self._vol20[t] = numpy.sqrt(260 * vars[1])
            
                self._ticker2vol20s[ticker] = numpy.append(self._ticker2vol20s[ticker], self._vol20[t])
                self._vol30[t] = numpy.sqrt(260 * vars[2])
                self._vol120[t] = numpy.sqrt(260 * vars[4])
                if numpy.isnan(self.__lastclose[t]):
                    self._onmarketdates[t] = 0
                else:
                    self._onmarketdates[t] = self._onmarketdates[t] + 1
      
        
    def initForNewInterval(self, current_tickerdatetime):
        
        model.Model.initForNewInterval(self, current_tickerdatetime)

        posdict_15m = {self.__contract2ticker[contract] : self._strategy.target_am15_pos_dict[contract] for  contract in self._strategy.target_list}
      
        # print(f"[DEBUG] initForNewInterval at {current_tickerdatetime}")
        # print(f"[DEBUG] posdict_15m: {posdict_15m}")

        if current_tickerdatetime.hour == 9 and current_tickerdatetime.minute == 00:
            posdict_daily = {self.__contract2ticker[contract] : self._strategy.target_am1d_pos_dict[contract] for  contract in self._strategy.target_list}
            for t, ticker in enumerate(self._dataUniverse):
                if ticker not in posdict_daily:
                    continue
                pos_daily = posdict_daily[ticker]
                if pos_daily < 0:
                    continue
                contract = self.__ticker2contract[ticker]
          
                if self._currentIntv[:8] > self.__msslastdate:
                    lastday_totalopi = self._strategy.target_am1d_dict[contract].idx_open_interest[pos_daily]
                    totalopis = self._ticker2totalopis[ticker]
                    totalopis = numpy.append(totalopis, lastday_totalopi)
                    self._ticker2totalopis[ticker] = totalopis
             
           
        for t, ticker in enumerate(self._dataUniverse):
            # print(f"\n[DEBUG] ticker={ticker}, index={t}")

            if ticker not in posdict_15m:
                print("  -> ticker not in posdict_15m, continue")
                continue
            contract = self.__ticker2contract[ticker]
            pos_15m = posdict_15m[ticker]
            print(f"  -> pos_15m={pos_15m}")

            if pos_15m < 0:
                print("  -> pos_15m < 0, continue")
                continue

            md = self._strategy.target_am15_dict[contract]
            if pos_15m >= len(md.close):
                print(f"[WARN] {ticker} pos_15m={pos_15m} exceeds close size {len(md.close)}")
                continue
            closeprice = md.close[pos_15m]

            ticker_intvtime = md.datetime_list[pos_15m]
            print(f"  -> ticker_intvtime={ticker_intvtime}")

            if not self._strategy.new_data_15m_dict[self.__ticker2contract[ticker]]:
                print("  -> new_data_15m_dict is False, continue")
                continue

            minutes = ticker_intvtime.minute
            print(f"  -> minutes={minutes}, updateintvduration={self._updateintvduration}")
            if minutes % self._updateintvduration != 0:
                print("  -> minute not on interval boundary, continue")
                continue

            closeprice = md.close[pos_15m]
            print(f"  -> closeprice={closeprice}, lastclose={self.__lastclose[t]}")
            if not numpy.isnan(self.__lastclose[t]):
                logret = numpy.round(numpy.log(closeprice / self.__lastclose[t]), 8)
                print(f"    -> logret={logret}")
                if self._currentIntv[:8] > self.__msslastdate:
                    self.__cumlagret[t] += logret
                    print(f"    -> cumlagret updated to {self.__cumlagret[t]}")
                    update = True
                else:
                    update = False
            else:
                update = False

            self.__lastclose[t] = closeprice

            print(f"  -> update flag={update}")
            if not update:
                print("  -> not update, continue")
                continue

            print(f"  -> vol120={self._vol120[t]}, onmarketdates={self._onmarketdates[t]}")
            if self._vol120[t] == 0 or numpy.isnan(self._vol120[t]):
                print("  -> bad vol120, continue")
                continue
            if self._onmarketdates[t] < 1:
                print("  -> onmarketdates < 1, continue")
                continue

            tsitsi = self._ticker2waves[ticker]
            print("  -> passing all filters, calling Waves.update/checkNewExtremum")
            tsitsi.update(self.__cumlagret[t], None, None)
            new = tsitsi.checkNewExtremum()
            print(f"  -> checkNewExtremum returned {new}")
           
            if new:
                if ticker in self._ticker2feature:
                    response = tsitsi.getSampleRet()
                    
                    feature = self._ticker2feature[ticker]
                    if feature is not None:
                        prediction = self._currentprediction[t]
                   
                        intv = self._currentIntv
                        label = ticker + "_" + intv
                    
                        self._xgboostLearner.update(feature, response, prediction, label)
          
                    
                feature = self._getTickerFeature(ticker)
               
                self._ticker2feature[ticker] = feature
                feature = self._getTickerFeature(ticker)
                # print(f"[DEBUG] {ticker} feature:", feature)
                if feature is not None:
                    pred = self._xgboostLearner.predict(feature)
                    # print(f"[DEBUG] {ticker} raw predict:", pred)
                    self._currentprediction[t] = pred
                else:
                    # print(f"[DEBUG] {ticker} feature is None, setting prediction to 0")
                    self._currentprediction[t] = 0


                tsitsi.setSampleIndex()

   
    def getTickerForecast(self, ticker): 
        signal = 0
        if ticker not in self._ticker2waves:
            signal = 0
        else:      
            if self._currentIntv[:8] <= self.__msslastdate:
                if ticker in self.__ticker2intv2forecast:
                    intv2forecast = self.__ticker2intv2forecast[ticker]
                    if self._currentIntv in intv2forecast:
                        signal  = intv2forecast[self._currentIntv]
                    else:
                        signal = 0
                else:
                    signal = 0
            else: 
              
                tsitis = self._ticker2waves[ticker]
                forecast = 0
                t = self._ticker2index[ticker]
               
                if ticker in self._ticker2feature:
                    risk = self._currentprediction[t]
                    if risk > 0:
                        forecast = -risk * tsitis.getLastStates(1)[-1]
                    elif risk < 0:
                        forecast = -0.4 * risk * tsitis.getLastStates(1)[-1]
                 
                    if self._currentIntv[:4] > "2020":
                        forecast = forecast * 1.6
                        
                    vol = self._vol10[t]
                    if vol == 0 or np.isnan(vol):
                        return 0  # 🚫 防止除以 0 或 NaN
                
                    if ticker in self.__bondset:
                        signal = forecast * 0.01 / self._vol10[t]
                    else:
                        signal = forecast * 0.1 / self._vol10[t] 
                    
                    # ✅ Optional log：讓你看每次預測細節
                    print(f"[{ticker}] prediction: {risk:.4f}, forecast: {forecast:.4f}, vol10: {vol:.4f}, signal: {signal:.4f}")
  
  
        return signal
     

class Uranus45XGBModel(WavesXGBModel):
    def __init__(self, strategy, universe, mss, ticker2contract, jsonpath):
        WavesXGBModel.__init__(self,  strategy, universe, mss, ticker2contract)
        self._h = 4.5
        self._span = 300
        self._winsorizePercentile = 1e-4
        self._normResponse = True
        self._demeanFactor = 1
        self._trainMinSize = 2000
        
        self._featureNames = ["f1", "f2", "f3", "f4", "f5", "f6", "meanf4",
                               "std4", "sr1", "adx40",
                               "std6", "std5", "adxratio", "atrRatio", " volratio", 
                               "cross_volratio_mean_long", 
                              "cross_volratio_std_long", 
                              "opiratio", "vol20zscore"
                              ]
     
        self._updateintvduration = 30
        self.__volratiorank = numpy.empty(len(self._dataUniverse))
        self.__volratiorank[:] = numpy.nan 
        self.__volratioranklong = numpy.empty(len(self._dataUniverse))
        self.__volratioranklong[:] = numpy.nan 
      
        self.__cross_volratio_mean_long = numpy.nan 
        self.__cross_volratio_std_long = numpy.nan 
        if self._xgboostLearner is None:
            self._xgboostLearner = mlengine.XgboostLearner(self._trainMinSize, 
                                                               self._winsorizePercentile,
                                                               demeanFactor = self._demeanFactor,
                                                                featureNames = self._featureNames,
                                                                training_interval_days = 91)
            if self._xgboostlearnermss is not None:
                self._xgboostLearner.setmodelspecificstate(self._xgboostlearnermss, jsonpath)
                
        if self._ticker2waves is None:
            self._ticker2waves = {ticker: waves.Waves(self._h, self._span, ticker) 
                                for ticker in self._dataUniverse}
            print(f"初始化waves对象，universe大小: {len(self._dataUniverse)}")
            
            if self._ticker2wavesmss is not None:
                for ticker in self._ticker2wavesmss:
                    ticker_wavesmss = self._ticker2wavesmss[ticker]
                    self._ticker2waves[ticker].setmodelspecificstate(ticker_wavesmss)
                    
    def initForNewDay(self, curdate):
        WavesXGBModel.initForNewDay(self, curdate) 
        valid = (self._vol20 != 0)
        volratio = numpy.full_like(self._vol20, numpy.nan)
        volratio[valid] = self._vol20[valid] / self._vol30[valid]
        volratio[~valid] = numpy.nan
        
        valid = (self._vol120 != 0)
        volratiolong = numpy.full_like(self._vol20, numpy.nan)
        volratiolong[valid] = self._vol20[valid] / self._vol120[valid]
        volratiolong[~valid] = numpy.nan
      
        valid = numpy.isfinite(volratiolong)
        validsum = numpy.sum(valid)
        
        if validsum > 15:
            self.__cross_volratio_mean_long = numpy.nanmean(volratiolong)
            self.__cross_volratio_std_long = numpy.nanstd(volratiolong)
        else:
            self.__cross_volratio_mean_long = numpy.nan 
            self.__cross_volratio_std_long = numpy.nan 
    
    def _getTickerFeature(self, ticker):
        
        t = self._ticker2index[ticker]
     
        
        tsitsi = self._ticker2waves[ticker]
        waves = tsitsi.getWaves()
        if len(waves) < 5:
            return None
        f1 = abs(waves[-1])
        f2 = abs(waves[-2])
        f3 = abs(waves[-3])
        f4 = abs(waves[-4])
        f5 = abs(waves[-5])
        if len(waves) == 5:
            f6 = numpy.nan 
            meanf4 = numpy.nan
        else:
            f6 = abs(waves[-6])
            meanf4 = (f1 + f2 + f3 + f4) / 4.0
        SRs = tsitsi.getLastWaveSR(4)
       
        sr1 = abs(SRs[-1])
  
        adxRatio = tsitsi.getADXRatio(15, 70)
        atrRatio = tsitsi.getATRRatio(130, 250)
        adx40 = tsitsi.getADX(40)
       
      
        std4 = numpy.std(numpy.array([f1, f2, f3, f4]))
        std5 = numpy.std(numpy.array([f1, f2, f3, f4, f5]))
        std6 = numpy.std(numpy.array([f1, f2, f3, f4, f5, f6]))

        totalopis = self._ticker2totalopis[ticker]
        opiratio = numpy.mean(totalopis[-10:]) / numpy.mean(totalopis[-130:])
        vol20s = self._ticker2vol20s[ticker]
    
        if len(vol20s) < 30:
            volRatio = numpy.nan 
        else:
            volRatio =  self._vol20[t] / self._vol30[t]
            
        if len(vol20s) < 40:
            vol20zscore = numpy.nan 
      
        else:
            vol20zscore = (self._vol20[t] - numpy.mean(vol20s[-40:])) / numpy.std(vol20s[-40:])
         
      
        
        feature = numpy.array([f1, f2, f3, f4, f5,  f6, meanf4,
                               std4, sr1, adx40,
                               std6, std5, adxRatio,  atrRatio, 
                               volRatio, self.__cross_volratio_mean_long, 
                               self.__cross_volratio_std_long, 
                                opiratio, vol20zscore
                               ])
  
        return feature



# 因子函数
def factor_func240(strategy):
    ONLINE = True

    if 'factor_func240' not in strategy.global_dict:
        strategy.global_dict['factor_func240'] = {}
        contracts = strategy.target_list
        universe = []
        ticker2contract = {}
        for contract in contracts:
            ticker = strategy.target_symbol_name_dict.get(contract, f"ticker_{contract}")
            universe.append(ticker)
            ticker2contract[ticker] = contract

        strategis_path = os.path.abspath(os.path.dirname(__file__))
        model_dir = os.path.join(strategis_path, "xgb45model")
        
        # 确保模型目录存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"创建模型存储目录: {model_dir}")

        msspath = None
        jsonpath = None
        for filename in sorted(os.listdir(model_dir)):
            if filename.startswith("mss"):
                msspath = os.path.join(model_dir, filename)
            elif filename.startswith("bst"):
                jsonpath = os.path.join(model_dir, filename)

        # 加载模型状态（带异常处理）
        mss = None
        if msspath and os.path.exists(msspath):
            try:
                with open(msspath, "rb") as f:
                    mss = pickle.load(f)
                print(f"成功加载历史模型状态: {msspath}")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"加载模型状态失败: {e}, 初始化新模型")
                mss = None

        # 初始化模型
        try:
            model = Uranus45XGBModel(
                strategy, universe, mss, ticker2contract, jsonpath
            )
            strategy.global_dict['factor_func240']["model"] = model
            strategy.global_dict['factor_func240']["universe"] = universe
            strategy.global_dict['factor_func240']['prev_date'] = None
            print("XGBoost 模型初始化成功")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise

    # 获取模型引用
    model = strategy.global_dict['factor_func240'].get("model")
    if model is None:
        raise ValueError("模型未正确初始化")

    # 日期处理
    current_intvtime = None
    for contract in strategy.target_list:
        pos = strategy.target_am15_pos_dict.get(contract, -1)
        if pos == -1:
            continue
        intvtime = strategy.target_am15_dict[contract].datetime_list[pos]
        if current_intvtime is None or intvtime > current_intvtime:
            current_intvtime = intvtime

    if current_intvtime is None:
        return {}

    # 日期更新检查
    date = current_intvtime.date()
    datestr = date.strftime("%Y%m%d")
    prev_date = strategy.global_dict['factor_func240'].get('prev_date')

    if date != prev_date:
        print(f"新交易日初始化: {datestr}")
        try:
            model.initForNewDay(datestr)
            strategy.global_dict['factor_func240']['prev_date'] = date
        except Exception as e:
            print(f"交易日初始化失败: {e}")
            raise

    # 时间点更新
    print(f"处理时间点: {current_intvtime}")
    try:
        model.initForNewInterval(current_intvtime)
    except Exception as e:
        print(f"时间点初始化失败: {e}")
        raise

    # 生成信号
    factor_dict = {}
    for contract in strategy.target_list:
        ticker = strategy.target_symbol_name_dict.get(contract)
        if not ticker:
            continue
        try:
            signal = model.getTickerForecast(ticker)
            factor_dict[contract] = np.clip(signal, -1.0, 1.0)  # 限制信号范围
        except KeyError:
            print(f"找不到合约 {contract} 对应的ticker")
            factor_dict[contract] = 0.0

    # 定期保存模型状态
    if ONLINE and current_intvtime.hour == 22 and current_intvtime.minute == 45:
        print("保存模型状态...")
        try:
            mss = model.getmodelspecificstate()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            mss_file = os.path.join(model_dir, f"mss_{timestamp}.pkl")
            with open(mss_file, "wb") as f:
                pickle.dump(mss, f)
            print(f"模型状态保存至: {mss_file}")
        except Exception as e:
            print(f"保存模型状态失败: {e}")

    return factor_dict
    
import datetime
