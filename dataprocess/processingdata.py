import datetime
import pandas as pd
import talib
import yfinance as yf
from talib import abstract
import matplotlib.pyplot as plt
import numpy as np
import math #slope運用

class processdata():
    '''
    處理資料
    '''
    def __init__(self, dfall):
        self.dfall = dfall
        self.totalrow = len(dfall)

    def createIndicator(self):
        '''
        創建漲跌指標
        '''
        

        self.dfall = self.dfall.loc[self.dfall.index > '2006-10-09']
        self.dfall['MA3'] =self. dfall['adj close'].rolling(window = 3).mean()
        self.dfall['MA5'] = self.dfall['adj close'].rolling(window = 5).mean()
        self.dfall['MA5volume'] = self.dfall['volume'].rolling(window = 5).mean()
        self.dfall['MA20'] =self.dfall['adj close'].rolling(window = 20).mean()
        self.dfall['MA10'] =self.dfall['adj close'].rolling(window = 10).mean()
        self.dfall["OBV"] = talib.OBV(self.dfall['adj close'].values,(self.dfall['volume'].values)/1000)
        self.dfall = self.dfall.dropna()
        self.dfall["x"]=(self.dfall['volume']-self.dfall['MA5volume'])
        self.dfall["volume_chage"]=self.dfall["x"]/self.dfall['MA5volume']
        # self.dfall['MA60'] = self.dfall['close'].rolling(window = 60).mean()
        # self.dfall['MA120'] = self.dfall['close'].rolling(window = 120).mean()
        # self.dfall['MA240'] = self.dfall['close'].rolling(window = 240).mean()
        self.dfall['macd'], self.dfall['macdsignal'], self.dfall['macdhist'] = talib.MACD(self.dfall.close, fastperiod = 12, slowperiod = 26, signalperiod = 9)
        # self.dfall['rsi6'] = talib.RSI(self.dfall.close, timeperiod = 6)
        # self.dfall['rsi14'] = talib.RSI(self.dfall.close, timeperiod = 14)
        self.dfall['k'], self.dfall['d'] = talib.STOCH(self.dfall['high'], self.dfall['low'], self.dfall['adj close'], fastk_period=9,slowk_period=3,slowk_matype=1,slowd_period=3,slowd_matype=1)
        # self.dfall['sma20']= talib.SMA(self.dfall.close, timeperiod=20)
        # self.dfall['sma60']= talib.SMA(self.dfall.close, timeperiod=60)
        # self.dfall['sma120']= talib.SMA(self.dfall.close, timeperiod=120)
        # self.dfall['sma6_volume'] = talib.SMA(self.dfall.volume, timeperiod=6)
        self.dfall['upperband'],self.dfall['middleban'],self.dfall['lowerband']= talib.BBANDS(self.dfall.close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        column_to_drop = 'OBV'
        # 使用 drop 方法刪除特定列
        self.dfall = self.dfall.drop(columns=[column_to_drop])
        column_to_drop = 'x'

        # 使用 drop 方法刪除特定列
        self.dfall = self.dfall.drop(columns=[column_to_drop])


    def distance(self,indexA,indexB):
        '''
        計算兩條線的距離
        '''
        self.dfall[indexA+'-'+indexB+' '+'distance']=self.dfall[indexA] - self.dfall[indexB]


    #特定指標

    def MAdistance(self):
        '''
        計算個數值間的距離
        '''
        # self.distance('MA240','MA120')
        # self.distance('MA240','MA60')
        # self.distance('MA240','MA20')
        # self.distance('MA240','MA5')
        # self.distance('MA120','MA60')
        # self.distance('MA120','MA20')
        # self.distance('MA120','MA5')
        # self.distance('MA60','MA20')
        # self.distance('MA60','MA5')
        self.distance('adj close','MA5')
        self.distance('adj close','MA10')
        self.distance('adj close','MA20')
        self.distance('adj close','MA3')

        self.distance('adj close','upperband')
        self.distance('adj close','middleban')
        self.distance('adj close','lowerband')

        self.distance('MA20','MA10')
        self.distance('MA20','MA5')
        self.distance('MA10','MA5')
        self.distance('MA10','MA3')
        self.distance('MA20','MA3')
        self.distance('MA5','MA3')
        self.distance('macd','macdsignal')
        # self.distance('rsi6','rsi14')
        self.distance('k','d')


    
        # # 計算價格百分比變化
        # self.dfall['PriceChange'] = self.dfall['adj close'].pct_change() * 100

        # # 將 Label 欄位初始化為 0
        # self.dfall['Label'] = 0

        # # 將符合條件的資料標記為 1
        # self.dfall.loc[self.dfall['PriceChange'] > 3.5, 'Label'] = 1

        # # 將 Label 欄位往後移一天
        # self.dfall['Label'] = self.dfall['Label'].shift(1)

        # # 刪除最後一天的資料（因為無法確定明天的漲跌）
        # self.dfall = self.dfall[:-1]

        # 印出結果
        # self.dfall.to_csv('label.csv', encoding = 'cp950')
    # end compareStockPrice        

    def upfractalbreakthrough(self):
        '''
        上碎形突破
        '''
        self.distance('adj close','MA240')
        self.distance('adj close','MA120')
        self.distance('adj close','MA60')
        self.distance('adj close','MA20')
        self.distance('adj close','MA5')

        #上碎形突破 == 突破前高
        #将这两者相减并除以平均交易量，再乘以100，就得到了交易量的变化百分比。
        #以每日看
        #volume代表当前的交易量，sma6Volume代表过去6个周期的平均交易量，即6周期移动平均线。
        #計算碎形變化量
        self.dfall['fractal_volume_change'] = ( self.dfall['volume']- self.dfall['sma6_volume']) / self.dfall['sma6_volume'] * 100
        #上碎形可以定义为当前周期的最高价超过了前两个周期的最高价，并且当前周期的最高价也是后面两个周期中的最高价之一，
        #同时前两个周期的最高价都小于当前周期的最高价。另外，还需要满足交易量变化率大于 5。
        #确定上碎形条件
        #high[3] > high[4] and high[4] > high[5]：判断最近三根蜡烛中的最高价是否形成了上升趋势，如果是，则该条件成立；
        #high[2] < high[3] and high[1] < high[2]：判断最近三根蜡烛中的前两根是否出现了下降趋势，如果是，则该条件成立；
        #fractalVolumeChange[3] > 5：判断最近四根蜡烛中的第四根蜡烛的交易量变化是否大于 5%，如果是，则该条件成立。
        self.dfall['fractals_up']= ((self.dfall['high'].shift(4) < self.dfall['high'].shift(3)) & (self.dfall['high'].shift(5) < self.dfall['high'].shift(4))
                            & (self.dfall['high'].shift(2) < self.dfall['high'].shift(3)) & (self.dfall['high'].shift(1) < self.dfall['high'].shift(2)) & (self.dfall['fractal_volume_change'].shift(3) > 5))

        self.dfall = self.dfall.replace({True: 1, False: 0})
        self.dfall = self.dfall.drop(labels=['sma6_volume'],axis=1)
        self.dfall = self.dfall.drop(labels=['fractal_volume_change'],axis=1)
        self.dfall.drop(self.dfall.head(240).index,inplace=True) 

    def fluidmoving(self):
        '''
        流體均線
        '''
                #sma均線產生函式        
        def sma (index):
            self.dfall[f'sma{str(index)}']= talib.SMA(self.dfall.close, timeperiod=index)

        for i in range(1,11):
            sma(i*12)

        ############################################################################
        ##########################下列為 (斜率角度與平均值) 實作#####################

        # slopeA2 = slope(sma24)
        # slopeA2 = slope(sma24)
        # slopeA3 = slope(sma36)
        # slopeA4 = slope(sma48)
        # slopeB1 = slope(sma84)
        # slopeB2 = slope(sma96)
        # slopeB3 = slope(sma108)
        # slopeB4 = slope(sma120)
        def slope_cal(_src):
            rad2Degree = 2/3.1416  #pi
            angle = rad2Degree * math.atan((math.log(_src[0]) - math.log(_src[20]))/math.log(20)) * 100
            return angle


        def slope(slope_name,index,sma_name):
            self.dfall[slope_name] = None
            list = []
            for i in range (index,len(self.dfall)-20):
                list = []
                for x in range(21):
                    list.append(self.dfall[sma_name].iloc[i + x])

                self.dfall[slope_name].iloc[i+20] = slope_cal( list )


        slopeA = []
        slopeB = []
        for i in range(1,5):
            avg = 12*i
            slope('slope' + str(avg), avg, 'sma' + str(avg))

        for i in range(1,5):
            avg = 12*i+72
            slope('slope' + str(avg), avg, 'sma' + str(avg))


        ##############################
        # avgSlopeA = (slopeA1 + slopeA2 + slopeA3 + slopeA4) / 4
        # avgSlopeB = (slopeB1 + slopeB2 + slopeB3 + slopeB4) / 4

        def avgSlope(slope_name,begin):
            for index, row in self.dfall.iterrows():
                if(row['slope'+str(begin+36)]!=None):
                    sum_slope = row['slope'+str(begin)] + row['slope'+str(begin+12)] + row['slope'+str(begin+24)] + row['slope'+str(begin+36)]
                    self.dfall.loc[index,slope_name] = sum_slope/4


        avgSlope('avgSlopeA',12)
        avgSlope('avgSlopeB',84)


        #################斜率角度差#################
        # hisSlope = (avgSlopeA - avgSlopeB)
        def hisSlope():
            for index, row in self.dfall.iterrows():
                if(row['avgSlopeB']!=None):
                    self.dfall.loc[index,'hisSlope'] = row['avgSlopeA'] - row['avgSlopeB']

        hisSlope()
            
        ############################################################################
        ##########################以下為 (均線壓力) 實作#############################

        def res(ABC,number,begin):
            for index, row in self.dfall.iterrows():
                if(row['sma' + str(begin+12)] != None):
                    self.dfall.loc[index,'res' + str(ABC) + str(number)] = abs(row['sma' + str(begin)] - row['sma' + str(begin+12)])  #abs() : 絕對值

        for i in range(1,4):
            res('A',i,i*12)

        for i in range(7,10):
            res('B',i-6,i*12)

        def resAvg(ABC,number):
            for index, row in self.dfall.iterrows():
                if(row['res' + str(ABC) + str(number+2)] != None):
                    res_sum = row['res' + str(ABC) + str(number)] + row['res' + str(ABC) + str(number+1)] + row['res' + str(ABC) + str(number+2)]
                    self.dfall.loc[index,'res' + str(ABC)] = res_sum/3

        resAvg('A',1)
        resAvg('B',1)

        for index, row in self.dfall.iterrows():
                if(row['resA']!=None):
                    self.dfall.loc[index,'resistance'] = row['resA'] - row['resB']


        ############################################################################
        ##########################以下為 (趨勢反轉訊號(trend_reversal_signal)) 實作#########################
        # signal = crossover(avgSlopeA,avgSlopeB) and avgSlopeA < 0 and avgSlopeB < 0 and resistance < 5

        # def crossover(A,B):
            
        for index, row in self.dfall.iterrows():
                def crossover(A,B):
                    row['avgSlopeA'] > row['avgSlopeB']
                if(row['avgSlopeA'] > row['avgSlopeB'] and row['avgSlopeA'] < 0 and row['avgSlopeA'] < 0 and row['resistance'] < 5):
                    self.dfall.loc[index,'signal'] = 1
                else:
                    self.dfall.loc[index,'signal'] = 0
        
    

    def smadeal(self):
        '''
        判斷sma多頭或空頭
        '''
        for index, row in self.dfall.iterrows():
            if row['sma20']> row['sma60'] and row['sma60'] > row['sma120']:
                self.dfall.loc[index,'bullish']=1
            elif row['sma20']< row['sma60'] and row['sma60'] < row['sma120']:
                self.dfall.loc[index,'bullish']=-1
            else:
                self.dfall.loc[index,'bullish']=0
            

    def getdfall(self):
        '''
        回傳dfall
        '''
        return self.dfall


