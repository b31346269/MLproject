import datetime
import pandas as pd
import talib
import yfinance as yf

class reptile :
    '''
    爬蟲
    '''
    def __init__(self):
        self.dfall = pd.DataFrame()
        self.totalrow = 0

    def changeFieldName(self):
        '''
        增加欄位
        '''
        self.dfall = self.dfall.assign(Label=0)
        self.dfall = self.dfall.assign(weekend=0)
        
    # end changeFieldName

   

    def _getWeekNumberOrYear(self, dateStr, wOrY = True):
        '''
        取得週數或年分
        getWeekNumberOrYear(dateStr, True: week / False: year)
        '''
        dateObj = datetime.datetime.strptime(dateStr, '%Y-%m-%d').date()
        if wOrY:
            return dateObj.isocalendar()[1]
        # end if
        else:
            return dateObj.year
        # end else
    # end _getWeekNumberOrYear

    def getLastDayOfWeek(self, dayNumber, lastWeekNumber):
        '''
        標出每週的最後一天
        '''
        self.totalrow = len(self.dfall)
        for i in range (dayNumber, self.totalrow):
            date_str = str(self.dfall.index[i]).split()[0]
            week_number = self._getWeekNumberOrYear(date_str)
            
            if(week_number == lastWeekNumber):  #同一週的
                self.dfall['weekend'].iloc[i] = 0
            # end if
            elif (week_number != lastWeekNumber):  #如果跟上週不一樣就是上一個就是最後一個
                
                self.dfall['weekend'].iloc[i - 1] = 1
                self.dfall['weekend'].iloc[i] = 0
            # end elif
            else:
                self.dfall['weekend'].iloc[i] = 0
            # end else  
            lastWeekNumber = week_number 
        # end for
            
    # end getLastDayOfWeek

    def prepareInitValue(self):
        '''
        日期初始值準備
        '''
        self.dfall['weekend'].iloc[4] = 1
        self.dfall['weekend'].iloc[9] = 1
        for i in range(0, 14):
            if(self.dfall['weekend'].iloc[i] != 1):
                self.dfall['weekend'].iloc[i] = 0
            # end if
        # end for
        for i in range(0, 14):
            self.dfall['Label'].iloc[i] = 2
        # end for
        return self.dfall
    # end prepareInitValue

    def stockDownload(self, codeName, startDate, dataPeriod = 'max'):
        '''
        下載股票資料
        stockDownload(股票代號, 開始日期, 結束日期)
        '''
        self.dfall = yf.download(codeName, start = startDate, period = dataPeriod).astype('float')
        self.dfall = self.dfall.rename(str.lower, axis = 'columns')
        dfbase = self.dfall[['open', 'high', 'low', 'close', 'volume']]
        ta_list = talib.get_functions()
        for x in ta_list:
            try:
                output = eval(f'abstract.{x}(self.dfall)')
                output.name = x.lower() if type(output) == pd.core.series.Series else None
                self.dfall = pd.merge(self.dfall, pd.DataFrame(output), left_on = self.dfall.index, right_on = output.index)
                self.dfall = self.dfall.set_index('key_0')
            # end try
            except:
                pass
            # end except
        # end for
        self.dfall = self.dfall.loc[:, self.dfall.columns.isin(['sma', 'ema' ,'sar', 'open', 'high', 'low', 'close', 'volume', 'trix', 'acos', 'slowd', 'slowk', 'MACDsignal', 'MACDhist', 'display.max_rows', 'k', 'd', 'cci', 'obv', 'mfi', 'adj close'])]
        pd.set_option('display.max_rows', None)
        return self.dfall, dfbase, ta_list
    # end stockDownload

    def dropna(self):
        '''
        刪除缺失值
        ''' 
        self.dfall.dropna(axis=0, how='all', inplace=True)
        self.totalrow = len(self.dfall)

    def dropnomarket(self, dayNumber):
        '''
        刪除沒有交易量的
        ''' 
        list1 = []
        for i in range (0,self.totalrow):
            if self.dfall.iloc[i]['volume'] == 0:
                list1.append(i)
        self.dfall = self.dfall.drop(self.dfall.index[list1],axis=0)

               
                
    def getdfall(self):
        '''
        取得dfall
        ''' 
        return self.dfall

    