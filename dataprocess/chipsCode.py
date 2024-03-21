import os                                                                        # class Backtest, Chart, Crawler, Process and Train use
from csv import writer                                                           # class Crawler use
from math import ceil, floor, sqrt                                               # class Backtest and Train use
from time import perf_counter                                                    # func timer use
from warnings import filterwarnings
import discord
from discord.ext import commands
import numpy as np                                                               # class Train use
import pandas as pd                                                              # class Process use
import seaborn as sns                                                        # class Chart use
from bs4 import BeautifulSoup                                                    # class Crawler use
from requests import get                                                         # class Crawler use

import matplotlib.pyplot as plt                                                  # class Backtest and Chart use
from IPython.display import display                                              # class Train use
from keras.callbacks import EarlyStopping, History                               # class Train use
from keras.layers import Dense, GRU, LSTM                                        # class Train use
from keras.models import Sequential, load_model                                  # class Train use
from keras.utils import np_utils, plot_model                                     # class Train use
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error  # class Train use
from sklearn.metrics import mean_squared_error, r2_score                         # class Train use
from sklearn.preprocessing import MinMaxScaler                                   # class Train use

import reptile as sa
import processingdata as po


class Backtest(object):
    '''
    回測用。
    Backtest(Files資料夾路徑, Images資料夾路徑)
    '''
    __money = 0
    __original = 0
    __rate = 0.0
    __shares_maximum = 0
    __money_list = []
    __shares_list = []

    def __init__(self, file_path: str, image_path: str):
        self.file_path = file_path
        self.image_path = image_path
        if not os.path.isdir(file_path):  # 如Files資料夾未建立則建立資料夾
            os.mkdir(file_path)
        # end if
        if not os.path.isdir(image_path):  # 如Images資料夾未建立則建立資料夾
            os.mkdir(image_path)
        # end if
    # end __init__

    def _process_price(self, price: float) -> float:
        '''
        處理股價。
        float <- _process_price(股價)
        '''
        if price < 10:
            price = round(price, 2)
        # end if
        elif price >= 10 and price < 50:
            price = round(price, 2)
            floor_temp = price - floor(price * 10) / 10
            decimal_temp = price - floor_temp
            if decimal_temp <= 0.03:
                price = floor_temp
            # end if
            elif decimal_temp > 0.03 and decimal_temp <= 0.06:
                price = floor_temp + 0.05
            # end elif
            else:
                price = floor_temp + 0.1
            # end else
        # end elif
        elif price >= 50 and price < 100:
            price = round(price, 1)
        # end elif
        elif price >= 100 and price < 500:
            floor_temp = floor(price)
            decimal_temp = price - floor_temp
            if decimal_temp <= 0.3:
                price = floor_temp
            # end if
            elif decimal_temp > 0.3 and decimal_temp <= 0.6:
                price = floor_temp + 0.5
            # end elif
            else:
                price = floor_temp + 1
            # end else
        # end elif
        elif price >= 500 and price < 1000:
            price = round(price)
        # end elif
        elif price >= 1000:
            price = round(price)
            digits_temp = price % 10
            if digits_temp <= 3:
                price -= digits_temp
            # end if
            elif decimal_temp > 3 and decimal_temp <= 6:
                price -= digits_temp
                price += 5
            # end elif
            else:
                price -= digits_temp
                price += 10
            # end else
        # end elif
        return price
    # end _process_price

    def set_money(self, value: int) -> None:
        '''
        設定初始持有金額。
        None <- set_money(金額)
        '''
        self.__original = value
        self.__money = self.__original
    # end set_money

    def set_rate(self, value: float) -> None:
        '''
        設定一次要使用多少比例的本金購買。
        None <- set_rate(比例)
        '''
        self.__rate = value
    # end set_rate

    def show_result(self, symbol: int) -> None:
        '''
        顯示回測後所持有金額。
        None <- show_result()
        '''
        roi = round((self.__money - self.__original) / self.__original * 100, 2)
        print(f'初始持有金額：{self.__original: ,} 元')
        print(f'回測後所持有：{self.__money: ,} 元')
        print(f'投資報酬率：{roi}%')
        folder_path = f'{self.file_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        save_path = f'{folder_path}/Result.txt'
        with open(save_path, 'a', encoding='utf-8-sig') as txt_file:  # 將回測結果寫入.txt檔
            txt_file.write(f'初始持有金額：{self.__original: ,} 元\n')
            txt_file.write(f'回測後所持有：{self.__money: ,} 元\n')
            txt_file.write(f'投資報酬率：{roi}%\n')
        # end with
    # end show_result

    def show_tendency(self, date: np.ndarray, symbol: int) -> None:
        '''
        顯示持有金額及股數走勢圖。
        None <- show_tendency(日期, 股票代號)
        '''
        print(u'生成回測走勢圖中……')
        plt.figure(figsize=(16, 8))
        plt.title(f'Backtesting Result ({symbol})')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Shares', fontsize=18)
        plt.bar(date, self.__shares_list, width=7.1, color='silver')
        plt.ylim(0, ceil(self.__shares_maximum * 1.1))
        plt.legend(['Shares'], loc='upper left')
        axis_two = plt.twinx()
        axis_two.set_ylabel('Money', fontsize=18)
        plt.plot(date, self.__money_list, color='gold')
        axis_two.set_ylim(0, ceil(self.__original * 1.1))
        plt.legend(['Money'], loc='upper right')
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        # end if
        plt.savefig(f'{folder_path}/Backtesting.png', dpi=300)
        plt.show()
        print(u'回測走勢圖生成完畢！')
    # end show_tendency

    def start_testing(self, ori_pri: np.ndarray, pre_pri: np.ndarray,
                      symbol: int) -> None:
        '''
        開始進行回測。
        None <- start_testing(原始股價, 預測股價)
        '''
        ori_pri, pre_pri = ori_pri.ravel(), pre_pri.ravel()  # 拉平資料
        transactions = 0  # 買賣次數
        shares = 0  # 持股數
        self.__shares_maximum = 0
        self.__money_list.clear()
        self.__shares_list.clear()
        pri_len = len(pre_pri)
        for i in range(pri_len):
            price = ori_pri[i]
            if i == pri_len - 1 and shares:  # 最後一週將持全數賣出
                total = floor(shares * price)
                charge = floor(total * 0.001425)  # 交易手續費
                service_charge = charge if charge >= 20 else 20  # 如果未滿20元則設為20
                taxes = floor(total * 0.003)  # 證交稅
                self.__money += total - service_charge - taxes
                shares = 0
                transactions += 1
            # end if
            elif i == pri_len - 1:  # 最後一週如果手上無持股則什麼都不做
                pass
            # end elif
            elif (i < pri_len - 3 and pre_pri[i + 3] > pre_pri[i + 2]
                                  and pre_pri[i + 2] > pre_pri[i + 1]
                                  and pre_pri[i + 1] > pre_pri[i]
                                  and pre_pri[i + 3] / pre_pri[i] >= 1.1):  # 如果預測連漲三週及漲幅大於10%則進行購買
                buy = floor(self.__money * self.__rate)  # 要花多少持有金額做購買
                self.__money -= buy
                shares = floor(buy / price)
                total = ceil(shares * price)
                charge = floor(total * 0.001425)  # 交易手續費
                service_charge = charge if charge >= 20 else 20  # 如果未滿20元則設為20
                while buy - total - service_charge < 0:  # 如果算上手續費還不足以購買則減少購買股數
                    shares -= 1
                    total = ceil(shares * price)
                    charge = floor(total * 0.001425)  # 交易手續費
                    service_charge = charge if charge >= 20 else 20  # 如果未滿20元則設為20
                # end while
                self.__money += buy - total - service_charge  # 加回買剩的金額
                transactions += 1
            # end elif
            elif (shares and i != pri_len - 2
                         and pre_pri[i + 2] < pre_pri[i + 1]
                         and pre_pri[i + 1] < pre_pri[i]
                         and pre_pri[i + 2] / pre_pri[i] <= 0.95):  # 如果預測連跌兩週及跌幅大於5%則賣掉全部持股
                total = floor(shares * price)
                charge = floor(total * 0.001425)  # 交易手續費
                service_charge = charge if charge >= 20 else 20  # 如果未滿20元則設為20
                taxes = floor(total * 0.003)  # 證交稅
                self.__money += total - service_charge - taxes
                shares = 0
                transactions += 1
            # end elif
            elif (shares and pre_pri[i + 1] < pre_pri[i]
                         and pre_pri[i + 1] / pre_pri[i] <= 0.975):  # 如果預測下週跌及跌幅大於2.5%則賣掉一半持股
                sells = shares >> 1  # 向右位移1-bit以做除2及無條件捨去
                shares -= sells
                total = floor(sells * price)
                charge = floor(total * 0.001425)  # 交易手續費
                service_charge = charge if charge >= 20 else 20  # 如果未滿20元則設為20
                taxes = floor(total * 0.003)  # 證交稅
                self.__money += total - service_charge - taxes
                transactions += 1
            # end elif
            if shares > self.__shares_maximum:  # 記錄歷史最高持股數以利畫圖
                self.__shares_maximum = shares
            # end if
            self.__money_list.append(self.__money)
            self.__shares_list.append(shares)
        # end for
        print(f'總共交易 {transactions} 次')
        folder_path = f'{self.file_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        save_path = f'{folder_path}/Result.txt'
        with open(save_path, mode='a', encoding='utf-8-sig') as txt_file:  # 將買賣次數寫入.txt檔
            txt_file.write(f'總共交易 {transactions} 次\n')
        # end with
    # end start_testing
# end class Backtesting

class Chart(object):
    '''
    繪製圖表用。
    Chart(Images資料夾路徑)
    '''
    def __init__(self, image_path: str):
        self.image_path = image_path
        if not os.path.isdir(image_path):  # 如Images資料夾未建立則建立資料夾
            os.mkdir(image_path)
        # end if
    # end __init__

    def heat_map(self, df: pd.core.frame.DataFrame, symbol: int) -> None:
        '''
        顯示熱力圖。
        None <- heat_map(DataFrame, 股票代號)
        '''
        print(u'生成熱力圖中……')
        data = df[['Number of Shareholders', 'Shares per Shareholders',
                   '>400 Holding Percentage', '>1000 Holding Percentage',
                   'Closing Price']].corr()
        sns.heatmap(data, cmap='coolwarm', annot=True, vmin=-1.0, vmax=1.0)
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        plt.savefig(f'{folder_path}/HeatMap.png', dpi=300, bbox_inches='tight')
        # plt.show()
        print(u'熱力圖生成完畢！')
    # end heat_map

    def line_graph(self, date: np.ndarray, price: np.ndarray, training: np.ndarray,
                   prediction: np.ndarray, symbol: int) -> None:
        '''
        將走勢資料以圖表呈現。
        None <- line_graph(日期, 原始股價, 訓練資料的預測價格, 測試資料的預測價格, 股票代號)
        '''
        print(u'生成資料走勢圖中……')
        total = len(price)
        start = total - len(prediction)
        plt.figure(figsize=(16, 8))
        plt.title(f'Model Prediction Result ({symbol})')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.plot(date, price, color='red')
        plt.plot(date[:start + 1], training, color='blue')
        plt.plot(date[start:], prediction, color='green')
        plt.legend(['Original Data', 'Predict on Training Data',
                    'Predict on Validation Data'], loc='lower right')
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        plt.savefig(f'{folder_path}/LineGraph.png', dpi=300)
        # plt.show()
        print(u'資料走勢圖生成完畢！')
    # end line_graph

    def pair_plot(self, df: pd.core.frame.DataFrame, symbol: int) -> None:
        '''
        顯示資料相關性分布圖。
        None <- pair_plot(DataFrame, 股票代號)
        '''
        print(u'生成資料相關性分布圖中……')
        data = df[['Number of Shareholders', 'Shares per Shareholders',
                   '>400 Holding Percentage', '>1000 Holding Percentage',
                   'Closing Price']]
        sns.pairplot(data, kind='reg')
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        plt.savefig(f'{folder_path}/PairPlot.png', dpi=300)
        # plt.show()
        print(u'資料相關性分布圖生成完畢！')
    # end pair_plot

    def train_loss(self, history: History, symbol: int) -> None:
        '''
        顯示Train時的Loss值。
        None <- train_loss(訓練歷史, 欄位名稱, 股票代號)
        '''
        print(u'生成Train時的Loss圖中……')
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(['Train'], loc='upper right')
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        plt.savefig(f'{folder_path}/TrainLoss.png', dpi=300)
        plt.show()
        print(u'Train時的Loss圖生成完畢！')
    # end train_loss

    def train_val_loss(self, history: History, symbol: int) -> None:
        '''
        顯示Train時的Val_loss值。
        None <- train_val_loss(訓練歷史, 欄位名稱, 股票代號)
        '''
        print(u'生成Train時的Val_loss圖中……')
        plt.plot(history.history['val_loss'])
        plt.title('Model Val_loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Val_loss', fontsize=14)
        plt.legend(['Train'], loc='upper right')
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        plt.savefig(f'{folder_path}/TrainValLoss.png', dpi=300)
        plt.show()
        print(u'Train時的Val_loss圖生成完畢！')
    # end train_val_loss
# end class Chart

class Crawler(object):
    '''
    抓取股權分散表用。
    Crawler(Files資料夾路徑)
    '''
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.isdir(file_path):  # 如Files資料夾未建立則建立資料夾
            os.mkdir(file_path)
        # end if
    # end __init__

    def stock_crawler(self, symbol: int) -> None:
        '''
        抓取神秘金字塔上的股權分散表。
        None <- stock_crawler(股票代號)
        '''
        print(u'抓取股權分散表中……')
        url = f'https://norway.twsthr.info/StockHolders.aspx?stock={symbol}'
        try:
            response = get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table_data = soup.find('table', {'id': 'Details'})  # 找到id為Deatils的table節點
            tr_data = table_data.select('tr')  # 選取tr子節點
        # end try
        except:
            raise CustomException(f'股票代號 {symbol} 不存在！')  # 丟出例外
        # end except
        else:
            folder_path = f'{self.file_path}/{symbol}'
            if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
                os.mkdir(folder_path)
            # end if
            save_path = f'{folder_path}/Table.csv'
            with open(save_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
                csv_writer = writer(csv_file)
                for tr in tr_data:
                    td_data = [td.text.replace('\xa0', '') for td in tr]  # .replace('\xa0', '')為去除&nbsp;
                    if td_data:  # 排除空list
                        csv_writer.writerow(td_data[2:15])
                    # end if
                # end for
            # end with
            print(u'股權分散表抓取完畢！')
        # end else
    # end stock_crawler
# end class Crawler

class Process(object):
    '''
    處理資料用。
    Process(Files資料夾路徑)
    '''
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.isdir(file_path):  # 如Files資料夾未建立則建立資料夾
            os.mkdir(file_path)
        # end if
    # end __init__

    def _insert_column(column: str, name: str):
        '''
        將股權分散表新增欄位用的裝飾器。
        @_insert_column(要插在哪個欄位後, 要插入的欄位名稱)
        def func_name(...):
            ...
        '''
        def decorator(func):
            '''
            接取傳入的Function。
            '''
            def wrapper(*args):
                '''
                將欄位新增後再執行傳入的Function。
                '''
                column_name = args[1].columns.tolist()
                column_name.insert(column_name.index(column) + 1, name)
                df = args[1].reindex(columns=column_name)
                return func(args[0], df)
            # end wrapper
            return wrapper
        # end decorator
        return decorator
    # end _insert_column

    def column_to_array(self, df: pd.core.frame.DataFrame, column: str) -> np.ndarray:
        '''
        取出DataFrame特定欄位。
        ndarray <- column_to_array(DataFrame, 欄位名稱)
        '''
        array = np.array(df[column])
        return array
    # end column_to_array

    def drop_columns(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        去除DataFrame中不需要的欄位。
        DataFrame <- drop_columns(DataFrame)
        '''
        df = df.drop(['Date', 'Total Shares', 'Number of Shareholders',
                      'Shares per Shareholders', '>400 Shares Held',
                      '>400 Holding Percentage', '>400 NoS', '400-600 NoS',
                      '600-800 NoS', '800-1000 NoS', '>1000 NoS',
                      '>1000 Holding Percentage', 'Closing Price'], axis=1)
        return df
    # end drop_columns

    def output_csv(self, df: pd.core.frame.DataFrame, symbol: int) -> None:
        '''
        輸出.csv檔。
        None <- output_csv(DataFrame)
        '''
        folder_path = f'{self.file_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        save_path = f'{folder_path}/Output.csv'
        df.to_csv(save_path, index=False)
    # end output_csv

    def price_processing(self, tra_pre: np.ndarray, val_pre: np.ndarray,
                         price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        將價差百分比變化轉回原始價格。
        ndarray, ndarray <- price_processing(tra_pre, val_ori, val_pre, 原始價格)
        '''
        length = len(tra_pre)
        pri_front, pri_back = price[:length], price[length - 1:]
        length = len(tra_pre)
        for i in range(length):
            tra_pre[i] = (tra_pre[i] + 1) * pri_front[i]
        # end for
        length = len(val_pre)
        for i in range(length):
            val_pre[i] = (val_pre[i] + 1) * pri_back[i]
        # end for
        return tra_pre, val_pre
    # end price_processing

    @_insert_column('>400 Holding Percentage', 'FHHP Change')  # 新增FHHP Change欄位
    def processing_fhhp_change(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        為股權分散表建立FHHP Change欄位並填值。
        DataFrame <- processing_fhhp_change(DataFrame)
        '''
        length = len(df)
        for i in range(length):
            if i == length - 1:  # 將最後一筆標0
                df.loc[i, 'FHHP Change'] = 0
            # end if
            else:
                df.loc[i, 'FHHP Change'] = (
                    df.loc[i, '>400 Holding Percentage']
                    - df.loc[i + 1, '>400 Holding Percentage']
                )  # 填入跟上週的差距
            # end else
        # end for
        return df
    # end processing_fhhp_change

    @_insert_column('Number of Shareholders', 'NoS Change')  # 新增NoS Change欄位
    def processing_nos_change(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        為股權分散表建立NoS Change欄位並填值。
        DataFrame <- processing_nos_change(DataFrame)
        '''
        length = len(df)
        for i in range(length):
            if i == length - 1:  # 將最後一筆標0
                df.loc[i, 'NoS Change'] = 0
            # end if
            else:
                df.loc[i, 'NoS Change'] = (
                    df.loc[i, 'Number of Shareholders']
                    - df.loc[i + 1, 'Number of Shareholders']
                )  # 填入跟上週的差距
            # end else
        # end for
        return df
    # end processing_nos_change

    @_insert_column('>1000 Holding Percentage', 'OTHP Change')  # 新增OTHP Change欄位
    def processing_othp_change(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        為股權分散表建立OTHP Change欄位並填值。
        DataFrame <- processing_othp_change(DataFrame)
        '''
        length = len(df)
        for i in range(length):
            if i == length - 1:  # 將最後一筆標0
                df.loc[i, 'OTHP Change'] = 0
            # end if
            else:
                df.loc[i, 'OTHP Change'] = (
                    df.loc[i, '>1000 Holding Percentage']
                    - df.loc[i + 1, '>1000 Holding Percentage']
                )  # 填入跟上週的差距
            # end else
        # end for
        return df
    # end processing_othp_change

    @_insert_column('Closing Price', 'Price Change Percentage')  # 新增Price Change Percentage欄位
    def processing_price_change(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        為股權分散表建立Price Change Percentage欄位並填值。
        DataFrame <- processing_price_change(DataFrame)
        '''
        length = len(df)
        for i in range(length):
            if i == length - 1:  # 將最後一筆標0
                df.loc[i, 'Price Change Percentage'] = 0
            # end if
            else:
                df.loc[i, 'Price Change Percentage'] = (
                    (df.loc[i, 'Closing Price']
                     - df.loc[i + 1, 'Closing Price'])
                    / df.loc[i + 1, 'Closing Price']
                )  # 填入跟上週的價差百分比
            # end else
        # end for
        return df
    # end processing_price_change

    @_insert_column('Shares per Shareholders', 'SpS Change')  # 新增SpS Change欄位
    def processing_sps_change(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        為股權分散表建立SpS Change欄位並填值。
        DataFrame <- processing_sps_change(DataFrame)
        '''
        length = len(df)
        for i in range(length):
            if i == length - 1:  # 將最後一筆標0
                df.loc[i, 'SpS Change'] = 0
            # end if
            else:
                df.loc[i, 'SpS Change'] = (
                    df.loc[i, 'Shares per Shareholders']
                    - df.loc[i + 1, 'Shares per Shareholders']
                )  # 填入跟上週的差距
            # end else
        # end for
        return df
    # end processing_sps_change

    def reverse(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        將資料順序倒反。
        DataFrame <- reverse(DataFrame)
        '''
        df = df.iloc[::-1]
        return df
    # end reverse

    def re_columns_name(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        更改欄位名稱。
        DataFrame <- re_columns_name(DataFrame)
        '''
        df.columns = ['Date', 'Total Shares', 'Number of Shareholders',
                      'Shares per Shareholders', '>400 Shares Held',
                      '>400 Holding Percentage', '>400 NoS', '400-600 NoS',
                      '600-800 NoS', '800-1000 NoS', '>1000 NoS',
                      '>1000 Holding Percentage', 'Closing Price']
        return df
    # end re_columns_name

    def read_csv(self, symbol: int) -> pd.core.frame.DataFrame:
        '''
        讀取股權分散表。
        DataFrame <- read_csv(股票代號)
        '''
        read_path = f'{self.file_path}/{symbol}/Table.csv'
        df = pd.read_csv(read_path, thousands=',', encoding='utf-8-sig')  # thousands=','為去掉數字間逗號
        df = df.dropna()  # 去除資料中的NaN項目
        df['資料日期'] = pd.to_datetime(df['資料日期'], format='%Y%m%d')  # 將int日期欄位轉datetime
        return df
    # end read_csv
# end class Process

class Train(object):
    '''
    訓練模型用。
    Train(Files資料夾路徑, Images資料夾路徑, Models資料夾路徑)
    '''
    def __init__(self, file_path: str, image_path: str, model_path: str):
        self.file_path = file_path
        self.image_path = image_path
        self.model_path = model_path
        if not os.path.isdir(file_path):  # 如Files資料夾未建立則建立資料夾
            os.mkdir(file_path)
        # end if
        if not os.path.isdir(image_path):  # 如Images資料夾未建立則建立資料夾
            os.mkdir(image_path)
        # end if
        if not os.path.isdir(model_path):  # 如Models資料夾未建立則建立資料夾
            os.mkdir(model_path)
        # end if
    # end __init__

    def build_model(self, category='lstm', timesteps=4, features=1) -> Sequential:
        '''
        建立模型。
        Sequential <- build_model(模型類別, Timesteps, Features)
        '''
        dictionary = {'lstm': 1, 1: 1, 
                      'gru': 2, 2: 2}
        try:
            choice = dictionary[category]
        # end try
        except:
            raise CustomException(f'模型建立參數 \'{category}\' 不存在！')  # 丟出例外
        # end except
        if choice == 1:  # LSTM Model
            model = Sequential()
            model.add(LSTM(units=50,
                           activation='tanh',
                           return_sequences=True,
                           input_shape=(timesteps, features)))
            model.add(LSTM(units=50,
                           activation='tanh'))
            model.add(Dense(units=1))
            model.compile(loss='mse',
                          optimizer='adam',
                          metrics=['accuracy'])
        # end if
        elif choice == 2:  # GRU Model
            model = Sequential()
            model.add(GRU(units=50,
                          activation='tanh',
                          return_sequences=True,
                          input_shape=(timesteps, features)))
            model.add(GRU(units=50,
                          activation='tanh'))
            model.add(Dense(units=1))
            model.compile(loss='mse',
                          optimizer='adam',
                          metrics=['accuracy'])
        # end elif
        return model
    # end build_model

    def build_training_data(self, df: pd.core.frame.DataFrame, past=4, 
                            future=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        製作Training Data。
        ndarray, ndarray, ndarray <- build_training_data(DataFrame, 製作參數, 前多少筆作Features, 未來多少筆要預測的作Features)
        '''
        X, y = [], []
        timestep, column = df.shape
        for i in range(timestep - future - past):
            X.append(df[i:i + past])
            y.append(df[i + past:i + past + future, column - 1])
        # end for
        return np.array(X), np.array(y)
    # end build_training_data

    def inverse_normalize(self, scaler: MinMaxScaler, train: np.ndarray,
                          prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        將正規化的資料轉回原始資料。
         ndarray, ndarray <- inverse_normalize(scaler, 訓練資料的預測價格, 測試資料的預測價格)
        '''
        train = scaler.inverse_transform(train)
        prediction = scaler.inverse_transform(prediction)
        return train, prediction
    # end inverse_normalize

    def load_model(self, symbol: int) -> Sequential:
        '''
        載入訓練好的模型。
        Sequential <- load_model(股票代號)
        '''
        save_path = f'{self.model_path}/{symbol}'
        # model = load_model(save_path)  # 從指定路徑載入模型
        model = load_model(f'{save_path}.h5')  # 載入舊版儲存方式模型
        return model
    # end load_model

    def normalize(self, scaler: MinMaxScaler, df: pd.core.frame.DataFrame) -> np.ndarray:
        '''
        將資料做正規化。
        ndarray <- normalize(scaler, DataFrame)
        '''
        col1 = np.array(df['NoS Change']).reshape(-1, 1)
        col2 = np.array(df['SpS Change']).reshape(-1, 1)
        col3 = np.array(df['FHHP Change']).reshape(-1, 1)
        col4 = np.array(df['OTHP Change']).reshape(-1, 1)
        col5 = np.array(df['Price Change Percentage']).reshape(-1, 1)
        nor1 = scaler.fit_transform(col1)
        nor2 = scaler.fit_transform(col2)
        nor3 = scaler.fit_transform(col3)
        nor4 = scaler.fit_transform(col4)
        nor5 = scaler.fit_transform(col5)
        normalize = np.hstack((nor1, nor2))
        normalize = np.hstack((normalize, nor3))
        normalize = np.hstack((normalize, nor4))
        normalize = np.hstack((normalize, nor5))
        return normalize
    # end normalize

    def one_hot_encode(self, df: pd.core.frame.DataFrame, column: str) -> np.ndarray:
        '''
        將欄位做One Hot編碼。
        ndarray <- one_hot_encode(DataFrame, 欄位名稱)
        '''
        data = np.array(df[column])
        one_hot_data = np_utils.to_categorical(data)
        return one_hot_data
    # end one_hot_encode

    def save_model(self, model: Sequential, symbol: int) -> None:
        '''
        儲存訓練好的模型。
        None <- save_model(模型, 股票代號)
        '''
        print(u'儲存模型中……')
        folder_path = f'{self.model_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        try:
            # model.save(folder_path)
            model.save(f'{folder_path}.h5')  # 舊版儲存方式
        # end try
        except Exception as e:
            print(u'模型儲存失敗！', e)
        # end except
        else:
            print(u'模型儲存完畢！')
        # end else
    # end save_model

    def show_model(self, model: Sequential, symbol: int) -> None:
        '''
        顯示模型摘要。
        None <- show_model(模型, 股票代號)
        '''
        print(model.summary())
        folder_path = f'{self.file_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        save_path = f'{folder_path}/Result.txt'
        with open(save_path, mode='w', encoding='utf-8-sig') as txt_file:  # 將模型摘要寫入.txt檔
            model.summary(print_fn=lambda x: txt_file.write(f'{x}\n'))
            txt_file.write('\n')
        # end with
        folder_path = f'{self.image_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        save_path = f'{folder_path}/Model.png'
        print(u'生成模型摘要圖中……')
        image = plot_model(model,
                           to_file=save_path,
                           show_shapes=True,
                           dpi=300)
        display(image)
        print(u'模型摘要圖生成完畢！')
    # end show_model

    def shuffle(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        將資料打散，而非照日期排序。
        ndarray, ndarray <- shuffle(訓練資料x, 訓練資料y)
        '''
        np.random.seed(10)  # 設定亂數種子
        randomList = np.arange(len(X))
        np.random.shuffle(randomList)
        return X[randomList], y[randomList]
    # end shuffle

    def scores(self, tra_ori: np.ndarray, tra_pre: np.ndarray,
               val_ori: np.ndarray, val_pre: np.ndarray, symbol: int) -> None:
        '''
        顯示迴歸模型的衡量標準。
        None <- scores(訓練原始資料, 訓練預測資料, 測試原始資料, 測試預測資料)
        '''
        tra_ori *= 100
        tra_pre *= 100
        val_ori *= 100
        val_pre *= 100
        print('Training')
        print('-' * 20)
        print('MSE:', round(mean_squared_error(tra_ori, tra_pre), 4))
        print('RMSE:', round(sqrt(mean_squared_error(tra_ori, tra_pre)), 4))
        print('MAE:', round(mean_absolute_error(tra_ori, tra_pre), 4))
        print('MAPE:', round(mean_absolute_percentage_error(tra_ori, tra_pre), 4))
        print('R-Squared:', round(r2_score(tra_ori, tra_pre), 4))
        print('-' * 20, end='\n\n')
        print('Validation')
        print('-' * 20)
        print('MSE:', round(mean_squared_error(val_ori, val_pre), 4))
        print('RMSE:', round(sqrt(mean_squared_error(val_ori, val_pre)), 4))
        print('MAE:', round(mean_absolute_error(val_ori, val_pre), 4))
        print('MAPE:', round(mean_absolute_percentage_error(val_ori, val_pre), 4))
        print('R-Squared:', round(r2_score(val_ori, val_pre), 4))
        print('-' * 20, end='\n\n')
        folder_path = f'{self.file_path}/{symbol}'
        if not os.path.isdir(folder_path):  # 如資料夾未建立則建立資料夾
            os.mkdir(folder_path)
        # end if
        save_path = f'{folder_path}/Result.txt'
        with open(save_path, mode='a', encoding='utf-8-sig') as txt_file:  # 將衡量標準寫出成.txt檔
            txt_file.write('Training\n')
            txt_file.write('-' * 20 + '\n')
            txt_file.write(f'MSE: {round(mean_squared_error(tra_ori, tra_pre), 4)}\n')
            txt_file.write(f'RMSE: {round(sqrt(mean_squared_error(tra_ori, tra_pre)), 4)}\n')
            txt_file.write(f'MAE: {round(mean_absolute_error(tra_ori, tra_pre), 4)}\n')
            txt_file.write(f'MAPE: {round(mean_absolute_percentage_error(tra_ori, tra_pre), 4)}\n')
            txt_file.write(f'R-Squared: {round(r2_score(tra_ori, tra_pre), 4)}\n')
            txt_file.write('-' * 20 + '\n\n')
            txt_file.write('Validation\n')
            txt_file.write('-' * 20 + '\n')
            txt_file.write(f'MSE: {round(mean_squared_error(val_ori, val_pre), 4)}\n')
            txt_file.write(f'RMSE: {round(sqrt(mean_squared_error(val_ori, val_pre)), 4)}\n')
            txt_file.write(f'MAE: {round(mean_absolute_error(val_ori, val_pre), 4)}\n')
            txt_file.write(f'MAPE: {round(mean_absolute_percentage_error(val_ori, val_pre), 4)}\n')
            txt_file.write(f'R-Squared: {round(r2_score(val_ori, val_pre), 4)}\n')
            txt_file.write('-' * 20 + '\n\n')
        # end with
    # end scores

    def split(self, X: np.ndarray, y: np.ndarray,
              rate=0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        將Training Data取一部份當作Validation Data。
        ndarray, ndarray, ndarray, ndarray <- split(訓練資料X, 訓練資料y, 分割比率)
        '''
        X_len, y_len = ceil(len(X) * rate), ceil(len(y) * rate)  # ceil為無條件進位
        X_train = X[:X_len]
        y_train = y[:y_len]
        X_val = X[X_len:]
        y_val = y[y_len:]
        return X_train, y_train, X_val, y_val
    # end split

    def train_predict(self, model: Sequential, rate: float, X: np.ndarray, y: np.ndarray,
                      train: np.ndarray, original: np.ndarray) -> tuple[Sequential, np.ndarray, np.ndarray]:
        '''
        進行股價預測。
        Sequential, ndarray, ndarray <- train_predict(model, 分割比率, 訓練資料X, 訓練資料y, 分割後的訓練資料X, 原始資料)
        '''
        ori_len = len(original)  # 抓取資料長度
        val_pre = [0] * ori_len  # 建立要儲存所預測股價的List
        counter = 1  # 紀錄執行次數
        X_train, y_train, X_val, y_val = self.split(X, y, rate)  # 分割資料
        for i in range(ori_len):  # 滾動式的進行預測
            print(f'模型進行第 {counter} 次訓練中……')
            callback = [EarlyStopping(monitor='val_loss', patience=6)]  # 建立EarlyStopping條件
            history = model.fit(X_train, y_train,
                                batch_size=64,
                                epochs=10,
                                callbacks=callback,
                                validation_data=(X_val, y_val))  # 將資料餵入模型做訓練
            tra_his = history.history['loss']
            val_his = history.history['val_loss']
            if (counter == 1 and round(tra_his[0], 4) == round(tra_his[1], 4)
                             and round(val_his[2], 4) == round(val_his[3], 4)):
                raise CustomException(u'模型訓練發生問題！')  # 丟出例外
            # end if
            elif counter == 1:
                tra_pre = model.predict(train)  # 預測訓練資料的股價
            # end elif
            result = model.predict(X_val)  # 預測測試資料的股價
            val_pre[i:] = result
            X_row = X_val[0].reshape(1, X_val.shape[1], X_val.shape[2])  # 儲存X_val第一筆資料更改矩陣形狀以便跟X_train合併
            y_row = y_val[0].reshape(1, -1)  # 儲存y_val第一筆資料並更改矩陣形狀以便跟y_train合併
            X_val = np.delete(X_val, 0, axis=0)  # 刪除X_va中要加入X_train的第一筆資料
            y_val = np.delete(y_val, 0, axis=0)  # 刪除y_val中要加入y_train的第一筆資料
            X_train = np.append(X_train, X_row, axis=0)  # 將資料加入X_train
            y_train = np.append(y_train, y_row, axis=0)  # 將資料加入y_train
            counter += 1  # 執行次數加一
        # end for
        tra_pre = np.concatenate((tra_pre, val_pre[:1]))  # 將一筆val_pre的第一筆資料加入tra_pre在畫圖時才會連貫
        tra_pre = np.array(tra_pre).reshape(-1, 1)  # 將tra_pre的shape改為(tra_pre大小, 1)
        val_pre = np.array(val_pre).reshape(-1, 1)  # 將val_pre的shape改為(val_pre大小, 1)
        print(u'模型訓練完畢！')
        return model, tra_pre, val_pre
    # end train_predict
# end class Train

class CustomException(Exception):
    '''
    自訂例外狀況。
    '''
    def __init__(self, msg: str):
        self.error_message = msg
    # end __init__

    def __str__(self):
        return self.error_message
    # end __str__
# end class CustomException

def timer(func):
    '''
    紀錄執行所費時長的裝飾器。
    '''
    def wrapper():
        '''
        紀錄Function執行時長。
        '''
        start = perf_counter()
        func()
        cost = round(perf_counter() - start, 2)  # 四捨五入到小數點後第二位
        print(f'Execution took {cost} seconds.')
    # end wrapper
    return wrapper
# end timer
def increase(df, a, b, day):

    # a為今天的index
    # b為前(day)天的index
    # 計算(a - b) / b，再乘以100%
    # result = (df[a] - df[b].shift(day))#測試用
    result = ((df[a] - df[b].shift(day)) / df[b].shift(day)) * 100
    return result
rep = sa.reptile()
@timer
def main():
    '''
    主程式。
    '''
    # 設定參數 #
    #client 是我們與 Discord 連結的橋樑，intents 是我們要求的權限
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    #調用event函式庫
    @client.event
    #當機器人完成啟動時
    async def on_ready():
        print('目前登入身份：',client.user)

    @client.event
    #當有訊息時
    async def on_message(message):
        #排除自己的訊息，避免陷入無限循環
        if message.author == client.user:
            return
        #如果以「說」開頭
        if message.content.startswith('!'):
        #分割訊息成兩份
            tmp = message.content.split(" ",2)
        #如果分割後串列長度只有1
        if len(tmp) < 2:
            await message.channel.send("請提供股票代號")
        else:
            SYMBOL = int(tmp[1])        # 要抓取的股票代號
            try:
                # PAST = 4             # 位移展開的訓練資料數
                # FUTURE = 1           # 位移展開的預測資料數
                # SPLIT_RATE = 0.8     # 訓練及測試資料的分割比例
                # MODEL_TYPE = 'lstm'  # 選取要建立的模型類別：lstm, gru
                # MONEY = 100000       # 初始持有金額
                # BUY_RATE = 0.1       # 每次購買占本金比例

                # 設定路徑 #
                await message.channel.send("資料建置中")
                image_path = f'C:/Users/b3134/Desktop/stockdc/dataprocess/Images/{SYMBOL}/'
                techcsvpath = f'C:/Users/b3134/Desktop/stockdc/dataprocess/Files/{SYMBOL}/'

                FILE_PATH = r'./Files'
                IMAGE_PATH = r'./Images'
                MODEL_PATH = r'./Models'
                HeatMap_file = 'HeatMap.png'
                PairPlot_file='PairPlot.png'

                HeatMap_path = os.path.join(image_path, HeatMap_file)
                PairPlot_path=os.path.join(image_path, PairPlot_file)
                # 建立物件 #
                
                chart = Chart(IMAGE_PATH)
                crawler = Crawler(FILE_PATH)
                process = Process(FILE_PATH)
                # backtest = Backtest(FILE_PATH, IMAGE_PATH)
                # train = Train(FILE_PATH, IMAGE_PATH, MODEL_PATH)
                
                # 抓取股權分散表 #
                crawler.stock_crawler(SYMBOL)  # 抓取股票並儲存為.csv檔

                # 處理股權分散表 #
                df = process.read_csv(SYMBOL)             # 讀取股權分散表
                df = process.re_columns_name(df)          # 重新命名欄位
                df = process.processing_fhhp_change(df)   # 新增四百張大戶百分比變化欄位
                df = process.processing_nos_change(df)    # 新增總股東人數變化欄位
                df = process.processing_othp_change(df)   # 新增千張大戶百分比變化欄位
                df = process.processing_price_change(df)  # 新增收盤價變化比例欄位
                df = process.processing_sps_change(df)    # 新增人均張數變化欄位
                df = process.reverse(df)                  # 將資料順序倒反
                process.output_csv(df, SYMBOL)            # 輸出修改後的資料
                chart.heat_map(df, SYMBOL)                # 顯示熱力圖
                chart.pair_plot(df, SYMBOL)               # 顯示資料相關性分布圖
                #技術分析資料處理並產出
                rep.stockDownload(str(SYMBOL)+'.TW','2015-10-07', 'max')
                dfall = rep.getdfall()
                pd.set_option('display.max_rows', None)
                processda = po.processdata(dfall)
                processda.createIndicator()
                processda.MAdistance()
                dfall = processda.getdfall()
                dfall2 = processda.getdfall()
                dfall2 = dfall2.dropna(axis=1)

                dfall2.to_csv(f'{techcsvpath}techdata.csv', encoding='utf-8-sig')


                # 假設 df 是你的資料框，並且 'Date' 欄位是字符串類型
                df['Date'] = pd.to_datetime(df['Date'])

                # 設置 'Date' 欄位為索引
                df.set_index('Date', inplace=True)
                dff=df
                # 使用 resample 方法，每天生成一條新的記錄，再使用 ffill 方法填充缺失值
                df_resampled = df.resample('D').ffill()

                # 如果需要的話，你可以將索引還原為欄位
                df_resampled.reset_index(inplace=True)
                df_merged = pd.merge(dfall2, df_resampled, on='Date', how='inner')
                df_merged.set_index('Date', inplace=True)
                df_merged.index.name = '年月日'
                df_merged.to_csv(f'{techcsvpath}mergedata.csv', encoding='utf-8-sig')

                df=df_merged
                #label
                df["adj"] = df["adj close"] / df["close"]
                # ----------------------------
                # 計算(adj open/adj high/adj low)的值
                df["adj open"] = df["open"] * df["adj"]
                df["adj high"] = df["high"] * df["adj"]
                df["adj low"] = df["low"] * df["adj"]


                # 計算(a - b) / b的前一項資料，再乘以100%，並將計算結果新增為新的欄位 df["?"]
                df["open_y"] = increase(df, "adj open", "adj close",1)
                df["high_y"] = increase(df, "adj high", "adj close",1)
                df["low_y"] = increase(df, "adj low", "adj close",1)
                df["close_y"] = increase(df, "adj close", "adj close",1)
                one_day_columns = ['open_y', 'high_y', 'low_y', 'close_y']
                # ----------------------------

                # 預測2天漲幅
                # 計算(a - b) / b的前一項資料，再乘以100%，並將計算結果新增為新的欄位 df["?"]
                df["open_y_2day"] = increase(df, "adj open", "adj close",2)
                df["high_y_2day"] = increase(df, "adj high", "adj close",2)
                df["low_y_2day"] = increase(df, "adj low", "adj close",2)
                df["close_y_2day"] = increase(df, "adj close", "adj close",2)
                two_day_columns = ['open_y_2day', 'high_y_2day', 'low_y_2day', 'close_y_2day']
                # ----------------------------

                # 預測一周漲幅
                # 計算(a - b) / b的前一項資料，再乘以100%，並將計算結果新增為新的欄位 df["?"]
                df["open_y_5day"] = increase(df, "adj open", "adj close",5)
                df["high_y_5day"] = increase(df, "adj high", "adj close",5)
                df["low_y_5day"] = increase(df, "adj low", "adj close",5)
                df["close_y_5day"] = increase(df, "adj close", "adj close",5)
                five_day_columns = ['open_y_5day', 'high_y_5day', 'low_y_5day', 'close_y_5day']
                # ----------------------------

                # 將欄位名稱"營收發布日"改成"年月日"
                df = df.rename(columns={"Date": "年月日"})
                for column in one_day_columns:
                    df[column] = df[column].shift(-1)

                for column in two_day_columns:
                    df[column] = df[column].shift(-2)

                for column in five_day_columns:
                    df[column] = df[column].shift(-5)
                df = df.dropna()
                selected_columns = ['年月日','open_y', 'high_y','low_y','close_y','open_y_2day','high_y_2day','low_y_2day',
                                    'close_y_2day','open_y_5day','high_y_5day','low_y_5day','close_y_5day']
                # 使用 filter 方法
                df_selected = df.filter(selected_columns)

                df_selected.to_csv(f'{techcsvpath}label.csv', encoding='utf-8-sig')
                # 取得兩個 DataFrame 的最後一行
                last_row_a = dfall2.iloc[-1]
                last_row_b = dff.iloc[-1]

                # 合併兩個 DataFrame，並指定以 df_a 的 index 為新 DataFrame 的 index
                # today = pd.merge(last_row_a, last_row_b, left_index=True, right_index=True, how='outer')
                today = pd.concat([last_row_a, last_row_b])
                today = today.to_frame().values.reshape(1, -1)
                today = pd.DataFrame(today, columns=last_row_a.index.tolist() + last_row_b.index.tolist())

                today.to_csv(f'{techcsvpath}today.csv', encoding='utf-8-sig', index=False)
                await message.channel.send(tmp[1])
                await message.channel.send("資料建置成功")
            except Exception as e:
                # 出現錯誤時發送錯誤訊息給使用者
                await message.channel.send(f"發生錯誤：{str(e)}")
            if os.path.exists(HeatMap_path and PairPlot_path):

                await message.channel.send(file=discord.File(HeatMap_path))
                await message.channel.send(file=discord.File(PairPlot_path))
                

            else:
                await message.channel.send("找不到相關圖片")
    
    
    # # 模型訓練及股價預測 #
    # date = process.column_to_array(df, 'Date')                                 # 儲存日期欄位
    # price = process.column_to_array(df, 'Closing Price')                       # 儲存收盤價欄位
    # precentage = process.column_to_array(df, 'Price Change Percentage')        # 儲存股價變化欄位
    # df = process.drop_columns(df)                                              # 去除不需要的欄位
    # scaler = MinMaxScaler(feature_range=(0, 1))                                # 將資料正規化至0到1之間
    # nor_df = train.normalize(scaler, df)                                       # 將資料做正規化
    # X, y = train.build_training_data(nor_df, PAST, FUTURE)                     # 製作Training Data
    # tra_ori, _, _, val_ori = train.split(X, y, SPLIT_RATE)                     # 儲存要預測部分的股價以便後續進行比對
    # tra_len, timesteps, fetures = tra_ori.shape                                # 紀錄Data的Training部分數量
    # model = train.build_model(MODEL_TYPE, timesteps, fetures)                  # 建立模型
    # train.show_model(model, SYMBOL)                                            # 顯示模型摘要
    # model, tra_pre, val_pre = train.train_predict(model, SPLIT_RATE, X, y,
    #                                               tra_ori, val_ori)            # 訓練模型並進行預測
    # train.save_model(model, SYMBOL)                                            # 儲存模型
    # # model = train.load_model(SYMBOL)                                           # 載入模型
    # tra_pre, val_pre = train.inverse_normalize(scaler, tra_pre, val_pre)       # 將正規化後的資料轉回
    # tra_pre_pre, val_pre_pre = tra_pre.copy(), val_pre.copy()                  # 儲存未轉回股價的股價變化
    # tra_pre, val_pre = process.price_processing(
    #     tra_pre, val_pre, price[PAST + FUTURE - 1:len(price) - 1]
    # )                                                                          # 將價差百分比轉回原始價格
    # all_pri = price[PAST + FUTURE:]                                            # 裁切收盤價欄位
    # all_pre = precentage[PAST + FUTURE:]                                       # 裁切股價變化欄位
    # chart.line_graph(date[PAST + FUTURE:], all_pri, tra_pre,
    #                  val_pre, SYMBOL)                                          # 顯示走勢資料圖
    # train.scores(all_pre[:tra_len], tra_pre_pre[:tra_len], all_pre[tra_len:],
    #              val_pre_pre, SYMBOL)                                          # 顯示迴歸模型的衡量標準

    # # 進行回測 #
    # backtest.set_money(MONEY)                                       # 設定初期持有金額
    # backtest.set_rate(BUY_RATE)                                     # 設定每次要用多少比例的本金做購買
    # backtest.start_testing(all_pri[tra_len:], val_pre, SYMBOL)      # 開始回測
    # backtest.show_result(SYMBOL)                                    # 顯示最終持有金額
    # backtest.show_tendency(date[PAST + FUTURE + tra_len:], SYMBOL)  # 顯示持有金額及股數走勢圖
# end main

if __name__ == '__main__':
    filterwarnings('ignore')  # 忽略警告
    main()
    
    
# end if