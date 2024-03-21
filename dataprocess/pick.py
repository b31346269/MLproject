import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import glob
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.dates as mdates
import mplfinance as mpf
from sklearn.metrics import r2_score
from warnings import filterwarnings
filterwarnings('ignore')
import plotly.graph_objects as go
import csv
import joblib
import discord
from discord.ext import commands
# def getcsv(path):
#     Stock_csv = []
#     csv_path = glob.glob(f"{path}*.csv")
#     num_csv = len(csv_path)  

#     for i in range(num_csv):
#         stock_data = pd.read_csv(csv_path[i])
#         Stock_csv.append(stock_data)


#     return Stock_csv

def getcsv(path):
    Stock_csv = []
    csv_path = glob.glob(f"{path}*.csv")
    num_csv = len(csv_path)

    if num_csv == 0:
        print(f"No CSV files found in path: {path}")
        return Stock_csv

    for i in range(num_csv):
        try:
            stock_data = pd.read_csv(csv_path[i])
            Stock_csv.append(stock_data)
        except Exception as e:
            print(f"Error reading CSV file {csv_path[i]}: {e}")

    return Stock_csv
#multiple_output
def multiple_output(features_df,merged_label,checkpath,processdatapath,stock_folder,feather120,label120,name,important,imagepath,today_df):
    print(merged_label.shape)
    print(features_df.shape)
    feather120 = feather120.drop(feather120.index[-1])
    
    # label120_df = label120.drop(columns=['年月日'])
    label120_df=label120
    # label120_df.to_csv(f'{backtraderpath}label120_df.csv', encoding='utf-8-sig')
    # new_dataframe = feather120.set_index('年月日')
    new_dataframe=feather120
    new_dataframe = new_dataframe.drop(columns=new_dataframe.columns)

    # features_important=features_df
    # merged_important=merged_label

    # features_df = features_df.drop(columns=['年月日'])
    # merged_label = merged_label.drop(columns=['年月日'])

    # feather120 = feather120.drop(columns=['年月日'])

    # uselastday_df=features_df
    label_columns = [col for col in features_df.columns if 'Unnamed' in col.lower()]
    features_df  = features_df.drop(columns=label_columns)


    features_df = features_df.dropna(axis=1)
    
    merged_label.drop(merged_label.index[-1], inplace=True)
    features_df.drop(features_df.index[-1], inplace=True)
    merged_label = merged_label.dropna(axis=1)

    # output_filename = f'{name}_processdata.csv'
    # features_df.to_csv(f'{processdatapath}{output_filename}', encoding='utf-8-sig')
    # output_filename = f'{name}_labeldata.csv'
    # merged_label.to_csv(f'{processdatapath}{output_filename}', encoding='utf-8-sig')
    # 将数据集拆分为训练集和测试集

    
    X_train, X_test, y_train, y_test = train_test_split(features_df, merged_label, test_size=0.2, random_state=42)

    if important == 1:

        
        sorted_importances = train_feature_xg(X_train, y_train)
        # 計算平均分數
        average_importance = sum(feature[1] for feature in sorted_importances) / len(sorted_importances)
        
        # 選取比平均分數還高的特徵
        top_features = [feature[0] for feature in sorted_importances if feature[1] > average_importance]
        
        # 在你的程式碼中使用該函數來繪製特徵重要性的圖表
        plot_feature_importance(sorted_importances, top_n=20, save_path='feature_importance_plot.png', imagepath=imagepath)
        # 印出重要特徵
        print(f"Important Features with Importance > {average_importance}:")
        for feature in top_features:
            print(feature)
        # 將特徵 DataFrame 僅保留重要特徵
        features_df = features_df[top_features]

    X_train.to_csv(f'{checkpath}X_train.csv', encoding='utf-8-sig')
    X_test.to_csv(f'{checkpath}X_test.csv', encoding='utf-8-sig')
    y_train.to_csv(f'{checkpath}y_train.csv', encoding='utf-8-sig')
    y_test.to_csv(f'{checkpath}y_test.csv', encoding='utf-8-sig')

    xgb_model = xgb.XGBRegressor(
    max_depth=5,         # 設定樹的最大深度
    min_child_weight=1,  # 設定子節點的最小樣本數
    n_estimators=100,    # 設定弱分類器的數量
    learning_rate=0.1    # 學習速率
    )

    multioutput_model = MultiOutputRegressor(xgb_model)

    # 訓練模型
    multioutput_model.fit(X_train, y_train)

    # 在測試集上進行預測
    # 預測訓練集
    # y_train_pred = multioutput_model.predict(X_train)


    string_array = ["open_y", "high_y", "low_y","close_y","open_y_2day","high_y_2day","low_y_2day","close_y_2day",
                    "open_y_5day", "high_y_5day","low_y_5day","close_y_5day"]
    # new_dataframe = pd.DataFrame(new_dataframe[0])
    

    # print(new_dataframe.iloc[0, 0])

    window_size=5
    datasize=120
    count=0
    # 印出每個輸出的訓練RMSE和R-squared分數
    # print("Before 120 data")
    # for i in range(y_train.shape[1]):
    #     train_rmse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i], squared=False)
    #     train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
    #     print(f'Training RMSE  {string_array[i]}: {train_rmse}')
    #     print(f'Training R-squared {string_array[i]}: {train_r2}')

    # # 測試部分
    # y_test_pred = multioutput_model.predict(X_test)

    # # 印出每個輸出的測試RMSE和R-squared分數
    # for i in range(y_test.shape[1]):
    #     test_rmse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i], squared=False)
    #     test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
    #     print(f'Testing RMSE {string_array[i]}: {test_rmse}')
    #     print(f'Testing R-squared {string_array[i]}: {test_r2}')


    new_dataframe = pd.DataFrame(index=range(datasize), columns=range(y_train.shape[1]))
    # new_dataframe = new_dataframe.set_index(label120['年月日'])
    new_dataframe = new_dataframe.set_index(label120.index)
    for u in range(5):
        column_name = u
        new_column_name = string_array[u]
        new_dataframe = new_dataframe.rename(columns={column_name: new_column_name})
    print("----------------------------------------------------")
    while(count<datasize):
        for k in range(0,window_size):
            try:
                # if((count+k)==60):
                    # print("12 round")
                    # y_train_pred = multioutput_model.predict(X_train)
                    # for i in range(y_train.shape[1]):
                    #     train_rmse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i], squared=False)
                    #     train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
                    #     print(f'Training RMSE  {string_array[i]}: {train_rmse}')
                    #     print(f'Training R-squared {string_array[i]}: {train_r2}')

                    # # 測試部分
                    # y_test_pred = multioutput_model.predict(X_test)

                    # # 印出每個輸出的測試RMSE和R-squared分數
                    # for i in range(y_test.shape[1]):
                    #     test_rmse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i], squared=False)
                    #     test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
                    #     print(f'Testing RMSE {string_array[i]}: {test_rmse}')
                    #     print(f'Testing R-squared {string_array[i]}: {test_r2}')

                today_features = feather120.iloc[count+k,:]
                label120_dfc = label120_df.iloc[count+k,:]
                today_features=pd.DataFrame(today_features)
                label120_dfc=pd.DataFrame(label120_dfc)
                today_features = today_features.transpose()
                label120_dfc = label120_dfc.transpose()
                tomorrow_prices = multioutput_model.predict(today_features)
                for f in range(0,y_train.shape[1]):
                    new_dataframe.iloc[count+k][f]=tomorrow_prices[0][f]
                X_train = pd.concat([X_train, today_features], axis=0, ignore_index=True)
                y_train = pd.concat([y_train, label120_dfc], axis=0, ignore_index=True)
                # X_train.to_csv(f'{processdatapath}X_train111.csv', encoding='utf-8-sig')
                # y_train.to_csv(f'{processdatapath}y_train111.csv', encoding='utf-8-sig')
            except:
                pass
        
        # print(count)
        multioutput_model.fit(X_train, y_train)
        count=count+5
    output_filename = f'{name}_yeti.csv'
    # save_to_csv(output_filename, os.path.join(stock_folder, output_filename))
    new_dataframe.to_csv(f'{processdatapath}{output_filename}', encoding='utf-8-sig')
    print("----------------------------------------------------")
    print("After 24 round")
    # y_train_pred = multioutput_model.predict(X_train)
    result_data = evaluate_model_performance(multioutput_model, X_train, y_train, X_test, y_test, string_array)



    # 結合變數name和文件名
    output_filename = f'{name}_model_performance.csv'

    # 假設result_data是包含模型性能數據的列表
   
    save_to_csv(result_data, os.path.join(stock_folder, output_filename))

    # for i in range(y_train.shape[1]):
    #     train_rmse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i], squared=False)
    #     train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
    #     print(f'Training RMSE  {string_array[i]}: {train_rmse}')
    #     print(f'Training R-squared {string_array[i]}: {train_r2}')

    # # 測試部分
    # y_test_pred = multioutput_model.predict(X_test)

    # # 印出每個輸出的測試RMSE和R-squared分數
    # for i in range(y_test.shape[1]):
    #     test_rmse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i], squared=False)
    #     test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
    #     print(f'Testing RMSE {string_array[i]}: {test_rmse}')
    #     print(f'Testing R-squared {string_array[i]}: {test_r2}')
    

    # today_features = uselastday_df.iloc[-1]
    
    # today_features=pd.DataFrame(today_features)

    # print(today_features.shape)
    # today_features = today_features.transpose()

    # today_features.to_csv(f'{processdatapath}today_features.csv', encoding='utf-8-sig')

    # 使用模型進行預測
    # today_features = today_features.values.reshape(1, -1)
    # print(today_features.shape)
    # 找到兩者共同的欄位名稱
    common_columns = today_df.columns.intersection(today_features.columns)

    # 從 todaydf 中選擇共同欄位名稱的部分
    result_df = today_df[common_columns]

    tomorrow_prices = multioutput_model.predict(result_df)


    # print("2330 2023/10/11 predict price")
    # print("Tomorrow open price:", tomorrow_prices[0][0])
    print(result_df.iloc[0, 3])

    print("Tomorrow close price:", result_df.iloc[0, 3] * (1 + tomorrow_prices[0][3]/100))
    print("明天會上漲或下跌",tomorrow_prices[0][3])
    
    return tomorrow_prices[0][0:12],result_df.iloc[0, 3] 
    # print("Tomorrow low price:", tomorrow_prices[0][2])#low
    # print("Tomorrow high price:", tomorrow_prices[0][3])#high
    # print("close_y_5day price:", tomorrow_prices[0][4])#high

def plot_feature_importance(sorted_importances, top_n=20, save_path=None, imagepath=None):
    top_features = sorted_importances[:top_n]
    feature_names, importance_scores = zip(*top_features)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importance_scores, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances')
    plt.gca().invert_yaxis()

    # 如果提供了保存路徑，則保存圖片到指定的 imagepath
    if save_path and imagepath:
        image_save_path = os.path.join(imagepath, save_path)
        plt.savefig(image_save_path)

    # plt.show()


def train_feature_xg(x_train, y_train):
    sorted_importances = []

    # 創建XGBoost迴歸器模型
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100)

    # 使用訓練集進行模型訓練
    model = model.fit(x_train, y_train)

    # 獲取特徵重要性
    importances = model.feature_importances_  # 特徵重要性得分
    feature_names = model.get_booster().feature_names  # 特徵名稱
    feature_importances = list(zip(feature_names, importances))  # 將特徵名稱和重要性得分組合為元組列表
    sorted_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)  # 按重要性得分降序排序
    return sorted_importances

def save_to_csv(data, filename):
    # 確保目錄存在
    folder_path = os.path.dirname(filename)
    os.makedirs(folder_path, exist_ok=True)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
def evaluate_model_performance(model, X_train, y_train, X_test, y_test, string_array):
    output_data = []

    # 訓練部分
    y_train_pred = model.predict(X_train)
    for i in range(y_train.shape[1]):
        train_rmse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i], squared=False)
        train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
        train_output = (f'Training RMSE  {string_array[i]}', train_rmse)
        train_output2 = (f'Training R-squared {string_array[i]}', train_r2)
        output_data.extend([train_output, train_output2])

    # 測試部分
    y_test_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        test_rmse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i], squared=False)
        test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        test_output = (f'Testing RMSE {string_array[i]}', test_rmse)
        test_output2 = (f'Testing R-squared {string_array[i]}', test_r2)
        output_data.extend([test_output, test_output2])

    return output_data

def preprocess(dfname,label_df,name,path):
    
    last_120_rows = dfname.tail(120)

    # 儲存最後 120 行資料到新的 CSV 檔案
    output_filename = f'{name}last_120_rows.csv'
    last_120_rows.to_csv(f'{path}{output_filename}', encoding='utf-8-sig')

    feather120=last_120_rows
    # 刪除原始 CSV 檔案的最後 120 行
    dfname = dfname.iloc[:-120]

    last_120_rows = label_df.tail(120)

    # 儲存最後 120 行資料到新的 CSV 檔案
    output_filename = f'{name}label_df.csv'
    last_120_rows.to_csv(f'{path}{output_filename}', encoding='utf-8-sig')

    label120=last_120_rows
    # 刪除原始 CSV 檔案的最後 120 行
    label_df = label_df.iloc[:-120]

    
    return dfname,label_df,feather120,label120

def find_csv_by_string(folder_path, sub_folder, target_string):
    matching_files = []
    
    # 檢查資料夾路徑是否存在
    if not os.path.exists(folder_path):
        print(f"指定的資料夾路徑不存在：{folder_path}")
        return matching_files
    
    # 組合子資料夾的完整路徑
    sub_folder_path = os.path.join(folder_path, sub_folder)

    # 檢查子資料夾是否存在
    if not os.path.exists(sub_folder_path):
        print(f"指定的子資料夾不存在：{sub_folder_path}")
        return matching_files

    # 搜尋子資料夾下的所有 CSV 檔案
    for file_name in os.listdir(sub_folder_path):
        if file_name.endswith(".csv") and target_string in file_name:
            matching_files.append(os.path.join(sub_folder_path, file_name))

    return matching_files

def find_csv_by_stock_code(base_path, stock_code):
    matching_folders = []
    
    # 搜尋基本路徑下的所有資料夾
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # 確保是資料夾而非檔案
        if os.path.isdir(folder_path):
            # 檢查資料夾名稱是否包含股票代號
            if stock_code in folder_name:
                matching_folders.append(folder_path)

    return matching_folders
def read_csv_filexce(file_path):
    try:
        df = pd.read_csv(file_path)


        # 將 DataFrame 放回列表中的正確位置
        # 在這裡可以使用 DataFrame 'df' 進行相關操作
        # print(df.head())  # 打印前幾行數據
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
    except pd.errors.ParserError:
        print(f"Error parsing CSV file: {file_path}")
    return df
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.set_index('年月日', inplace=True)
    
        # 將 DataFrame 放回列表中的正確位置
        # 在這裡可以使用 DataFrame 'df' 進行相關操作
        # print(df.head())  # 打印前幾行數據
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
    except pd.errors.ParserError:
        print(f"Error parsing CSV file: {file_path}")
    return df
def align_csv_data(file1, file2):
    

    # 取得每個 CSV 檔案的日期範圍
    date_range1 = file1.index
    date_range2 = file2.index
    
    # 將日期進行排序
    file1_sorted = file1.sort_index()
    file2_sorted = file2.sort_index()
    
    # 找到兩個日期範圍的最小共同範圍
    common_date_range = list(set(date_range1) & set(date_range2))
    
    common_date_range_str = sorted([str(date) for date in common_date_range])
    # print(common_date_range_str)
    
    # 選擇每個 CSV 檔案中的最小共同範圍
    df1_aligned = file1_sorted.loc[common_date_range_str]
    df2_aligned = file2_sorted.loc[common_date_range_str]
    # print(df1_aligned)
    return df1_aligned, df2_aligned

    
def find_csv_by_name(folder_path, name):
    matching_files = []

    # 檢查資料夾路徑是否存在
    if not os.path.exists(folder_path):
        print(f"指定的資料夾路徑不存在：{folder_path}")
        return matching_files

    # 搜尋資料夾下的所有 CSV 檔案
    for file_name in os.listdir(folder_path):
        if name in file_name and file_name.endswith(".csv"):
            matching_files.append(os.path.join(folder_path, file_name))

    return matching_files
def main():
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
        if message.content.startswith('$'):
        #分割訊息成兩份
            tmp = message.content.split(" ",2)
        #如果分割後串列長度只有1
        if len(tmp) < 2:
            await message.channel.send("請提供股票代號")
        else:
            stock_code = (tmp[1])        # 要抓取的股票代號
    
            # 搜尋包含股票代號的資料夾
            all_datapath='C:/Users/b3134/Desktop/stockdc/dataprocess/Files/'
            matching_folders = find_csv_by_stock_code(all_datapath, stock_code)

            # 顯示結果
            if matching_folders:
                print(f"包含股票代號 {stock_code} 的資料夾：")
                for stock_folder  in matching_folders:
                    print(stock_folder )
            else:
                print(f"在指定路徑中未找到包含股票代號 {stock_code} 的資料夾。")

            mergedata_files = find_csv_by_name(stock_folder, "mergedata")
            print(f"包含名稱 'mergedata' 的 CSV 檔案：")
            for mergedata_path in mergedata_files:
                print(mergedata_path)

            # 找到包含名稱 'label' 的 CSV 檔案
            label_files = find_csv_by_name(stock_folder, "label")
            print(f"\n包含名稱 'label' 的 CSV 檔案：")
            for label_path in label_files:
                print(label_path)
                # 找到包含名稱 'today' 的 CSV 檔案
            today_path = find_csv_by_name(stock_folder, "today")
            print(f"\n包含名稱 'today' 的 CSV 檔案：")
            for today_path in today_path:
                print(today_path)

            

            alldatapath=mergedata_path
            processdatapath='C:/Users/b3134/Desktop/stockdc/dataprocess/processdata/' #路徑
            labelpath=label_path
            checkpath='C:/Users/b3134/Desktop/stockdc/dataprocess/checkcsv/' #路徑
            backtraderpath='C:/Users/b3134/Desktop/stockdc/dataprocess/backtrader/' #路徑
            imagepath=f'C:/Users/b3134/Desktop/stockdc/dataprocess/Images/{int(stock_code)}/'
            feature_importance_plot='feature_importance_plot.png'
            feature_importance_path=os.path.join(imagepath, feature_importance_plot)

            #----------------------------
            print(imagepath)
            # 讀取原始CSV文件

            alldata = read_csv_file(alldatapath)
            label_df= read_csv_file(labelpath)
            today_df= read_csv_filexce(today_path)
            
            # tech_df=getcsv(newdata)
            print(today_df)
            #創建一個新的DataFrame來存儲包含"label"的欄位

            # alldata = pd.DataFrame(alldata[0])
            # label_df = pd.DataFrame(label_df[0])

            first = ['年月日','open', 'high','low','volume','adj close']
            second = ['年月日','adj close-MA5 distance', 'adj close-MA10 distance','adj close-MA20 distance',
                            'adj close-MA3 distance','MA20-MA10 distance','MA20-MA5 distance','MA10-MA5 distance',
                            'MA10-MA3 distance','MA20-MA3 distance','MA5-MA3 distance','volume']
            firstkdboolmacd=['年月日','open', 'high','low','volume','adj close','k','d','macdsignal','macdhist',
                            'upperband','middleban','lowerband']
            secondkdboolmacd = ['年月日','adj close-MA5 distance', 'adj close-MA10 distance','adj close-MA20 distance',
                            'adj close-MA3 distance','MA20-MA10 distance','MA20-MA5 distance','MA10-MA5 distance',
                            'MA10-MA3 distance','MA20-MA3 distance','MA5-MA3 distance','volume','macd','k-d distance',
                            'adj close-upperband distance','adj close-middleban distance','adj close-lowerband distance']


            first = alldata.filter(first, axis=1)
            second = alldata.filter(second, axis=1)
            firstkdboolmacd=alldata.filter(firstkdboolmacd, axis=1)
            secondkdboolmacd=alldata.filter(secondkdboolmacd, axis=1)

            alldata, label_df = align_csv_data(alldata, label_df)
            first, label_df = align_csv_data(first, label_df)
            second, label_df = align_csv_data(second, label_df)
            firstkdboolmacd, label_df = align_csv_data(firstkdboolmacd, label_df)
            secondkdboolmacd, label_df = align_csv_data(secondkdboolmacd, label_df)


            # option = int(input("Please input 1~5 :"))
            # important=int(input("Do feature important 1 or 0:"))
            option=5
            important=1

            if option == 1:
                #一階未加其他東西特徵
                result_string = stock_code + '_first'
                first,label_df,feather120,label120=preprocess(first,label_df,result_string,backtraderpath)
                multiple_output(first,label_df,checkpath,processdatapath,stock_folder,feather120,label120,result_string,important,imagepath)
            elif option == 2:
                #二階未加其他東西特徵
                result_string = stock_code + '_second'
                second,label_df,feather120,label120=preprocess(second,label_df,result_string,backtraderpath)
                multiple_output(second,label_df,checkpath,processdatapath,stock_folder,feather120,label120,result_string,important,imagepath)
            elif option == 3:
                #一階且加入kd和bool和macd
                result_string = stock_code + '_firstkdboolmacd'
                firstkdboolmacd,label_df,feather120,label120=preprocess(firstkdboolmacd,label_df,result_string,backtraderpath,imagepath)
                multiple_output(firstkdboolmacd,label_df,checkpath,processdatapath,stock_folder,feather120,label120,result_string,important,imagepath)
            elif option == 4:
                #二階且加入kd和bool和macd
                result_string = stock_code + '_secondkdboolmacd'
                secondkdboolmacd,label_df,feather120,label120=preprocess(secondkdboolmacd,label_df,result_string,backtraderpath)
                multiple_output(secondkdboolmacd,label_df,checkpath,processdatapath,stock_folder,feather120,label120,result_string,important,imagepath)
            elif option == 5:
                #全部特徵
                await message.channel.send("特徵抽取&開始訓練")
                
                
                result_string = stock_code + '_all'
                alldata,label_df,feather120,label120=preprocess(alldata,label_df,result_string,backtraderpath)
                percange,closeprice=multiple_output(alldata,label_df,checkpath,processdatapath,stock_folder,feather120,label120,result_string,important,imagepath,today_df)
                await message.channel.send("資料解析&訓練完畢")
                print(stock_folder)
                all_model_performance_path = find_csv_by_name(stock_folder, "all_model_performance")
                print(f"\n包含名稱 'all_model_performance' 的 CSV 檔案：")
                for all_model_performance_path in all_model_performance_path:
                    print(all_model_performance_path)
                all_model_performance_df= read_csv_filexce(all_model_performance_path)
                await message.channel.send(file=discord.File(feature_importance_path))
                await message.channel.send("明天會上漲或下跌的比例(%)")
                percange_str = [str(percange_value) + "%" for percange_value in percange]
                string_array = ["明日開盤", "明日最高", "明日最低", "明日收盤", "2日後開盤", "2日後最高", "2日後最低", "2日後收盤",
                                "5日後開盤", "5日後最高", "5日後最低", "5日後收盤"]
                string_arraymse = ["Testing RMSE open_y", "Testing RMSE high_y", "Testing RMSE low_y","Testing RMSE close_y",
                                   "Testing RMSE open_y_2day","Testing RMSE high_y_2day","Testing RMSE low_y_2day","Testing RMSE close_y_2day",
                    "Testing RMSE open_y_5day", "Testing RMSE high_y_5day","Testing RMSE low_y_5day","Testing RMSE close_y_5day"]
                predicted_prices_list = []
                count=23

                for i in range(12):  # 修正為範圍 12
                    try:
                        percange_value = percange_str[i].replace('%', '')  # 移除百分比符號
                        percange_float = float(percange_value)
                        predicted_price = closeprice * (1 + percange_float / 100)
                        all_model_performance_price=closeprice*(all_model_performance_df.iloc[count,1]/100)
                        
                        await message.channel.send(f"價格變化百分比:{string_array[i]}  {percange_str[i]}誤差為+-{all_model_performance_df.iloc[count,1]}%")
                        await message.channel.send("轉換之預測價格為")
                        await message.channel.send(f"{string_array[i]}價 : {predicted_price}+-{all_model_performance_price}")
                        predicted_prices_list.append(predicted_price)
                        print(all_model_performance_df.iloc[count,1])
                        print(count)
                        count=count+2
                    except ValueError:
                        await message.channel.send(f"價格變化百分比分別是:{string_array[i]} : 無效的數值")
                await message.channel.send("預測結束")
                # 繪製k棒圖

                # fig = go.Figure()
                # # 獲取預測價格數據（假設你的預測價格是predicted_price）
                # predicted_price_data = {'open': [predicted_prices_list[0]],
                #                         'high': [predicted_prices_list[1] ],  
                #                         'low': [predicted_prices_list[2] ],   
                #                         'close': [predicted_prices_list[3]]}
                # predicted_price_2day_data = {'open': [predicted_prices_list[4]],
                #                         'high': [predicted_prices_list[5] ],  
                #                         'low': [predicted_prices_list[6] ],   
                #                         'close': [predicted_prices_list[7]]}
                # predicted_price_5day_data = {'open': [predicted_prices_list[8]],
                #                         'high': [predicted_prices_list[9] ],  
                #                         'low': [predicted_prices_list[10] ],   
                #                         'close': [predicted_prices_list[11]]}
                # # 創建 3 個子圖，排列方式為 3 行 1 列
                # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

                # # 將 data 轉換為 Pandas DataFrame
                # predicted_price_data_df = pd.DataFrame(predicted_price_data)
                # predicted_price_2day_data_df = pd.DataFrame(predicted_price_2day_data)
                # predicted_price_5day_data_df = pd.DataFrame(predicted_price_5day_data)

                # # 確保 index 是日期
                # predicted_price_data_df.index = pd.to_datetime(predicted_price_data_df.index)
                # predicted_price_2day_data_df.index = pd.to_datetime(predicted_price_2day_data_df.index)
                # predicted_price_5day_data_df.index = pd.to_datetime(predicted_price_5day_data_df.index)

                # # 創建 3 個子圖，排列方式為 3 行 1 列
                # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

                # # 繪製 K 棒圖
                # axes[0].set_title('明日預測價格的K棒圖')
                # axes[0].plot(predicted_price_data_df.index, predicted_price_data_df['open'], label='Open')
                # # Add other candlestick components as needed...

                # axes[1].set_title('2日後預測價格的K棒圖')
                # axes[1].plot(predicted_price_2day_data_df.index, predicted_price_2day_data_df['open'], label='Open')
                # # Add other candlestick components as needed...

                # axes[2].set_title('5日後預測價格的K棒圖')
                # axes[2].plot(predicted_price_5day_data_df.index, predicted_price_5day_data_df['open'], label='Open')
                # # Add other candlestick components as needed...

                # # 設定共同的 x 軸標籤
                # for ax in axes:
                #     ax.set_xlabel('日期')

                # # 顯示圖表
                # plt.tight_layout()  # 確保子圖之間的間距適當
                # plt.show()
            else:
                print("無效的選項")
            
            # except ValueError:
            #     print("請輸入有效的數字")

            # multiple_output(alldata,label_df,checkpath,processdatapath,backtraderpath,feather120,label120,'all')
            # multiple_output(firstnotech,label_df,checkpath,processdatapath,backtraderpath,feather120,label120,'firstnotech')
    
# 設定k棒圖的數據
def plot_candlestick(fig, data, name):
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=name))
if __name__ == '__main__':
    main()

