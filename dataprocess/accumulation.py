import os   
from warnings import filterwarnings
import discord
from discord.ext import commands
import numpy as np                                                               # class Train use
import pandas as pd        
import glob
def main():
    '''
    主程式。
    '''
    # 設定參數 #
    #client 是我們與 Discord 連結的橋樑，intents 是我們要求的權限
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    # 調用 event 函式庫
    @client.event
    # 當機器人完成啟動時
    async def on_ready():
        print('目前登入身份：', client.user)

    @client.event
    # 當有訊息時
    async def on_message(message):
        # 排除自己的訊息，避免陷入無限循環
        if message.author == client.user:
            return
        # 如果以「#」開頭
        if message.content.startswith('#'):
            # 分割訊息成兩份
            tmp = message.content.split(" ", 2)
            # 如果分割後串列長度至少為 3（包含 #、x、y）
            if len(tmp) >= 3:
                # 取得 x 和 y 的數值
                x = tmp[1]
                y = tmp[2]
                # 將 x 和 y 轉換為適當的數值類型（這裡假設都是整數）
                x = int(x)
                y = int(y)
                # 進行你的操作，例如印出 x 和 y
                csv_path = f'C:/Users/b3134/Desktop/stock_chip_data/stock_chip_data'
                csv_files = glob.glob(csv_path + '/*.csv')
                await message.channel.send(f"近{x}周大戶累積買進大於{y}%，開始展開並計算漲幅")
                await message.channel.send(f"符合條件之股票代號有")
                for file_path in csv_files:
                    # 使用 Pandas 的 read_csv 函數讀取檔案
                    df = pd.read_csv(file_path)
                    file_name = os.path.basename(file_path)
                    # 在這裡可以進行你的操作，例如處理資料框 df
                    
                    # 印出檔案名稱和前幾筆資料
                    # print(f"檔案名稱: {file_path}")
                    # 假設你的 DataFrame 名稱為 df
                    first_row = df.iloc[0]  # 取得第一列資料
                    first_closing_price = first_row['收盤價']  # 假設收盤價的欄位名稱是 '收盤價'

                    if first_closing_price > 100:
                        # 取得最新一列中目標欄位的數值
                        latest_value = df.iloc[0, 5]
                        # 取得第四列中目標欄位的數值
                        fourth_value = df.iloc[int(x)-1, 5]

                        # 判斷是否比第四列高 y%
                        if latest_value > (fourth_value * (1 + y/100)):
                            z = latest_value - fourth_value * (1 + y/100)
                            print("==========")
                            print(latest_value)
                            print(fourth_value)
                            print(f"{z} (檔案名稱: {file_name}")
                            print("==========")
                            await message.channel.send(f"{file_name}")
                    else:
                        # 取得最新一列中目標欄位的數值
                        latest_value = df.iloc[0, 11]
                        # 取得第四列中目標欄位的數值
                        fourth_value = df.iloc[int(x)-1, 11]
                        # 判斷是否比第四列高 y%

                        if latest_value > (fourth_value * (1 + y/100)):
                            z = latest_value - fourth_value * (1 + y/100)
                            print("==========")
                            print(latest_value)
                            print(fourth_value)
                            print(f"{z} (檔案名稱: {file_name}")
                            print("==========")
                            await message.channel.send(f"{file_name}")

            else:
                await message.channel.send("請提供正確格式數值")

if __name__ == '__main__':
    filterwarnings('ignore')  # 忽略警告
    main()