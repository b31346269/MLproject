建議先閱讀論文
此程式緣起由我在TCSE2023台灣軟體工程研討會，針對台灣股權分散表結合機器學習衍伸出來的專案
程式功能為:
使用者可在discord輸入指令，! 股票代號，我寫的機器人即可抓此代號
抓到代號後，上網爬籌碼及技術面資料
進行探索性eda分析，![image](https://github.com/b31346269/MLproject/assets/104146065/2f05a74f-dea4-4a9b-a30b-fdf0da5d27d1)
籌碼即可找到其相關性還有特徵重要性
![image](https://github.com/b31346269/MLproject/assets/104146065/c5d5a476-e9a6-44d0-b313-51b5ed73194d)

![image](https://github.com/b31346269/MLproject/assets/104146065/53106b92-de10-40b6-ace6-b3aa0a32f30c)

輸入$ 股票代號
本地端執行機器學習之流程，輸入籌碼及技術面，輸出預測之5日內漲跌幅並回傳
過程包含特徵工程，模型建置，調整參數，訓練模型(滾動式訓練)
並回傳其預測結果及誤差給使用者參考
![image](https://github.com/b31346269/MLproject/assets/104146065/faea34e0-fb96-4006-8240-7a22e2421841)




