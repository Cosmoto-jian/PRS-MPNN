本資料夾 (result/) 包含了 9 個目標蛋白質在不同溫度 (T=1.0, 2.0, 5.0, 10.0) 下的 DFI Z-score 柔性變化分析結果。每個蛋白質皆擁有獨立的專屬資料夾，檔案依據類型分類如下：

result/
 ├── 2OAR_ChainA/
 │    ├── 2OAR_ChainA_ExtremeTemp_Clean.png
 │    ├── Top20_T_1.0.csv
 │    ├── Compare_1.0_to_2.0_Combined.csv
 │    ├── Compare_1.0_to_2.0_Combined_chainA.png
 │    ├── Compare_1.0_to_2.0_Combined_all.png
 │    └── ... (其他溫度條件檔案)
 ├── 4NYK_ChainA/
 └── ... (如此類推)

1. Top20_T_[溫度].csv 
在特定溫度下，Z-score最高的前 20% 殘基。

2. Compare_[T1]_to_[T2]_Increased.csv
升溫激發位點。記錄從 T1 升溫至 T2 時，新擠進Top 20% 的殘基。

3. Compare_[T1]_to_[T2]_Decreased.csv
消失位點。記錄從 T1 升溫至 T2 時，跌出 Top 20% 的殘基。

4. Compare_[T1]_to_[T2]_Combined.csv
上述 Increased 與 Decreased 的合併表。內含 Status column。


-------------------------------------------------------------------

png 為上述所有的 CSV 條件生成的 3D 圖 (用 Sticks 標記)。每種條件有兩種視角：

*_chainA.png：單鏈。

*_all.png：全鏈。

橙色：該特定溫度的 Top 20% 位點。
紅色：升溫後新增 (Increased) 的位點。
藍色：升溫後跌出 (Decreased) 的位點。
