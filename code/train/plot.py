# 導入必要的函式庫
import numpy as np  # 用於數值計算
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端，適合在無圖形界面環境中生成圖表
import matplotlib.pyplot as plt  # 用於繪圖
import collections  # 提供特殊的容器數據類型
import time  # 時間相關函數（但在此程式碼中未使用）
import sys  # 系統相關函數（但在此程式碼中未使用）
import os.path as osp  # 用於路徑處理

# 使用 defaultdict 創建兩個字典，用於存儲指標數據
# 這允許在不存在鍵時自動創建空字典
_since_beginning = collections.defaultdict(lambda: {})  # 存儲從程式開始到當前的所有數據
_since_last_flush = collections.defaultdict(lambda: {})  # 存儲自上次刷新以來的數據

# 使用列表來模擬可變的迭代計數器
_iter = [0]

# 遞增迭代計數器的函數
def tick():
    _iter[0] += 1

# 記錄指標值的函數
def plot(name, value):
    # 在 _since_last_flush 字典中，以迭代次數為鍵記錄指標值
    _since_last_flush[name][_iter[0]] = value

# 繪圖並保存結果的函數
def flush(save_path):
    prints = []
    
    # 遍歷最近一次刷新以來的所有指標
    for name, vals in _since_last_flush.items():
        # 計算並打印該指標的平均值
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        
        # 更新從程式開始以來的數據
        _since_beginning[name].update(vals)
        
        # 準備繪圖數據
        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]
        
        # 清除當前圖形
        plt.clf()
        
        # 繪製指標隨迭代變化的折線圖
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        
        # 保存圖表為 JPG 文件
        plt.savefig(osp.join(save_path, name.replace(' ', '_')+'.jpg'))
    
    # 打印當前迭代次數和所有指標的平均值
    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    
    # 清空最近一次刷新的數據
    _since_last_flush.clear()