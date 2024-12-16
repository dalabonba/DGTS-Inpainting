import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端，確保可以在無圖形界面環境中生成圖表
import matplotlib.pyplot as plt
import collections
import os.path as osp
import os

class MetricsTracker:
    def __init__(self, save_path):
        # 初始化數據結構
        self._since_beginning = collections.defaultdict(lambda: {}) # 存儲從程式開始到當前的所有數據
        self._since_last_flush = collections.defaultdict(lambda: {}) # 存儲自上次刷新以來的數據
        self._iter = [0]
        self.save_path = save_path
        
        # 確保保存路徑存在
        os.makedirs(save_path, exist_ok=True)
    
    def tick(self):
        # 遞增迭代計數器
        self._iter[0] += 1
    
    def plot(self, name, value):
        # 記錄指標值
        self._since_last_flush[name][self._iter[0]] = value
    
    def flush(self):
        # 如果沒有數據，直接返回
        if not self._since_last_flush:
            return
        
        # 打開或創建 txt 文件用於記錄所有指標
        with open(osp.join(self.save_path, 'metrics_log.txt'), 'a') as log_file:
            # 當前迭代的指標值
            log_file.write(f"Iteration {self._iter[0]}:\n")
            for name, vals in self._since_last_flush.items():
                log_file.write(f"{name}: {list(vals.values())}\n")
            log_file.write("\n")
        
        # 更新從程式開始以來的數據
        for name, vals in self._since_last_flush.items():
            self._since_beginning[name].update(vals)
        
        # 保留原本的單指標圖表
        for name, vals in self._since_last_flush.items():
            x_vals = np.sort(list(self._since_beginning[name].keys()))
            y_vals = [self._since_beginning[name][x] for x in x_vals]
            
            plt.figure(figsize=(8, 4))
            plt.plot(x_vals, y_vals)
            plt.xlabel('Iteration')
            plt.ylabel(name)
            plt.title(f'{name} Progression')
            plt.grid(True)
            plt.savefig(osp.join(self.save_path, f'{name.replace(" ", "_")}.jpg'))
            plt.close()
        
        # 繪製所有指標的綜合折線圖
        plt.figure(figsize=(12, 6))
        
        # 遍歷所有已記錄的指標
        for name in self._since_beginning.keys():
            x_vals = np.sort(list(self._since_beginning[name].keys()))
            y_vals = [self._since_beginning[name][x] for x in x_vals]
            plt.plot(x_vals, y_vals, label=name)
        
        plt.xlabel('Iteration')
        plt.ylabel('Metrics Value')
        plt.title('All Metrics Progression')
        plt.legend()
        plt.grid(True)
        
        # 保存所有指標的綜合圖表
        plt.savefig(osp.join(self.save_path, 'all_metrics.jpg'))
        plt.close()
        
        # 清空最近一次刷新的數據
        self._since_last_flush.clear()

# 使用範例
def main():
    # 創建 MetricsTracker 實例
    save_path = './logs'
    tracker = MetricsTracker(save_path)
    
    # 模擬訓練過程
    for epoch in range(100):
        tracker.tick()
        
        # 模擬不同的指標
        tracker.plot('loss', np.random.random())
        tracker.plot('accuracy', np.random.random() * 100)
        tracker.plot('learning_rate', 0.01 * (0.99 ** epoch))
        
        # 每 10 個 epoch 保存一次數據
        if epoch % 10 == 0:
            tracker.flush()

if __name__ == "__main__":
    main()