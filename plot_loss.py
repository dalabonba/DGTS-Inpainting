import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_loss_from_filenames(directory, model_prefix='Generator'):
    """
    從模型檔名中提取epoch和loss值
    
    Parameters:
        directory (str): 模型文件所在的目錄
        model_prefix (str): 模型檔名的前綴，預設為'Generator'
    
    Returns:
        tuple: (epochs, losses) - 兩個列表，分別包含epoch數和對應的loss值
    """
    # 編譯正則表達式模式
    pattern = f"{model_prefix}_(\d+)_([\d]+)"
    
    epochs = []
    losses = []
    
    # 遍歷目錄中的所有文件
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)
    
    # 根據epoch排序
    sorted_pairs = sorted(zip(epochs, losses))
    epochs, losses = zip(*sorted_pairs)
    
    return np.array(epochs), np.array(losses)

def plot_training_loss(directory, save_path=None, model_prefix='Generator', 
                      figure_size=(10, 6), style='dark_background'):
    """
    繪製訓練loss曲線
    
    Parameters:
        directory (str): 模型文件所在的目錄
        save_path (str): 圖表保存路徑，如果為None則顯示圖表
        model_prefix (str): 模型檔名的前綴
        figure_size (tuple): 圖表大小
        style (str): matplotlib樣式
    """
    # 設置畫布風格和大小
    plt.style.use(style)
    plt.figure(figsize=figure_size)
    
    # 獲取數據
    epochs, losses = extract_loss_from_filenames(directory, model_prefix)
    
    # 繪製主要曲線
    plt.plot(epochs, losses, 'c-', linewidth=2, label='Training Loss')
    
    # 添加趨勢線
    z = np.polyfit(epochs, losses, 3)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), 'r--', linewidth=1, label='Trend')
    
    # 標註最小loss點
    min_loss_idx = np.argmin(losses)
    min_epoch = epochs[min_loss_idx]
    min_loss = losses[min_loss_idx]
    plt.plot(min_epoch, min_loss, 'go', markersize=10, label=f'Min Loss: {min_loss:.4f}')
    
    # 設置圖表屬性
    current_date = datetime.now().strftime('%Y-%m-%d')
    plt.title(f'Training Loss Over Time ({current_date})', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 美化圖表
    plt.tight_layout()
    
    # 保存或顯示圖表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存至: {save_path}")
    else:
        plt.show()
    
    # 關閉圖表，釋放記憶體
    plt.close()

# 使用示例
if __name__ == "__main__":
    # 假設模型文件位於 "models" 目錄
    directory = "logs/trainedChpt/trainNO01"
    
    # 確保輸出目錄存在
    os.makedirs("plots", exist_ok=True)
    
    # 繪製並保存圖表
    plot_training_loss(
        directory=directory,
        save_path=f"plots/training_loss_{datetime.now().strftime('%Y%m%d')}.png",
        model_prefix="Generator",
        figure_size=(12, 8)
    )
    
    # 輸出一些統計信息
    epochs, losses = extract_loss_from_filenames(directory)
    print(f"訓練總epoch數: {len(epochs)}")
    print(f"最低loss: {min(losses):.4f} (epoch {epochs[np.argmin(losses)]})")
    print(f"最終loss: {losses[-1]:.4f} (epoch {epochs[-1]})")