""" Main function for this repo. """
import argparse
import numpy as np
import torch
from test import Trainer

if __name__ == '__main__': # 程式的入口點，確保只有在此作為主程式運行時才執行以下程式碼，而不是被導入其他模組時執行
    parser = argparse.ArgumentParser() # 創建一個 argparse 解析器物件
    # basic parameters 基本參數
    parser.add_argument('--num_work', type=int, default=12) # 工作線程數
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu') # 設備類型，默認為 CUDA 如果可用的話，否則為 CPU
    parser.add_argument('--image_size', type=int, default=256) # 圖片尺寸
    parser.add_argument('--crop_size', type=int, default=128) # 裁剪尺寸
    parser.add_argument('--seed', type=int, default=0) # 隨機種子
    parser.add_argument('--max_epoch', type=int, default=50) # 最大訓練時期數
    parser.add_argument('--batch_size', type=int, default=1) # 批次大小
    parser.add_argument('--start_epoch', type=int, default=0) # 開始時期
    parser.add_argument('--dataset_dir', type=str, default='data/teeth_seem_inlay') # ！！！資料集目錄！！！
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--mask_dir', type=str, default='data/mask_teeth_seem_inlay')  # ！！！遮罩(mask)資料集目錄！！！
    parser.add_argument('--file_name', type=str, default='my_model_test')  # ！！！設置測試文件夾名稱以保存圖像！！！
    parser.add_argument('--dgts_path', type=str, default='logs/trainedChpt/trainNO01/Generator_428_0.5445051230490208.pth')  # ！！！設置使用的模型！！！

    args = parser.parse_args() # 解析命令列參數並存儲在 args 中
    print("args:",args)

    # Set manual seed for PyTorch 設置 PyTorch 的手動種子
    if args.seed==0: # 如果種子為0
        print ('Using random seed.') # 使用隨機種子
        torch.backends.cudnn.benchmark = True # 使用CUDA加速，自動找到最佳的算法配置(適合model的輸入尺寸固定時)
    else:
        print ('Using manual seed:', args.seed) # 使用手動設置的種子
        torch.manual_seed(args.seed) # 設置種子
        torch.cuda.manual_seed(args.seed) # 設置 CUDA 種子
        torch.backends.cudnn.deterministic = True # 指定使用特定的算法以確保一致的結果
        torch.backends.cudnn.benchmark = False # 關閉自動找尋最佳算法

    runTimes = 3 # 要嘗試修復幾輪(每一輪新建一個結果資料夾)
    runAllOrNot = 0 # 要從1開始跑嗎？否則只跑一次(即只輸出一次流水號為"runTimes"的資料夾)
    start = 1 if runAllOrNot == 1 else runTimes
    for i in range(start,runTimes+1):
        trainer = Trainer(args,i) # 創建 Trainer 實例，傳入參數 args 和循環變量 i(用於在輸出的資料夾後方加上流水號)
        trainer.train() # 調用 Trainer 實例的 train() 方法
