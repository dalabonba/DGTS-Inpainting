""" Main function for this repo. """
import argparse #用於處理命令行參數的模組
import numpy as np
import torch
from train import Trainer
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #argparse 模組會建立一個ArgumentParser 物件，用於管理命令行參數。
    # basic parameters 
    parser.add_argument('--num_work', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=256) # 圖像大小
    parser.add_argument('--crop_size', type=int, default=128) # 裁剪大小
    parser.add_argument('--seed', type=int, default=0) # 隨機種子
    parser.add_argument('--dataset_dir', type=str, default='data/teeth_depthmaps') # ！！！輸入的資料集(執行run_train.py時會賦值給此參數)！！！
    parser.add_argument('--max_epoch', type=int, default=1000) # 最大epoch數
    parser.add_argument('--batch_size', type=int, default=1) # 批次大小
    parser.add_argument('--start_epoch', type=int, default=1) # 開始epoch數
    parser.add_argument('--file_name', type=str, default='trainNO02') # ！！！設定儲存的資料夾名稱！！！
    parser.add_argument('--mask_dir', type=str, default='data/mask_teeth_depthmaps')  # ！！！遮罩(mask)資料集目錄！！！
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test']) # Phase 此屬性對程式不影響(沒用到)
    parser.add_argument('--gpu_devices', type=str, default='0,1,2,3', help='指定要使用的 GPU 設備')

    args = parser.parse_args()
    print("屬性args:",args)

    # 設置可見的 CUDA 設備
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    # Set manual(手動) seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    trainer = Trainer(args)
    trainer.train()
