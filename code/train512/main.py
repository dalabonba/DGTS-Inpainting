""" Main function for this repo. """
import argparse #用於處理命令行參數的模組
import numpy as np
import torch
from train import Trainer
import os

if __name__ == '__main__':
    # 動態檢測並選擇 GPU
    available_gpus = list(range(torch.cuda.device_count()))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
    print(f"可用 GPU: {available_gpus}")

    parser = argparse.ArgumentParser() #argparse 模組會建立一個ArgumentParser 物件，用於管理命令行參數。
    parser.add_argument('--num_work', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # 圖像大小原始輸入圖像的完整大小（像素），如果設定為 256，表示所有輸入圖像都會被調整到 256x256 像素
    parser.add_argument('--image_size', type=int, default=256)

    # 實際用於神經網絡處理的圖像區域大小（像素），從原始 image_size 中心區域裁剪出來的子區域，
    # 如果 image_size 是 256，crop_size 可能是 224，意味著只使用圖像中心的 224x224 像素。
    # 只有用在建立rgular mask
    parser.add_argument('--crop_size', type=int, default=128)

    parser.add_argument('--seed', type=int, default=0) # 隨機種子
    parser.add_argument('--dataset_dir', type=str, default='data/teeth_depthmaps') # ！！！輸入的資料集(執行run_train.py時會賦值給此參數)！！！
    parser.add_argument('--max_epoch', type=int, default=1000) # 最大epoch數
    parser.add_argument('--batch_size', type=int, default=1) # 批次大小
    parser.add_argument('--start_epoch', type=int, default=1) # 開始epoch數
    parser.add_argument('--file_name', type=str, default='trainNO03') # ！！！設定儲存的資料夾名稱！！！
    parser.add_argument('--mask_dir', type=str, default='data/mask03')  # ！！！遮罩(mask)資料集目錄！！！
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test']) # Phase 此屬性對程式不影響(沒用到)

    args = parser.parse_args()
    print("屬性args:",args)

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
