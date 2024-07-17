""" Main function for this repo. """
import argparse #用於處理命令行參數的模組
import numpy as np
import torch
from train import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #argparse 模組會建立一個ArgumentParser 物件，用於管理命令行參數。
    # basic parameters 
    parser.add_argument('--num_work', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='data/places2') # Dataset folder 輸入的資料集(執行run_train.py時會賦值給此參數)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--file_name', type=str, default='trainNO01') # ## set train Folder name for save the pretrained model 設定儲存的資料夾名稱
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test']) # Phase

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
