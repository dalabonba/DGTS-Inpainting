""" Trainer for meta-train phase. 
這個檔案定義了一個 `Trainer` 類，
負責設置訓練環境、初始化模型、定義損失函數，
並實現了整個訓練循環，
包括數據加載、前向傳播、損失計算和反向傳播等核心訓練邏輯。
"""
# GPT編輯版
# import json
import os
import os.path as osp
import torch.nn.functional as F
# import numpy as np
import torch
import torch.nn as nn
# import torchvision.transforms as T
# from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

#import plot
from dataset_loader import DatasetLoader as Dataset   # 引入資料集的自訂義 DatasetLoader 類，負責加載訓練和測試資料
from transformer_64 import Generator  # 引入模型 Generator 類，負責生成影像
from loss import PerceptualLoss, StyleLoss  # 引入感知損失與風格損失，這是訓練過程中的重要損失函數
# from itertools import cycle

import time
from pathlib import Path
import statistics

import random

class Trainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # 設置訓練結果保存路徑
        log_base_dir = 'logs/trainedChpt'  # 設定模型訓練結果的存放資料夾
        meta_base_dir = osp.join(log_base_dir, args.file_name)
        print("輸出儲存路徑:",meta_base_dir)
        self.save_path = meta_base_dir
        if os.path.exists(self.save_path):
            pass
        else:
            os.makedirs(self.save_path)

        self.args = args  # 將外部參數儲存在 self.args 中，方便後續使用

        # data
        # 初始化訓練資料集
        self.trainset = Dataset('train', self.args)  # 加載訓練資料集
        self.train_loader = None

        # model
        # 初始化生成器模型並移到設備上 (如 GPU)
        self.netG = Generator().to(self.args.device)

        # 設定規則遮罩(rgular mask)
        self.mask = torch.ones(self.args.batch_size, 1, self.args.image_size, self.args.image_size, device = self.args.device)
        self.mask[:, :, int((self.args.image_size - self.args.crop_size)//2): int((self.args.image_size + self.args.crop_size)//2), 
        int((self.args.image_size - self.args.crop_size)//2): int((self.args.image_size + self.args.crop_size)//2)] = 0.0

        # 初始化損失函數
        self.perceptual_loss = PerceptualLoss().to(self.args.device)
        self.style_loss = StyleLoss().to(self.args.device)
        self.l1_loss = nn.L1Loss()  # 使用 L1 損失

        # 設置優化器，對不同層賦予不同的學習率
        param_dicts = [
            {"params": [p for n, p in self.netG.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.netG.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]
        self.optimizer_g = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)

        # 將生成器設置為 DataParallel 以支援多 GPU 訓練
        self.netG = torch.nn.DataParallel(self.netG)

        ## transform for mask
        # 設置影像處理過程，用於對遮罩進行預處理
        self.transform = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) 

    # 訓練核心邏輯
    def train(self):
        # 設置模型為訓練模式
        self.netG.train()

        # 構建資料加載器，從訓練資料集中提取資料
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, drop_last=True) 
        for epoch in range(self.args.start_epoch, self.args.max_epoch + 1):
            start_time = time.time()
            print(f"Epoch {epoch} Now...")
            loss1_list = []
            loss2_list = []
            loss3_list = []
            lossa_list = []
            for data_in, data_in_path in self.train_loader: # 此處會調用__getitem__ (一次回傳一個batch的資料)

                # 查看當前輸入資料
                # for i in range(len(data_in)):
                #     for j in range(len(data_in[i])):
                #         image_tensor = data_in[i][j]  # 取出影像
                #         image_tensor = (image_tensor + 1) / 2  # 將張量的值範圍從 [-1, 1] 轉換到 [0, 1]
                #         image_pil = transforms.ToPILImage()(image_tensor)  # 轉換為 PIL 影像
                #         image_pil.save(f"debug/data_in/data_in{i}_{j}.png")  # 保存影像

                real = data_in.to(self.args.device)  # 將真實資料移到設備上 (如 GPU)
                B, C, H, W = real.size()  # 提取資料的尺寸

                # 隨機選取遮罩
                # tmp = random.sample(range(0, 12000), 1)
                # THE_PATH = osp.join('data/mask','%05d.png' % tmp[0])
                # mask_in = self.transform(Image.open(THE_PATH).convert('1')).to(self.args.device)
                # mask = mask_in.resize(1, 1, H, W)
                # mask = torch.repeat_interleave(1 - mask, repeats=B, dim=0)
                # print("！！隨機mask！！mask.shape:",mask.shape)

                # rgular mask
                # mask = self.mask
                # print(mask.shape)

                # my mask
                try:
                    THE_PATH = data_in_path
                    mask_list = []  # 用於存儲每張處理後的遮罩影像

                    for path in THE_PATH:
                        # 取得原始檔案名稱
                        original_name = Path(path).name

                        # 創建新的遮罩路徑
                        mask_path = str(Path(f'{self.args.mask_dir}/masked_' + original_name))

                        # 打開遮罩影像，並將其轉換為單通道（黑白）圖像，再來應用 transform 轉換，並移動到指定裝置
                        mask_in = self.transform(Image.open(mask_path).convert('1')).to(self.args.device)
                        
                        # 將遮罩 resize 成指定大小 (1, 1, H, W)
                        mask = mask_in.resize(1, 1, H, W)

                        # 反轉遮罩的黑白 (1 - mask)
                        mask = (1 - mask)
                        
                        # 將處理後的遮罩添加到列表
                        mask_list.append(mask)

                    # 使用 torch.cat 沿著第 0 維進行合併
                    mask = torch.cat(mask_list, dim=0)
                except Exception as e:
                    print(f"處理遮罩時發生錯誤: {str(e)}")
                    raise
                # print("！！my mask！！mask.shape:",mask.shape)

                # 建立縮小的真實影像與遮罩
                real1 = F.interpolate(real, scale_factor=0.25)
                mask1 = F.interpolate(mask, scale_factor=0.25)

                # 生成影像
                fakes = self.netG(real, mask)

                # 融合生成的影像與遮罩影像
                fake3 = fakes * (1. - mask1) + mask1 * real1

                # 計算損失
                loss1 = self.l1_loss(fake3 * (1. - mask1), real1 * (1. - mask1)) / (1 - mask1).mean()  # L1 損失
                loss2 = self.perceptual_loss(fake3, real1)  # 感知損失
                loss3 = self.style_loss(fake3 * (1. - mask1), real1 * (1. - mask1))  # 風格損失
                lossa = loss1 * 10 + loss2 * 0.1 + loss3 * 250  # 將各種損失加權求和
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())
                loss3_list.append(loss3.item())
                lossa_list.append(lossa.item())

                # 反向傳播與優化步驟
                self.optimizer_g.zero_grad()
                lossa.backward()
                self.optimizer_g.step()
            
            avg_loss1 = statistics.mean(loss1_list)
            avg_loss2 = statistics.mean(loss2_list)
            avg_loss3 = statistics.mean(loss3_list)
            avg_lossa = statistics.mean(lossa_list)
            print("L1損失損失(loss1)平均",avg_loss1)
            print("感知損失(loss2)平均",avg_loss2)
            print("風格損失(loss3)平均",avg_loss3)
            print("加權損失(lossa)平均",avg_lossa)
            # 儲存模型檔案
            torch.save(self.netG.state_dict(), os.path.join(self.save_path, 'Generator_{}.pth'.format(int(epoch))))
            end_time = time.time()
            execution_time = end_time - start_time
            print("執行時間為:", execution_time, "秒")
            print("--------------------")
