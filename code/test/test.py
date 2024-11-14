
import json
import os
import os.path as osp
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid,save_image

import torchvision.utils as vutils
import plot
from dataset_loader import DatasetLoader as Dataset
from transformer_64 import Generator
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from glob import glob
from PIL import Image
import argparse
import random
from torchvision import transforms
from models.networks.generator import UpsamplerGenerator

from torchvision.transforms import ToPILImage
from util.getFolderFilesPath import get_all_filenames as gafn
from util.depth_map_adjustment import adjust_depth_map_brightness as admb

class Options():
        netG = 'Upsampler'
        ngf = 64
        norm_G = 'spectralspadeposition3x3'
        resnet_n_blocks = 6
        use_attention = True
        input_nc = 4
        gpu_ids = [0]
        semantic_nc = 4
    

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def postprocess(x):

    x = (x + 1.) / 2.
    x.clamp_(0, 1)
    return x

class Trainer(object):
    def __init__(self, args, folder_number):
        # Set the folder to save the records and checkpoints

        log_base_dir = 'logs/'
        meta_base_dir = osp.join(log_base_dir, f"{args.file_name}{folder_number:02d}")
        self.save_path = meta_base_dir
        if os.path.exists(self.save_path):
            pass
        else:
            os.makedirs(self.save_path)
        self.args = args
        ### data
        self.trainset = Dataset('test', self.args)
        self.train_loader = None
        ####### model #######
        self.netG = Generator().to(self.args.device)
        self.netG = torch.nn.DataParallel(self.netG)
        ##square
        self.mask = torch.ones(self.args.batch_size, 1, self.args.image_size, self.args.image_size, device = self.args.device)
        self.mask[:, :, int((self.args.image_size - self.args.crop_size)//2): int((self.args.image_size + self.args.crop_size)//2), 
        int((self.args.image_size - self.args.crop_size)//2): int((self.args.image_size + self.args.crop_size)//2)] = 0.0

        
        self.transform = transforms.Compose([
        	transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
        	transforms.ToTensor(),
        ])
        opt = Options()
        self.up = UpsamplerGenerator(opt)

    def train(self):
        start_time = time.time()
        # Set the meta-train log
        upsample_path ="Places2/latest_net_G.pth"# The path of Upsampler pretrained network
        print("upsample_path：",upsample_path)
        self.up.load_state_dict(torch.load(upsample_path))
        self.up = self.up.cuda()
        self.up.eval()
        
        dgts_path = self.args.dgts_path #The path of our pretrained network
        print("dgts_path：",dgts_path)
        self.netG.load_state_dict(torch.load(dgts_path))
        self.netG.eval()
        self.train_loader = DataLoader(self.trainset, batch_size=1, shuffle=False, num_workers=self.args.num_work, drop_last=False) 
        
        maskList = gafn(self.args.mask_dir) # gafn：輸入 > 遮罩(mask)資料夾路徑。 輸出 > 資料夾中所有檔案的路徑(path物件清單)
        for i, (data_in, data_in_path) in enumerate(self.train_loader):
            real = data_in.to(self.args.device)
            B,C,H,W = real.size()

            #隨機mask
            # tmp = random.sample(range(0,12000),1)
            # MASK_PATH = osp.join('data/mask','%05d.png'%tmp[0])

            MASK_PATH = osp.join(maskList[i])
            THE_FILENAME = maskList[i].stem # 沒有後綴的檔名
            # print(maskList[i].name)

            mask_in = self.transform(Image.open(MASK_PATH).convert('1')).to(self.args.device)
            mask = mask_in.resize(1,1,H,W)
            mask = torch.repeat_interleave(1-mask, repeats=B, dim=0)

            #mask =self.mask
            fakes = self.netG(real,mask)
            
            
            real1 = F.interpolate(real, scale_factor = 0.25)
            mask1 = F.interpolate(mask, scale_factor = 0.25)
            fake3 = fakes* (1. - mask1) + mask1 * real1
            
            vis = fake3.detach().cpu()
            vis = make_grid(vis, nrow =1, padding = 0, normalize = True)
            vis = T.ToPILImage()(vis)
            vis.save(os.path.join(self.save_path,f'{THE_FILENAME}_f.png'))

            vis = (real).detach().cpu()
            vis = make_grid(vis, nrow =1, padding = 0, normalize = True)
            vis = T.ToPILImage()(vis)
            real_file_path = os.path.join(self.save_path,f'{THE_FILENAME}_r.png')
            vis.save(real_file_path)

            img = torch.from_numpy(np.array(Image.open(os.path.join(self.save_path,f'{THE_FILENAME}_r.png')).resize([256,256], Image.BICUBIC)))
            img_r = img.permute(2,0,1)/127.5 - 1.
            
            img_f = torch.from_numpy(np.array(Image.open(os.path.join(self.save_path,f'{THE_FILENAME}_f.png')).resize([32,32], Image.BICUBIC)))
            masked_img = torch.cat([(img_r.unsqueeze(0).cuda()* mask+(1-mask)), (1-mask)], 1)
            
            # 製作缺陷圖+遮罩圖
            to_pil = ToPILImage()
            pil_img = to_pil(masked_img.cpu().squeeze())
            mask_file_path = os.path.join(self.save_path,f'{THE_FILENAME}_m.png')
            pil_img.save(mask_file_path)
            # 將兩個圖像打開
            vis_img = Image.open(os.path.join(self.save_path,f'{THE_FILENAME}_r.png'))
            pil_img = Image.open(os.path.join(self.save_path,f'{THE_FILENAME}_m.png'))
            # 將 PNG 圖像貼到 png 圖像上，使用 paste 方法
            vis_img.paste(pil_img, (0, 0), mask=pil_img.split()[3])
            # 保存疊加後的圖像
            # vis_img.save(os.path.join(self.save_path,f'{THE_FILENAME}_c.png'))
            # 刪除原始圖
            os.remove(real_file_path)
            # 刪除遮罩圖
            os.remove(mask_file_path)

            sample_tensor =  torch.from_numpy(img_f.squeeze(0).data.cpu().numpy()).permute(2,0,1).unsqueeze(0).float()
            sample_tensor = sample_tensor/127.5 - 1
            
            _, sample_up = self.up([masked_img.cuda(), sample_tensor.cuda()])
            sample_up = sample_up * (1-mask) + masked_img[:,:3] * mask
            sample_up = sample_up[0].permute(1,2,0).detach().cpu().numpy()
            sample_up = ((sample_up+1)*127.5).astype(np.uint8)
            Image.fromarray(sample_up).save(os.path.join(self.save_path, f'{THE_FILENAME}_f.png'))

            # 將輸出的修復後圖片亮度校正回歸
            admb(os.path.join(self.save_path,f'{THE_FILENAME}_f.png'), data_in_path[0], MASK_PATH, os.path.join(self.save_path,f'{THE_FILENAME}_f.png'))

        end_time = time.time()
        execution_time = end_time - start_time
        print("execution_time:", execution_time, "s")
