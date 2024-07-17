""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DatasetLoader(Dataset): #繼承pytorch的Dataset
    """The class to load the dataset"""
    def __init__(self, setname, args):
        # Set the path according to train, val and test   
        self.args = args
        THE_PATH = None     
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train')
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'test')
        else:
            raise ValueError('Wrong setname.') 
         
        data = []
        #遍歷THE_PATH目錄中所有檔案並儲存每個檔案路徑至data[]中
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                data.append(osp.join(root, name))
                
        self.data = data
        print("輸入資料集的樣本數量：",len(self.data))

        self.image_size = args.image_size

        '''
        定義self.transform:圖像操作
        transforms.Compose:接收一個預處理列表作為參數，按順序將每個操作應用於輸入圖像
        RandomResizedCrop:將圖像隨機裁減為指定尺寸
        RandomHorizontalFlip:隨機左右翻轉圖像
        ToTensor:圖像轉張量
        Normalize:正規化(不是很理解參數)
        '''
        if setname=='train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
        	    transforms.ToTensor(),
          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    #返回資料集中的樣本數
    def __len__(self):
        return len(self.data)

    #對圖像執行self.transform定義的圖像操作
    def __getitem__(self, i):
        path = self.data[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image
