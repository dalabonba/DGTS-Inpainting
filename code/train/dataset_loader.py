""" Dataloader for all datasets. 
這個檔案定義了一個名為 `DatasetLoader` 的類，
負責加載數據集，並對圖像
進行隨機裁剪、翻轉和正規化等預處理操作，以便於訓練和測試模型。
"""
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
        功用：
            動態擴增：
                資料擴增是在訓練過程中動態進行的。
                每次讀取一張圖片時，都會隨機應用這些變換。
                這意味著同一張圖片在不同的訓練週期中可能會呈現不同的形式。

            虛擬擴增效果：
                雖然數據集的實際大小沒有增加，但模型在訓練過程中會"看到"更多變化的版本。
                這等效於使用了一個更大的數據集，因為每次訓練週期圖像都可能不同。
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

    '''
    使用 PyTorch 的 DataLoader 來加載數據集時，
    __getitem__ 方法會被調用。
    DataLoader 會根據指定的批次大小和其他參數，
    依次請求數據集中的樣本。
    '''
    #對圖像執行self.transform定義的圖像操作
    def __getitem__(self, i):
        path = self.data[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image
