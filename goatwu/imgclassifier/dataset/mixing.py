import sys
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

mixing = sys.modules[__name__]


class TrainDataset(data.Dataset):
    def __init__(self, csv_path, img_path, cls2num, transform=None, resize=224):
        self.img_path = img_path
        self.resize = resize
        self.transform = transform
        self.cls2num = cls2num
        self.df = pd.read_csv(csv_path, header=None)   # 去掉表头
        self.data_len = len(self.df.index) - 1         # 表头会放进列表，所以减1
        self.image = np.asarray(self.df.iloc[1:, 0])
        self.label = np.asarray(self.df.iloc[1:, 1])
        
    def __getitem__(self, index):
        single_image_name = self.image[index]
        # 读取图像文件
        img_as_img = Image.open(self.img_path + single_image_name)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img_as_img = self.transform(img_as_img)
        num_label = self.cls2num[self.label[index]]
        return img_as_img, num_label
    
    def __len__(self):
        return self.data_len
    
    
class TestDataset(data.Dataset):
    def __init__(self, csv_path, img_path, num2cls, transform=None, resize=224):
        self.img_path = img_path
        self.resize = resize
        self.transform = transform
        self.num2cls = num2cls
        self.df = pd.read_csv(csv_path, header=None)   # 去掉表头
        self.data_len = len(self.df.index) - 1         # 表头会放进列表，所以减1
        self.image = np.asarray(self.df.iloc[1:, 0])
        
    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image[index]  #self.image_arr[0]='images/0.jpg'
        # 读取图像文件
        img_as_img = Image.open(self.img_path + single_image_name)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img_as_img = self.transform(img_as_img)
        return img_as_img
    
    def __len__(self):
        return self.data_len
    