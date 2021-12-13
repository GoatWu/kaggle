import sys
sys.path.append("../..")
import goatwu.basicfunc as basefunc

import goatwu.imgclassifier.dataset.loader as loader
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

loader = sys.modules[__name__]

def default_train_loader(dataset, batch_size, sampler=None):
    return data.DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           num_workers=basefunc.get_num_workers(),
                           sampler=sampler)


def default_valid_loader(dataset, batch_size, sampler=None):
    return data.DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           num_workers=basefunc.get_num_workers(),
                           sampler=sampler)

def default_test_loader(dataset, batch_size, sampler=None):
    return data.DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=basefunc.get_num_workers(),
                           sampler=sampler)

def generate_train_valid_set(dataset, split_ratio):
    lenth = dataset.__len__()
    valid_lenth = int(lenth * split_ratio)
    train_lenth = lenth - valid_lenth
    trainset, validset = data.random_split(dataset, [train_lenth, valid_lenth])
    return trainsetr, validset


def generate_cutmix_loader_func(num_class, beta=1.0, prob=0.5, num_mix=2):
    def cutmix_train_loader(dataset, batch_size, sampler=None):
        return data.DataLoader(
            CutMix(dataset, num_class=num_class, beta=beta, prob=prob, num_mix=num_mix), 
            batch_size=batch_size, sampler=sampler, num_workers=basefunc.get_num_workers())
    return cutmix_train_loader