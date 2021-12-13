import sys
import numpy as np
import copy
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from tqdm import tqdm
import ttach as tta

trans = sys.modules[__name__]

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss


def transform_224():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return train_transform, valid_transform

def ttafunc(net, height):
    return tta.ClassificationTTAWrapper(net, tta.aliases.five_crop_transform(height, height))
