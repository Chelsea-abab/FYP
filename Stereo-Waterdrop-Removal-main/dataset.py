from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
from skimage import measure
from torch.nn import init
from torchvision import transforms
import torchvision.transforms.functional as transformsF


#./FYP/datasetQ/train/train
class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, resize=False):
        super(TrainSetLoader, self).__init__()
        self.resize = resize
        file_lists=[]
        data_lists=[]
        gt_lists=[]
        datanames=os.listdir(os.path.join(dataset_dir,'data'))
        gtnames=os.listdir(os.path.join(dataset_dir,'gt'))
        
        for img in datanames:
            data_lists.append(os.path.join(os.path.join(dataset_dir,'data'),img))
            print("img_name: ",img)
            
        for img in gtnames:
            gt_lists.append(os.path.join(os.path.join(dataset_dir,'gt'),img))
            print("gt_name: ",img)
        file_lists.append(data_lists)
        file_lists.append(gt_lists)
        self.file_list = file_lists
    def __getitem__(self, index):
        img=Image.open(self.file_list[0][index])
        gt=Image.open(self.file_list[1][index])
        if self.resize:
            h = 576
            w = 288
            img  = img.resize((h,w))
            gt  = gt.resize((h,w))
        if random.random()<0.5:
            img = transformsF.hflip(img)
            gt = transformsF.hflip(gt)
        if random.random()<0.5:
            img = transformsF.vflip(img)
            gt = transformsF.vflip(gt)

        return transforms.ToTensor()(np.array(img)), transforms.ToTensor()(np.array(gt))
        
    def __len__(self):
        return len(self.file_list[0])

#./FYP/datasetQ/test_a/test_a
class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, resize=False, val=False):
        super(TestSetLoader, self).__init__()
        self.resize = resize
        self.val = val
        file_lists=[]
        data_lists=[]
        gt_lists=[]
        datanames=os.listdir(os.path.join(dataset_dir,'data'))
        gtnames=os.listdir(os.path.join(dataset_dir,'gt'))
        
        for img in datanames:
            data_lists.append(os.path.join(os.path.join(dataset_dir,'data'),img))
            print("img_name: ",img)
            
        for img in gtnames:
            gt_lists.append(os.path.join(os.path.join(dataset_dir,'gt'),img))
            print("gt_name: ",img)
        file_lists.append(data_lists)
        file_lists.append(gt_lists)
        self.file_list = file_lists
    def __getitem__(self, index):
        img  = Image.open(self.file_list[0][index])
        if self.val:
            gt  = Image.open(self.file_list[1][index])
        if self.resize:
            h = 576
            w = 288
            img  = img.resize((h,w))
            if self.val:
                gt  = gt.resize((h,w))
        if self.val:
            return transforms.ToTensor()(np.array(img)), transforms.ToTensor()(np.array(gt))
        else:
            return transforms.ToTensor()(np.array(img))
    def __len__(self):
        return len(self.file_list[0])

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)