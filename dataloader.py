import os
import cv2
import glob
from PIL import Image

from torchvision import transforms
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class Alaska2TestDataset(Dataset):

    def __init__(self, df, augmentations=None):

        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data.loc[idx]
        im = Image.open(fn)
        if self.augment:
            im = self.augment(im)
        return im, label
    

class Alaska2Dataset(Dataset):
    def __init__(self, df, data_dir, augmentations=None):

        self.data = df
        self.augment = augmentations
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #fn, label = self.data.loc[idx]
        fn = self.data.loc[idx][0]
        label = self.data.loc[idx][1]
        fn = fn.split('/')[-2:]

        path = ''
        for i in fn:
            path = os.path.join(path, i)
        im = Image.open(os.path.join(self.data_dir, path))
        if self.augment:
            # Apply transformations
            im = self.augment(im)
        return im, label
