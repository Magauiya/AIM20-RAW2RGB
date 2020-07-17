import glob
import os

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def extract_bayer_channels(raw):
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir, test=False):

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.test = test

    def __len__(self):
        return len(glob.glob(os.path.join(self.dslr_dir, "*.jpg")))

    def _transform(self, raw, rgb):
        if np.random.rand() > 0.5:
            raw = np.fliplr(raw)
            rgb = np.fliplr(rgb)
        if np.random.rand() > 0.5:
            raw = np.flipud(raw)
            rgb = np.flipud(rgb)

        raw = np.copy(raw)
        rgb = np.copy(rgb)

        return raw, rgb

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')), dtype=np.float32)
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        dslr_image = Image.open(os.path.join(self.dslr_dir, str(idx) + ".jpg"))
        dslr_image = np.asarray(dslr_image, dtype=np.float32)
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))

        if not self.test:
            raw_image, dslr_image = self._transform(raw_image, dslr_image)

        return raw_image, dslr_image / 255.


class LoadTestData(Dataset):

    def __init__(self, dataset_dir):
        self.raw_dir = dataset_dir

    def __len__(self):
        return len(glob.glob(os.path.join(self.raw_dir, "*.png")))

    def __getitem__(self, idx):
        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')), dtype=np.float32)
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image, str(idx)