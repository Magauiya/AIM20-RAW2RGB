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


class LoadTestData(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(glob.glob(os.path.join(self.dataset_dir, "*.jpg")))

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.dataset_dir, str(idx) + '.jpg')), dtype=np.float32)
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image, str(idx)


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


class LoadTestData_Organizer(Dataset):

    def __init__(self, data_dir, level, full_resolution=False):

        self.raw_dir = os.path.join(data_dir, 'test', 'huawei_full_resolution')

        self.level = level
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)

        if level > 1 or full_resolution:
            self.image_height = 1440
            self.image_width = 1984
        elif level > 0:
            self.image_height = 1280
            self.image_width = 1280
        else:
            self.image_height = 960
            self.image_width = 960

    def __len__(self):
        return len(glob.glob(os.path.join(self.raw_dir, "*.png")))

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = extract_bayer_channels(raw_image)

        if self.level > 1 or self.full_resolution:
            raw_image = raw_image[0:self.image_height, 0:self.image_width, :]
        elif self.level > 0:
            raw_image = raw_image[80:self.image_height + 80, 352:self.image_width + 352, :]
        else:
            raw_image = raw_image[240:self.image_height + 240, 512:self.image_width + 512, :]

        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image
