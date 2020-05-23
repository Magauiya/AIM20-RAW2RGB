import os
import time
import glob 
import random
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from dataloader import *
from model import UNet as model

# Hydra <- yaml
import hydra
from hydra import utils
from omegaconf import DictConfig
from losses import loss 

'''
seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

torch.backends.cudnn.deterministic = True


class ImageProcessor:
    def __init__(self, cfg):
    	self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        print(f"Path: {self.data_dir}")


    def build(self):
        self.criterion = loss(self.cfg.loss)
        self.model = model().to(self.device)
        self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.lr_sch = lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_step, gamma=self.cfg.lr_gamma)

        self._load_ckpt()
        self._load_data()
        self._make_dir()
        self.writer=SummaryWriter(log_dir=self.logs_path)
        
    def train(self):
        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.cfg.num_epochs - 1))
            print('-' * 10)
            self.evaluate(epoch)

            self.model.train()
            running_loss = 0
            tk0 = tqdm(self.train_loader, total=int(len(self.train_loader)))
            for noisy, clean in tk0:
            	noisy = noisy.to(self.device)
            	clean = clean.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                tk0.set_postfix(loss=(loss.item()))

            epoch_loss = running_loss / (len(self.train_loader)/self.cfg.batch_size)
            print('Training Loss: {:.8f}'.format(epoch_loss))

            self.writer.add_scalar("Loss/Train", epoch_loss)
            self.writer.add_scalar("LR", self.lr_sch.get_lr()[0])


	def evaluate(self, epoch):
        tk1 = tqdm(self.valid_loader, total=int(len(self.valid_loader)))
        self.model.eval()
        running_loss = 0
        running_psnr = 0

        with torch.no_grad():
            for noisy, clean in tk1:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                running_loss += loss.item()
                tk1.set_postfix(loss=(loss.item()))

                
                clean = clean.cpu().detach().numpy()
                output = np.clip(output.cpu().detach().numpy(), 0., 1.)

                # ------------ PSNR ------------
                for m in range(self.batch_size):
                	running_psnr += PSNR(clean[m], output[m])

            epoch_loss = running_loss / (len(self.valid_loader)/self.batch_size)
            epoch_psnr = running_psnr / (len(self.valid_loader)/self.batch_size)

            print(f'Val Loss: {epoch_loss:.3}, PSNR:{epoch_psnr:.3}')
            
            ckpt_name = f"epoch_{epoch}_val_loss_{epoch_loss:.3}_psnr_{epoch_psnr:.3}.pth"
            self._save_ckpt(epoch, ckpt_name)

            self.writer.add_scalar("Loss/Validation", epoch_loss)
            self.writer.add_scalar("PSNR/Validation", epoch_psnr)

    def test(self):
        return 0
    
    def _save_ckpt(self, epoch, save_file):
        state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
            }
        torch.save(state, save_file)
        del state
        
    def _load_ckpt(self):
        self.start_epoch = 1
        if self.cfg.resume:
            if os.path.isfile(self.cfg.resume):
                print("[&] LOADING CKPT '{}'".format(self.cfg.resume))
                checkpoint = torch.load(self.resume, map_location='cpu')
                self.start_epoch = checkpoint['epoch'] + 1
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("[*] CKPT Loaded '{}' (epoch {})".format(self.cfg.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("[!] NO checkpoint found at '{}'".format(self.cfg.resume))

    def _load_data(self):
        
        AUGMENTATIONS_TRAIN = transforms.Compose([
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor() 
                                        ])

        AUGMENTATIONS_TEST = transforms.Compose([
                                        transforms.ToTensor()
                                        ])

        # TODO: make dataloader
        train_dataset = RAW2RGBDataset(train_df, self.data_dir, augmentations=AUGMENTATIONS_TRAIN)
        valid_dataset = RAW2RGBDataset(val_df.sample(1000).reset_index(drop=True), self.data_dir, augmentations=AUGMENTATIONS_TEST)
        test_dataset = RAW2RGBTestDataset(self.test_df, augmentations=AUGMENTATIONS_TEST)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.cfg.batch_size,
                                                   num_workers=self.cfg.num_workers,
                                                   shuffle=True)

        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.cfg.batch_size,
                                                   num_workers=self.cfg.num_workers,
                                                   shuffle=False)
        
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  num_workers=self.cfg.num_workers,
                                                  shuffle=False,
                                                  drop_last=False)

    def _make_dir(self):
        # Output path: ckpts, imgs, etc.
        if not os.path.isdir(self.cfg.output_path):
            os.mkdir(self.cfg.output_path)

        if not os.path.isdir(os.path.join(self.cfg.output_path, self.cfg.exp_name)):
            os.mkdir(os.path.join(self.cfg.output_path, self.cfg.exp_name))
            os.mkdir(os.path.join(self.cfg.output_path, self.cfg.exp_name, 'ckpt'))
            os.mkdir(os.path.join(self.cfg.output_path, self.cfg.exp_name, 'logs'))
            print("[*] Save Directory created!")
        else:
            print("[*] Save Directory already exist!")

        self.ckpt_path = os.path.join(self.cfg.output_path, self.cfg.exp_name, 'ckpt')
        self.logs_path = os.path.join(self.cfg.output_path, self.cfg.exp_name, 'logs')


@hydra.main(config_path="./default.yaml")
def main(cfg: DictConfig) -> None:
    app = ImageProcessor(cfg.parameters)
    app.build()
    app.train()
    
    # app.test() # [WIP]
    
if __name__=="__main__":
    main()

