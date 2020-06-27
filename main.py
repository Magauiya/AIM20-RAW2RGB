import os
import random
import hydra
import time
import numpy as np
from sklearn import metrics
from skimage.measure import compare_psnr as PSNR

# PyTorch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# OWN
import loss
import model
from dataloader import LoadData


class ImageProcessor:
    def __init__(self, cfg):
        self.cfg = cfg.parameters
        self.steplr = cfg.steplr
        self.plateau = cfg.plateau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self):
        
        # MODEL
        model_name = self.cfg.model_name
        if model_name not in dir(model):
            model_name = "PDANet"
        self.model = getattr(model, model_name)(cfg=self.cfg).to(self.device)
        self.model = nn.DataParallel(self.model)
        if self.cfg.net_verbose:
            summary(self.model, (4, 224, 224))

        print('-' * 40)
        print(f"[*] Model: {self.cfg.model_name}")
        print(f"[*] Device: {self.device}")
        print(f"[*] Path: {self.cfg.data_dir}")

        # OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_init)

        # LOSS
        self.criterion = loss.Loss(self.cfg, self.device)

        # SCHEDULER
        if self.cfg.scheduler == "step":
            self.lr_sch = lr_scheduler.StepLR(self.optimizer, **self.steplr)
        else:
            self.lr_sch = lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.plateau)

        # INITIALIZE: dirs, ckpts, data loaders
        self._make_dir()
        self._load_ckpt()
        self._load_data()
        self.writer = SummaryWriter(log_dir=self.cfg.logs_path)

    def train(self):
        self.model.train()
        val_loss, val_psnr = self.evaluate(self.step - 1, self.start_epoch)
        print("[*] Preliminary check: Epoch: {} Step: {} Validation Loss: {} PSNR: {}".format(
            self.start_epoch,
            (self.step-1),
            val_loss,
            val_psnr
            ))
        print('-' * 40)

        # Resume training from stopped epoch
        for epoch in range(self.start_epoch, self.cfg.num_epochs):

            step_loss = 0
            start_time = time.time()

            for idx, (noisy, clean) in enumerate(self.train_loader, start=1):
                # Input/Target 
                noisy = noisy.to(self.device, dtype=torch.float)
                clean = clean.to(self.device, dtype=torch.float)

                # BackProp
                self.optimizer.zero_grad()
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                loss.backward()
                self.optimizer.step()

                # STATS
                step_loss += loss.item()
                if idx % self.cfg.verbose_step == 0:
                    val_loss, val_psnr = self.evaluate(self.step, epoch)
                    self.writer.add_scalar("Loss/Train", step_loss / self.cfg.verbose_step, self.step)
                    self.writer.add_scalar("Loss/Validation", val_loss, self.step)
                    self.writer.add_scalar("Stats/LR", self.optimizer.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar("Stats/PSNR", val_psnr, self.step)


                    print("[{}/{}/{}] Loss [T/V]: [{:.5f}/{:.5f}] PSNR: {:.3f} LR: {} Time: {:.1f} Output: [{}-{}]".format(
                        epoch, self.step, idx,
                        (step_loss/self.cfg.verbose_step), val_loss,
                        val_psnr,
                        self.optimizer.param_groups[0]['lr'],
                        (time.time()-start_time),
                        torch.min(output).item(),
                        torch.max(output).item()
                        ))

                    self.step += 1
                    if self.cfg.scheduler == "step":
                        self.lr_sch.step()
                    elif self.cfg.scheduler == "plateau":
                        self.lr_sch.step(metrics=val_loss) 
                    
                    step_loss, start_time = 0, time.time()
                    self.model.train()


    @torch.no_grad()
    def evaluate(self, step, epoch):
        self.model.eval()
        running_loss = 0
        running_psnr = 0

        for idx, (noisy, clean) in enumerate(self.valid_loader):
            noisy = noisy.to(self.device, dtype=torch.float)
            clean = clean.to(self.device, dtype=torch.float)
            output = self.model(noisy)

            if idx == 1:
                save_image(output, 'step_%d_output.png'%step, nrow=4)
                save_image(clean, 'step_%d_clean.png'%step, nrow=4)
            
            loss = self.criterion(output, clean)
            running_loss += loss.item()

            clean = clean.cpu().detach().numpy()
            output = np.clip(output.cpu().detach().numpy(), 0., 1.)

            # ------------ PSNR ------------
            for m in range(self.cfg.batch_size):
                running_psnr += PSNR(clean[m], output[m])

        epoch_loss = running_loss / (len(self.valid_loader)*self.cfg.batch_size)
        epoch_psnr = running_psnr / (len(self.valid_loader)*self.cfg.batch_size)

        ckpt_name = f"step_{step}_epoch_{epoch}_val_loss_{epoch_loss:.3}_psnr_{epoch_psnr:.3}.pth"

        self._save_ckpt(epoch, ckpt_name)
        self.model.train()
        
        return epoch_loss, epoch_psnr


    @torch.no_grad()
    def test(self):
        self.model.eval()
        print("Test f-n is WIP")


    def _load_ckpt(self):
        self.step = 1
        self.start_epoch = 0
        if os.path.exists(self.cfg.resume):
            resume_path = self.cfg.resume
        else:
            ckpts = [[f, int(f.split("_")[1])] for f in os.listdir(self.cfg.ckpt_path) if f.endswith(".pth")]
            ckpts.sort(key=lambda x: x[1], reverse=True)
            resume_path = None if len(ckpts) == 0 else os.path.join(self.cfg.ckpt_path, ckpts[0][0])

        if resume_path and os.path.exists(resume_path):
            print("[&] LOADING CKPT {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu')
            self.step = checkpoint['step'] + 1
            self.model.load_state_dict(checkpoint['model'])

            if not self.cfg.optim_reset:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # To support previously trained models w\o epoch parameter
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch']

            print("[*] CKPT Loaded '{}' (Start Epoch: {} Step {})".format(resume_path, self.start_epoch,
                                                                          checkpoint['step']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("[!] NO checkpoint found at '{}'".format(resume_path))


    def _save_ckpt(self, epoch, save_file):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': epoch
        }
        torch.save(state, os.path.join(self.cfg.ckpt_path, save_file))
        del state


    def _load_data(self):

        if not self.cfg.inference:
            train_dataset = LoadData(self.cfg.data_dir, test=False)
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                drop_last=True    # easier to estimate PSNR, loss, etc. 
            )

            valid_dataset = LoadData(self.cfg.data_dir, test=True)
            self.valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                drop_last=True 
            )

            print(f"[*] Trainset:    {self.cfg.batch_size} x {len(self.train_loader)}")
            print(f"[*] Validset:    {self.cfg.batch_size} x {len(self.valid_loader)}")

        else:
            test_dataset = LoadTestData(self.cfg.data_dir, level=2, full_resolution=False)
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                drop_last=False
            )


    def _make_dir(self):
        # Output path: ckpts, imgs, etc.
        if not os.path.exists(self.cfg.ckpt_path):
            os.mkdir(self.cfg.ckpt_path)
            print("[*] Checkpoints Directory created!")
        else:
            print("[*] Checkpoints Directory already exist!")

        if not os.path.exists(self.cfg.logs_path):
            os.mkdir(self.cfg.logs_path)
            print("[*] Logs Directory created!")
        else:
            print("[*] Logs Directory already exist!")


@hydra.main(config_path="./default.yaml")
def main(cfg):
    seed = cfg.parameters.random_seed
    print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    app = ImageProcessor(cfg)
    app.build()

    if cfg.parameters.inference:
        app.test()
    else:
        app.train()


if __name__ == "__main__":
    main()
