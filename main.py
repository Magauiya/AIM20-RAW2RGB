import os
import random
import numpy as np
from sklearn import metrics
from skimage.measure import compare_psnr as PSNR

# Hydra <- yaml
import hydra
import numpy as np
# PyTorch
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as PSNR
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
# my files
import loss
import model
from dataloader import LoadData


class ImageProcessor:
    def __init__(self, cfg):
        self.cfg = cfg.parameters
        self.steplr = cfg.steplr
        self.plateau = cfg.plateau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        print(f"Path: {self.cfg.data_dir}")

    def build(self):
        self.criterion = loss.Loss(self.cfg, self.device)
        model_name = self.cfg.model_name

        if model_name not in dir(model):
            model_name = "PDANet"

        print(f"Model choice: {self.cfg.model_name}")
        self.model = getattr(model, model_name)(cfg=self.cfg).to(self.device)

        self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_init)
        if self.cfg.scheduler == "step":
            self.lr_sch = lr_scheduler.StepLR(self.optimizer, **self.steplr)
        else:
            self.lr_sch = lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.plateau)

        self._make_dir()
        self._load_ckpt()
        self._load_data()
        self.writer = SummaryWriter(log_dir=self.cfg.logs_path)

    def train(self):
        tb_iter = self.step
        # Validation of the loaded ckpt (if it exists) before the training starts
        self.evaluate(tb_iter - 1, self.start_epoch)

        # Resume training from stopped epoch
        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.cfg.num_epochs - 1))
            print('-' * 10)

            self.model.train()

            step_loss = 0
            for idx, (noisy, clean) in enumerate(self.train_loader, start=1):
                noisy = noisy.to(self.device, dtype=torch.float)
                clean = clean.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                loss.backward()
                self.optimizer.step()
                step_loss += loss.item()

                if idx % self.cfg.verbose_step == 0:
                    self.evaluate(tb_iter, epoch)
                    self.writer.add_scalar("Train/Loss", step_loss / self.cfg.verbose_step, tb_iter)
                    self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]['lr'], tb_iter)
                    print(f'[{tb_iter:3}/{epoch:3}/{idx:4}] Train loss: {loss:.5e}')

                    tb_iter += 1
                    step_loss = 0

    def evaluate(self, step, epoch):
        self.model.eval()
        running_loss = 0
        running_psnr = 0

        with torch.no_grad():
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

            print(f'Val Loss: {epoch_loss:.3}, PSNR:{epoch_psnr:.3}')

            ckpt_name = f"epoch_{step}_val_loss_{epoch_loss:.3}_psnr_{epoch_psnr:.3}.pth"
            self._save_ckpt(epoch, step, ckpt_name)

            self.writer.add_scalar("Validation/Loss", epoch_loss, step)
            self.writer.add_scalar("Validation/PSNR", epoch_psnr, step)

        self.lr_sch.step(metrics=epoch_loss)
        self.model.train()

    def test(self):
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

    def _save_ckpt(self, step, epoch, save_file):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'epoch': epoch
        }
        torch.save(state, os.path.join(self.cfg.ckpt_path, save_file))
        del state

    def _load_data(self):
        train_dataset = LoadData(self.cfg.data_dir, test=False)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True
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
    if not cfg.parameters.inference:
        app.train()
    app.test()


if __name__ == "__main__":
    main()
