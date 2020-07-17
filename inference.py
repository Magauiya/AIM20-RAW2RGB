import os
import time

# Hydra <- yaml
import hydra
import numpy as np
# PyTorch
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

# my files
import model
from dataloader import LoadTestData


class ImageProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Device: {self.device}")
        print(f"[*] Path: {self.cfg.test_dir}")

    def build(self):
        model_name = self.cfg.model_name

        if model_name not in dir(model):
            model_name = "RRDUNet"

        print(f"[*] Model: {self.cfg.model_name}")
        self.model = getattr(model, model_name)(cfg=self.cfg).to(self.device)
        self.model = nn.DataParallel(self.model)

        self._load_ckpt()
        self._load_data()

    def inference(self):
        self.model.eval()

        if not os.path.exists(os.path.join(self.cfg.output_dir, self.cfg.subfile)):
            os.mkdir(self.cfg.output_dir + self.cfg.subfile)

        start = time.time()
        with torch.no_grad():
            for idx, (noisy, img_name) in enumerate(self.test_loader):
                noisy = noisy.to(self.device, dtype=torch.float)
                output = self.model(noisy)

                if self.cfg.tta:
                    for i in range(1, 4):
                        inputs = torch.rot90(noisy, i, dims=[2, 3])
                        output += torch.rot90(self.model(inputs), -i, dims=[2, 3])

                    noisy = noisy.flip(2)
                    for i in range(4):
                        inputs = torch.rot90(noisy, i, dims=[2, 3])
                        tmp = torch.rot90(self.model(inputs), -i, dims=[2, 3])
                        output += tmp.flip(2)

                    output /= 8

                output = np.clip(output.cpu().detach().numpy(), 0., 1.)
                output = 255. * np.squeeze(output).transpose((1, 2, 0))
                output = output.astype(np.uint8)

                img = Image.fromarray(output)
                img.save(f"{os.path.join(self.cfg.output_dir, self.cfg.subfile)}/{img_name[0]}.png")
                print(f"[{idx}] range: [{np.amin(output)} {np.amax(output)}]", end="\r")

        runtime = (time.time() - start) / len(self.test_loader)

        print(f"[*] Inference is done! Total time: {runtime} per img")

        cpu_or_gpu = 0  # 0 - GPU
        use_metadata = 0  # 0: no use of metadata, 1: metadata used
        other = '(optional) any additional description or information'

        readme_fn = os.path.join(os.path.join(self.cfg.output_dir, self.cfg.subfile), 'readme.txt')
        with open(readme_fn, 'w') as readme_file:
            readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
            readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
            readme_file.write('Metadata[1] / No Metadata[0]: %s\n' % str(use_metadata))
            readme_file.write('Other description: %s\n' % str(other))

    def _load_ckpt(self):
        resume_path = os.path.join(self.cfg.resume)
        if resume_path and os.path.exists(resume_path):
            print("[&] LOADING CKPT {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            print("[*] CKPT Loaded: {}".format(resume_path))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("[!] NO checkpoint found at '{}'".format(resume_path))

    def _load_data(self):
        test_dataset = LoadTestData(self.cfg.test_dir)
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False
        )
        print(f"Testset:    {self.cfg.batch_size} x {len(self.test_loader)}")


@hydra.main(config_path="./default.yaml")
def main(cfg):
    app = ImageProcessor(cfg.test)
    app.build()
    app.inference()


if __name__ == "__main__":
    main()
