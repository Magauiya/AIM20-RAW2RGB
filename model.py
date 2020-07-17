from utils import *


class UNET(nn.Module):
    def __init__(self, cfg):
        super(UNET, self).__init__()
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.inc = DoubleConv(cfg.in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, cfg.out_channel)

    def forward(self, x):
        x = self.upscale(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)

        return out


'''
Raw2Rgb from CycleISP
source: https://github.com/swz30/CycleISP/blob/master/networks/cycleisp.py
'''


class RRGNet(nn.Module):
    def __init__(self, cfg):
        super(RRGNet, self).__init__()

        self.num_rrg = 3
        self.num_dab = 5
        self.n_feats = 96
        self.kernel_size = 3
        reduction = 8

        activation = nn.PReLU(self.n_feats)

        modules_head = [conv(cfg.in_channel, self.n_feats, kernel_size=self.kernel_size, stride=1)]

        modules_body = [
            RRG(
                conv, self.n_feats, self.kernel_size, reduction, act=activation, num_dab=self.num_dab) \
            for _ in range(self.num_rrg)]

        modules_body.append(conv(self.n_feats, self.n_feats, self.kernel_size))
        modules_body.append(activation)

        modules_tail = [conv(self.n_feats, self.n_feats, self.kernel_size), activation]
        modules_tail_rgb = [conv(self.n_feats, cfg.out_channel * 4, kernel_size=1)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.tail_rgb = nn.Sequential(*modules_tail_rgb)

        conv1x1 = [conv(self.n_feats * 2, self.n_feats, kernel_size=1)]
        self.conv1x1 = nn.Sequential(*conv1x1)

    def forward(self, x):
        x = self.head(x)
        for i in range(len(self.body)):
            x = self.body[i](x)
        x = self.tail(x)
        x = self.tail_rgb(x)
        x = nn.functional.pixel_shuffle(x, 2)
        return x


class RRDUNet(nn.Module):
    def __init__(self, cfg):
        super(RRDUNet, self).__init__()

        self.num_rrg = 3
        self.num_dab = 5
        self.n_feats = 96
        self.kernel_size = 3
        in_channel = cfg.in_channel
        out_channel = cfg.out_channel

        reduction = 8

        activation = nn.PReLU(self.n_feats)

        modules_head = [conv(in_channel, self.n_feats, kernel_size=self.kernel_size, stride=1)]

        modules_body_up = [
            RRG(
                conv, self.n_feats, self.kernel_size, reduction, act=activation, num_dab=self.num_dab) for _ in
            range(self.num_rrg)
        ]

        self.down = conv(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1)
        modules_body_down = [
            RRG(
                conv, self.n_feats, self.kernel_size, reduction, act=activation, num_dab=self.num_dab) for _ in
            range(self.num_rrg)
        ]

        self.up = Up(2 * self.n_feats, self.n_feats)

        modules_tail = [
            conv(self.n_feats, self.n_feats, self.kernel_size),
            activation,
            conv(self.n_feats, out_channel * 4, kernel_size=1)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body_up = nn.Sequential(*modules_body_up)
        self.body_down = nn.Sequential(*modules_body_down)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        out = self.head(x)
        out_down = self.down(out)

        for i in range(len(self.body_up)):
            out = self.body_up[i](out)

        for i in range(len(self.body_down)):
            out_down = self.body_down[i](out_down)

        out = self.up(out_down, out)
        out = self.tail(out)
        out = nn.functional.pixel_shuffle(out, 2)
        return out
