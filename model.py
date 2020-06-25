
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils

import utils

# Inspired by CycleISP and PaNet
class PDANet(nn.Module):
    def __init__(self, cfg):
        super(PDANet, self).__init__()
        self.kernel_size = 3
        self.features = 64
        self.in_channel = cfg.in_channel
        self.out_channel = cfg.out_channel

        m_head = [
                    utils.default_conv(self.in_channel, self.features, self.kernel_size),
                    nn.PReLU(),
                    utils.default_conv(self.features, self.features, self.kernel_size)
                 ]

        m_tail = [
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    utils.default_conv(self.features, self.out_channel, self.kernel_size),
                    nn.ReLU(),
                    utils.default_conv(self.out_channel, self.out_channel, kernel_size=1)
                 ]

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

        self.down1 = Down(64, 128, level=4)
        self.down2 = Down(128, 256, level=3)
        self.down3 = Down(256, 384, level=2)
        self.up3 = Up(640, 256)
        self.up2 = Up(384, 128)
        self.up1 = Up(192, 64)


    def forward(self, x):
        x1 = self.head(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        out = self.up3(x4, x3)
        out = self.up2(out, x2)
        out = self.up1(out, x1) + x1 
        out = self.tail(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then RPAB"""

    def __init__(self, in_channels, out_channels, level):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            utils.BasicBlock(utils.default_conv, in_channels, out_channels),
            RPAB(out_channels, level)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention = nn.Sequential(
            utils.BasicBlock(utils.default_conv, in_channels, out_channels)
        )


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.attention(x)


class PyramidAttention(nn.Module):
    def __init__(self, level=4, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True, conv=utils.default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1-i/10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = utils.BasicBlock(conv,channel,channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = utils.BasicBlock(conv,channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = utils.BasicBlock(conv,channel, channel,1,bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        #theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base,1,dim=0)
        # patch size for matching 
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bilinear')
            #feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            #sampling
            raw_w_i = utils.extract_image_patches(base, ksizes=[kernel, kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            #sampling
            w_i = utils.extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            #group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))],dim=0)  # [L, C, k, k]
            #normalize
            max_wi = torch.max(torch.sqrt(utils.reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi
            #matching
            xi = utils.same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1,wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)
            
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()
            
            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))],dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride,padding=1)/4.
            y.append(yi)
      
        y = torch.cat(y, dim=0)+res*self.res_scale  # back to the mini-batch
        return y


class DAU(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False):
        super(DAU, self).__init__()
        out_feat = n_feat // reduction
        self.head = nn.Sequential(
                nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias),
                nn.PReLU(),
                nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1, 0, bias=bias),
            nn.Sigmoid()
        )

        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, out_feat, 1, 1, 0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_feat, n_feat, 1, 1, 0, bias=bias),
            nn.Sigmoid()
        )

        self.tale = nn.Conv2d(2*n_feat, n_feat, 3, 1, 1, bias=bias)

    def forward(self, x):
        feat = self.head(x)

        # Channel Attention
        ch_at = torch.mul(self.channel(feat), feat)

        # Spatial Attention
        gmp, _ = torch.max(feat, dim=1, keepdim=True)  # max pool over channels
        gap = torch.mean(feat, dim=1, keepdim=True)    # average pool over channels

        sp_map = self.spatial(torch.cat([gap, gmp], axis=1))
        sp_at = torch.mul(sp_map, feat)

        out = torch.cat([ch_at, sp_at], axis=1)
        out = self.tale(out)

        return out


class RPAB(nn.Module):
    def __init__(self, features, level):
        super(RPAB, self).__init__()
        m_body = [
            utils.ResBlock(utils.default_conv, features, 3),
            PyramidAttention(level=level, channel=features),
            utils.ResBlock(utils.default_conv, features, 3)
        ]
        self.body = nn.Sequential(*m_body)
    
    def forward(self, x):
        out = self.body(x)
        return out


            