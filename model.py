import numpy as np

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



            


# MIRNet based model WIP
class MIRNet(nn.Module):
    def __init__(self, cfg):
        super(MIRNet, self).__init__()

        # Params
        self.in_channel = cfg.in_channel
        self.out_channel = cfg.out_channel
        self.features = cfg.features
        self.kernel_size = cfg.kernel_size
        self.bias = cfg.bias

        N_rrg = cfg.n_resgroups
        scale = 2

        # Shallow features
        self.head = nn.Conv2d(self.in_channel, self.features, 3, 1, 1, bias=self.bias)
        # self.rrg = nn.ModuleList([RRG(cfg) for _ in range(N_rrg)])
        self.rrg = RRG(cfg)

        self.tale = nn.Sequential(
                Upscaling(scale, n_feat=self.features),
                nn.Conv2d(self.features//scale, self.out_channel, 3, 1, 1, bias=self.bias),
                nn.ReLU(),
                nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0, bias=self.bias)
                )

    def forward(self, x):
        x = self.head(x)
        out = self.rrg(x) + x
        out = self.tale(out)
        return out


class MRB(nn.Module):
    def __init__(self, n_feat=64, reduction=8, scale=2, dw_type='antialias'):
        super(MRB, self).__init__()

        downX2 = n_feat*scale
        downX4 = downX2*scale

        # Pre DAU
        self.top_dau_pre = DAU(n_feat, reduction)
        self.mid_dau_pre = nn.Sequential(
                                Downscaling(scale, n_feat, dw_type),
                                DAU(downX2, reduction)
                                )
        self.btm_dau_pre = nn.Sequential(
                                Downscaling(scale, n_feat, dw_type),
                                Downscaling(scale, downX2, dw_type),
                                DAU(downX4, reduction)
                                )

        # to the TOP
        self.btm2top = nn.Sequential(
                                Upscaling(2, downX4),
                                Upscaling(2, downX2)
                                )
        self.mid2top = Upscaling(2, downX2)

        # to the MID
        self.top2mid = Downscaling(2, n_feat, dw_type)
        self.btm2mid = Upscaling(2, downX4)

        # to the BTM
        self.top2btm = nn.Sequential(
                                Downscaling(2, n_feat, dw_type),
                                Downscaling(2, downX2, dw_type)
                                )

        self.mid2btm = Downscaling(2, downX2, dw_type)

        # SKFF
        self.top_skff = SKFF(n_feat)
        self.mid_skff = SKFF(downX2)
        self.btm_skff = SKFF(downX4)

        # POST DAU
        self.top_dau_pos = DAU(n_feat, reduction)
        self.mid_dau_pos = nn.Sequential(
                                DAU(downX2, reduction),
                                Upscaling(2, downX2)
                                )
        self.btm_dau_pos = nn.Sequential(
                                DAU(downX4, reduction),
                                Upscaling(2, downX4),
                                Upscaling(2, downX2)
                                )
        # Tale
        self.skff_last = SKFF(n_feat)
        self.tale = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)


    def forward(self, x):
        top = self.top_dau_pre(x)
        mid = self.mid_dau_pre(x)
        btm = self.btm_dau_pre(x)
        #print("DAU pre: top {} mid {} btm {}".format(top.size(), mid.size(), btm.size()))

        top_skff = self.top_skff(top, self.mid2top(mid), self.btm2top(btm))
        mid_skff = self.mid_skff(mid, self.top2mid(top), self.btm2mid(btm))
        btm_skff = self.btm_skff(btm, self.top2btm(top), self.mid2btm(mid))
        #print("SKFF top_skff {} mid_skff {} btm_skff {}".format(top_skff.size(), mid_skff.size(), btm_skff.size()))

        top = self.top_dau_pos(top_skff)
        mid = self.mid_dau_pos(mid_skff)
        btm = self.btm_dau_pos(btm_skff)
        #print("DAU pos top {} mid {} btm {}".format(top.size(), mid.size(), btm.size()))

        out = self.tale(self.skff_last(top, mid, btm)) + x # long skip
        return out


class RRG(nn.Module):
    def __init__(self, cfg):
        super(RRG, self).__init__()

        n_feat = cfg.features
        reduction = cfg.reduction
        scale = cfg.scale
        N_mrb = cfg.n_resblocks
        bias  = cfg.bias
        dw_type = cfg.dw_type

        self.head = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias)
        self.mrb = nn.ModuleList(
                    [MRB(n_feat, reduction, scale, dw_type) for _ in range(N_mrb)]
                    )
        self.tale = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias)

    def forward(self, x):
        out = self.head(x)
        for i, block in enumerate(self.mrb):
            out = block(out)

        out = self.tale(out)
        return out


class SKFF(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False):
        super(SKFF, self).__init__()
        out_feat = n_feat // reduction

        self.gap = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(n_feat, out_feat, 1, 1, 0, bias=bias),
                nn.PReLU()
                )

        self.conv_top = nn.Sequential(
                            nn.Conv2d(out_feat, n_feat, 1, 1, 0, bias=bias),
                            nn.Softmax(dim=1)
                            )
        self.conv_mid = nn.Sequential(
                            nn.Conv2d(out_feat, n_feat, 1, 1, 0, bias=bias),
                            nn.Softmax(dim=1)
                            )
        self.conv_btm = nn.Sequential(
                            nn.Conv2d(out_feat, n_feat, 1, 1, 0, bias=bias),
                            nn.Softmax(dim=1)
                            )

    def forward(self, top, mid, btm):
        tmp = self.gap(top + mid + btm) # z
        out = torch.mul(top, self.conv_top(tmp))  # top fusion -> selected
        out += torch.mul(mid, self.conv_mid(tmp)) # mid fusion -> selected
        out += torch.mul(btm, self.conv_btm(tmp)) # btm fusion -> selected
        return out


class Upscaling(nn.Module):
    def __init__(self, scale, n_feat, bias=True):
        super(Upscaling, self).__init__()
        out_feat = int(n_feat // scale)

        self.topline = nn.Sequential(
                        nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=bias),
                        nn.PReLU(),
                        nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias),
                        nn.PReLU(),
                        nn.Upsample(scale_factor=scale, mode='bilinear'),
                        nn.Conv2d(n_feat, out_feat, 1, 1, 0, bias=bias)
        )

        self.skipline = nn.Sequential(
                        nn.Upsample(scale_factor=scale, mode='bilinear'),
                        nn.Conv2d(n_feat, out_feat, 1, 1, 0, bias=bias)
        )

    def forward(self, x):
        skip = self.skipline(x)
        out = self.topline(x) + skip
        return out


class Downscaling(nn.Module):
    def __init__(self, scale, n_feat, dw_type='antialias', bias=True):
        super(Downscaling, self).__init__()
        out_feat = scale*n_feat

        if dw_type == 'antialias':
            self.downsample = Antialiased_downsample(filt_size=3, channels=out_feat)
        else:
            self.downsample = nn.MaxPool2d(scale)

        self.topline = nn.Sequential(
                        nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias),
                        nn.PReLU(),
                        nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias),
                        nn.PReLU(),
                        self.downsample,
                        nn.Conv2d(out_feat, out_feat, kernel_size=1, padding=0, bias=bias)
        )

        self.skipline = nn.Sequential(
                        self.downsample,
                        nn.Conv2d(out_feat, out_feat, kernel_size=1, padding=0, bias=bias)
        )

    def forward(self, x):
        skip = self.skipline(x)
        out = self.topline(x) + skip
        return out

'''
Paper: http://proceedings.mlr.press/v97/zhang19a/zhang19a.pdf
Code: https://richzhang.github.io/antialiased-cnns/
'''
class Antialiased_downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Antialiased_downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
