from utils import *

'''
Raw2Rgb from CycleISP
source: https://github.com/swz30/CycleISP/blob/master/networks/cycleisp.py
'''

class miniRRGNet(nn.Module):
    def __init__(self, cfg):
        super(miniRRGNet, self).__init__()

        self.num_rrg = 2
        self.num_dab = 4
        self.n_feats = 64
        self.kernel_size = 3
        
        reduction = 8
        in_channel = cfg.out_channel
        out_channel = cfg.out_channel
        activation = nn.PReLU(self.n_feats)

        # NETWORK
        modules_head = [
                        conv(in_channel, self.n_feats, kernel_size=self.kernel_size, stride=1),
                        activation,
                        conv(self.n_feats, self.n_feats, kernel_size=self.kernel_size, stride=2),
                        ]
        
        modules_body = [
            RRG(
                conv, self.n_feats, self.kernel_size, reduction, act=activation, num_dab=self.num_dab) \
            for _ in range(self.num_rrg)]

        modules_body.append(conv(self.n_feats, self.n_feats, self.kernel_size))
        modules_body.append(activation)

        modules_tail = [
                        conv(self.n_feats, self.n_feats, self.kernel_size, bias=True), 
                        activation,
                        conv(self.n_feats, out_channel*4, kernel_size=1, bias=True)
                        ]
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        out = self.head(x)
        
        # miniRRGNet
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.tail(out)

        # Pixel Shuffle + Long skip connection
        out = nn.functional.pixel_shuffle(out, 2) + x
        
        return out


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
                conv, self.n_feats, self.kernel_size, reduction, act=activation, num_dab=self.num_dab) for _ in range(self.num_rrg)
            ]

        self.down = conv(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1)
        modules_body_down = [
            RRG(
                conv, self.n_feats, self.kernel_size, reduction, act=activation, num_dab=self.num_dab) for _ in range(self.num_rrg)
            ]

        self.up = Up(2*self.n_feats, self.n_feats)
        
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

        out_down = self.up(out_down, out)
        out = self.tail(out)
        out = nn.functional.pixel_shuffle(out, 2)
        return out