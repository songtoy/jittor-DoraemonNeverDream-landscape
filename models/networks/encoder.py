"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from jittor.models import resnet
class ConvEncoder(BaseNetwork):
    ' Same architecture as the image discriminator '

    def __init__(self, opt):
        super().__init__()
        kw = 3
        pw = int(np.ceil(((kw - 1.0) / 2)))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv(3, ndf, kw, stride=2, padding=pw, bias=False))
        self.layer2 = norm_layer(nn.Conv((ndf * 1), (ndf * 2), kw, stride=2, padding=pw, bias=False))
        self.layer3 = norm_layer(nn.Conv((ndf * 2), (ndf * 4), kw, stride=2, padding=pw, bias=False))
        self.layer4 = norm_layer(nn.Conv((ndf * 4), (ndf * 8), kw, stride=2, padding=pw, bias=False))
        self.layer5 = norm_layer(nn.Conv((ndf * 8), (ndf * 8), kw, stride=2, padding=pw, bias=False))
        if (opt.crop_size >= 256):
            self.layer6 = norm_layer(nn.Conv((ndf * 8), (ndf * 8), kw, stride=2, padding=pw, bias=False))
        self.so = s0 = 4
        self.fc_mu = nn.Linear((((ndf * 8) * s0) * s0), 256)
        self.fc_var = nn.Linear((((ndf * 8) * s0) * s0), 256)
        self.actvn = nn.LeakyReLU(scale=0.2)
        self.opt = opt

    def execute(self, x):
        if ((x.shape[2] != 256) or (x.shape[3] != 256)):
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if (self.opt.crop_size >= 256):
            x = self.layer6(self.actvn(x))
            
        x = self.actvn(x)
        x = x.view((x.shape[0], (- 1)))
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return (mu, logvar)


class LabelEncoder(BaseNetwork):
    ' Same architecture as the image discriminator '

    def __init__(self, opt):
        super().__init__()
        kw = 3
        pw = int(np.ceil(((kw - 1.0) / 2)))
        ndf = opt.ngf // 2
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv(29, ndf, kw, stride=2, padding=pw, bias=False))
        self.layer2 = norm_layer(nn.Conv((ndf * 1), (ndf * 2), kw, stride=2, padding=pw, bias=False))
        self.layer3 = norm_layer(nn.Conv((ndf * 2), (ndf * 4), kw, stride=2, padding=pw, bias=False))
        self.layer4 = nn.AvgPool2d(2)
        self.so = s0 = 16
        self.fc_mu = nn.Linear((((ndf * 4) * s0) * s0), 256)
        self.fc_var = nn.Linear((((ndf * 4) * s0) * s0), 256)
        self.actvn = nn.LeakyReLU(scale=0.2)
        self.opt = opt

    def execute(self, x):
        if ((x.shape[2] != 256) or (x.shape[3] != 256)):
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        
        x = self.layer4(x)
            
        x = self.actvn(x)
        x = x.view((x.shape[0], (- 1)))
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return (mu, logvar)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

class ResLabelEncoder(BaseNetwork):
    '''
    Same architecture as the encoder of bicyclegan
    '''
    def __init__(self, opt):
        super().__init__()
        resnet18_model = resnet.Resnet18()
        self.embedding_to_3_channels=nn.Embedding(opt.label_nc, 3)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:(- 3)])
        self.pooling = nn.Pool(kernel_size=8, stride=8, padding=0, op='mean')
        self.fc_mu = nn.Linear(256, 256)
        self.fc_logvar = nn.Linear(256, 256)
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, labelimg):
        # labelimg should be a tensor of shape (batch_size, 1, 512, 384)
        
        out=jt.squeeze(labelimg,1)
        out=self.embedding_to_3_channels(out).transpose(1,3)
        out = self.feature_extractor(out)
        out = self.pooling(out)
        out = jt.reshape(out, [out.shape[0], (- 1)])
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return (mu, logvar)