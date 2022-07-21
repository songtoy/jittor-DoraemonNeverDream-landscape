"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import nn
from models.networks.architecture import VGG19

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.gan_mode = gan_mode
        self.opt = opt
        if (gan_mode == 'ls'):
            self.loss = nn.MSELoss()
        elif (gan_mode == 'original'):
            self.loss = nn.BCEWithLogitsLoss()
        elif (gan_mode == 'w'):
            pass
        elif (gan_mode == 'hinge'):
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if (self.real_label_tensor is None):
                self.real_label_tensor = jt.float32(self.real_label)
                self.real_label_tensor.requires_grad = False
            return self.real_label_tensor.expand_as(input)
        else:
            if (self.fake_label_tensor is None):
                self.real_label_tensor = jt.float32(self.fake_label)
                self.fake_label_tensor.requires_grad = False
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if (self.zero_tensor is None):
            self.zero_tensor = jt.float32(0)
            self.zero_tensor.requires_grad = False
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if (self.gan_mode == 'original'):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)
        elif (self.gan_mode == 'ls'):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)
        elif (self.gan_mode == 'hinge'):
            if for_discriminator:
                if target_is_real:
                    minval = jt.minimum((input - 1), self.get_zero_tensor(input))
                    loss = (- jt.mean(minval))
                else:
                    minval = jt.minimum(((- input) - 1), self.get_zero_tensor(input))
                    loss = (- jt.mean(minval))
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = (- jt.mean(input))
            return loss
        elif target_is_real:
            return (- input.mean())
        else:
            return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[(- 1)]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = (1 if (len(loss_tensor.shape) == 0) else loss_tensor.shape[0])
                new_loss = jt.mean(loss_tensor.view((bs, (- 1))), dim=1)
                loss += new_loss
            return (loss / len(input))
        else:
            return self.loss(input, target_is_real, for_discriminator)

class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [(1.0 / 32), (1.0 / 16), (1.0 / 8), (1.0 / 4), 1.0]

    def execute(self, x, y):
        (x_vgg, y_vgg) = (self.vgg(x), self.vgg(y))
        loss = 0
        for i in range(len(x_vgg)):
            loss += (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
        return loss

class KLDLoss(nn.Module):

    def execute(self, mu, logvar):
        return ((- 0.5) * jt.sum((((1 + logvar) - mu.pow(2)) - logvar.exp())))
