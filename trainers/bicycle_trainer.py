"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
from models.bicycle_model import BicycleModel


class BicycleTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = BicycleModel(opt)
        self.generated = None
        if opt.isTrain:
            (self.optimizer_G, self.optimizer_D) = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_and_encoder_one_step(self, data):
        self.pix2pix_model.stop_grad() # stop grad for Discriminator
        (g_losses, generated) = self.pix2pix_model(data)
        g_loss = sum(g_losses.values()).mean()
        #print("--------------------------")
        #print(sum([param.data.mean() for param in self.pix2pix_model.netE.parameters()]).mean())
        self.optimizer_G.step(g_loss)
        #print(sum([param.data.mean() for param in self.pix2pix_model.netE.parameters()]).mean())
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.pix2pix_model.start_grad() # start grad for Discriminator
        d_losses = self.pix2pix_model.backward_D()
        d_loss = sum(d_losses.values()).mean()
        self.optimizer_D.step(d_loss)
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model.save(epoch)

    def update_learning_rate(self, epoch):
        if (epoch > self.opt.niter):
            lrd = (self.opt.lr / self.opt.niter_decay)
            new_lr = (self.old_lr - lrd)
        else:
            new_lr = self.old_lr
        if (new_lr != self.old_lr):
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = (new_lr / 2)
                new_lr_D = (new_lr * 2)
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print(('update learning rate: %f -> %f' % (self.old_lr, new_lr)))
            self.old_lr = new_lr