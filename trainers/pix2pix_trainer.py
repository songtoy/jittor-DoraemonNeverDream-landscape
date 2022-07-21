"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
import math
from models.pix2pix_model import Pix2PixModel
from models.labelenc_model import LabelEncModel,ResLabelEncModel

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        self.generated = None
        if opt.isTrain:
            (self.optimizer_G, self.optimizer_D) = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr
            self.eta_min = opt.eta_min
            self.T_max = opt.Tmax
            self.last_epoch = 0

    def run_generator_one_step(self, data):
        #self.pix2pix_model.stop_grad() # stop grad for Discriminator
        if self.opt.latent_reg:
            (g_losses, generated) = self.pix2pix_model(data, mode='latent regressor')
        else:
            (g_losses, generated) = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        self.optimizer_G.step(g_loss)

        # Grad
        grad_record = []
        for p in self.pix2pix_model.netE.parameters():
            grad = p.opt_grad(self.optimizer_G)
            grad_record.append(grad.detach().norm().mean())
        g_losses['grad-E'] = jt.Var(sum(grad_record) / len(grad_record))

        self.g_losses = g_losses
        self.generated = generated
        #self.pix2pix_model.start_grad() # start grad for Discriminator

    def run_discriminator_one_step(self, data):
        d_losses = self.pix2pix_model(data, mode='discriminator')
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


    def get_lr(self, base_lr, now_lr):
        # cosine_annealing_lr
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return (now_lr + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2)
        return  ((1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (now_lr - self.eta_min) + self.eta_min)


    def update_learning_rate(self, epoch):
        if self.opt.cosine_lr:
            self.last_epoch += 1
            new_lr = self.get_lr(self.opt.lr, self.old_lr)
            print(f"change lr to {new_lr}")
        elif (epoch > self.opt.niter):
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


class LabelEncoderTrainer():
    """
    Trainer for subencoder, which replace the original style encoder.
    It is used to provide the style info with label map.
    """
    def __init__(self, opt):
        assert opt.train_enc
        self.opt = opt
        self.pix2pix_model = LabelEncModel(opt)
        self.generated = None
        if opt.isTrain:
            self.optimizer_E = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr
            self.eta_min = opt.eta_min
            self.T_max = opt.Tmax
            self.last_epoch = 0
    
    def run_label_encoder_one_step(self, data):
        e_losses = self.pix2pix_model(data, mode='label_enc_only')
        e_loss = sum(e_losses.values()).mean()
        self.optimizer_E.step(e_loss)
        self.e_losses = e_losses

    def get_latest_losses(self):
        return {**self.e_losses}

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model.save(epoch)

    def get_lr(self, base_lr, now_lr):
        # cosine_annealing_lr
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return (now_lr + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2)
        return  ((1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (now_lr - self.eta_min) + self.eta_min)

    def update_learning_rate(self, epoch):
        if self.opt.cosine_lr:
            self.last_epoch += 1
            new_lr = self.get_lr(self.opt.lr, self.old_lr)
            print(f"change lr to {new_lr}")
        elif (epoch > self.opt.niter):
            lrd = (self.opt.lr / self.opt.niter_decay)
            new_lr = (self.old_lr - lrd)
        else:
            new_lr = self.old_lr
        if (new_lr != self.old_lr):
            for param_group in self.optimizer_E.param_groups:
                param_group['lr'] = new_lr
            print(('update learning rate: %f -> %f' % (self.old_lr, new_lr)))
            self.old_lr = new_lr

class ResLabelEncoderTrainer():
    """
    Trainer for subencoder, which replace the original style encoder.
    It is used to provide the style info with label map.
    This subencoder uses the resnet50 as the backbone.
    """
    def __init__(self, opt):
        assert opt.train_enc
        self.opt = opt
        self.pix2pix_model = ResLabelEncModel(opt)
        self.generated = None
        if opt.isTrain:
            self.optimizer_E = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr
    
    def run_label_encoder_one_step(self, data):
        e_losses = self.pix2pix_model(data, mode='label_enc_only')
        e_loss = sum(e_losses.values()).mean()
        self.optimizer_E.step(e_loss)
        self.e_losses = e_losses

    def get_latest_losses(self):
        return {**self.e_losses}

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
            for param_group in self.optimizer_E.param_groups:
                param_group['lr'] = new_lr
            print(('update learning rate: %f -> %f' % (self.old_lr, new_lr)))
            self.old_lr = new_lr