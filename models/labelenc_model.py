"""
Copyright (C) Zhou Songtao. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import nn
import models.networks as networks
import util.util as util
import jittor.transform as transforms
class LabelEncModel(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def stop_grad(self):
        for param in self.netE.parameters():
            param.stop_grad()
        
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        (self.netE, self.labelE) = self.initialize_networks(opt)
        self.stop_grad() # stop grad for original encoder
        if opt.isTrain:
            self.criterionFeat = jt.nn.MSELoss()
            self.KLDLoss = networks.KLDLoss()

    def execute(self, data, mode):
        (input_semantics, real_image) = self.preprocess_input(data)

        # set to different mode
        if (mode == 'generator'):
            (g_loss, generated) = self.compute_generator_loss(input_semantics, real_image)
            return (g_loss, generated)
        elif (mode == 'discriminator'):
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif (mode == 'encode_only'):
            (z, mu, logvar) = self.encode_z(real_image)
            return (mu, logvar)
        elif (mode == 'inference'):
            with jt.no_grad():
                (fake_image, _) = self.generate_fake(input_semantics, real_image)
            return fake_image
        elif (mode == 'label_enc_only'):
            e_loss = self.compute_encoder_loss(input_semantics, real_image)
            return e_loss
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        if opt.train_enc:
            E_params = list(self.labelE.parameters())
            (beta1, beta2) = (opt.beta1, opt.beta2)
            E_lr = opt.lr
            optimizer_E = jt.optim.Adam(E_params, lr=E_lr, betas=(beta1, beta2))
            return optimizer_E
            
    # save network
    def save(self, epoch):
        util.save_network(self.labelE, 'LabelE', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################
    def initialize_networks(self, opt):
        assert True
        netE = networks.define_E(opt)
        netLabelE = networks.define_LE(opt)
        netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        if ((not opt.isTrain) or opt.continue_train):
            netLabelE = util.load_network(netLabelE, 'LabelE', opt.which_epoch, opt)
        return (netE, netLabelE)



    # preprocess the input, transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        label_map = data['label']
        (bs, _, h, w) = label_map.shape
        nc = ((self.opt.label_nc + 1) if True else self.opt.label_nc)
        # assure dontcare_label is label_nc + 1

        input_label =  jt.zeros(shape=(bs, nc, h, w), dtype='float32')
        input_semantics = input_label.scatter_(1, label_map, jt.ones(shape=(bs, nc, h, w), dtype='float32'))
        return (input_semantics, data['image'])

    def compute_encoder_loss(self, input_semantics, real_image):
        E_losses = {}
        (image_mu, image_var) = self.netE(real_image)
        (label_mu, label_var) = self.labelE(input_semantics)
        E_losses['KLD'] = self.KLDLoss(label_mu, label_var)
        E_losses['E_mu'] = self.criterionFeat(image_mu, label_mu)
        E_losses['E_var'] = self.criterionFeat(image_var, label_var)
        return E_losses

class ResLabelEncModel(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def stop_grad(self):
        for param in self.netE.parameters():
            param.stop_grad()
        
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        (self.netE, self.labelE) = self.initialize_networks(opt)
        self.stop_grad() # stop grad for original encoder
        if opt.isTrain:
            self.criterionFeat = jt.nn.MSELoss()

    def execute(self, data, mode):
        (input_semantics, real_image) = self.preprocess_input(data)

        # set to different mode
        if (mode == 'generator'):
            (g_loss, generated) = self.compute_generator_loss(input_semantics, real_image)
            return (g_loss, generated)
        elif (mode == 'discriminator'):
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif (mode == 'encode_only'):
            (z, mu, logvar) = self.encode_z(real_image)
            return (mu, logvar)
        elif (mode == 'inference'):
            with jt.no_grad():
                (fake_image, _) = self.generate_fake(input_semantics, real_image)
            return fake_image
        elif (mode == 'label_enc_only'):
            e_loss = self.compute_encoder_loss(input_semantics, real_image)
            return e_loss
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        if opt.train_enc:
            E_params = list(self.labelE.parameters())
            (beta1, beta2) = (opt.beta1, opt.beta2)
            E_lr = opt.lr
            optimizer_E = jt.optim.Adam(E_params, lr=E_lr, betas=(beta1, beta2))
            return optimizer_E
            
    # save network
    def save(self, epoch):
        util.save_network(self.labelE, 'ResLabelE', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################
    def initialize_networks(self, opt):
        assert True
        netE = networks.define_E(opt)
        netLabelE = networks.define_RLE(opt)
        netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        if ((not opt.isTrain) or opt.continue_train):
            netLabelE = util.load_network(netLabelE, 'ResLabelE', opt.which_epoch, opt)
        return (netE, netLabelE)



    # preprocess the input, transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        return (data['label'], data['image'])

    def compute_encoder_loss(self, input_semantics, real_image):
        E_losses = {}
        (image_mu, image_var) = self.netE(real_image)
        (label_mu, label_var) = self.labelE(input_semantics)
        E_losses['E_mu'] = self.criterionFeat(image_mu, label_mu)
        E_losses['E_var'] = self.criterionFeat(image_var, label_var)
        return E_losses