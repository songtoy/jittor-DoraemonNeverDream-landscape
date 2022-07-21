"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
import models.networks as networks
from models.networks.diffaugment import DiffAugment
import util.util as util

class Pix2PixModel(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def start_grad(self):
        for param in self.netD.parameters():
            param.start_grad()

    def stop_grad(self):
        for param in self.netD.parameters():
            param.stop_grad()
        
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        (self.netG, self.netD, self.netE) = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt=self.opt)
            self.criterionFeat = jt.nn.L1Loss()
            if (not opt.no_vgg_loss):
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if True:
                self.KLDLoss = networks.KLDLoss()

    def execute(self, data, mode):
        (input_semantics, real_image) = self.preprocess_input(data)

        # set to different mode
        if (mode == 'generator'):
            (g_loss, generated) = self.compute_generator_loss(input_semantics, real_image)
            return (g_loss, generated)
        elif (mode == 'latent regressor'):
            (g_loss, generated) = self.compute_latent_loss(input_semantics, real_image)
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
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if True:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        (beta1, beta2) = (opt.beta1, opt.beta2)
        if opt.no_TTUR:
            (G_lr, D_lr) = (opt.lr, opt.lr)
        else:
            (G_lr, D_lr) = ((opt.lr / 2), (opt.lr * 2))
        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return (optimizer_G, optimizer_D)


    # save network
    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if True:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################
    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = (networks.define_D(opt) if opt.isTrain else None)
        if True: # and opt.isTrain:
            if opt.use_label_enc:
                netE = networks.define_LE(opt)
            else:
                netE = networks.define_E(opt)
        else:
            netE = None
        if ((not opt.isTrain) or opt.continue_train):
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if True: #and opt.isTrain:
                if opt.use_label_enc:
                    netE = util.load_network(netE, 'LabelE', opt.which_epoch, opt)
                else:
                    netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return (netG, netD, netE)


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
        # Never use instance map, annotate following code
        # if (not True):
        #     inst_map = data['instance']
        #     instance_edge_map = self.get_edges(inst_map)
        #     input_semantics = jt.contrib.concat((input_semantics, instance_edge_map), dim=1)
        return (input_semantics, data['image'])

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        (fake_image, KLD_loss) = self.generate_fake(input_semantics, real_image, compute_kld_loss=True)
        if True:
            G_losses['KLD'] = KLD_loss

        
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if (not self.opt.no_ganFeat_loss):
            num_D = len(pred_fake)
            GAN_Feat_loss = jt.zeros(shape=(1), dtype='float32')
            for i in range(num_D):
                num_intermediate_outputs = (len(pred_fake[i]) - 1)
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += ((unweighted_loss * self.opt.lambda_feat) / num_D)
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if (not self.opt.no_vgg_loss):
            G_losses['VGG'] = (self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg)
        return (G_losses, fake_image)

    def compute_latent_loss(self, input_semantics, real_image):
        # latent regressor
        # GAN_LOSS(G,D) + 
        assert True and self.opt.isTrain
        G_losses = {}
        z = jt.randn((input_semantics.shape[0], self.opt.z_dim), dtype="float32")
        fake_image = self.netG(input_semantics, z=z)
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if (not self.opt.no_ganFeat_loss):
            num_D = len(pred_fake)
            GAN_Feat_loss = jt.zeros(shape=(1), dtype='float32')
            for i in range(num_D):
                num_intermediate_outputs = (len(pred_fake[i]) - 1)
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += ((unweighted_loss * self.opt.lambda_feat) / num_D)
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if (not self.opt.no_vgg_loss):
            G_losses['VGG'] = (self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg)

        (mu, _) = self.netE(fake_image)
        latent_loss = self.opt.lambda_latent * self.criterionFeat(mu, z) # l1 loss
        G_losses['latent'] = latent_loss

        return (G_losses, fake_image)

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with jt.no_grad():
            (fake_image, _) = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            #fake_image.requires_grad_() # ?? why?
        
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        (mu, logvar) = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return (z, mu, logvar)

    def generate_fake(self, input_semantics, real_image=None, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if True and (self.opt.isTrain or self.opt.use_sea_style):
            #print(input_semantics.shape)
            #print(real_image.shape)
            (z, mu, logvar) = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = (self.KLDLoss(mu, logvar) * self.opt.lambda_kld)
        elif True and not self.opt.isTrain and self.opt.use_label_enc:
            (z, mu, logvar) = self.encode_z(input_semantics)

        fake_image = self.netG(input_semantics, z=z)
        assert ((not compute_kld_loss) or True), 'You cannot compute KLD loss if True == False'
        return (fake_image, KLD_loss)

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.contrib.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.contrib.concat([input_semantics, real_image], dim=1)
        fake_and_real = DiffAugment(jt.contrib.concat([fake_concat, real_concat], dim=0))
        
        discriminator_out = self.netD(fake_and_real)
        (pred_fake, pred_real) = self.divide_pred(discriminator_out)
        return (pred_fake, pred_real)

    def divide_pred(self, pred):
        if (type(pred) == list):
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:(tensor.shape[0] // 2)] for tensor in p])
                real.append([tensor[(tensor.shape[0] // 2):] for tensor in p])
        else:
            fake = pred[:(pred.shape[0] // 2)]
            real = pred[(pred.shape[0] // 2):]
        return (fake, real)

    def get_edges(self, t):
        edge = jt.zeros(shape=t.shape, dtype='uint8')
        edge[:, :, :, 1:] = (edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :(- 1)]))
        edge[:, :, :, :(- 1)] = (edge[:, :, :, :(- 1)] | (t[:, :, :, 1:] != t[:, :, :, :(- 1)]))
        edge[:, :, 1:, :] = (edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :(- 1), :]))
        edge[:, :, :(- 1), :] = (edge[:, :, :(- 1), :] | (t[:, :, 1:, :] != t[:, :, :(- 1), :]))
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp((0.5 * logvar))
        eps = jt.randn_like(std)
        return (eps.multiply(std) + mu)
