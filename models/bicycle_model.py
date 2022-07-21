"""
Copyright (C) Zhou Songtao. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
import models.networks as networks
import util.util as util

class BicycleModel(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        assert True
        super().__init__()
        self.opt = opt
        (self.netG, self.netD, self.netE) = self.initialize_networks(opt)
        assert True and self.opt.isTrain
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt=self.opt)
            self.criterionFeat = jt.nn.L1Loss()
            self.criterionLatent = jt.nn.L1Loss()
            if (not opt.no_vgg_loss):
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            self.KLDLoss = networks.KLDLoss()

    def start_grad(self, for_discriminator=True):
        if for_discriminator:
            for param in self.netD.parameters():
                param.start_grad()
        else:
            for param in self.netE.parameters():
                param.start_grad()

    def stop_grad(self, for_discriminator=True):
        if for_discriminator:
            for param in self.netD.parameters():
                param.stop_grad()
        else:
            for param in self.netE.parameters():
                param.stop_grad()

    def create_optimizers(self, opt):
        assert opt.isTrain
        G_params = list(self.netG.parameters()) + list(self.netE.parameters())
        D_params = list(self.netD.parameters())
        (beta1, beta2) = (opt.beta1, opt.beta2)
        if opt.no_TTUR:
            (G_lr, D_lr) = (opt.lr, opt.lr.lr)
        else:
            (G_lr, D_lr) = ((opt.lr / 2), (opt.lr * 2))
        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return (optimizer_G, optimizer_D)

    # save network
    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = (networks.define_D(opt) if opt.isTrain else None)
        netE = networks.define_E(opt)
        if ((not opt.isTrain) or opt.continue_train):
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if True and opt.isTrain:
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



    def execute(self, data):
        (input_semantics, real_image) = self.preprocess_input(data)
        self.forward(input_semantics, real_image)
        return self.backward_EG()

    def compute_encoder_loss(self, input_semantics, real_image):
        E_losses = {}
        (image_mu, image_var) = self.netE(real_image)
        E_losses['E_mu'] = self.criterionFeat(image_mu, label_mu)
        E_losses['E_var'] = self.criterionFeat(image_var, label_var)
        return E_losses

    def encode_z(self, real_image):
        (mu, logvar) = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return (z, mu, logvar)

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.contrib.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.contrib.concat([input_semantics, real_image], dim=1)
        fake_and_real = jt.contrib.concat([fake_concat, real_concat], dim=0)
        
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

    def forward(self, input_semantics, real_image): 
        assert self.opt.isTrain and real_image is not None
        # get real images
        half_size = self.opt.batchSize // 2
        # A1, B1 for encoded (cVAE-GAN); A2, B2 for random (cLR-GAN)
        # A for label, B for image
        self.real_A_encoded = input_semantics[0:half_size]
        self.real_B_encoded = real_image[0:half_size]
        self.real_A_random = input_semantics[half_size:]
        self.real_B_random = real_image[half_size:]
        # get encoded z from real image
        self.z_encoded, self.mu, self.logvar = self.encode_z(self.real_B_encoded) 
        # get random z
        self.z_random = jt.randn((self.real_A_encoded.shape[0], self.opt.z_dim), dtype="float32")
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_random, self.z_random)

        # compute z_predict
        with jt.no_grad():
            self.mu2, self.logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self):
        real_A = jt.concat([self.real_A_encoded, self.real_A_random], dim=0)
        fake_B = jt.concat([self.fake_B_encoded.detach(), self.fake_B_random.detach()], dim=0)
        # Fake, stop backprop to the generator by detaching fake_B
        real_B = jt.concat([self.real_B_encoded, self.real_B_random], dim=0)
        (pred_fake, pred_real) = self.discriminate(real_A, fake_B, real_B)
        D_losses = {}
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def backward_EG(self):
        G_losses = {}
        # 1, G(A) should fool D
        (pred_fake, pred_real) = self.discriminate(self.real_A_encoded, self.fake_B_encoded, self.real_B_encoded)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        (pred_fake_lr, pred_real_lr) = self.discriminate(self.real_A_random, self.fake_B_random, self.real_B_random)
        G_losses['GAN2'] = self.criterionGAN(pred_fake_lr, True, for_discriminator=False)

        # 2. GANFeat loss for encoded image
        if (not self.opt.no_ganFeat_loss):
            num_D = len(pred_fake)
            GAN_Feat_loss = jt.zeros(shape=(1), dtype='float32')
            for i in range(num_D):
                num_intermediate_outputs = (len(pred_fake[i]) - 1)
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += ((unweighted_loss * self.opt.lambda_feat) / num_D)
            G_losses['GAN_Feat'] = GAN_Feat_loss
        # 3. vgg loss for encoded image
        if (not self.opt.no_vgg_loss):
            G_losses['VGG'] = (self.criterionVGG(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_vgg)        

        # 4. KL loss
        G_losses['KLD'] = (self.KLDLoss(self.mu, self.logvar) * self.opt.lambda_kld)
        
        # 5, reconstruction |(E(G(A, z_random)))-z_random|
        G_losses['latent'] = self.criterionLatent(self.mu2, self.z_random) * self.opt.lambda_latent
        return G_losses, self.fake_B_encoded
