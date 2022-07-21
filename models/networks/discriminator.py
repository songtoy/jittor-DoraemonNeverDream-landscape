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
import util.util as util

class MultiscaleDiscriminator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        (opt, _) = parser.parse_known_args()
        subnetD = util.find_class_in_module((opt.netD_subarch + 'discriminator'), 'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = nn.Sequential()
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.layers.add_module(('discriminator_%d' % i), subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if (subarch == 'n_layer'):
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError(('unrecognized discriminator subarchitecture %s' % subarch))
        return netD

    def downsample(self, input):
        return nn.avg_pool2d(input, kernel_size=3, stride=2, padding=(1, 1), count_include_pad=False)

    def execute(self, input):
        result = []
        get_intermediate_features = (not self.opt.no_ganFeat_loss)
        for (name, D) in self.layers.items():
            out = D(input)
            if (not get_intermediate_features):
                out = [out]
            result.append(out)
            input = self.downsample(input)
        return result

class NLayerDiscriminator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 4
        padw = int(np.ceil(((kw - 1.0) / 2)))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv(input_nc, nf, kw, stride=2, padding=padw), nn.LeakyReLU(scale=0.2)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min((nf * 2), 512)
            stride = (1 if (n == (opt.n_layers_D - 1)) else 2)
            sequence += [[norm_layer(nn.Conv(nf_prev, nf, kw, stride=stride, padding=padw, bias=False)), nn.LeakyReLU(scale=0.2)]]

        sequence += [[nn.Conv(nf, 1, kw, stride=1, padding=padw)]]

        self.models = nn.ModuleList()
        for n in range(len(sequence)):
            self.models.add_module(('model' + str(n)), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = (opt.label_nc + opt.output_nc)
        if True:
            input_nc += 1
        if (not True):
            input_nc += 1
        return input_nc

    def execute(self, input):
        results = [input]
        for submodel in self.models.children():
            intermediate_output = submodel(results[(- 1)])
            results.append(intermediate_output)
        get_intermediate_features = (not self.opt.no_ganFeat_loss)
        if get_intermediate_features:
            return results[1:]
        else:
            return results[(- 1)]
