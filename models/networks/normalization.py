"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""
import jittor as jt
from jittor import init
import re
from jittor import nn
from models.networks.spectral_norm import spectral_norm

def get_nonspade_norm_layer(opt, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.shape[0]

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if ((subnorm_type == 'none') or (len(subnorm_type) == 0)):
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization


        if (subnorm_type == 'batch'):
            norm_layer = nn.BatchNorm(get_out_channel(layer), affine=True)
        elif (subnorm_type == 'sync_batch'):
            raise NotImplementedError('SynchronizedBatchNorm2d is needed. Please use mpi')
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif (subnorm_type == 'instance'):
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError(('normalization layer %s is not recognized' % subnorm_type))
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer



# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE

class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if (param_free_norm_type == 'instance'):
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif (param_free_norm_type == 'syncbatch'):
            raise NotImplementedError('SynchronizedBatchNorm2d is needed. Please use mpi')
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif (param_free_norm_type == 'batch'):
            self.param_free_norm = nn.BatchNorm(norm_nc, affine=False)
        else:
            raise ValueError(('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type))


        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        pw = (ks // 2)

        self.mlp_shared = nn.Sequential(nn.Conv(label_nc, nhidden, ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv(nhidden, norm_nc, ks, padding=pw)
        self.mlp_beta = nn.Conv(nhidden, norm_nc, ks, padding=pw)

    def execute(self, x, segmap):
        # Step.1 normalize the activations without param
        normalized = self.param_free_norm(x)

        # Step.2 calculate the param derived by segmap
        segmap = nn.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Step.3 denormalize the activations with param
        out = ((normalized * (1 + gamma)) + beta)
        return out
