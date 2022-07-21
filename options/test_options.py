
import jittor as jt
from jittor import init
from jittor import nn
"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""
from .base_options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--output_path', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float('inf'), help='how many test images to run')
        parser.add_argument('--use_sea_style', action='store_true', help='if specified, use sea style')
        parser.set_defaults(preprocess_mode='fixed', crop_size=512, load_size=1024, display_winsize=256)
        parser.set_defaults(name='pretrain-model')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
