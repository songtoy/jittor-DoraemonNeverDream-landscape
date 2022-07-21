"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

class CustomDataset(Pix2pixDataset):
    """
    Dataset that loads images from directories
    Use option --label_dir, --image_dir, --instance_dir to specify the directories.
    The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        load_size = (1024 if is_train else 1024)
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=28)
        parser.set_defaults(contain_dontcare_label=True)
        parser.add_argument('--label_dir', type=str, default='', help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default='', help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='', help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        if opt.isTrain or opt.use_sea_style:
            image_dir = opt.image_dir
            image_paths = make_dataset(image_dir, recursive=False, read_cache=True)
            assert (len(label_paths) == len(image_paths)), 'The #images in %s and %s do not match. Is there something wrong?'
        else:
            image_paths = []
        if (len(opt.instance_dir) > 0):
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []
        return (label_paths, image_paths, instance_paths)
