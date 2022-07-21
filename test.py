"""
Copyright (C) Zhou Songtao. All rights reserved.
Translate from 2019 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0 license.
"""

import jittor as jt
from jittor import init
from jittor import nn
import os
from collections import OrderedDict
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

jt.flags.use_cuda = 1

opt = TestOptions().parse()
opt.label_dir = opt.input_path

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

#web_dir = os.path.join(opt.results_dir, opt.name, ('%s_%s' % (opt.phase, opt.which_epoch)))
web_dir = opt.results_dir

webpage = html.HTML(web_dir, ('Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch)))

for (i, data_i) in enumerate(dataloader):
    if ((i * opt.batchSize) >= opt.how_many):
        break
    generated = model(data_i, mode='inference')
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print(('process image... %s' % img_path[b]))
        visuals = OrderedDict([('synthesized_image', generated[b]),])
        visualizer.save_images(webpage, visuals, img_path[b:(b + 1)])
        
webpage.save()
