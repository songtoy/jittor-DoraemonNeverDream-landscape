# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import jittor as jt


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = jt.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1])
    translation_y = jt.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1])
    grid_batch, grid_x, grid_y = jt.meshgrid(
        jt.arange(x.size(0), dtype=jt.long),
        jt.arange(x.size(2), dtype=jt.long),
        jt.arange(x.size(3), dtype=jt.long),
    )
    grid_x = jt.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = jt.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = jt.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1])
    offset_y = jt.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1])
    grid_batch, grid_x, grid_y = jt.meshgrid(
        jt.arange(x.size(0), dtype=jt.long),
        jt.arange(cutout_size[0], dtype=jt.long),
        jt.arange(cutout_size[1], dtype=jt.long),
    )
    grid_x = jt.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = jt.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = jt.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}