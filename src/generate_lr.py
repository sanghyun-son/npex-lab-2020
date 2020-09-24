import os
import glob
import tqdm

import numpy as np
import imageio
import torch
from torch import cuda
from bicubic_pytorch import core

scales = [2, 4]
for split in ('train', 'eval'):
    img_list = sorted(glob.glob('../dataset/DIV2K_sub/{}/target/*.png'.format(split)))
    for img_name in tqdm.tqdm(img_list):
        hr = imageio.imread(img_name)
        hr = np.transpose(hr, (2, 0, 1))
        hr = np.ascontiguousarray(hr)
        hr = torch.from_numpy(hr)
        hr = hr.unsqueeze(0)
        hr = hr.float()
        if cuda.is_available():
            hr = hr.cuda()

        for s in scales:
            os.makedirs('../dataset/DIV2K_sub/{}/input_x{}'.format(split, s), exist_ok=True)
            lr = core.imresize(hr, scale=(1 / s))
            lr = lr.round()
            lr = lr.clamp(min=0, max=255)
            lr = lr.squeeze(0)
            lr = lr.byte()
            lr = lr.cpu()
            lr = lr.numpy()
            lr = np.transpose(lr, (1, 2, 0))
            imageio.imwrite(img_name.replace('target', 'input_x{}'.format(s)), lr)


