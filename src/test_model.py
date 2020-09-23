import os
from os import path
import glob
import argparse
import importlib

import imageio
import numpy as np

import torch
from torch import cuda

def main() -> None:
    parser = argparse.ArgumentParser('NPEX Image Restoration Lab')
    parser.add_argument('-i', '--input', type=str, default='example')
    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('-m', '--model', type=str, default='simple')
    parser.add_argument('-p', '--pretrained', type=str)
    cfg = parser.parse_args()

    imgs = sorted(glob.glob(path.join(cfg.input, '*.png')))
    os.makedirs(cfg.output, exist_ok=True)

    # CUDA configuration
    if cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net_module = importlib.import_module('.' + cfg.model, package='model')
    net = net_module.Net()
    net = net.to(device)
    print(net)

    # Implement below
    return

if __name__ == '__main__':
    main()
