# -*- coding: utf-8 -*-
"""Script to report the MACs and parameter count of a trained DenseUAV model.

Loads a model checkpoint specified via CLI flags (merged with the run's
``opts.yaml``), then uses ``thop`` to profile a single forward pass and prints
the Multiply-Accumulate operations (MACs) and total parameter count.

Example:
    Measure FLOPs and params for a 224×224 input::

        python tool/get_model_flops_params.py \\
            --name my_run --checkpoint net_119.pth \\
            --test_h 224 --test_w 224

    The script must be run from the ``baseline/`` directory so that
    ``opts.yaml`` and the checkpoint are resolvable.

Outputs:
    Prints to stdout::

        model MACs=<value>, Params=<value>
"""

import sys
sys.path.append("../../")
import yaml
import argparse

from tool.utils import load_network, calc_flops_params


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='resnet',
                    type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_119.pth',
                    type=str, help='save model path')
parser.add_argument('--test_h', default=224, type=int, help='height')
parser.add_argument('--test_w', default=224, type=int, help='width')
opt = parser.parse_args()

config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
for cfg, value in config.items():
    setattr(opt, cfg, value)

model = load_network(opt)
model = model.eval()

# Compute MACs with thop
macs, params = calc_flops_params(
    model, (1, 3, opt.test_h, opt.test_w), (1, 3, opt.test_h, opt.test_w))
print("model MACs={}, Params={}".format(macs, params))
