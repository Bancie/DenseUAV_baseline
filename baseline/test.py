# -*- coding: utf-8 -*-
"""Feature extraction script for DenseUAV geo-localization evaluation.

Loads a trained dual-branch model, extracts L2-normalised embeddings for all
query and gallery images, and serialises them to a MATLAB ``.mat`` file for
downstream evaluation.

Configuration is read from CLI flags **and** merged with the ``opts.yaml``
produced during training (located in the current working directory).

Example:
    Extract drone-to-satellite (mode 1) features::

        python test.py --name my_run --checkpoint net_119.pth \\
            --test_dir /path/to/DenseUAV/test --mode 1 --batchsize 128

    Extract satellite-to-drone (mode 2) features::

        python test.py --name my_run --checkpoint net_119.pth \\
            --test_dir /path/to/DenseUAV/test --mode 2

Outputs:
    - ``pytorch_result_1.mat`` (mode 1) or ``pytorch_result_2.mat`` (mode 2)
      — MATLAB file containing ``gallery_f``, ``gallery_label``,
      ``gallery_path``, ``query_f``, ``query_label``, and ``query_path``.
    - ``gallery_name.txt`` — list of absolute gallery image paths.
    - ``query_name.txt``   — list of absolute query image paths.
"""

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tool.utils import load_network
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from datasets.queryDataset import Dataset_query,Query_transforms

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir',default='/home/dmmm/Dataset/DenseUAV/data_2022/test',type=str, help='./test_data')
parser.add_argument('--name', default='resnet', type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_119.pth', type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--mode',default='1', type=int,help='1:drone->satellite   2:satellite->drone')
parser.add_argument('--num_worker',default=4, type=int,help='1:drone->satellite   2:satellite->drone')
# parser.add_argument('--LPN', default=True, type=bool, help='')
# parser.add_argument('--block', default=2, type=int, help='')
parser.add_argument('--box_vis', default=False, type=int, help='')

opt = parser.parse_args()
print(opt.name)
###load config###
# load the training config
config_path = 'opts.yaml'
# with open(config_path, 'r') as stream:
#         config = yaml.load(stream)
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
for cfg,value in config.items():
    setattr(opt,cfg,value)

str_ids = opt.gpu_ids.split(',')
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# if len(gpu_ids)>0:
#     torch.cuda.set_device(gpu_ids[0])
#     cudnn.benchmark = True
if len(gpu_ids) > 0 and torch.cuda.is_available():
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_query_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        # Query_transforms(pad=10,size=opt.w),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



data_dir = test_dir

# # Requried two folder query_satellite and query_drone
# image_datasets_query = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_query_transforms) for x in ['query_satellite','query_drone']}

# # Required two folder gallery_satellite and gallery_drone
# image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone']}

# image_datasets = {**image_datasets_query, **image_datasets_gallery}

# # Required four folder gallery_satellite, gallery_drone, query_satellite, query_drone
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                          shuffle=False, num_workers=opt.num_worker) for x in ['gallery_satellite', 'gallery_drone','query_satellite','query_drone']}

# Mode 1 only: required folders query_drone and gallery_satellite
image_datasets = {
    'query_drone': datasets.ImageFolder(
        os.path.join(data_dir, 'query_drone'),
        data_query_transforms,
    ),
    'gallery_satellite': datasets.ImageFolder(
        os.path.join(data_dir, 'gallery_satellite'),
        data_transforms,
    ),
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_worker,
    )
    for x in ['query_drone', 'gallery_satellite']
}

use_gpu = torch.cuda.is_available()

def fliplr(img):
    """Flip a batch of images horizontally.

    Args:
        img (torch.Tensor): Image tensor of shape ``(N, C, H, W)``.

    Returns:
        torch.Tensor: Horizontally flipped image tensor of the same shape.
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    """Map a dataset split name to an integer view index.

    Args:
        name (str): Dataset split name, e.g. ``'gallery_satellite'``,
            ``'query_drone'``, or ``'query_street'``.

    Returns:
        int: ``1`` for satellite, ``2`` for street, ``3`` for drone,
            ``-1`` if the view cannot be determined.
    """
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model,dataloaders, view_index = 1):
    """Extract L2-normalised feature embeddings from a dataloader.

    For each mini-batch the feature is computed **twice** — once for the
    original image and once for its horizontal flip — and the two raw
    feature vectors are summed before L2-normalisation.  This test-time
    augmentation (TTA) typically improves retrieval accuracy.

    For 3-D part-based features of shape ``(N, D, P)`` the norm is scaled by
    ``sqrt(P)`` so that the full concatenated vector has unit cosine norm.

    Args:
        model (torch.nn.Module): Trained dual-branch model in eval mode.
        dataloaders (torch.utils.data.DataLoader): DataLoader for the target
            split (query or gallery).
        view_index (int, optional): Which model branch to use.  ``1`` invokes
            the satellite branch; ``3`` invokes the drone branch.
            Defaults to ``1``.

    Returns:
        torch.Tensor: CPU tensor of shape ``(N, D)`` containing the
            L2-normalised feature for every image in the dataloader.
    """
    # Use CPU when CUDA is not available (e.g. PyTorch CPU-only build); avoids AssertionError
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, _ = data
        batchsize = img.size()[0]
        count += batchsize
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            # Original (GPU-only): input_img = Variable(img.cuda())
            # New: move input to same device as model so it runs on CPU when CUDA is not available
            input_img = Variable(img.to(device))
            if view_index == 1:
                outputs, _ = model(input_img, None)
            elif view_index ==3:
                _, outputs = model(None, input_img)
            outputs = outputs[1]
            if i==0:ff = outputs
            else:ff += outputs
        # Normalise features
        if len(ff.shape)==3:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    """Parse integer class labels and file paths from an ImageFolder sample list.

    Args:
        img_path (list[tuple[str, int]]): List of ``(absolute_path, class_index)``
            tuples as produced by ``torchvision.datasets.ImageFolder.imgs``.
            The integer label is derived from the **parent folder name** (which
            must be a numeric string, e.g. ``'000123'``).

    Returns:
        tuple:
            - labels (list[int]): Integer class labels, one per image.
            - paths (list[str]): Corresponding absolute file paths.
    """
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model = load_network(opt)
print("Results for checkpoint: %s" % opt.checkpoint)
# model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

if opt.mode==1:
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
elif opt.mode==2:
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
else:
    raise Exception("opt.mode is not required")


which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))
print(query_name.split("_")[-1],"->",gallery_name.split("_")[-1])

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    with torch.no_grad():
        query_feature = extract_feature(model,dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model,dataloaders[gallery_name], which_gallery)

    # For street-view image, we use the avg feature as the final feature.

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result_{}.mat'.format(opt.mode),result)


    # print(opt.name)
    # result = 'result.txt'
    # os.system('python evaluate_gpu.py | tee -a %s'%result)
