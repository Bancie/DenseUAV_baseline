"""GPS-distance-based evaluation metrics (SDM@K and MA@K) for DenseUAV.

Loads pre-computed query/gallery features from a ``.mat`` file (produced by
``test.py``) together with GPS coordinates from ``Dense_GPS_ALL.txt``, then
computes two distance-aware retrieval metrics:

- **SDM@K** (Soft Distance Metric at K): weighted exponential decay score
  averaged over the top-K retrieved gallery items.
- **MA@K** (Meter Accuracy at K): fraction of queries whose top-1 retrieved
  gallery is within K metres of the true location.

Example:
    Evaluate drone-to-satellite retrieval (mode 1)::

        python evaluateDistance.py \\
            --root_dir /path/to/DenseUAV/ \\
            --mode 1 --K 1 3 5 10 --M 5000

    Evaluate satellite-to-drone retrieval (mode 2)::

        python evaluateDistance.py --root_dir /path/to/DenseUAV/ --mode 2

Outputs:
    - Prints SDM@K values for each K in ``--K`` to stdout.
    - ``SDM@K(1,100).json`` — full SDM@K curve from K=1 to K=100.
    - ``MA@K(1,100)``       — full MA@K curve from K=1 to K=100 (metres).
"""

import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import math

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
# parser.add_argument('--query_index', default=10, type=int, help='test_image_index')
parser.add_argument(
    '--root_dir', default='/home/dmmm/Dataset/DenseUAV/data_2022/', type=str, help='./test_data')
parser.add_argument('--K', default=[1, 3, 5, 10], type=str, help='./test_data')
parser.add_argument('--M', default=5e3, type=str, help='./test_data')
parser.add_argument('--mode', default="1", type=str,
                    help='1:drone->satellite 2:satellite->drone')
opts = parser.parse_args()

opts.config = os.path.join(opts.root_dir, "Dense_GPS_ALL.txt")
opts.test_dir = os.path.join(opts.root_dir, "test")
configDict = {}
with open(opts.config, "r") as F:
    context = F.readlines()
    for line in context:
        splitLineList = line.split(" ")
        configDict[splitLineList[0].split("/")[-2]] = [float(splitLineList[1].split("E")[-1]),
                                                       float(splitLineList[2].split("N")[-1])]

if opts.mode == "1":
    gallery_name = 'gallery_satellite'
    query_name = 'query_drone'
else:
    gallery_name = 'gallery_drone'
    query_name = 'query_satellite'

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [
    gallery_name, query_name]}


#####################################################################
# Show result
def imshow(path, title=None):
    """Display an image from disk using Matplotlib.

    Args:
        path (str): Absolute or relative path to the image file.
        title (str, optional): Title to display above the image.
            Defaults to ``None``.
    """
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


######################################################################
if opts.mode == "1":
    result = scipy.io.loadmat('pytorch_result_1.mat')
else:
    result = scipy.io.loadmat('pytorch_result_2.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


#######################################################################
# sort the images and return topK index
def sort_img(qf, ql, gf, gl, K):
    """Rank gallery images by cosine similarity to a query and return the top-K indices.

    Junk gallery entries (label ``-1``) are removed from the ranked list
    before the top-K selection.

    Args:
        qf (torch.Tensor): Query feature vector of shape ``(D,)``, on GPU.
        ql (int): Integer class label of the query (used to identify positive
            matches; not used for filtering here).
        gf (torch.Tensor): Gallery feature matrix of shape ``(N, D)``, on GPU.
        gl (numpy.ndarray): Integer class labels of all gallery images,
            shape ``(N,)``.
        K (int): Number of top results to return.

    Returns:
        numpy.ndarray: Indices of the top-K gallery images (after junk
            removal) ranked by descending cosine similarity, shape ``(K,)``.
    """
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl == -1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index[:K]


def getLatitudeAndLongitude(imgPath):
    """Look up the GPS coordinates for one or more image paths.

    Coordinates are retrieved from the pre-loaded ``configDict`` which maps
    each location folder name (e.g. ``'000123'``) to ``[longitude_E,
    latitude_N]``.

    Args:
        imgPath (str | list[str]): A single image path string or a list of
            image path strings.  The second-to-last path component is used as
            the lookup key.

    Returns:
        list | list[list]: For a single path, a two-element list
            ``[longitude_E, latitude_N]``.  For a list of paths, a list of
            such two-element lists.
    """
    if isinstance(imgPath, list):
        posInfo = [configDict[p.split("/")[-2]] for p in imgPath]
    else:
        posInfo = configDict[imgPath.split("/")[-2]]
    return posInfo


def euclideanDistance(query, gallery):
    """Compute the Euclidean distance between a query position and each gallery position.

    Operates in the raw coordinate space (longitude/latitude degrees or any
    2-D Euclidean space).  For geodesic (metre) distances use
    :func:`latlog2meter`.

    Args:
        query (array_like): Query position of shape ``(2,)`` — ``[lon, lat]``.
        gallery (array_like): Gallery positions of shape ``(K, 2)``.

    Returns:
        numpy.ndarray: 1-D array of shape ``(K,)`` containing the Euclidean
            distance from the query to each of the K gallery positions.
    """
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=np.bool8)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance


def evaluateSingle(distance, K):
    """Compute the SDM score for a single query given top-K gallery distances.

    The score is a weighted mean of exponential decay values::

        score = sum(w_k * exp(-d_k * M)) / sum(w_k)

    where ``w_k = 1 - k/K`` (linearly decreasing rank weights) and ``M`` is a
    scale factor (``opts.M``).

    Args:
        distance (numpy.ndarray): Array of shape ``(K,)`` containing the
            coordinate-space distances from the query to each of the top-K
            gallery samples.
        K (int): Number of top retrieved results to include in the score.

    Returns:
        float: SDM score in ``[0, 1]``; higher is better.
    """
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0, K, 1))/K
    # m1 = distance / maxDistance
    m2 = 1 / np.exp(distance*opts.M)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


def latlog2meter(lata, loga, latb, logb):
    """Compute the geodesic distance between two GPS coordinates using the Haversine formula.

    Args:
        lata (float): Latitude of point A in decimal degrees.
        loga (float): Longitude of point A in decimal degrees.
        latb (float): Latitude of point B in decimal degrees.
        logb (float): Longitude of point B in decimal degrees.

    Returns:
        float: Great-circle distance between the two points in **metres**,
            assuming an Earth radius of 6378.137 km (WGS-84 equatorial).
    """
    # Note: variable names follow the convention lon=longitude, lat=latitude
    EARTH_RADIUS =6378.137
    PI = math.pi
    # Convert degrees to radians
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance


def evaluate_SDM(indexOfTopK, queryIndex, K):
    """Compute the SDM@K score for a single query.

    Looks up GPS coordinates for the query and its top-K gallery results,
    computes coordinate-space Euclidean distances, and returns the SDM score
    via :func:`evaluateSingle`.

    Args:
        indexOfTopK (numpy.ndarray): Gallery indices of the top-K ranked
            results (output of :func:`sort_img`).
        queryIndex (int): Index of the query image within
            ``image_datasets[query_name].imgs``.
        K (int): Number of top results to include in the score.

    Returns:
        float: SDM@K score for this query; higher is better.
    """
    query_path, _ = image_datasets[query_name].imgs[queryIndex]
    galleryTopKPath = [image_datasets[gallery_name].imgs[i][0]
                       for i in indexOfTopK[:K]]
    # get position information including latitude and longitude
    queryPosInfo = getLatitudeAndLongitude(query_path)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath)
    # compute Euclidean distance of query and gallery
    distance = euclideanDistance(queryPosInfo, galleryTopKPosInfo)
    # compute single query evaluate result
    P = evaluateSingle(distance, K)
    return P


def evaluate_MA(indexOfTop1, queryIndex):
    """Compute the geodesic distance (metres) between a query and its top-1 gallery match.

    Used to build the MA@K curve: a query contributes to MA@K if the returned
    distance is less than K metres.

    Args:
        indexOfTop1 (int): Gallery index of the top-1 ranked result for this
            query.
        queryIndex (int): Index of the query image within
            ``image_datasets[query_name].imgs``.

    Returns:
        float: Haversine distance in metres between the query GPS position
            and the top-1 gallery GPS position.
    """
    query_path, _ = image_datasets[query_name].imgs[queryIndex]
    galleryTopKPath = image_datasets[gallery_name].imgs[indexOfTop1][0]
    # get position information including latitude and longitude
    queryPosInfo = getLatitudeAndLongitude(query_path)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath)
    # get real distance
    distance_meter = latlog2meter(queryPosInfo[1],queryPosInfo[0],galleryTopKPosInfo[1],galleryTopKPosInfo[0])
    return distance_meter



indexOfTopK_list = []
for i in range(len(query_label)):
    indexOfTopK = sort_img(
        query_feature[i], query_label[i], gallery_feature, gallery_label, 100)
    indexOfTopK_list.append(indexOfTopK)

SDM_dict = {}
for K in tqdm(range(1, 101, 1)):
    metric = 0
    for i in range(len(query_label)):
        P_ = evaluate_SDM(indexOfTopK_list[i], i, K)
        metric += P_
    metric = metric / len(query_label)
    if K in opts.K:
        print("metric{} = {:.2f}%".format(K, metric * 100))
    SDM_dict[K] = metric

MA_dict = {}
for meter in tqdm(range(1,101,1)):
    MA_K = 0
    for i in range(len(query_label)):
        MA_meter = evaluate_MA(indexOfTopK_list[i][0],i)
        if MA_meter<meter:
            MA_K+=1
    MA_K = MA_K/len(query_label)
    MA_dict[meter]=MA_K
        

with open("SDM@K(1,100).json", 'w') as F:
    json.dump(SDM_dict, F, indent=4)

with open("MA@K(1,100)", 'w') as F:
    json.dump(MA_dict, F, indent=4)
