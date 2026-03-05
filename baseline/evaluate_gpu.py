"""GPU-accelerated CMC and mAP evaluation for DenseUAV geo-localization.

Loads pre-computed query and gallery features from ``pytorch_result_1.mat``
(produced by ``test.py``), computes per-query Average Precision and Cumulative
Matching Characteristic (CMC) curves using GPU dot-product similarity, and
prints Recall@K and mAP metrics.

Example:
    Run standard single-query evaluation::

        python evaluate_gpu.py

    The script expects ``pytorch_result_1.mat`` in the current working
    directory.  Optionally, if ``multi_query.mat`` is present, multi-query
    evaluation is also performed.

Outputs:
    Prints to stdout::

        Recall@1:<val> Recall@5:<val> Recall@10:<val> Recall@top1:<val> AP:<val>
"""

import scipy.io
import torch
import numpy as np
#import time
import os

#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    """Compute Average Precision and CMC curve for a single query.

    Ranks all gallery samples by cosine similarity to the query, removes
    junk samples (label ``-1``), then delegates to :func:`compute_mAP`.

    Args:
        qf (torch.Tensor): Query feature vector of shape ``(D,)``, residing
            on GPU.
        ql (int): Integer class label of the query.
        gf (torch.Tensor): Gallery feature matrix of shape ``(N, D)``,
            residing on GPU.
        gl (numpy.ndarray): Integer class labels for all gallery samples,
            shape ``(N,)``.

    Returns:
        tuple:
            - ap (float): Average Precision for this query.
            - cmc (torch.IntTensor): CMC curve of length ``N``; ``cmc[k]``
              is 1 if the correct match appears in the top ``k+1`` results.
    """
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index
    #print(good_index)
    #print(index[0:10])
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    """Compute Average Precision and CMC for a single ranked list.

    Junk samples are excluded from the ranked list before computing metrics.
    AP is computed via the trapezoidal rule over the precision–recall curve.

    Args:
        index (numpy.ndarray): Gallery indices sorted by descending similarity,
            shape ``(N,)``.
        good_index (numpy.ndarray): Indices of true-positive gallery samples,
            shape ``(G, 1)`` or ``(G,)``.
        junk_index (numpy.ndarray): Indices of junk gallery samples to ignore,
            shape ``(J, 1)`` or ``(J,)``.

    Returns:
        tuple:
            - ap (float): Average Precision; ``0.0`` if ``good_index`` is
              empty.
            - cmc (torch.IntTensor): Binary CMC array of length ``N``;
              ``cmc[k] = 1`` means a correct match was found within the top
              ``k+1`` retrieved results.  ``cmc[0] = -1`` signals an empty
              ``good_index``.
    """
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    # mask = np.in1d(index, junk_index, invert=True)
    mask = np.isin(index, junk_index, invert=True) # np.in1d now replaced with np.isin
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

# ######################################################################
# result = scipy.io.loadmat('pytorch_result_1.mat')
# query_feature = torch.FloatTensor(result['query_f'])
# query_label = result['query_label'][0]
# gallery_feature = torch.FloatTensor(result['gallery_f'])
# gallery_label = result['gallery_label'][0]
# multi = os.path.isfile('multi_query.mat')

# # New: use CPU when CUDA is not available (avoids AssertionError on PyTorch CPU-only build)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if multi:
#     m_result = scipy.io.loadmat('multi_query.mat')
#     mquery_feature = torch.FloatTensor(m_result['mquery_f'])
#     mquery_label = m_result['mquery_label'][0]
#     # mquery_feature = mquery_feature.cuda()
#     mquery_feature = mquery_feature.to(device)  # New: to(device) for CPU fallback

# # query_feature = query_feature.cuda(0)
# # gallery_feature = gallery_feature.cuda(0)
# query_feature = query_feature.to(device)   # New: to(device) for CPU fallback
# gallery_feature = gallery_feature.to(device)

# print(query_feature.shape)
# print(gallery_feature.shape)
# #print(gallery_feature[0,:])
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# # print(query_label)
# for i in range(len(query_label)):
#     ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
#     if CMC_tmp[0]==-1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     # print(i, CMC_tmp[0])

# CMC = CMC.float()
# CMC = CMC/len(query_label) #average CMC
# print(round(len(gallery_label)*0.01))
# print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f'%(CMC[0]*100,CMC[4]*100,CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))

# # multiple-query
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# if multi:
#     for i in range(len(query_label)):
#         mquery_index1 = np.argwhere(mquery_label==query_label[i])
#         mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
#         mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
#         mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
#         ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#         if CMC_tmp[0]==-1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#         #print(i, CMC_tmp[0])
#     CMC = CMC.float()
#     CMC = CMC/len(query_label) #average CMC
#     print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
