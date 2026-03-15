"""GPU-accelerated CMC and mAP evaluation for DenseUAV geo-localization.

Loads pre-computed query and gallery features from Modal Dict (key from
``test_modal.py``), computes per-query Average Precision and Cumulative
Matching Characteristic (CMC) curves using GPU/CPU dot-product similarity,
and prints Recall@K and mAP metrics. Data source is Modal Dict only (no .mat).

Example:
    python evaluate_gpu.py --run_name Swinv2S_256 --wandb_mode online
    python evaluate_gpu.py --result_key ViTS_224_mode_1

Outputs:
    Prints to stdout::

        Recall@1:<val> Recall@5:<val> Recall@10:<val> Recall@top1:<val> AP:<val>
"""

import argparse
import sys
# import time  # old code
import numpy as np
import torch

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


def _normalize_labels(labels):
    """Ensure labels are 1D numpy array (list or array from Modal Dict)."""
    a = np.atleast_1d(labels)
    return a.flatten()


def load_result_from_modal_dict(result_key, dict_name):
    """Load query/gallery result from Modal Dict only (no .mat).

    Returns dict with keys: query_f, query_label, gallery_f, gallery_label (numpy).
    Labels are normalized to 1D numpy arrays.
    """
    import modal
    d = modal.Dict.from_name(dict_name, create_if_missing=False)
    result = d[result_key]
    query_f = np.asarray(result["query_f"])
    gallery_f = np.asarray(result["gallery_f"])
    query_label = _normalize_labels(result["query_label"])
    gallery_label = _normalize_labels(result["gallery_label"])
    return {
        "query_f": query_f,
        "gallery_f": gallery_f,
        "query_label": query_label,
        "gallery_label": gallery_label,
    }


def main():
    parser = argparse.ArgumentParser(description="CMC and mAP evaluation (Modal Dict only)")
    parser.add_argument("--run_name", default="", type=str, help="Run name; key becomes run_name_mode_1")
    parser.add_argument("--result_key", default="", type=str, help="Modal Dict key (overrides run_name if set)")
    parser.add_argument("--dict_name", default="denseuav-test-results", type=str, help="Modal Dict name")
    parser.add_argument("--wandb_project", default="denseuav-eval", type=str, help="W&B project (empty to skip)")
    parser.add_argument("--wandb_mode", default="disabled", type=str, help="W&B mode: disabled, online, offline")
    args = parser.parse_args()

    result_key = args.result_key or (f"{args.run_name}_mode_1" if args.run_name else "")
    if not result_key:
        sys.exit("Error: provide --run_name or --result_key (e.g. --run_name Swinv2S_256 or --result_key ViTS_224_mode_1)")
    data = load_result_from_modal_dict(result_key, args.dict_name)

    query_feature = torch.FloatTensor(data["query_f"])
    gallery_feature = torch.FloatTensor(data["gallery_f"])
    query_label = data["query_label"]
    gallery_label = data["gallery_label"]

    # new code: use CPU when CUDA is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    recall_1 = CMC[0].item() * 100
    recall_5 = CMC[4].item() * 100
    recall_10 = CMC[9].item() * 100
    recall_top1 = CMC[round(len(gallery_label) * 0.01)].item() * 100
    ap_pct = ap / len(query_label) * 100

    print(
        "Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f"
        % (recall_1, recall_5, recall_10, recall_top1, ap_pct)
    )

    # new code: W&B logging
    use_wandb = getattr(args, "wandb_mode", "disabled") != "disabled"
    wandb_run = None
    if use_wandb and (args.wandb_project or "denseuav-eval"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project or "denseuav-eval",
                name=result_key,
                config={
                    "result_key": result_key or None,
                    "run_name": args.run_name or None,
                    "dict_name": args.dict_name,
                },
                mode=args.wandb_mode,
            )
            wandb_run.log(
                {
                    "Recall@1": recall_1,
                    "Recall@5": recall_5,
                    "Recall@10": recall_10,
                    "Recall@top1": recall_top1,
                    "AP": ap_pct,
                },
                step=0,
            )
        except Exception as e:
            print(f"WandB init failed: {e}")
            wandb_run = None
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
