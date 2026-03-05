import evaluate_gpu
import math
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import yaml

from tool.utils import load_network

class UAVnonGPS:
    def __init__(self, EmbeddingDatabase, model, batchsize, mode):
        """
        Initialize the UAVnonGPS class.

        Parameters
        ----------
        EmbeddingDatabase : str
            A string that contains the path to the embedding database file. Example: `pytorch_result_1.mat`
        model : str
            A string that contains the path to the model file. Example: `net_119.pth`
        Batchsize : int
            A integer that contains the batch size for the model.
        Mode : int
            A integer that contains the mode of the model. 1 for drone-to-satellite, 2 for satellite-to-drone.

        Notes
        ------
        The embedding database file is a `.mat` file that contains the embedding vectors of the gallery images used for retrieval on a query image. It is a dictionary with the keys: `{'gallery_f': gallery_feature, 'gallery_label': gallery_label, 'gallery_path': gallery_path}`.
        """
        self.EmbeddingDatabase = EmbeddingDatabase
        self.Model = model
        self.Batchsize = batchsize
        self.Mode = mode

        # Internal cache: model & preprocessing, only init once
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._opt = None
        self._query_transform = None

    def _load_model_and_transform(self):
        """
        Internal helper to lazily load the trained model and query transform.

        This mirrors the configuration + model loading flow in `test.py`,
        but uses the parameters provided in `__init__` instead of CLI args.
        """
        if self._model is not None and self._query_transform is not None:
            return

        # Load training config from opts.yaml (same as test.py)
        config_path = "opts.yaml"
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

        # Base options (equivalent to argparse defaults in test.py)
        opt = SimpleNamespace(
            gpu_ids="0",
            test_dir=".",  # not needed here but kept for compatibility
            name="resnet",
            checkpoint=self.Model,
            batchsize=self.Batchsize,
            h=256,
            w=256,
            ms="1",
            mode=self.Mode,
            num_worker=0,
            box_vis=False,
        )

        # Merge with training config (opts.yaml can override / add fields)
        for cfg, value in config.items():
            setattr(opt, cfg, value)

        self._opt = opt

        # Load network using the same util as test.py
        model = load_network(opt)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.to(self._device)

        self._model = model

        # Build query transforms exactly like `data_query_transforms` in test.py
        self._query_transform = transforms.Compose(
            [
                transforms.Resize((opt.h, opt.w), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _fliplr(img: torch.Tensor) -> torch.Tensor:
        """Horizontal flip for a 4D tensor (N, C, H, W)."""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        return img.index_select(3, inv_idx)
        
    def query_embedding(self, query_image):
        """
        Embed the query image using the model specified in the `Model` parameter.

        Parameters
        ----------
        query_image : str
            A string that contains the path to the query image. Example: `data/DenseUAV_data/test/query_drone/000000000000.jpg`

        Returns
        -------
        query_embedding : torch.Tensor
            A tensor that contains the L2-normalised embedding of the query image.

        Notes
        -----
        - Use the same network architecture, image size, and normalization as in `test.py`.
        - Use the parameters passed to `__init__`: `Model` (checkpoint), `Batchsize` (not critical for a single image but keep for consistency), and `Mode` (1: drone->satellite, 2: satellite->drone) to select the appropriate branch.
        """
        # Lazy init model & transform (chỉ load 1 lần, tái dùng cho các lần gọi sau)
        self._load_model_and_transform()

        # Load & preprocess query image
        img = Image.open(query_image).convert("RGB")
        img = self._query_transform(img)  # (C, H, W)
        img = img.unsqueeze(0)  # (1, C, H, W)

        device = self._device
        model = self._model
        opt = self._opt

        # Determine which branch to use:
        # - mode == 1: query is drone  -> view_index = 3 (drone branch)
        # - mode == 2: query is satellite -> view_index = 1 (satellite branch)
        if self.Mode == 1:
            view_index = 3
        elif self.Mode == 2:
            view_index = 1
        else:
            raise ValueError(f"Unsupported mode {self.Mode}. Expected 1 or 2.")

        # Test-time augmentation: original + horizontal flip (same logic as extract_feature in test.py)
        with torch.no_grad():
            img = img.to(device)
            ff = None
            for i in range(2):
                if i == 1:
                    img = self._fliplr(img)

                input_img = Variable(img)
                if view_index == 1:
                    outputs, _ = model(input_img, None)
                elif view_index == 3:
                    _, outputs = model(None, input_img)
                else:
                    raise ValueError(f"Unsupported view_index {view_index}.")

                outputs = outputs[1]
                if ff is None:
                    ff = outputs
                else:
                    ff = ff + outputs

            # L2 normalisation (giữ nguyên logic như test.py)
            if len(ff.shape) == 3:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * math.sqrt(
                    getattr(opt, "block", ff.shape[-1])
                )
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

        # Trả về tensor 1-D (D,) trên CPU cho tiện xử lý tiếp
        return ff.squeeze(0).cpu()

    def ranking_by_similarity(self, query_embedding):
        """
        Rank gallery trong EmbeddingDatabase theo similarity với query embedding.

        Logic giống evaluate_gpu.evaluate(): dùng dot-product (cosine vì đã L2-norm)
        giữa query và gallery_f, xếp theo score giảm dần.

        Parameters
        ----------
        query_embedding : torch.Tensor
            Vector embedding 1-D (D,) trả về từ method `query_embedding(...)`.

        Returns
        -------
        ranking : list of dict
            Danh sách gallery đã xếp hạng, mỗi phần tử là dict với keys:
            - "gallery_f": feature vector (numpy),
            - "gallery_label": label (int),
            - "gallery_path": đường dẫn ảnh (str),
            - "score": similarity với query (float),
            - "rank": thứ hạng 1-based (int).
        """
        # Load EmbeddingDatabase (.mat): gallery_f, gallery_label, gallery_path
        db = scipy.io.loadmat(self.EmbeddingDatabase)
        gallery_f = db["gallery_f"]   # (N, D) hoặc (D, N) tùy .mat
        gallery_label = db["gallery_label"]
        gallery_path = db["gallery_path"]

        # Chuẩn hóa shape: gallery_label và gallery_path thường (1, N) hoặc (N,) từ .mat
        if gallery_label.ndim > 1:
            gallery_label = gallery_label.flatten()
        if gallery_path.ndim > 1:
            gallery_path = gallery_path.flatten()
        # gallery_path có thể là array of array of str; lấy str
        paths = []
        for p in gallery_path:
            p = p if isinstance(p, str) else (p.item() if hasattr(p, "item") else str(p).strip())
            paths.append(p)

        # Đảm bảo gallery_f là (N, D)
        if gallery_f.shape[0] != len(paths):
            gallery_f = gallery_f.T

        # Tensor: query (D,) và gallery (N, D)
        qf = query_embedding.float()
        if qf.dim() == 1:
            qf = qf.unsqueeze(1)  # (D, 1)
        gf = torch.from_numpy(gallery_f).float()

        # Dot-product similarity (như evaluate_gpu.evaluate: score = torch.mm(gf, query))
        score = torch.mm(gf, qf)
        score = score.squeeze(1)  # (N,)
        score_np = score.numpy()

        # Xếp theo score giảm dần
        index = np.argsort(score_np)[::-1]

        # Build danh sách EmbeddingDatabase đã rank
        ranking = []
        for r, idx in enumerate(index, start=1):
            ranking.append({
                "gallery_f": gallery_f[idx],
                "gallery_label": int(gallery_label[idx]),
                "gallery_path": paths[idx],
                "score": float(score_np[idx]),
                "rank": r,
            })
        return ranking