import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import glob


class Dataloader_University(Dataset):
    """Paired satellite-drone image dataset for the University-1652 benchmark.

    Each call to ``__getitem__`` returns one randomly sampled satellite image
    and one randomly sampled drone image that share the same geo-location
    class, together with the integer class index.  This design supports
    cross-view geo-localisation training where both views are available at
    every iteration.

    Directory structure expected under ``root``::

        root/
          satellite/
            <class_name>/
              *.jpg  (or any image format)
          drone/
            <class_name>/
              *.jpg

    Args:
        root (str): Path to the dataset root directory.
        transforms (dict[str, torchvision.transforms.Compose]): A mapping
            that must contain at least the keys ``"train"`` (applied to
            drone images) and ``"satellite"`` (applied to satellite images).
        names (list[str], optional): Sub-directory names to scan.
            Defaults to ``['satellite', 'drone']``.

    Attributes:
        cls_names (list[str]): Sorted list of class folder names.
        map_dict (dict[int, str]): Integer index to class-name mapping.
        dict_path (dict[str, dict[str, list[str]]]): Nested mapping from
            view name → class name → list of absolute image paths.
    """

    def __init__(self, root, transforms, names=['satellite', 'drone']):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names = names
        # Collect all image paths grouped by view and class name.
        # Structure: {view_name: {class_name: [absolute_path, ...]}}
        dict_path = {}
        for name in names:
            dict_ = {}
            # old code: gather all entries (including non-image files like .DS_Store)
            # for cls_name in os.listdir(os.path.join(root, name)):
            #     cls_dir = os.path.join(root, name, cls_name)
            #     if not os.path.isdir(cls_dir):
            #         continue
            #     img_list = os.listdir(cls_dir)
            #     img_path_list = [os.path.join(
            #         root, name, cls_name, img) for img in img_list]
            #     dict_[cls_name] = img_path_list

            # new code: only treat subdirectories as class folders and only keep image files
            for cls_name in os.listdir(os.path.join(root, name)):
                cls_dir = os.path.join(root, name, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                # old: chỉ cho phép ảnh định dạng phổ biến JPG/PNG/BMP
                # img_list = [
                #     img for img in os.listdir(cls_dir)
                #     if os.path.splitext(img.lower())[1] in {".jpg", ".jpeg", ".png", ".bmp"}
                # ]
                # new: thêm cả GeoTIFF để dùng trực tiếp .tif/.tiff
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                img_list = [
                    img for img in os.listdir(cls_dir)
                    if os.path.splitext(img.lower())[1] in valid_exts
                ]
                img_path_list = [
                    os.path.join(root, name, cls_name, img) for img in img_list
                ]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_

        # Build a bidirectional index ↔ class-name mapping.
        # new: only directories (skip .DS_Store etc.)
        cls_names = [x for x in os.listdir(os.path.join(root, names[0]))
                     if os.path.isdir(os.path.join(root, names[0], x))]
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    def sample_from_cls(self, name, cls_num):
        """Randomly sample one image from a given view and class.

        Args:
            name (str): View name (e.g. ``"satellite"`` or ``"drone"``).
            cls_num (str): Class folder name (geo-location identifier).

        Returns:
            PIL.Image.Image: The loaded image converted to RGB.
        """
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, index):
        """Return a (satellite_tensor, drone_tensor, class_index) triple.

        Args:
            index (int): Integer class index in ``[0, len(self) - 1]``.

        Returns:
            tuple:
                * **img_s** (torch.Tensor): Augmented satellite image tensor.
                * **img_d** (torch.Tensor): Augmented drone image tensor.
                * **index** (int): Integer class label.
        """
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_satellite(img)

        img = self.sample_from_cls("drone", cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s, img_d, index

    def __len__(self):
        """Return the total number of geo-location classes.

        Returns:
            int: Number of unique classes in the dataset.
        """
        return len(self.cls_names)


class DataLoader_Inference(Dataset):
    """Dataset for inference over a directory of GeoTIFF images.

    Scans ``root`` for all ``*.tif`` files and exposes them for forward
    passes.  The label for each sample is the filename stem (without
    extension).

    Args:
        root (str): Directory containing ``*.tif`` images.
        transforms (torchvision.transforms.Compose): Transform pipeline
            applied to every loaded image.

    Attributes:
        imgs (list[str]): Sorted list of absolute paths to ``.tif`` files.
        labels (list[str]): Corresponding filename stems used as identifiers.
    """

    def __init__(self, root, transforms):
        super(DataLoader_Inference, self).__init__()
        self.root = root
        self.imgs = glob.glob(root+"/*.tif")
        self.tranforms = transforms
        sorted(self.imgs)
        self.labels = [os.path.basename(img).split(".tif")[
            0] for img in self.imgs]

    def __getitem__(self, index):
        """Return a (transformed_tensor, label_str) pair for one image.

        Args:
            index (int): Sample index.

        Returns:
            tuple:
                * **tensor** (torch.Tensor): Transformed image tensor.
                * **label** (str): Filename stem of the source image.
        """
        img = Image.open(self.imgs[index])
        return self.tranforms(img), self.labels[index]

    def __len__(self):
        """Return the number of images in the directory.

        Returns:
            int: Total number of ``.tif`` files found.
        """
        return len(self.imgs)


class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.

    This sampler shuffles all class indices once per epoch and repeats each
    index ``sample_num`` times so that the DataLoader sees multiple random
    views of the same class within a single batch.

    Args:
        data_source (Dataset): Dataset whose length determines the index range.
        batchsize (int, optional): Batch size (stored for reference).
            Defaults to ``8``.
        sample_num (int, optional): Number of times each class index is
            repeated per epoch.  Defaults to ``4``.
    """

    def __init__(self, data_source, batchsize=8, sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        """Yield shuffled and repeated class indices for one epoch.

        Returns:
            iterator: An iterator over an array of length
            ``data_len * sample_num`` where each class index appears
            ``sample_num`` consecutive times after a global shuffle.
        """
        list = np.arange(0, self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list, self.sample_num, axis=0)
        return iter(nums)

    def __len__(self):
        """Return the total number of samples yielded per epoch.

        Returns:
            int: ``data_len * sample_num``.
        """
        return len(self.data_source)


def train_collate_fn(batch):
    """Collate a list of ``(satellite, drone, label)`` tuples into two batches.

    The DataLoader calls this function with a list whose length equals the
    batch size.  Each element is the output of
    ``Dataloader_University.__getitem__``.  The function stacks images and
    wraps labels into tensors, then returns separate satellite and drone
    sub-batches so the training loop can process each view independently.

    Args:
        batch (list[tuple]): List of ``(img_s, img_d, id)`` tuples where
            ``img_s`` and ``img_d`` are ``torch.Tensor`` images and ``id``
            is an integer class label.

    Returns:
        tuple:
            * **satellite_batch** (list): ``[stacked_satellite_imgs, ids]``
              where ``stacked_satellite_imgs`` has shape
              ``(B, C, H, W)`` and ``ids`` is a ``torch.int64`` tensor of
              shape ``(B,)``.
            * **drone_batch** (list): ``[stacked_drone_imgs, ids]`` with
              the same shape conventions as ``satellite_batch``.
    """
    img_s, img_d, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [torch.stack(img_s, dim=0), ids], [torch.stack(img_d, dim=0), ids]


if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list = {"satellite": transforms.Compose(transform_train_list),
                            "train": transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="/home/dmmm/University-Release/train",
                                     transforms=transform_train_list, names=['satellite', 'drone'])
    samper = Sampler_University(datasets, 8)
    dataloader = DataLoader(datasets, batch_size=8, num_workers=0,
                            sampler=samper, collate_fn=train_collate_fn)
    for data_s, data_d in dataloader:
        print()
