from torchvision import transforms
from .Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from .autoaugment import ImageNetPolicy
import torch
from .queryDataset import RotateAndCrop, RandomCrop, RandomErasing


def make_dataset(opt):
    """Build the training DataLoader with view-specific augmentation pipelines.

    Constructs separate augmentation transform lists for drone (UAV) and satellite
    images based on the flags in ``opt``.  Augmentations are applied selectively
    per view using ``opt.rr`` (rotate-and-crop), ``opt.ra`` (random affine),
    ``opt.re`` (random erasing), and ``opt.cj`` (color jitter).  When
    ``opt.DA`` is ``True``, ImageNet AutoAugment is prepended to the drone
    pipeline.  All pipelines end with a standard ImageNet normalisation.

    Args:
        opt: Configuration namespace with the following required attributes:

            * ``data_dir`` (str): Root directory of the dataset.
            * ``h`` (int): Target image height in pixels.
            * ``w`` (int): Target image width in pixels.
            * ``pad`` (int): Edge-padding size applied before random crop.
            * ``batchsize`` (int): Number of samples per batch.
            * ``sample_num`` (int): Number of times each class index is
              repeated per epoch by the sampler.
            * ``num_worker`` (int): Number of DataLoader worker processes.
            * ``erasing_p`` (float): Probability for ``RandomErasing``.
            * ``rr`` (str): Comma-separated view names (``"uav"``,
              ``"satellite"``) that receive ``RotateAndCrop``.
            * ``ra`` (str): View names that receive ``RandomAffine(180)``.
            * ``re`` (str): View names that receive ``RandomErasing``.
            * ``cj`` (str): View names that receive ``ColorJitter``.
            * ``DA`` (bool): Whether to apply ImageNet AutoAugment to the
              drone pipeline.

    Returns:
        tuple:
            * **dataloaders** (torch.utils.data.DataLoader): Training
              DataLoader that yields paired ``([satellite_batch, ids],
              [drone_batch, ids])`` tuples.
            * **class_names** (list[str]): Sorted list of geo-location class
              folder names.
            * **dataset_sizes** (dict[str, int]): Effective dataset size per
              view (``len(dataset) * sample_num``) keyed by ``"satellite"``
              and ``"drone"``.
    """
    transform_train_list = []
    transform_satellite_list = []
    if "uav" in opt.rr:
        transform_train_list.append(RotateAndCrop(0.5))
    if "satellite" in opt.rr:
        transform_satellite_list.append(RotateAndCrop(0.5))
    transform_train_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_satellite_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),
                          interpolation=3),  # Image.BICUBIC
    ]

    if "uav" in opt.ra:
        transform_train_list = transform_train_list + \
            [transforms.RandomAffine(180)]
    if "satellite" in opt.ra:
        transform_satellite_list = transform_satellite_list + \
            [transforms.RandomAffine(180)]

    if "uav" in opt.re:
        transform_train_list = transform_train_list + \
            [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.re:
        transform_satellite_list = transform_satellite_list + \
            [RandomErasing(probability=opt.erasing_p)]

    if "uav" in opt.cj:
        transform_train_list = transform_train_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
    if "satellite" in opt.cj:
        transform_satellite_list = transform_satellite_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]

    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list

    last_aug = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list += last_aug
    transform_satellite_list += last_aug
    transform_val_list += last_aug

    print(transform_train_list)
    print(transform_satellite_list)

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)}

    # custom Dataset
    image_datasets = Dataloader_University(
        opt.data_dir, transforms=data_transforms)
    samper = Sampler_University(
        image_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                              sampler=samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    dataset_sizes = {x: len(image_datasets) *
                     opt.sample_num for x in ['satellite', 'drone']}
    class_names = image_datasets.cls_names
    return dataloaders, class_names, dataset_sizes
