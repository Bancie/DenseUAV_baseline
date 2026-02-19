from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
from math import cos, sin, pi
import random
import math


class Dataset_query(Dataset):
    """Placeholder query dataset for test-time loading.

    Provides the skeleton for loading query images from a file list.
    Subclasses or future implementations should override ``__getitem__``
    with the actual loading logic.

    Args:
        filename (str): Path to the file that lists query image paths.
        transformer (callable): Transform applied to each loaded image.
        basedir (str): Root directory prepended to relative image paths.
    """

    def __init__(self, filename, transformer, basedir):
        super(Dataset_query, self).__init__()
        self.filename = filename
        self.transformer = transformer
        self.basedir = basedir

    def __getitem__(self, item):
        """Return the sample at position ``item``.

        Args:
            item (int): Sample index.

        Returns:
            None: Not yet implemented; subclasses should provide loading logic.
        """
        pass

    def __len__(self):
        """Return the number of query samples.

        Returns:
            int: Length of this dataset instance.
        """
        return len(self)


class Query_transforms(object):
    """Prepend a horizontally-mirrored pad to a query image.

    Pads the left side of an image by mirroring the first ``pad`` columns,
    then crops the result back to ``size`` pixels wide.  This is used to
    handle the wraparound boundary condition for panoramic/cylindrical UAV
    query images.

    Args:
        pad (int, optional): Number of columns to mirror and prepend.
            Defaults to ``20``.
        size (int, optional): Output width in pixels after prepending.
            Defaults to ``256``.
    """

    def __init__(self, pad=20, size=256):
        self.pad = pad
        self.size = size

    def __call__(self, img):
        """Apply the mirror-pad transform to a PIL image.

        Args:
            img (PIL.Image.Image): Input RGB image.

        Returns:
            PIL.Image.Image: Image with the mirror-padded left border,
            cropped to width ``size``.
        """
        img_ = np.array(img).copy()
        img_part = img_[:, 0:self.pad, :]
        img_flip = cv2.flip(img_part, 1)  # horizontal mirror
        image = np.concatenate((img_flip, img_), axis=1)
        image = image[:, 0:self.size, :]
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return image


class CenterCrop(object):
    """Crop an image to a square by removing equal margins from the longer side.

    Computes the largest square that fits in the centre of the image and
    returns it.  The aspect ratio is preserved only in the sense that no
    scaling is applied—pixels are simply discarded from the longer axis.

    Note:
        This transform calls ``cv2.imshow`` during execution, which is
        intended for debugging purposes only.
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """Crop the image to a centre-aligned square region.

        Args:
            img (PIL.Image.Image): Input RGB image of arbitrary aspect ratio.

        Returns:
            PIL.Image.Image: Square-cropped image.

        Raises:
            AssertionError: If the resulting crop is not square.
        """
        img_ = np.array(img).copy()
        h, w, c = img_.shape
        min_edge = min((h, w))
        if min_edge == h:
            edge_lenth = int((w - min_edge) / 2)
            new_image = img_[:, edge_lenth:w - edge_lenth, :]
        else:
            edge_lenth = int((h - min_edge) / 2)
            new_image = img_[edge_lenth:h - edge_lenth, :, :]
        assert new_image.shape[0] == new_image.shape[1], "the shape is not correct"
        cv2.imshow("query", cv2.resize(new_image, (512, 512)))
        image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image


class RotateAndCrop(object):
    """Randomly rotate a satellite image and extract a square crop via perspective warp.

    With probability ``rate`` the transform selects a random rotation angle,
    computes a quadrilateral inscribed in the image circle at that angle, and
    applies a perspective warp to map the quadrilateral to a canonical
    ``output_size`` square.  This simulates the effect of a drone viewing the
    same location from a different heading.

    Args:
        rate (float): Probability of applying the rotation-crop.  When a
            uniform random sample exceeds ``rate`` the original image is
            returned unchanged.
        output_size (tuple[int, int], optional): ``(height, width)`` of the
            output square.  Defaults to ``(512, 512)``.
        rotate_range (int, optional): Upper bound (exclusive) for the
            random rotation angle in degrees.  Defaults to ``360``.
    """

    def __init__(self, rate, output_size=(512, 512), rotate_range=360):
        self.rate = rate
        self.output_size = output_size
        self.rotate_range = rotate_range

    def __call__(self, img):
        """Apply the random rotate-and-crop augmentation.

        Args:
            img (PIL.Image.Image): Input RGB image.

        Returns:
            PIL.Image.Image: Either the perspective-warped crop (with
            probability ``rate``) or the original image unchanged.
        """
        img_ = np.array(img).copy()

        def getPosByAngle(img, angle):
            h, w, c = img.shape
            y_center = h // 2
            x_center = w//2
            r = h // 2
            angle_lt = angle - 45
            angle_rt = angle + 45
            angle_lb = angle + 135
            angle_rb = angle + 225
            angleList = [angle_lt, angle_rt, angle_lb, angle_rb]
            pointsList = []
            for angle in angleList:
                x1 = x_center + r * cos(angle * pi / 180)
                y1 = y_center + r * sin(angle * pi / 180)
                pointsList.append([x1, y1])
            pointsOri = np.float32(pointsList)
            pointsListAfter = np.float32(
                [[0, 0], [0, self.output_size[0]], [self.output_size[0], self.output_size[1]], [self.output_size[1], 0]])
            M = cv2.getPerspectiveTransform(pointsOri, pointsListAfter)
            res = cv2.warpPerspective(
                img, M, (self.output_size[0], self.output_size[1]))
            return res

        if np.random.random() > self.rate:
            image = img
        else:
            angle = int(np.random.random()*self.rotate_range)
            new_image = getPosByAngle(img_, angle)
            image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image


class RandomCrop(object):
    """Randomly crop a fraction of pixels from each border of an image.

    Selects a random inset from each edge proportional to ``rate`` and
    returns the interior crop.  Unlike ``torchvision.transforms.RandomCrop``
    this operates in proportion to the image dimensions rather than in
    absolute pixels.

    Args:
        rate (float, optional): Maximum fraction of width/height that can be
            removed from each side.  Defaults to ``0.2``.
    """

    def __init__(self, rate=0.2):
        self.rate = rate

    def __call__(self, img):
        """Apply the random proportional crop.

        Args:
            img (PIL.Image.Image): Input RGB image.

        Returns:
            PIL.Image.Image: Cropped image with random borders removed.
        """
        img_ = np.array(img).copy()
        h, w, c = img_.shape
        random_width = int(np.random.random()*self.rate*w)
        random_height = int(np.random.random()*self.rate*h)
        x_l = random_width
        x_r = w-random_width
        y_l = random_height
        y_r = h-random_height
        new_image = img_[y_l:y_r, x_l:x_r, :]
        image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image


class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.

    Implements the *Random Erasing Data Augmentation* technique described in:
    Zhong et al., "Random Erasing Data Augmentation", AAAI 2020.
    See https://arxiv.org/pdf/1708.04896.pdf.

    The erased pixels are filled with the per-channel mean of the image,
    which empirically outperforms filling with a fixed constant.

    Args:
        probability (float): Probability that the erasing operation is
            applied to a given image.  Defaults to ``0.5``.
        sl (float): Minimum proportion of the erased area relative to the
            full image area.  Defaults to ``0.02``.
        sh (float): Maximum proportion of the erased area relative to the
            full image area.  Defaults to ``0.3``.
        r1 (float): Minimum aspect ratio of the erased region; the maximum
            is ``1 / r1``.  Defaults to ``0.3``.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.3, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """Apply random erasing to a PIL image.

        Attempts up to 100 times to find a rectangle whose area and aspect
        ratio fall within the specified ranges.  If a valid rectangle is
        found the region is filled with the per-channel image mean and the
        modified image is returned.  If no valid rectangle is found within
        100 attempts the original image is returned unchanged.

        Args:
            img (PIL.Image.Image): Input RGB image (may be a ``torch.Tensor``
                if used after ``ToTensor``; operates on numpy arrays
                internally).

        Returns:
            PIL.Image.Image: Image with one erased rectangle, or the original
            image if erasing was skipped or no valid region was found.
        """
        if random.uniform(0, 1) > self.probability:
            return img

        img_ = np.array(img).copy()

        mean = np.mean(np.mean(img_, 0), 1)

        for _ in range(100):
            area = img_.shape[0] * img_.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_.shape[1] and h < img_.shape[0]:
                x1 = random.randint(0, img_.shape[1] - h)
                y1 = random.randint(0, img_.shape[0] - w)

                img_[x1:x1+h, y1:y1+w, 0] = mean[0]
                img_[x1:x1+h, y1:y1+w, 1] = mean[1]
                img_[x1:x1+h, y1:y1+w, 2] = mean[2]
                img = Image.fromarray(img_.astype('uint8')).convert('RGB')
                return img

        return img


if __name__ == '__main__':
    image_path = "/home/ming.dai/workspace/dataset/DenseUAV/data_2022/test/drone/002810/H80.JPG"
    image = Image.open(image_path)

    re = RandomErasing(probability=1.0)

    from torchvision import transforms
    ra = transforms.RandomAffine(180)

    rac = RotateAndCrop(rate=1.0)

    target_dir = "/home/ming.dai/workspace/code/DenseUAV/visualization/rotate_crop"
    import os
    os.makedirs(target_dir, exist_ok=True)
    for ind in range(10):
        image_ = rac(image)
        image_ = np.array(image_)
        image_ = image_[:, :, [2, 1, 0]]
        h, w = image_.shape[:2]
        image_ = cv2.circle(
            image_.copy(), (int(w/2), int(h/2)), 3, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(target_dir, "{}.jpg".format(ind)), image_)
