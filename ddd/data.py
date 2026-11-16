"""Image dataset and transforms. The transforms follow the public ViT-DD code, see
results/vitdd_recipe_check.md."""
import random
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CODES = [f"c{i}" for i in range(10)]


class _Blur:
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


class _Solarize:
    def __call__(self, img):
        return ImageOps.solarize(img)


def train_transform(size=224, color_jitter=0.3, three_augment=True, flip=True, random_crop=True):
    """With all four options off, the steps equal `eval_transform`: no random step (decision P15)."""
    crop = (transforms.RandomCrop(size, padding=4, padding_mode="reflect") if random_crop
            else transforms.CenterCrop(size))
    steps = [transforms.Resize(size, interpolation=InterpolationMode.BICUBIC), crop]
    if flip:
        steps.append(transforms.RandomHorizontalFlip())
    if three_augment:
        steps.append(transforms.RandomChoice([transforms.Grayscale(3), _Solarize(), _Blur()]))
    if color_jitter:
        steps.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    return transforms.Compose(steps)


def eval_transform(size=224, gray=False):
    """`gray` gives each image in gray scale with 3 channels, as the gray step of 3-Augment."""
    steps = [transforms.Resize(size, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size)]
    if gray:
        steps.append(transforms.Grayscale(3))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    return transforms.Compose(steps)


class StateFarmImages(Dataset):
    """Images of the label rows. The target is the index of the StateFarm code."""

    def __init__(self, rows, root, transform):
        self.paths = [Path(root) / "imgs" / "train" / r["classname"] / r["img"] for r in rows]
        self.targets = [CODES.index(r["classname"]) for r in rows]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), self.targets[i]
