
import math
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import numpy as np
import os


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2471, 0.2435, 0.2616])

CIFAR100_MEAN = np.array([0.5070758,  0.4865503,  0.44091913])
CIFAR100_STD = np.array([0.26733097, 0.25643396, 0.27614763])

TINYIMAGENET_MEAN = np.array([0.48024783, 0.44806924, 0.39754707])
TINYIMAGENET_STD = np.array([0.27698126, 0.2690698, 0.28208217])

GTSRB_MEAN = np.array([0.34169674, 0.31255573, 0.3215503])
GTSRB_STD = np.array([0.28010625, 0.26819786, 0.2746381])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

device = torch.device('cuda:0')     # use GPU


def set_config(obj, config_dic):
    for k, v in config_dic.items():
        setattr(obj, k, v)


class ConfigBase:
    def __init__(self, **kwargs):
        set_config(self, kwargs)


def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))


class MyCIFAR10(CIFAR10):
    def __init__(
            self,
            root,
            train,
            transform=None,
            target_transform=None,
            download=False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        # self.targets return torch.Tensor instead of list to be the same as MNIST
        self.targets = torch.LongTensor(self.targets)


class MyNumpyData(Dataset):
    def __init__(self, np_data, np_labels, preprocess=None):
        super().__init__()

        self.np_data = np_data
        self.np_labels = np_labels

        self.preprocess = preprocess
        self.targets = torch.LongTensor(self.np_labels)     # to be compatible with other datasets

    def __len__(self):
        return self.np_data.shape[0]

    def __getitem__(self, idx):

        x, y = torch.FloatTensor(self.np_data[idx]), self.np_labels[idx]

        if self.preprocess is not None:
            x = self.preprocess(x)

        return x, y

    def add_extra_data(self, extra_data, extra_labels):

        if self.np_data is None:
            self.np_data = extra_data
            self.np_labels = extra_labels
            return

        self.np_data = np.concatenate([self.np_data, extra_data], axis=0)
        self.np_labels = np.concatenate([self.np_labels, extra_labels])


def save_image_batch(imgs, labels, num_classes, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs =imgs.detach().cpu().numpy()
    imgs = (np.clip(imgs, a_min=0, a_max=1)*255).astype('uint8')

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_labelled_images(imgs, labels, num_classes).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, (img, y) in enumerate(zip(imgs, labels)):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename+f'_{idx}_label_{y}.png')

    return


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def pack_labelled_images(images, labels, num_classes, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    uniq_labels = np.unique(labels)
    col = max([(labels == x).sum() for x in uniq_labels])
    row = num_classes

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    img_cnt = {x: 0 for x in uniq_labels}

    for idx, (img, y) in enumerate(zip(images, labels)):
        h = y * (H + padding)

        w = (img_cnt[y]) * (W + padding)
        img_cnt[y] += 1

        pack[:, h:h + H, w:w + W] = img

    return pack









